import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
from typing import Dict, Tuple

os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models_pretrain
import tools
import wrappers

import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()


class DreamerPretrain(nn.Module):

  def __init__(self, config, logger, offline_datasets, action_space):
    super(DreamerPretrain, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._metrics = {}
    self._step = 0
    
    # Schedules
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    
    self._offline_datasets = offline_datasets
    self._wm = models_pretrain.WorldModelTeacher(self._step, config, action_space)

  def train_step(self, data, task_id):
    """Single training step on data from task_id"""
    metrics = {}
    post, context, mets = self._wm._train(data, task_id=task_id)
    metrics.update(mets)
    
    for name, value in metrics.items():
      if name not in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)
    
    return post, context, metrics
  
  def _shared_params_for_conflicts(self, include_heads: bool = False):
    """Choose which shared params define your 'update direction' space."""
    params = []
    params += list(self._wm.encoder.parameters())     # shared encoder
    params += list(self._wm.dynamics.parameters())    # shared dynamics
    if include_heads:
      # heads are task-conditional but weights are shared modules
      for _, head in self._wm.heads.items():
        params += list(head.parameters())
    return params
  
  def gather_losses_and_conflicts(self, include_heads: bool = False):
    """
    Returns:
      per_task: {tid: {'total': float, '<head>_loss': float, ...}}
      conflicts: {(tid_i, tid_j): cosine_sim}
    """
    self._wm.eval()
    per_task_numeric: Dict[int, Dict[str, float]] = {}
    per_task_loss_tensor: Dict[int, torch.Tensor] = {}

    # 1) collect one batch per task and compute the loss tensor
    for tid in range(len(self._config.source_tasks)):
      data = next(self._offline_datasets[tid])
      out = self._wm._train(
        data, task_id=tid,
        optimize=False,
        return_loss_tensor=True
      )
      # unpack 4-tuple (see the change above)
      _post, _ctx, mets, loss_tensor = out

      # numeric snapshot for logging
      # pick up your existing per-head logs (they end with '_loss') and total
      snap = {}
      for k, v in mets.items():
        if k.endswith('_loss'):
          snap[k] = float(v)
      # total (prefer model_loss if present)
      snap['total'] = float(mets.get('model_loss', sum(val for k, val in snap.items())))
      per_task_numeric[tid] = snap

      # raw tensor for gradients
      per_task_loss_tensor[tid] = loss_tensor

    # 2) compute conflicts on chosen shared params
    shared_params = self._shared_params_for_conflicts(include_heads=include_heads)
    for p in shared_params:
      p.requires_grad_(True)
    conflicts = tools.compute_gradient_conflicts(per_task_loss_tensor, shared_params)

    # 3) (optional) push a few scalars into your logger buffers
    print(per_task_numeric)
    for tid, d in per_task_numeric.items():
      self._metrics.setdefault(f't{tid}/total_loss_snapshot', []).append(d['total'])
    for (i, j), sim in conflicts.items():
      self._metrics.setdefault(f'grad_cos_sim_{i}_{j}', []).append(sim)

    return per_task_numeric, conflicts


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == "metaworld":
      task = "-".join(task.split("_"))
      env = wrappers.MetaWorld(
          task,
          config.seed,
          config.action_repeat,
          config.size,
          config.camera,
      )
      env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  env = wrappers.RewardObs(env)
  return env


def main(config):
  import setup_utils as setup_utils
  device_mapper = setup_utils.setup_device(config)
  config.device = str(config.device)
  
  tools.set_seed_everywhere(config.seed)
  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.act = getattr(torch.nn, config.act)
  
  # Save configuration and command info
  setup_utils.save_config_to_json(config, logdir)
  setup_utils.save_cmd(str(logdir))
  setup_utils.save_git(str(logdir))

  print('Logdir', logdir)
  logger = tools.Logger(logdir, 0)

  print('Create dummy env to get action space.')
  suite, task = config.task.split('_', 1)
  if suite == "metaworld":
    task = "-".join(task.split("_"))
    env = wrappers.MetaWorld(task, config.seed, config.action_repeat, config.size, config.camera)
    env = wrappers.NormalizeActions(env)
  elif suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(f"Add environment setup for {suite}")
  
  acts = env.action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  print('Load offline datasets for pretraining.')
  offline_datasets = []
  
  if hasattr(config, 'source_task_dirs') and config.source_task_dirs:
    # Load from explicitly specified source directories
    for task_id, task_dir in enumerate(config.source_task_dirs):
      task_dir = pathlib.Path(task_dir).expanduser()
      task_name = config.source_tasks[task_id]
      print(f'Loading task {task_id}: {task_name} from {task_dir}')
      task_eps = tools.load_episodes(task_dir, limit=config.dataset_size)
      task_dataset = make_dataset(task_eps, config)
      offline_datasets.append(task_dataset)
  else:
    raise ValueError("Please specify source_task_dirs in config")
  
  print(f'Loaded {len(offline_datasets)} offline task datasets.')

  # Initialize pretrain agent
  print('Initialize DreamerPretrain agent.')
  agent = DreamerPretrain(config, logger, offline_datasets, acts).to(config.device)
  # agent.requires_grad_(requires_grad=False)
  
  if (logdir / 'teacher_model.pt').exists():
    print('Loading existing teacher model checkpoint.')
    checkpoint = torch.load(logdir / 'teacher_model.pt')
    agent._wm.load_state_dict(checkpoint)

  # Pretraining loop
  print('Start pretraining teacher models.')
  num_iterations = getattr(config, 'pretrain_iterations', 200000)
  
  for i in range(num_iterations):
    
    # Evaluation
    if i % 1000 == 0:
      print(f'Step {i}: Evaluation')
      eval_task_id = np.random.randint(0, len(config.source_tasks))
      eval_data = next(offline_datasets[eval_task_id])
      video_pred = agent._wm.video_pred(eval_data, task_id=eval_task_id)
      logger.video(f'pretrain_task_{eval_task_id}', to_np(video_pred))
    
    # Training
    task_id = np.random.randint(0, len(config.source_tasks))
    
    data = next(offline_datasets[task_id])
    agent.train_step(data, task_id=task_id)
    
    # Logging
    if i % config.log_every == 0 and i > 0:
      for name, values in agent._metrics.items():
        logger.scalar(name, float(np.mean(values)))
        
      # NEW: snapshot losses & gradient conflicts
      per_task, conflicts = agent.gather_losses_and_conflicts(include_heads=False)
      # Log a compact set of summary scalars
      for tid, d in per_task.items():
        logger.scalar(f't{tid}/total_loss_snapshot', d['total'])
      for (a, b), sim in conflicts.items():
        logger.scalar(f'grad_cos_sim_{a}_{b}', sim)
        
      agent._metrics = {}
      logger.write()
      
    # Save checkpoint
    if i % 1000 == 0:
      torch.save(agent._wm.state_dict(), logdir / 'teacher_model.pt')
      torch.save(agent._wm.vae.state_dict(), logdir / 'vae_model.pt') 
      print(f'Step {i}: Saved teacher checkpoint')
    
    logger.step = i
  
  print('Pretraining complete!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))
  
# python -u dreamer_pretrain.py --configs defaults metaworld --logdir ./logs/moe_teacher_new