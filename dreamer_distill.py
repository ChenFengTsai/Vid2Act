import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models_distill
import models_nodistill

import tools
import wrappers

import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()


class DreamerDistill(nn.Module):

  def __init__(self, config, logger, dataset, action_space):
    super(DreamerDistill, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
  # Initialize CSV file for incident counts logging
    self._logdir = pathlib.Path(config.logdir).expanduser()
    self._csv_file = self._logdir / "incident_counts.csv"
    
    scheduled_params = ['actor_entropy', 'actor_state_entropy', 'imag_gradient_mix', 'moe_temperature']
    for param in scheduled_params:
        if param in config:
            setattr(config, f'{param}_str', getattr(config, param))
            
    # Schedules
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    config.moe_temperature = (
        lambda x=config.moe_temperature: tools.schedule(x, self._step))
      
    self._dataset = dataset
    # self._offline_datasets = offline_datasets  # For VAE training
    
    # Student world model with distillation
    if config.use_distill:
      self._wm = models_distill.WorldModelStudent(self._step, config, action_space)
    else:
      self._wm = models_nodistill.WorldModelStudent(self._step, config, action_space)
    total = sum(p.numel() for p in self._wm.parameters())  
    trainable = sum(p.numel() for p in self._wm.parameters() if p.requires_grad)

    print(f"Total: {total/1e6:.2f}M")
    print(f"Trainable: {trainable/1e6:.2f}M")
    
    # Load pretrained teacher
    if config.teacher_model_path:
      print(f'Loading pretrained teacher from {config.teacher_model_path}')
      teacher_checkpoint = torch.load(config.teacher_model_path, map_location=config.device)
      
      # Load VAE if path is provided
      vae_checkpoint = None
      if hasattr(config, 'vae_model_path') and config.vae_model_path:
        print(f'Loading pretrained VAE from {config.vae_model_path}')
        vae_checkpoint = torch.load(config.vae_model_path, map_location=config.device)
        
      self._wm.load_teacher(teacher_checkpoint, vae_checkpoint)
    else:
      raise ValueError("Must provide teacher_model_path for distillation")
    
    print("\n===== TEACHER ENCODER =====")
    print(self._wm.encoder_teachers)

    print("\n===== TEACHER DYNAMICS =====")
    print(self._wm.dynamics_teachers)
    
    # Behavior learning
    if config.use_distill:
      self._task_behavior = models_distill.ImagBehavior(
          config, self._wm, config.behavior_stop_grad)
    else:
      self._task_behavior = models_nodistill.ImagBehavior(
          config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    
    self.softmax_1 = nn.Softmax(dim=1)

  def __call__(self, obs, reset, state=None, reward=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)
        
        incident_list = self._task_behavior._incident_counts_running.tolist()
        
        tools.save_incident_counts_to_csv(incident_list=incident_list, step=step, csv_file=self._csv_file)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  # def _policy(self, obs, state, training):
  #   if state is None:
  #     batch_size = len(obs['image'])
  #     latent = self._wm.dynamics.initial(len(obs['image']))
  #     action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
  #   else:
  #     latent, action = state
  #   embed = self._wm.encoder(self._wm.preprocess(obs))
  #   latent, _ = self._wm.dynamics.obs_step(
  #       latent, action, embed, self._config.collect_dyn_sample)
  #   if self._config.eval_state_mean:
  #     latent['stoch'] = latent['mean']
  #   feat = self._wm.dynamics.get_feat(latent)
  #   if not training:
  #     actor = self._task_behavior.actor(feat)
  #     action = actor.mode()
  #   elif self._should_expl(self._step):
  #     actor = self._expl_behavior.actor(feat)
  #     action = actor.sample()
  #   else:
  #     actor = self._task_behavior.actor(feat)
  #     action = actor.sample()
  #   logprob = actor.log_prob(action)
  #   latent = {k: v.detach()  for k, v in latent.items()}
  #   action = action.detach()
  #   if self._config.actor_dist == 'onehot_gumble':
  #     action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
  #   action = self._exploration(action, training)
  #   policy_output = {'action': action, 'logprob': logprob}
  #   state = (latent, action)
  #   return policy_output, state
  
  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    data = self._wm.preprocess(obs)
    embed = self._wm.encoder(data)
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)

    teacher_feat = []
    for index in range(self._config.num_teachers):
      teacher_embed = self._wm.encoder_teachers(data, label=index)
      latent_, _ = self._wm.dynamics_teachers.obs_step(
      latent, action, teacher_embed, self._config.collect_dyn_sample, label=index)

      teacher_i = self._wm.dynamics_teachers.get_feat(latent_)
      teacher_feat.append(teacher_i)

    t_weight = torch.stack(teacher_feat, axis=1)

    student_weight = feat.unsqueeze(1).repeat(1,self._config.num_teachers,1)
    all_weight = torch.cat([t_weight, student_weight], -1) # 1, 6, 500
    all_weight = self._wm.imp(all_weight).squeeze(-1)  # 1, 6
    all_weight = self.softmax_1(all_weight)  # 1, 6
    weight_max = torch.argmax(all_weight, dim=1)
    
    if self._config.use_vae: 
      sampled_actions, sampled_feats = self._wm.vae.decode(feat, weight_max)  ## 1 * 50
      feat = torch.cat([feat, sampled_feats], -1) 
    
    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data):
    metrics = {}
    post, context, mets, weight = self._wm._train(data)
    metrics.update(mets)
    
    start = post
    if self._config.pred_discount:
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior._train(start, reward, weight=weight)[-1])
    
    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset

import os
import imageio
import numpy as np

class SaveFrameWrapper:
    """
    Wraps an environment so each reset/step call saves the RGB image observation.
    """
    def __init__(self, env, save_dir="env_frames", env_id=0):
        self._env = env
        self.save_dir = os.path.join(save_dir, f"env{env_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

    def reset(self):
        obs = self._env.reset()
        self._save_obs(obs, prefix="reset")
        return obs

    def step(self, action):
        result = self._env.step(action)
        obs, reward, done = result[:3]
        self._save_obs(obs)
        return result

    def _save_obs(self, obs, prefix="step"):
        if "image" not in obs:
            return
        img = obs["image"]
        if img.dtype != np.uint8:
            img = np.clip((img * 255.0), 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] in (1,3):  # NCHW → HWC
            img = np.transpose(img, (1,2,0))
        path = os.path.join(self.save_dir, f"{prefix}_{self.frame_count:06d}.png")
        imageio.imwrite(path, img)
        self.frame_count += 1

    def __getattr__(self, name):
        # Forward all other methods to the wrapped env
        return getattr(self._env, name)


def make_env(config, logger, mode, train_eps, eval_eps, save_frames=False, save_dir="env_frames"):
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
      print('Current task:', task)
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
  if (mode == 'train') or (mode == 'eval'):
    callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks, logger=logger, mode=mode)
  env = wrappers.RewardObs(env)
  if save_frames:
    env = SaveFrameWrapper(env, save_dir=save_dir, env_id=config.seed)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  import setup_utils as setup_utils
  device_mapper = setup_utils.setup_device(config)
  config.device = str(config.device)
  
  tools.set_seed_everywhere(config.seed)
  logdir = pathlib.Path(config.logdir).expanduser()
  
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act)
  

  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)

  
  setup_utils.save_cmd(str(logdir))
  setup_utils.save_git(str(logdir))
  
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)

  print('Create envs.')
  # if config.offline_traindir:
  #   directory = config.offline_traindir.format(**vars(config))
  # else:
  if config.online_mode:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.online_mode:
  #   directory = config.offline_evaldir.format(**vars(config))
  # else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  train_envs = [make('train') for _ in range(config.envs)]
  eval_envs = [make('eval') for _ in range(config.envs)]
  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  # ---- PREFILL (ported from v3, adapted for v2) ----
  # Only do online prefill if we're NOT using an offline dataset.
  if not config.offline_traindir:
    # How many env steps to add to the replay before we start training.
    prefill = max(0, config.prefill - count_steps(config.traindir))
    if prefill > 0:
      print(f"Prefill dataset ({prefill} steps).")

      # Random policy for prefill; match your action space type.
      acts = train_envs[0].action_space
      if hasattr(acts, 'n'):  # discrete
        def random_agent(obs, done, state, reward):
          # Use a uniform OneHotDist over actions.
          logits = torch.zeros(config.envs, config.num_actions, device=config.device)
          dist = tools.OneHotDist(logits)
          action = dist.sample()
          logprob = dist.log_prob(action)
          return {'action': action, 'logprob': logprob}, None
      else:  # continuous
        low = torch.tensor(acts.low, device=config.device).repeat(config.envs, 1)
        high = torch.tensor(acts.high, device=config.device).repeat(config.envs, 1)
        def random_agent(obs, done, state, reward):
          action = low + torch.rand_like(low) * (high - low)
          # keep API consistent; logprob unused by your wrappers
          logprob = torch.zeros(action.shape[0], device=action.device)
          return {'action': action, 'logprob': logprob}, None

      # Roll out the random agent for `prefill` steps. Because your envs are
      # wrapped with CollectDataset(process_episode(...)), episodes are saved
      # to disk and also inserted into the in-memory `train_eps` cache.
      _ = tools.simulate(random_agent, train_envs, prefill)

      # Keep the logger step aligned with env steps.
      logger.step += prefill * config.action_repeat
      print(f"Logger: ({logger.step} steps).")
  # ---- END PREFILL ----

  print('Create datasets.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = make_dataset(eval_eps, config)
  

  print('Initialize agent.')
  agent = DreamerDistill(config, logger, train_dataset, acts).to(config.device)
  
  # Save configuration and command info
  setup_utils.save_config_to_json(config, logdir)
  
  agent.requires_grad_(requires_grad=False)
  
  if (logdir / 'latest_model.pt').exists():
    print('Loading checkpoint.')
    agent.load_state_dict(torch.load(logdir / 'latest_model.pt', map_location=config.device))
    agent._should_pretrain._once = False

  # Online training loop
  state = None
  while agent._step < config.steps:
    logger.write()
    print('Start evaluation.')
    tools.simulate(
        functools.partial(agent, training=False),
        eval_envs, episodes=config.eval_num)
    
    print('Start training.')
    state = tools.simulate(agent, train_envs, config.eval_every, state=state)

    incident_list = agent._task_behavior._incident_counts_running.tolist()
    total_sum = sum(incident_list)
    
    print("Current incident counts:", incident_list)
    print(f"Total sum: {total_sum}")
    percentages = [(count / total_sum * 100) for count in incident_list]
    print(f"step {agent._step} Percentages: {percentages}")
    
    agent._task_behavior._incident_counts_running.zero_()
    
    torch.save(agent.state_dict(), logdir / 'latest_model.pt')
  
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


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
  # python dreamer_distill.py --configs defaults metaworld --logdir /storage/ssd1/richtsai1103/vid2act/log/metaworld/open3/window_close/new_moe/original --teacher_encoder_mode original_conv --device cuda:4 --teacher_model_path /storage/ssd1/richtsai1103/vid2act/models/original_teacher/teacher_model.pt --vae_model_path /storage/ssd1/richtsai1103/vid2act/models/original_teacher/vae_model.pt --task metaworld_window_close --seed 0
  # python dreamer_distill.py --configs defaults metaworld --logdir debug --teacher_encoder_mode original_conv --device cuda:4 --teacher_model_path /home/richtsai1103/CRL/Vid2Act/logs/original_teacher/teacher_model.pt --vae_model_path /home/richtsai1103/CRL/Vid2Act/logs/original_teacher/vae_model.pt --task metaworld_drawer_close --seed 0

  # Total: 7.74M
# Trainable: 5.41M

# Total: 3.83M
# Trainable: 1.50M