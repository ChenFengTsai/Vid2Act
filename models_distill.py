import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import torch.nn.functional as F
from pathlib import Path

import networks
import tools
to_np = lambda x: x.detach().cpu().numpy()


class WorldModelStudent(nn.Module):
  """
  Student world model for online training with distillation from frozen teachers.
  """

  def __init__(self, step, config, action_space):
    super(WorldModelStudent, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    # self._offline_dataset = offline_dataset  # List of datasets for VAE
    
    # ===== Load teacher config (if available) =====
    self._teacher_config = None

    # Try to get teacher config path from config; otherwise derive from teacher_model_path
    teacher_model_path = getattr(config, "teacher_model_path", None)
    teacher_config_path = getattr(config, "teacher_config_path", None)

    # If teacher_config_path not explicitly given, assume it's "config.json" next to the model
    if teacher_config_path is None and teacher_model_path is not None:
      from pathlib import Path
      teacher_config_path = Path(teacher_model_path).parent / "config.json"

    if teacher_config_path is not None:
      from pathlib import Path
      teacher_config_path = Path(teacher_config_path)
      if teacher_config_path.is_file():
        import json
        from types import SimpleNamespace

        print(f"[WorldModelStudent] Loading teacher config from {teacher_config_path}")
        with open(teacher_config_path, "r") as f:
          teacher_cfg_dict = json.load(f)

        # Store teacher config object
        self._teacher_config = SimpleNamespace(**teacher_cfg_dict)

        # Copy the relevant hyperparameters to config as "teacher_*"
        # so the teacher networks use teacher’s own sizes.
        for name in ["teacher_cnn_depth", "teacher_dyn_hidden", "teacher_dyn_deter", "teacher_dyn_stoch"]:
          if hasattr(self._teacher_config, name):
            setattr(config, f"{name}", getattr(self._teacher_config, name))
      else:
        print(f"[WorldModelStudent] WARNING: teacher config not found at {teacher_config_path}")

    # Calculate embed size
    if config.size[0] == 64 and config.size[1] == 64:
      student_embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      student_embed_size *= 2 * 2
      teacher_embed_size = 2 ** (len(config.encoder_kernels)-1) * getattr(config, 'teacher_cnn_depth', config.cnn_depth)
      teacher_embed_size *= 2 * 2
    # elif config.size[0] == 84 and config.size[1] == 84:
    #   def conv_down(h, k, s=2, p=0, d=1):
    #     return (h + 2*p - d*(k-1) - 1)//s + 1
    #   H, W = int(config.size[0]), int(config.size[1])
    #   for k in config.encoder_kernels:
    #       H = conv_down(H, k, s=2, p=0)
    #       W = conv_down(W, k, s=2, p=0)
    #   channels_last = (2 ** (len(config.encoder_kernels)-1)) * config.cnn_depth
    #   embed_size = channels_last * H * W
    else:
      raise NotImplemented(f"{config.size} is not applicable now")

    # ========== TEACHER MODELS (Frozen) ==========
    # NEW (Solution - separate configs):
    teacher_cnn_depth = getattr(config, 'teacher_cnn_depth', config.cnn_depth)
    teacher_dyn_hidden = getattr(config, 'teacher_dyn_hidden', config.dyn_hidden)
    teacher_dyn_deter = getattr(config, 'teacher_dyn_deter', config.dyn_deter)
    teacher_dyn_stoch = getattr(config, 'teacher_dyn_stoch', config.dyn_stoch)
    
    self.encoder_teachers = networks.ConvEncoder(
        config.grayscale, teacher_cnn_depth, config.act, 
        config.encoder_kernels, label_num=config.num_teachers)
    
    self.dynamics_teachers = networks.RSSM(
        teacher_dyn_stoch, teacher_dyn_deter, teacher_dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, teacher_embed_size, config.device, label_num=config.num_teachers)
    
    # Freeze teacher parameters
    for param in self.encoder_teachers.parameters():
      param.requires_grad = False
    for param in self.dynamics_teachers.parameters():
      param.requires_grad = False

    # ========== DISTILLATION COMPONENTS ==========
    # self.imp = nn.Linear(2*(config.dyn_stoch+config.dyn_deter), 1)
    # self.distiller = nn.Linear(config.dyn_stoch+config.dyn_deter, 
    #                            config.dyn_stoch+config.dyn_deter)
    
    
    teacher_feat_dim = teacher_dyn_stoch + teacher_dyn_deter
    student_feat_dim = config.dyn_stoch + config.dyn_deter
    
    self.imp = nn.Sequential(
    nn.Linear(teacher_feat_dim + student_feat_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
    )
    
    if self._config.use_distill:
      self.distiller = nn.Sequential(
          nn.Linear(teacher_feat_dim, 256),
          nn.ReLU(),
          # nn.Linear(256, 256),
          # nn.ReLU(),
          nn.Linear(256, student_feat_dim)
      )

    vae_latent = getattr(config, 'vae_latent_dim', 32)
    vae_action_dim = config.num_actions
    self.vae = networks.VAE(
        state_dim=teacher_feat_dim,
        action_dim=vae_action_dim,
        latent_dim=vae_latent,
        max_action=action_space.high[0],
        device=config.device,
        config=config
    )
    
    # Freeze VAE params during online training
    for p in self.vae.parameters():
      p.requires_grad = False
    self.vae.eval()
    
    self.softmax = nn.Softmax(dim=0)
    self.m = torch.tensor([0.1]).to(config.device)

    # ========== STUDENT MODELS (Trainable) ==========
    self.encoder = networks.ConvEncoder(
        config.grayscale, config.cnn_depth, config.act, config.encoder_kernels)
    
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, student_embed_size, config.device)
    
    # Student heads
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size
    
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    
    self.heads['image'] = networks.ConvDecoder(
        feat_size, config.cnn_depth, config.act, shape, 
        config.decoder_kernels, config.decoder_thin)
    self.heads['reward'] = networks.DenseHead(
        feat_size, [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          feat_size, [], config.discount_layers, config.units, config.act, 
          dist='binary')
    
    for name in config.grad_heads:
      assert name in self.heads, name
    
    # # Student optimizer
    # self.module_para = (
    #     list(self.encoder.parameters()) + 
    #     list(self.dynamics.parameters()) +
    #     list(self.imp.parameters()) +
    #     # list(self.distiller.parameters()) +
    #     # list(self.vae.parameters()) +
    #     list(self.heads['image'].parameters()) +
    #     list(self.heads['reward'].parameters())
    # )
    
    # if self._config.use_distill:
    #   self.module_para += list(self.distiller.parameters())
    
    self.modules_to_optimize = nn.ModuleList([
      self.encoder,
      self.dynamics,
      self.imp,
      self.heads['image'],
      self.heads['reward']
    ])

    if self._config.use_distill:
        self.modules_to_optimize.append(self.distiller)

    self.module_para = list(self.modules_to_optimize.parameters())
    
    self._model_opt = tools.Optimizer(
        'model', self.module_para, config.model_lr, config.opt_eps, 
        config.grad_clip, config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    print(
      f"Optimizer model_opt has {sum(param.numel() for param in self.module_para)} variables."
    )
    
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)
    self.l2_loss = torch.nn.MSELoss(reduction='none')


  
  def load_teacher(self, teacher_state_dict, vae_state_dict=None):
    """Load pretrained teacher weights and optional VAE weights
    
    Args:
        teacher_state_dict: State dict from teacher model checkpoint
        vae_state_dict: State dict from VAE checkpoint (optional)
    """
    # Map teacher checkpoint to student's teacher models
    teacher_dict = {}
    for key, value in teacher_state_dict.items():
      # Rename keys from teacher checkpoint to match student's teacher models
      if key.startswith('encoder.'):
        teacher_dict['encoder_teachers.' + key[8:]] = value
      elif key.startswith('dynamics.'):
        teacher_dict['dynamics_teachers.' + key[9:]] = value
      elif key.startswith('heads.'):
        # Skip heads, we don't need them in student
        pass
      else:
        # Handle any other keys (like MoE components)
        pass
    
    # Load teacher parameters with strict=False to ignore missing student-only params
    self.load_state_dict(teacher_dict, strict=False)
    print(f"Loaded {len(teacher_dict)} teacher parameters")
    
    # Load VAE weights if provided
    if vae_state_dict is not None:
      try:
        self.vae.load_state_dict(vae_state_dict, strict=True)
        print(f"Loaded VAE checkpoint with {len(vae_state_dict)} parameters")
      except RuntimeError as e:
        print(f"Warning: Could not load VAE checkpoint - {e}")
        print("Continuing with random VAE initialization")
    
    # Freeze teacher models
    for param in self.encoder_teachers.parameters():
      param.requires_grad = False
    for param in self.dynamics_teachers.parameters():
      param.requires_grad = False

  def _train(self, data):
    """Train student model with distillation"""
    data = self.preprocess(data)
    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        # Student forward pass
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free    = tools.schedule(self._config.kl_free,    self._step)
        kl_scale   = tools.schedule(self._config.kl_scale,   self._step)

        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)

        feat = self.dynamics.get_feat(post)

        # ========== DISTILLATION LOSS ==========
        d_loss = torch.tensor(0.0, device=feat.device)
        vae_loss = torch.tensor(0.0, device=feat.device)
        
        if self._config.is_adaptive:
          teacher_feat = []
          imp_weights = []

          # Get features from all teachers
          for index in range(self._config.num_teachers):
            with torch.no_grad():  # Teachers are frozen
              teacher_embed = self.encoder_teachers(data, label=index)
              t_post, t_prior = self.dynamics_teachers.observe(
                  teacher_embed, data["action"], label=index)
              teacher_i = self.dynamics_teachers.get_feat(t_post)
            
            teacher_feat.append(teacher_i)

            # Compute importance weights
            imp_input = torch.cat([teacher_i, feat], dim=-1)
            imp_weight = self.imp(imp_input)
            imp_weights.append(imp_weight)

          # Compute weighted distillation loss
          imp_weights = torch.stack(imp_weights, dim=0)
          imp_weights = torch.squeeze(imp_weights)
          imp_weights = self.softmax(imp_weights)
          all_weight = imp_weights.reshape((self._config.num_teachers, self._config.batch_size * self._config.batch_length)) # 50*50=2500
          out_weight = torch.argmax(all_weight, dim=0) # 2500

          d_loss_val = 0.0
          for index in range(self._config.num_teachers):
            teacher_feature = teacher_feat[index]
            if self._config.use_distill:
              d_t_feat = self.distiller(teacher_feature)
              mse = torch.mean(self.l2_loss(d_t_feat, feat), dim=-1)
            else:
              mse = torch.mean(self.l2_loss(teacher_feature, feat), dim=-1)
            weight = imp_weights[index]
            
            weight = torch.max(self.m, weight)
            # print(weight)
            d_loss_val += torch.mean(mse * weight)

          d_loss = d_loss_val

          # ========== VAE LOSS ==========
          # if self._offline_dataset is not None:
          #   vae_accum = 0.0
          #   for max_weight in range(self._config.num_teachers):
          #     source_data = next(self._offline_dataset[max_weight])
          #     source_data = self.preprocess(source_data)
              
          #     with torch.no_grad():
          #       source_embed = self.encoder_teachers(source_data, label=max_weight)
          #       source_post, _ = self.dynamics_teachers.observe(
          #           source_embed, source_data["action"], label=max_weight)
          #       source_feat = self.dynamics_teachers.get_feat(source_post)
          #       source_feat = self.distiller(source_feat).detach()

          #     batch, seq, dim = source_feat.shape
          #     z_flat = source_feat[:, :-1, :].reshape(batch * (seq - 1), dim)
          #     act = source_data['action'][:, 1:, :].reshape(batch * (seq - 1), -1)
          #     label = [max_weight] * (batch * (seq - 1))

          #     recon, mean, std = self.vae(z_flat, act, label)
          #     recon_loss = torch.nn.functional.mse_loss(recon, act)
          #     vae_kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
          #     vae_accum += recon_loss + 0.5 * vae_kl_loss

          #   vae_loss = vae_accum

        # ========== STUDENT HEADS ==========
        losses = {'kl': kl_loss}
        if self._config.use_distill:
          losses['distillation'] = d_loss # multiplied by self._config.distill_weight
        # losses['vae'] = vae_loss

        likes = {}
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          feat_in = feat if grad_head else feat.detach()
          pred = head(feat_in)
          like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

        model_loss = sum(losses.values())
        
      metrics = self._model_opt(model_loss, self.module_para)

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = to_np(torch.mean(kl_value))
    
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      context = dict(
          embed=embed, feat=feat,
          kl=kl_value, postent=self.dynamics.get_dist(post).entropy())

    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics, out_weight.tolist()

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)

    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](
        self.dynamics.get_feat(states)).mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)
  



class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward

    self._incident_counts_running = torch.zeros(3, dtype=torch.long)
    self._local_step = 0
    
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.actor = networks.ActionHead(
        # feat_size+50,  # pytorch version
        feat_size,
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    self.value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)
      self._updates = 0
    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    print(
      f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
    )
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)
    print(
      f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
    )


  def _train(
      self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None, weight=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}
    self._local_step += 1

    with tools.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(self._use_amp):
        # imag_feat, imag_feat_action, imag_state, imag_action = self._imagine(
        #     start, self.actor, self._config.imag_horizon, repeats, weight)
        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats, weight)
        reward = objective(imag_feat, imag_state, imag_action)
        # actor_ent = self.actor(imag_feat_action).entropy()
        actor_ent = self.actor(imag_feat).entropy()
        state_ent = self._world_model.dynamics.get_dist(
            imag_state).entropy()
        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target)
        imag_feat_action = None
        actor_loss, mets = self._compute_actor_loss(
            imag_feat, imag_feat_action, imag_state, imag_action, target, actor_ent, state_ent,
            weights)
        metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_value_target)
        value_input = imag_feat

    with tools.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())
        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))
    return imag_feat, imag_state, imag_action, weights, metrics

    
  def _imagine(self, start, policy, horizon, repeats=None, weight=None):
    dynamics = self._world_model.dynamics
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      # state, _, _, _ = prev
      state, _, _= prev
      feat = dynamics.get_feat(state)
      # print(weight)
    
    
      sampled_actions, sampled_feat = self._world_model.vae.decode(feat, weight)
      # inp = torch.cat([feat, sampled_feat], -1)
      inp = feat.detach()
      action = policy(inp).sample()
      # action = policy(inp.detach()).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      # return succ, feat, inp, action
      return succ, feat, action
    # feat = 0 * dynamics.get_feat(start)
    # sam_action, sam_feat = self._world_model.vae.decode(feat, weight)
    # feat_action = torch.cat([feat, sam_feat], -1)
    # action = policy(feat_action).mode()

    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon)], (start, None, None))
    # succ, feats, feat_actions, actions = tools.static_scan(
    #     step, [torch.arange(horizon)], (start, feat, feat_action, action))
    
    
    
    # ---------- NEW: count & save incidents ----------
    if weight is not None:
      if not isinstance(weight, torch.Tensor):
        t = torch.tensor(weight)
      # Make sure it’s a 1D integer array. Adjust dtype / reshape if needed.
      self._incident_counts_running += torch.bincount(t, minlength=3)
      if self._local_step % 100 == 0:
        incident_list = self._incident_counts_running.tolist()
        total_sum = sum(incident_list)
        
        print("Current incident counts:", incident_list)
        print(f"Total sum: {total_sum}")
        
        percentages = [(count / total_sum * 100) for count in incident_list]
        print(f"Percentages: {percentages}")
    # ---------- /NEW ----------
  
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    
    return feats, states, actions
    # return feats, feat_actions, states, actions

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_feat_action, imag_state, imag_action, target, actor_ent, state_ent,
      weights):
    metrics = {}
    # inp = imag_feat_action.detach() if self._stop_grad_actor else imag_feat_action
    inp = imag_feat.detach()
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1


