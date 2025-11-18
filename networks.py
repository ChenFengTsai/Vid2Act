import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools

      
import torch
from torch import nn
from torch import distributions as torchd

import tools


class RSSM_Teacher(nn.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=nn.ELU,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='gru',
      num_actions=None, embed=None, device=None, label_num=None):
    super(RSSM_Teacher, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = embed
    self._device = device

    # Task labels
    self.label_num = label_num
    if self.label_num:
      label_tmp = torch.eye(label_num).unsqueeze(0)
      self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)

    # Input MLP (for img_step)
    inp_layers = []
    if self._discrete:
      inp_dim = self._stoch * self._discrete + num_actions
    else:
      if self.label_num:
        inp_dim = self._stoch + num_actions + label_num
      else:
        inp_dim = self._stoch + num_actions
    if self._shared:
      inp_dim += self._embed
    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._inp_layers = nn.Sequential(*inp_layers)

    # Recurrent cell
    if cell == 'gru':
      self._cell = GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    # Prior (imagined) output MLP
    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._img_out_layers = nn.Sequential(*img_out_layers)

    # Posterior (observed) output MLP
    obs_out_layers = []
    if self._temp_post:
      inp_dim = self._deter + self._embed
    else:
      inp_dim = self._embed
    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    # ===== Multi-head stat layers (per task), or shared head if no label_num =====
    if self.label_num:
      if self._discrete:
        self._ims_stat_layers = nn.ModuleList([
            nn.Linear(self._hidden, self._stoch * self._discrete)
            for _ in range(self.label_num)
        ])
        self._obs_stat_layers = nn.ModuleList([
            nn.Linear(self._hidden, self._stoch * self._discrete)
            for _ in range(self.label_num)
        ])
      else:
        self._ims_stat_layers = nn.ModuleList([
            nn.Linear(self._hidden, 2 * self._stoch)
            for _ in range(self.label_num)
        ])
        self._obs_stat_layers = nn.ModuleList([
            nn.Linear(self._hidden, 2 * self._stoch)
            for _ in range(self.label_num)
        ])
    else:
      if self._discrete:
        self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
      else:
        self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
        self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)

  # ---------------------------------------------------------------------------

  def initial(self, batch_size):
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=deter)
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=deter)
    return state

  def observe(self, embed, action, state=None, label=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev_state, prev_act, emb: self.obs_step(
            prev_state[0], prev_act, emb, label=label),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None, label=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tools.static_scan_withlabel(self.img_step, [action], state, label=label)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = tools.ContDist(torchd.independent.Independent(
          torchd.normal.Normal(mean, std), 1))
    return dist

  # ---------------------------------------------------------------------------

  def obs_step(self, prev_state, prev_action, embed, sample=True, label=None):
    prior = self.img_step(prev_state, prev_action, None, sample, label=label)
    if self._shared:
      # shared = use same transition for prior/posterior
      post = self.img_step(prev_state, prev_action, embed, sample, label=label)
    else:
      if self._temp_post:
        x = torch.cat([prior['deter'], embed], -1)
      else:
        x = embed
      x = self._obs_out_layers(x)
      stats = self._suff_stats_layer('obs', x, label=label)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action, embed=None, sample=True, label=None):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = prev_stoch.reshape(shape)
    if self._shared:
      if embed is None:
        shape = list(prev_action.shape[:-1]) + [self._embed]
        embed = torch.zeros(shape)
      x = torch.cat([prev_stoch, prev_action, embed], -1)
    else:
      x = torch.cat([prev_stoch, prev_action], -1)

    # add label information
    if label is not None:
      bs, _ = x.size()
      # label_use: [1, label_num, label_num], select along last dim
      label_use = self.label_use[:, :, label]  # shape [1, label_num]
      label_use = label_use.repeat(bs, 1)
      x = torch.cat((x, label_use.detach()), -1)

    x = self._inp_layers(x)
    for _ in range(self._rec_depth):  # rec depth is not correctly implemented
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x, label=label)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def fill_action_with_zero(self, action):
    B, D = action.shape[0], action.shape[1]
    zeros = torch.zeros([B, 20 - D]).to(self._device)
    return torch.cat([action, zeros], -1)

  # ---------------------------------------------------------------------------

  def _get_head_layer(self, name, label):
    """
    Select the appropriate stat head (ims/obs) given the task label.
    If no label_num is set, fall back to the shared head.
    """
    # No multi-head: single shared layer
    if not self.label_num:
      return self._ims_stat_layer if name == 'ims' else self._obs_stat_layer

    # Multi-head: choose head index
    if isinstance(label, int):
      idx = label
    elif isinstance(label, torch.Tensor):
      if label.dim() == 0:
        idx = int(label.item())
      else:
        # If label is [B], assume all in batch use same label; take first.
        idx = int(label[0].item())
    elif label is None:
      # Fallback if label is missing: use head 0
      idx = 0
    else:
      raise ValueError(f"Unsupported label type: {type(label)}")

    if not (0 <= idx < self.label_num):
      raise ValueError(f"Label index {idx} out of range (0..{self.label_num-1})")

    if name == 'ims':
      return self._ims_stat_layers[idx]
    elif name == 'obs':
      return self._obs_stat_layers[idx]
    else:
      raise NotImplementedError(name)

  def _suff_stats_layer(self, name, x, label=None):
    """
    Produce sufficient statistics for stochastic state:
      - discrete: logits
      - continuous: mean / std (with activations)
    Uses per-task head if label_num is set, otherwise shared head.
    """
    layer = self._get_head_layer(name, label)

    if self._discrete:
      x = layer(x)
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = layer(x)
      mean, std = torch.split(x, [self._stoch] * 2, -1)
      mean = {
          'none':  lambda: mean,
          'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: torch.softplus(std),
          'abs':      lambda: torch.abs(std + 1),
          'sigmoid':  lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = torchd.kl.kl_divergence
    dist = lambda x: self.get_dist(x)
    sg = lambda x: {k: v.detach() for k, v in x.items()}
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      loss = torch.mean(torch.maximum(value, torch.as_tensor(free, device=value.device)))
    else:
      value_lhs = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                      dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      loss_lhs = torch.maximum(torch.mean(value_lhs), torch.as_tensor(free, device=value_lhs.device))
      loss_rhs = torch.maximum(torch.mean(value_rhs), torch.as_tensor(free, device=value_rhs.device))
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
      value = value_lhs  # just to return something consistent
    loss *= scale
    return loss, value
     
class RSSM(nn.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=nn.ELU,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='gru',
      num_actions=None, embed = None, device=None, label_num=None):
    super(RSSM, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = embed
    self._device = device
    
    self.label_num = label_num
    if self.label_num:
      label_tmp = torch.eye(label_num).unsqueeze(0)
      self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)    

    inp_layers = []
    if self._discrete:
      inp_dim = self._stoch * self._discrete + num_actions
    else:
      if self.label_num:
        inp_dim = self._stoch + num_actions + label_num
      else:
        inp_dim = self._stoch + num_actions
    if self._shared:
      inp_dim += self._embed
    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._inp_layers = nn.Sequential(*inp_layers)

    if cell == 'gru':
      self._cell = GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._img_out_layers = nn.Sequential(*img_out_layers)

    obs_out_layers = []
    if self._temp_post:
      inp_dim = self._deter + self._embed
    else:
      inp_dim = self._embed
    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    if self._discrete:
      self._ims_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
      self._obs_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
    else:
      self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
  def initial(self, batch_size):
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=deter)
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=deter)
    return state

  def observe(self, embed, action, state=None, label=None):
    print(embed.shape)
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev_state, prev_act, embed: self.obs_step(
            prev_state[0], prev_act, embed, label=label),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None, label=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = action
    action = swap(action)
    prior = tools.static_scan_withlabel(self.img_step, [action], state, label=label)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = tools.ContDist(torchd.independent.Independent(
          torchd.normal.Normal(mean, std), 1))
    return dist

  def obs_step(self, prev_state, prev_action, embed, sample=True, label=None):
    prior = self.img_step(prev_state, prev_action, None, sample, label=label)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        x = torch.cat([prior['deter'], embed], -1)
      else:
        x = embed
      x = self._obs_out_layers(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action, embed=None, sample=True, label=None):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = prev_stoch.reshape(shape)
    if self._shared:
      if embed is None:
        shape = list(prev_action.shape[:-1]) + [self._embed]
        embed = torch.zeros(shape)
      x = torch.cat([prev_stoch, prev_action, embed], -1)
    else:
      x = torch.cat([prev_stoch, prev_action], -1)

    # add label information
    if label is not None:
      bs, _ = x.size()
      # label_use = self.label_use[:, label]
      label_use = self.label_use[:, :, label]
      # if bs != 1:
      label_use = label_use.repeat(bs, 1)
      x = torch.cat((x, label_use.detach()), -1)
    
    x = self._inp_layers(x)
    for _ in range(self._rec_depth): # rec depth is not correctly implemented
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def fill_action_with_zero(self, action):
      B, D = action.shape[0], action.shape[1]
      zeros = torch.zeros([B, 20 - D]).to(self._device)
      return torch.cat([action, zeros], -1)

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      mean, std = torch.split(x, [self._stoch]*2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: torch.softplus(std),
          'abs': lambda: torch.abs(std + 1),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = torchd.kl.kl_divergence
    dist = lambda x: self.get_dist(x)
    sg = lambda x: {k: v.detach() for k, v in x.items()}
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      loss = torch.mean(torch.maximum(value, free))
    else:
      value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                              dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
      loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value
  
# class MoEOrthogonalConvEncoder(nn.Module):
#     """
#     Mixture of Experts Orthogonal Convolutional Encoder.
#     Creates multiple parallel convolutional encoders whose outputs are orthogonalized.
#     """
    
#     def __init__(self, n_experts=5, grayscale=False, depth=32, 
#                  act=nn.ReLU, kernels=(4, 4, 4, 4), label_num=None,
#                  use_gating=True, temperature=1.0):
#         """
#         Args:
#             n_experts: Number of expert encoders in the mixture
#             grayscale: Whether input images are grayscale
#             depth: Base depth for convolutional layers
#             act: Activation function class
#             kernels: Kernel sizes for each conv layer
#             label_num: Number of labels for conditional encoding
#             use_gating: Whether to use gating mechanism for expert selection
#             temperature: Temperature for gating softmax
#         """
#         super(MoEOrthogonalConvEncoder, self).__init__()
        
#         self.n_experts = n_experts
#         self.use_gating = use_gating
#         self.temperature = temperature
#         self._act = act
#         self._depth = depth
#         self._kernels = kernels
#         self.label_num = label_num
        
#         # Create parallel expert encoders
#         self.expert_encoders = nn.ModuleList()
#         for _ in range(n_experts):
#             self.expert_encoders.append(self._create_single_encoder(
#                 grayscale, depth, act, kernels, label_num
#             ))
        
#         # Gating network (if enabled)
#         if self.use_gating:
#             gate_input_dim = 3 if not grayscale else 1
#             if label_num is not None:
#                 gate_input_dim += label_num
            
#             self.gate_network = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),  # Global average pooling
#                 nn.Flatten(),
#                 nn.Linear(gate_input_dim, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, n_experts)
#             )
        
#         # Orthogonalization layer
#         self.orthogonal_layer = OrthogonalLayer()
        
#         # Final aggregation layer (learnable weighted sum after orthogonalization)
#         self.aggregation_weights = nn.Parameter(torch.ones(n_experts) / n_experts)
        
#         # Label embedding for conditional encoding
#         if label_num is not None:
#             label_tmp = torch.eye(label_num).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#             self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)
    
#     def _create_single_encoder(self, grayscale, depth, act, kernels, label_num):
#         """Create a single convolutional encoder."""
#         layers = []
        
#         for i, kernel in enumerate(kernels):
#             if i == 0:
#                 if grayscale:
#                     inp_dim = 1
#                 else:
#                     inp_dim = 3 + (label_num if label_num is not None else 0)
#             else:
#                 inp_dim = 2 ** (i-1) * depth
            
#             out_depth = 2 ** i * depth
#             layers.append(nn.Conv2d(inp_dim, out_depth, kernel, 2))
#             layers.append(act())
        
#         return nn.Sequential(*layers)
    
#     def forward(self, obs, label=None, return_expert_outputs=False):
#         """
#         Forward pass through MoE orthogonal encoder.
        
#         Args:
#             obs: Dictionary with 'image' key containing input images
#             label: Optional label for conditional encoding
#             return_expert_outputs: Whether to return individual expert outputs
        
#         Returns:
#             Encoded representation (and optionally expert outputs and gates)
#         """
#         # Prepare input
#         x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
#         x = x.permute(0, 3, 1, 2)
#         batch_size = x.shape[0]
        
#         # Add label conditioning if provided
#         if label is not None and self.label_num is not None:
#             label_input = self.label_use[:, label]
#             bs, c, h, w = x.size()
#             label_input = label_input.repeat(bs, 1, h, w)
#             x_with_label = torch.cat((label_input.detach(), x), 1)
#         else:
#             x_with_label = x
        
#         # Compute gating weights if enabled
#         if self.use_gating:
#             gate_input = x_with_label
#             gate_logits = self.gate_network(gate_input)
#             gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)
#         else:
#             gate_weights = torch.ones(batch_size, self.n_experts).to(x.device) / self.n_experts
        
#         # Process through each expert encoder
#         expert_outputs = []
#         for i, encoder in enumerate(self.expert_encoders):
#             expert_out = encoder(x_with_label)
#             expert_out = expert_out.reshape(batch_size, -1)
#             expert_outputs.append(expert_out)
        
#         # Stack expert outputs: [n_experts, batch_size, feature_dim]
#         expert_outputs = torch.stack(expert_outputs, dim=0)
        
#         # Apply orthogonalization
#         orthogonal_outputs = self.orthogonal_layer(expert_outputs)
        
#         # Weighted aggregation
#         # Apply gating weights and aggregation weights
#         weighted_outputs = []
#         for i in range(self.n_experts):
#             weight = gate_weights[:, i:i+1] * self.aggregation_weights[i]
#             weighted_outputs.append(orthogonal_outputs[i] * weight)
        
#         # Sum weighted expert outputs
#         final_output = torch.sum(torch.stack(weighted_outputs, dim=0), dim=0)
        
#         # Reshape to match original output format
#         shape = list(obs['image'].shape[:-3]) + [final_output.shape[-1]]
#         final_output = final_output.reshape(shape)
        
#         if return_expert_outputs:
#             return final_output, {
#                 'expert_outputs': orthogonal_outputs,
#                 'gate_weights': gate_weights,
#                 'aggregation_weights': self.aggregation_weights
#             }
        
#         return final_output


# class OrthogonalLayer(nn.Module):
#     """
#     Orthogonalization layer using Gram-Schmidt process.
#     Adapted for convolutional encoder outputs.
#     """
    
#     def __init__(self):
#         super(OrthogonalLayer, self).__init__()
    
#     def forward(self, x):
#         """
#         Apply Gram-Schmidt orthogonalization.
        
#         Args:
#             x: Tensor of shape [n_experts, batch_size, feature_dim]
        
#         Returns:
#             Orthogonalized tensor of same shape
#         """
#         # Transpose to [batch_size, n_experts, feature_dim]
#         x = x.transpose(0, 1)
        
#         # Normalize first expert output
#         basis = x[:, 0:1, :] / (torch.norm(x[:, 0:1, :], dim=2, keepdim=True) + 1e-8)
        
#         # Gram-Schmidt process for remaining experts
#         for i in range(1, x.shape[1]):
#             v = x[:, i:i+1, :]
            
#             # Project v onto existing basis vectors and subtract
#             projections = torch.sum(v * basis, dim=2, keepdim=True) * basis
#             w = v - torch.sum(projections, dim=1, keepdim=True)
            
#             # Normalize and add to basis
#             w_norm = w / (torch.norm(w, dim=2, keepdim=True) + 1e-8)
#             basis = torch.cat([basis, w_norm], dim=1)
        
#         # Transpose back to [n_experts, batch_size, feature_dim]
#         return basis.transpose(0, 1)


class OrthogonalLayer(nn.Module):
    """
    Orthogonalize expert features along the expert axis using Gram–Schmidt.
    Input:  feats  [B, N, F]  (batch, n_experts, feature_dim)
    Output: ortho  [B, N, F]  where experts for each batch are orthonormal
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
      B, N, Fdim = feats.shape
      Q = torch.zeros_like(feats)
      for i in range(N):
        v = feats[:, i, :]  # [B, F]
        if i > 0:
          proj = torch.zeros_like(v)
          for j in range(i):
            qj = Q[:, j, :]                            # [B, F]
            alpha = (v * qj).sum(dim=-1, keepdim=True) # [B, 1]
            proj = proj + alpha * qj
          v = v - proj
        norm = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        Q[:, i, :] = v / norm
      return Q


class MoEOrthogonalConvEncoder(nn.Module):
    """
    Task-weighted Mixture-of-Experts Conv Encoder with optional orthogonalization.

    - Builds N parallel conv encoders (experts).
    - Computes per-task mixture weights w from label one-hots via a linear map.
    - (Optional) Orthogonalizes expert features across experts before aggregation.
    - Aggregates with a single weighted sum: y = (w @ experts) ∈ [..., F]
    - Output shape matches a standard ConvEncoder: leading dims from obs['image'].

    Args:
        n_experts: number of experts
        grayscale: if True, first conv expects 1 channel else 3
        depth: base channels for conv stacks
        act: activation class (e.g., nn.ReLU)
        kernels: tuple of kernel sizes for the conv stack
        label_num: number of discrete contexts; if provided, enables task weights
        agg_activation: (pre, post) activations around aggregation, e.g. ('ReLU','ReLU')
        use_orthogonal: apply OrthogonalLayer before weighted sum
    """
    def __init__(
      self,
      n_experts: int = 4,
      grayscale: bool = False,
      depth: int = 32,
      act=nn.ReLU,
      kernels=(4, 4, 4, 4),
      label_num=1,
      agg_activation=('ReLU', 'ReLU'),
      use_orthogonal: bool = True,
      ):
      super().__init__()
      self.n_experts = n_experts
      self.grayscale = grayscale
      self.kernels = kernels
      self._act = act
      self.depth = depth
      self.label_num = label_num
      self.use_orthogonal = use_orthogonal

      # Expert encoders
      self.experts = nn.ModuleList(
        [self._make_encoder(grayscale, depth, act, kernels, label_num) for _ in range(n_experts)]
      )

      # Task weight encoder (one-hot label -> weights over experts)
      if label_num is not None:
        self._task_encoder = nn.Linear(label_num, n_experts, bias=False)
        nn.init.xavier_uniform_(self._task_encoder.weight, gain=nn.init.calculate_gain('linear'))
      else:
        self._task_encoder = None

      # Optional orthogonalization
      self.orthogonal_layer = OrthogonalLayer() if use_orthogonal else None

      # Pre/post aggregation activations
      self._agg_activation = tuple(agg_activation) if isinstance(agg_activation, (list, tuple)) else (str(agg_activation), 'ReLU')

    def _make_encoder(self, grayscale, depth, act, kernels, label_num):
      return ConvEncoder(
        grayscale=grayscale,
        depth=depth,
        act=act,
        kernels=kernels,
        label_num=label_num,
        include_label=False
      )
      # layers = []
      
      # for i, kernel in enumerate(self.kernels):
      #   if i == 0:
      #     if grayscale:
      #       inp_dim = 1
      #     else:
      #         inp_dim = 3
      #   else:
      #     inp_dim = 2 ** (i-1) * self.depth
      #   d = 2 ** i * self.depth
      #   layers.append(nn.Conv2d(inp_dim, d, kernel, 2))
      #   layers.append(act())
      # self.layers = nn.Sequential(*layers)

      # return self.layers

    def forward(self, obs: dict, label=None, return_expert_outputs: bool = False):
      # obs['image']: [..., H, W, C]  ->  [B, C, H, W]
      tmp = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
      tmp = tmp.permute(0, 3, 1, 2)  # NCHW
      B = tmp.shape[0]
      # x_in = x

      # Per-expert features -> list of [B, T, F], stack -> [B, N, T, F]
      expert_feats = [enc(obs) for enc in self.experts]
      feats = torch.stack(expert_feats, dim=1)

      # Pre-aggregation activation
      if self._agg_activation and self._agg_activation[0] and self._agg_activation[0].lower() != 'linear':
        feats = getattr(torch, self._agg_activation[0].lower())(feats)

      # ------- ORTHOGONALIZE PER (B, T) --------     
      if self.orthogonal_layer is not None:
        B, N, T, Fdim = feats.shape
        # Flatten (B, T) into batch: [B*T, N, F]
        feats_bt = feats.permute(0, 2, 1, 3).contiguous().view(B * T, N, Fdim)
        feats_bt = self.orthogonal_layer(feats_bt)          # [B*T, N, F]
        # Back to [B, N, T, F]
        feats = feats_bt.view(B, T, N, Fdim).permute(0, 2, 1, 3).contiguous()
      # -----------------------------------------

      # Compute task weights w: [B, N]
      if (self._task_encoder is not None) and (label is not None):
        if not isinstance(label, torch.Tensor):
          label = torch.tensor(label, device=feats.device)
        label = label.long().view(-1)
        c_onehot = F.one_hot(label, num_classes=self.label_num).float()
        w = self._task_encoder(c_onehot)  # [B, N]
      else:
        w = torch.ones(B, self.n_experts, device=feats.device) / float(self.n_experts)

      # Weighted sum across experts: [B, 1, N] @ [B, N, F] -> [B, F]
      # out = torch.bmm(w.unsqueeze(1), feats).squeeze(1)
      
      # feats: [B, N, T, F]; w: [B, N]
      # Weighted sum over experts -> out: [B, T, F]
      out = torch.einsum('bn,bntf->btf', w, feats)

      # Post-aggregation activation
      if self._agg_activation and self._agg_activation[1] and self._agg_activation[1].lower() != 'linear':
        out = getattr(torch, self._agg_activation[1].lower())(out)

      # Restore leading dims to match ConvEncoder: [..., F]
      shape = list(obs['image'].shape[:-3]) + [out.shape[-1]]
      out = out.reshape(shape)

      if return_expert_outputs:
        return out, {"expert_outputs": expert_feats, "task_weights": w}
      return out
      
      
class ConvEncoder(nn.Module):

  def __init__(self, grayscale=False,
               depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4), label_num=None, include_label=True):
    super(ConvEncoder, self).__init__()
    self._act = act
    self._depth = depth
    self._kernels = kernels
    self._include_label = include_label

    layers = []
    # print(label_num)
    for i, kernel in enumerate(self._kernels):
      if i == 0:
        if grayscale:
          inp_dim = 1
        else:
          if label_num is not None and self._include_label:
            inp_dim = 3 + label_num
          else:
            inp_dim = 3
      else:
        inp_dim = 2 ** (i-1) * self._depth
      depth = 2 ** i * self._depth
      layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
      layers.append(act())
    self.layers = nn.Sequential(*layers)

    if include_label:
      if label_num is not None:
        label_tmp = torch.eye(label_num).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)

  def __call__(self, obs, label=None):
    x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
    x = x.permute(0, 3, 1, 2)
    
    if self._include_label:
      if label is not None:
        label_input = self.label_use[:, label]
        bs, c, h , w = x.size()
        label_input = label_input.repeat(bs, 1, h, w)
        x = torch.cat((label_input.detach(), x), 1)
    
    x = self.layers(x)
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    shape = list(obs['image'].shape[:-3]) + [x.shape[-1]]
    return x.reshape(shape)



class ConvDecoder(nn.Module):

  def __init__(
      self, inp_depth,
      depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6),
      thin=True, label_num=None):
    super(ConvDecoder, self).__init__()
    self._inp_depth = inp_depth
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

    if self._thin:
      if label_num is not None:
        self._linear_layer = nn.Linear(inp_depth + label_num, 32 * self._depth)
      else:
        self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
    else:
      if label_num is not None:
        self._linear_layer = nn.Linear(inp_depth + label_num, 128 * self._depth)
      else:
        self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
    inp_dim = 32 * self._depth

    cnnt_layers = []
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        #depth = self._shape[-1]
        depth = self._shape[0]
        act = None
      if i != 0:
        inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
      cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
      if act is not None:
        cnnt_layers.append(act())
    self._cnnt_layers = nn.Sequential(*cnnt_layers)

    if label_num is not None:
      label_tmp = torch.eye(label_num).unsqueeze(0).unsqueeze(0)
      self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)

  def __call__(self, features, label=None, dtype=None):
    if label is not None:
      bs, num, c = features.size()
      label_use = self.label_use[:, :, label]
      label_use = label_use.repeat(bs, num, 1)
      features = torch.cat((label_use.detach(), features), -1)

    if self._thin:
      x = self._linear_layer(features)
      x = x.reshape([-1, 1, 1, 32 * self._depth])
      x = x.permute(0,3,1,2)
    else:
      x = self._linear_layer(features)
      x = x.reshape([-1, 2, 2, 32 * self._depth])
      x = x.permute(0,3,1,2)
    x = self._cnnt_layers(x)
    mean = x.reshape(features.shape[:-1] + self._shape)
    mean = mean.permute(0, 1, 3, 4, 2)
    return tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mean, 1), len(self._shape)))
    

class DenseHead(nn.Module):

  def __init__(
      self, inp_dim,
      shape, layers, units, act=nn.ELU, dist='normal', std=1.0, label_num=None):
    super(DenseHead, self).__init__()
    self._shape = (shape,) if isinstance(shape, int) else shape
    if len(self._shape) == 0:
      self._shape = (1,)
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

    mean_layers = []

    if label_num is not None:
      inp_dim += label_num
    for index in range(self._layers):
      mean_layers.append(nn.Linear(inp_dim, self._units))
      mean_layers.append(act())
      if index == 0:
        inp_dim = self._units
    mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
    self._mean_layers = nn.Sequential(*mean_layers)

    if self._std == 'learned':
      self._std_layer = nn.Linear(self._units, np.prod(self._shape))

    if label_num is not None:
      label_tmp = torch.eye(label_num).unsqueeze(0).unsqueeze(0)
      self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)

  def __call__(self, features, label=None, dtype=None):

    if label is not None:
      bs, num, c = features.size()
      label_use = self.label_use[:, :, label]
      label_use = label_use.repeat(bs, num, 1)
      features = torch.cat((label_use.detach(), features), -1)

    x = features
    mean = self._mean_layers(x)
    if self._std == 'learned':
      std = self._std_layer(x)
      std = torch.softplus(std) + 0.01
    else:
      std = self._std
    if self._dist == 'normal':
      return tools.ContDist(torchd.independent.Independent(
        torchd.normal.Normal(mean, std), len(self._shape)))
    if self._dist == 'huber':
      return tools.ContDist(torchd.independent.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
    if self._dist == 'binary':
      return tools.Bernoulli(torchd.independent.Independent(
        torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
    raise NotImplementedError(self._dist)


class ActionHead(nn.Module):

  def __init__(
      self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    super(ActionHead, self).__init__()
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

    pre_layers = []
    for index in range(self._layers):
      pre_layers.append(nn.Linear(inp_dim, self._units))
      pre_layers.append(act())
      if index == 0:
        inp_dim = self._units
    self._pre_layers = nn.Sequential(*pre_layers)

    if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
      self._dist_layer = nn.Linear(self._units, 2 * self._size)
    elif self._dist in ['normal_1','onehot','onehot_gumbel']:
      self._dist_layer = nn.Linear(self._units, self._size)

  def __call__(self, features, dtype=None):
    x = features
    x = self._pre_layers(x)
    if self._dist == 'tanh_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = torch.tanh(mean)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = 5 * torch.tanh(mean / 5)
      std = F.softplus(std + 5) + 5
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'normal_1':
      x = self._dist_layer(x)
      dist = torchd.normal.Normal(mean, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'trunc_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, [self._size]*2, -1)
      mean = torch.tanh(mean)
      std = 2 * torch.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'onehot':
      x = self._dist_layer(x)
      dist = tools.OneHotDist(x)
    elif self._dist == 'onehot_gumble':
      x = self._dist_layer(x)
      temp = self._temp
      dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1/temp))
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(nn.Module):

  def __init__(self, inp_size,
               size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]
  

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
  def __init__(self, state_dim, action_dim, latent_dim, max_action, device, config):
    super(VAE, self).__init__()
    self.config = config
    self.e1 = nn.Linear(state_dim + action_dim + config.num_teachers, 750)
    self.e2 = nn.Linear(750, 750)

    self.mean = nn.Linear(750, latent_dim)
    self.log_std = nn.Linear(750, latent_dim)

    self.d1 = nn.Linear(state_dim + latent_dim + config.num_teachers, 750)
    self.d2 = nn.Linear(750, 50)
    self.d3 = nn.Linear(50, action_dim)

    self.max_action = max_action
    self.latent_dim = latent_dim
    self.device = device

    self.label_use = torch.eye(config.num_teachers).to(device)

  # def forward(self, state, action, label):  

  #   label_use = self.label_use[label]  
  #   z = F.relu(self.e1(torch.cat([state, action, label_use], -1)))
  #   z = F.relu(self.e2(z))

  #   mean = self.mean(z)
  #   # Clamped for numerical stability 
  #   log_std = self.log_std(z).clamp(-4, 15)
  #   std = torch.exp(log_std)
  #   z = mean + std * torch.randn_like(std)  ## sigma
    
  #   u, s = self.decode(state, label, z)

  #   return u, mean, std
  def forward(self, state, action, label):
    # Extract label embedding
    label_use = self.label_use[label]  # Shape: [num_teachers]
    
    # IMPORTANT: Broadcast label_use to match state/action dimensions
    # If state is [B, T, D_s] and action is [B, T, A], we need label_use to be [B, T, num_teachers]
    
    if len(state.shape) == 3:
        # Temporal dimension exists: [B, T, D]
        batch_size, time_steps, _ = state.shape
        # Reshape label_use from [num_teachers] to [1, 1, num_teachers]
        # then broadcast to [B, T, num_teachers]
        label_use = label_use.unsqueeze(0).unsqueeze(0)
        label_use = label_use.expand(batch_size, time_steps, -1)
    elif len(state.shape) == 2:
        # No temporal dimension: [B, D]
        batch_size, _ = state.shape
        # Reshape label_use from [num_teachers] to [1, num_teachers]
        # then broadcast to [B, num_teachers]
        label_use = label_use.unsqueeze(0)
        label_use = label_use.expand(batch_size, -1)
    
    # Now all tensors have matching dimensions
    z = F.relu(self.e1(torch.cat([state, action, label_use], -1)))
    z = F.relu(self.e2(z))
    
    mean = self.mean(z)
    log_std = self.log_std(z).clamp(-4, 15)
    std = torch.exp(log_std)
    z = mean + std * torch.randn_like(std)
    
    u, s = self.decode(state, label, z)
    return u, mean, std

  # def decode(self, state, label, z=None):  
  #   # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
  #   if z is None:
  #     z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
  #   label_use = self.label_use[label]  
    
  #   if len(state.shape) == 3:
  #     seq, _, _ = state.size()
  #     label_use = label_use.squeeze(0).repeat(seq, 1, 1)

  #   a = F.relu(self.d1(torch.cat([state, z, label_use], -1)))
  #   a = F.relu(self.d2(a))
  #   return self.max_action * torch.tanh(self.d3(a)), a
  
  # def decode(self, state, label, z=None):
  #   if z is None:
  #       z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
    
  #   label_use = self.label_use[label]  # Shape: [num_teachers]
    
  #   # Broadcast label_use to match state dimensions
  #   if len(state.shape) == 3:
  #       batch_size, seq, _ = state.size()
  #       # Reshape from [num_teachers] to [1, 1, num_teachers]
  #       # then broadcast to [batch_size, seq, num_teachers]
  #       label_use = label_use.unsqueeze(0).unsqueeze(0)
  #       label_use = label_use.expand(batch_size, seq, -1)
  #   elif len(state.shape) == 2:
  #       batch_size = state.shape[0]
  #       # Reshape from [num_teachers] to [1, num_teachers]
  #       # then broadcast to [batch_size, num_teachers]
  #       label_use = label_use.unsqueeze(0)
  #       label_use = label_use.expand(batch_size, -1)
    
  #   # Broadcast z if needed
  #   if z is not None and len(state.shape) == 3 and len(z.shape) == 2:
  #       seq = state.shape[1]
  #       z = z.unsqueeze(1).expand(-1, seq, -1)
    
  #   a = F.relu(self.d1(torch.cat([state, z, label_use], -1)))
  #   a = F.relu(self.d2(a))
  #   return self.max_action * torch.tanh(self.d3(a)), a
  
  def decode(self, state, label, z=None):
    if z is None:
      z = torch.randn((state.shape[0], self.latent_dim), device=self.device).clamp(-0.5, 0.5) # z is never used

    # label -> [num_teachers]
    # print(self.label_use.shape)
    label_use = self.label_use[label]
    # print(label)
    # print('t', label_use.shape)

    if state.dim() == 3:
      # state: [B, T, Ds]  -> label_use: [B, T, num_teachers]
      B, T, _ = state.shape
      label_use = label_use.view(1, 1, -1).expand(B, T, -1)
      if z is not None and z.dim() == 2:
        # z: [B, Dz] -> [B, T, Dz]
        z = z.unsqueeze(1).expand(B, T, -1)
    # elif state.dim() == 2:
    #   # state: [B, Ds] -> label_use: [B, num_teachers]
    #   B, _ = state.shape
    #   print('h', label_use.shape)
    #   label_use = label_use.view(1, -1).expand(B, -1)
    #   print(label_use.shape)
      # label_use = label_use.unsqueeze(0)
      # label_use = label_use.expand(B, -1)
    elif state.dim() == 2:
      # state: [B, Ds]
      B, _ = state.shape

      # --- shape label_use to [B, num_teachers] WITHOUT flattening ---
      if label_use.dim() == 1:
        # one global label (num_teachers,)
        label_use = label_use.view(1, -1).expand(B, -1)
      elif label_use.dim() == 2:
        if label_use.shape[0] == 1:
          label_use = label_use.expand(B, -1)
        elif label_use.shape[0] == B:
          pass  # already per-sample
        else:
          raise ValueError(f"Unexpected label_use shape for [B]: {label_use.shape}")
    else:
      raise ValueError(f"Unexpected state shape: {state.shape}")

    a = F.relu(self.d1(torch.cat([state, z, label_use], dim=-1)))
    a = F.relu(self.d2(a))
    return self.max_action * torch.tanh(self.d3(a)), a

  
# class MoEBlock(nn.Module):
#   """
#   Mixture-of-Experts feature block placed **after RSSM**.
#   - Supports task-weight gating (pass `gate_weight` of shape [..., n_experts])
#     or learned gating from the input when `gate_weight` is None.
#   - Returns (y, losses) where losses contains:
#       - 'orth_loss': expert output orthogonality penalty (Gram off-diagonal).
#       - 'load_balance_loss': entropy regularizer for gating distribution.
#   """
#   def __init__(self, inp_dim, out_dim=None, n_experts=4, units=256, layers=1, act=nn.ELU, gate_learned=True, orth_lambda=0.0, lb_lambda=0.0):
#     super(MoEBlock, self).__init__()
#     self.inp_dim = inp_dim
#     self.out_dim = out_dim or inp_dim
#     self.n_experts = n_experts
#     self.units = units
#     self.layers = layers
#     self.act = act
#     self.orth_lambda = orth_lambda
#     self.lb_lambda = lb_lambda
#     # Experts
#     self.experts = nn.ModuleList()
#     for _ in range(n_experts):
#       layers_list = []
#       d = inp_dim
#       for _ in range(layers):
#         layers_list += [nn.Linear(d, units), act()]
#         d = units
#       layers_list += [nn.Linear(d, self.out_dim)]
#       self.experts.append(nn.Sequential(*layers_list))
#     # Gating network (only used if gate_weight not provided)
#     self.gate_learned = gate_learned
#     if gate_learned:
#       self.gate_net = nn.Sequential(
#           nn.Linear(inp_dim, max(64, self.n_experts)),
#           act(),
#           nn.Linear(max(64, self.n_experts), self.n_experts)
#       )
#     # Orthogonal init for stability
#     for m in self.modules():
#       if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight)
#         if m.bias is not None:
#           nn.init.zeros_(m.bias)

#   def _shape_flat(self, x):
#     # Accept [B,D] or [T,B,D] or [*,D]
#     orig = x.shape
#     x = x.reshape(-1, x.shape[-1])
#     return x, orig

#   def forward(self, x, gate_weight=None):
#     # x: [..., D]
#     x_flat, orig = self._shape_flat(x)
#     # Expert outputs: list of [N, out_dim] -> tensor [E, N, out_dim]
#     outs = []
#     for e in self.experts:
#       outs.append(e(x_flat))
#     E = torch.stack(outs, dim=0)  # [E, N, O]

#     # Gating probs
#     if gate_weight is not None:
#       gw = gate_weight
#       # Support [B,E] or [T,B,E] -> [N,E]
#       if gw.dim() == 2 and len(orig) == 3:  # [B,E] and x is [T,B,D]
#         T = orig[0]
#         gw = gw.unsqueeze(0).expand(T, *gw.shape)
#       gw_flat = gw.reshape(-1, gw.shape[-1])
#       logits = gw_flat
#     else:
#       if not self.gate_learned:
#         raise ValueError("gate_weight is None and gate_learned=False")
#       logits = self.gate_net(x_flat)  # [N,E]
#     probs = torch.softmax(logits, dim=-1)  # [N,E]

#     # Combine: y_n = sum_e p_{n,e} * y_{e,n}
#     # E: [E,N,O], probs: [N,E]
#     y = torch.einsum('eno,ne->no', E, probs)

#     # Losses
#     losses = {}
#     if self.orth_lambda > 0:
#       # compute Gram across experts for each sample: G_n = Y_n^T Y_n with Y_n in R^{E x O}
#       # Normalize each expert output vector to unit norm for cosine sim.
#       Yn = F.normalize(E.permute(1,0,2), dim=-1)  # [N,E,O]
#       G = torch.matmul(Yn, Yn.transpose(1,2))     # [N,E,E]
#       off_diag = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
#       orth_loss = (off_diag**2).mean()
#       losses['orth_loss'] = self.orth_lambda * orth_loss
#     else:
#       losses['orth_loss'] = x_flat.new_tensor(0.0)

#     if self.lb_lambda > 0:
#       # Encourage high-entropy, balanced gating (maximally mixed)
#       ent = -(probs * (probs.clamp_min(1e-8).log())).sum(-1).mean()
#       # Max entropy is log(E). Penalize negative gap.
#       max_ent = math.log(self.n_experts)
#       lb = (max_ent - ent).clamp_min(0.0)
#       losses['load_balance_loss'] = self.lb_lambda * lb
#     else:
#       losses['load_balance_loss'] = x_flat.new_tensor(0.0)

#     y = y.reshape(*orig[:-1], self.out_dim)
#     return y, losses
