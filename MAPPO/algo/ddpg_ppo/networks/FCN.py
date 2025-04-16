from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, gain=np.sqrt(2), bias=0.):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer

class QNetwork(nn.Module):
    def __init__(self, share_dim, action_dim, value_arch):
        super().__init__()
        last_layer_dim = share_dim + 1
        value_net = []
        for current_layer_dim in value_arch:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))
        self.q_network = nn.Sequential(*value_net)
        
    def forward(self, x, a):
        return self.q_network(torch.cat([x, a], 1))

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, policy_arch):
        super().__init__()
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=0.01))
        policy_net.append(nn.Tanh())
        self.actor_network = nn.Sequential(*policy_net)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((1 - 0) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((1 + 0) / 2.0, dtype=torch.float32)
        )
        
    def forward(self, x):
        return self.actor_network(x) * self.action_scale + self.action_bias
    
class cNetwork(nn.Module):
    def __init__(
            self, 
            state_dim, share_dim, action_dim, 
            action_list, policy_arch: List, value_arch: List,
            args
        ):
        super(cNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_dim = share_dim
        self.action_list = action_list

        self.actor = ActorNetwork(state_dim, action_dim, policy_arch)
        self.actor_target = ActorNetwork(state_dim, action_dim, policy_arch)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qf = QNetwork(share_dim, action_dim, value_arch)
        self.qf_target = QNetwork(share_dim, action_dim, value_arch)
        self.qf_target.load_state_dict(self.qf.state_dict())

        self.exploration_noise = 0.5
    # def get_value(self, share_state):
    #     value = self.value_net(share_state)
    #     return value
    
    def get_distribution(self, state, train=False):
        if self.action_list.dim() == state.dim():
            state_soc = state[1]
        else:
            state_soc = state[:, 1].unsqueeze(1)
        actions = self.actor(state)
        if train:
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = torch.clip(actions, 0, 1)
        actions[actions < state_soc] = 0
        # actions = torch.where(actions < state_soc, 0, actions)
        return actions

class rNetwork(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List, value_arch: List,
        args
    ):
        super(rNetwork, self).__init__()
        # self.state_dim = state_shape[1]
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.action_dim = action_dim
        self.action_list = action_list
        # -------- init policy network --------
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.append(layer_init(nn.Linear(last_layer_dim, action_dim), gain=0.01))
        # -------- init value network --------
        last_layer_dim = share_dim
        value_net = []
        for current_layer_dim in value_arch:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))

        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)
        
    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

    def get_distribution(self, state, mask):
        log_prob = self.policy_net(state)
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)