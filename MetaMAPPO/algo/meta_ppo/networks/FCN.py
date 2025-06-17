from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def layer_init(layer, gain=np.sqrt(2), bias=0.):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer

class cActor(nn.Module):
    def __init__(
            self, 
            state_dim, share_dim, action_dim, 
            action_list, policy_arch: List,
            args
        ):
        super(cActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_dim = share_dim
        self.action_list = action_list
        # -------- init policy network --------
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.append(layer_init(nn.Linear(last_layer_dim, action_dim), gain=0.01))
        self.policy_net = nn.Sequential(*policy_net)

    def get_distribution(self, state):
        if self.action_list.dim() == state.dim():
            mask = (self.action_list > state[1]+0.05).long().bool()
        else:
            action_list = self.action_list.unsqueeze(0).repeat_interleave(state.shape[0], 0)
            state_soc = state[:, 1].unsqueeze(1).repeat_interleave(self.action_dim, 1)
            mask = (action_list > state_soc+0.05).long().bool()
        log_prob = self.policy_net(state)
        masked_logit = log_prob.masked_fill((~mask).bool(), -1e32)
        return Categorical(logits=masked_logit)

    def get_valid(self, state, theta):
        if self.action_list.dim() == state.dim():
            mask = (self.action_list > state[1]+0.05).long().bool()
        else:
            action_list = self.action_list.unsqueeze(0).repeat_interleave(state.shape[0], 0)
            state_soc = state[:, 1].unsqueeze(1).repeat_interleave(self.action_dim, 1)
            mask = (action_list > state_soc+0.05).long().bool()
        theta = list(theta.values())
        for i in range(0, len(theta)-2, 2):
            state = F.linear(state, theta[i], theta[i+1])
            state = F.tanh(state)
        log_prob = F.linear(state, theta[-2], theta[-1])
        masked_logit = log_prob.masked_fill((~mask).bool(), -1e32)
        return Categorical(logits=masked_logit)

class cCritic(nn.Module):
    def __init__(
            self, 
            state_dim, share_dim, action_dim, 
            action_list, value_arch: List,
            args
        ):
        super(cCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_dim = share_dim
        self.action_list = action_list
        # -------- init value network --------
        last_layer_dim = share_dim
        value_net = []
        for current_layer_dim in value_arch:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))

        self.value_net = nn.Sequential(*value_net)

    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

class rActor(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List,
        args
    ):
        super(rActor, self).__init__()
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
        self.policy_net = nn.Sequential(*policy_net)
        
    def get_distribution(self, state, mask):
        log_prob = self.policy_net(state)
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)

    def get_valid(self, state, mask, theta):
        theta = list(theta.values())
        for i in range(0, len(theta)-2, 2):
            state = F.linear(state, theta[i], theta[i+1])
            state = F.tanh(state)
        log_prob = F.linear(state, theta[-2], theta[-1])
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)

class rCritic(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, value_arch: List,
        args
    ):
        super(rCritic, self).__init__()
        # self.state_dim = state_shape[1]
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.action_dim = action_dim
        self.action_list = action_list
        # -------- init value network --------
        last_layer_dim = share_dim
        value_net = []
        for current_layer_dim in value_arch:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))

        self.value_net = nn.Sequential(*value_net)
        
    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value
