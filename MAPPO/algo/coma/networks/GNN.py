from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_geometric.nn as gnn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv

def layer_init(layer, gain=np.sqrt(2), bias=0.):
    if isinstance(layer, GCNConv):
        nn.init.orthogonal_(layer.lin.weight, gain=gain)
    else:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, bias)
    return layer

class GNet(nn.Module):
    def __init__(
        self, 
        state_shape, share_shape, state_dim, share_dim,
        action_dim, action_list, 
        args
    ):
        super(GNet, self).__init__()
        self.obs_dim = state_shape[1]
        self.action_dim = action_dim
        self.share_obs_dim = share_shape[1]
        self.num_pos = state_shape[0]
        self.action_list = action_list
        self.state_dim = state_dim
        self.share_dim = share_dim
        # -------- init policy network --------
        last_layer_dim = self.obs_dim
        policy_net = []
        for current_layer_dim in [32, 32]:
            policy_net.append(
                (layer_init(GCNConv(last_layer_dim, current_layer_dim)), 'x, edge_index -> x')
                )
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.extend(
            [
                (layer_init(nn.Linear(last_layer_dim, last_layer_dim)), 'x -> x'),
                nn.Tanh(),
                (self.transpose, 'x -> x'),
                (global_add_pool, 'x, batch -> x'),
                (layer_init(nn.Linear(action_dim, 32)), 'x -> x')
            ]
        )
        self.policy_gnet = gnn.Sequential('x, edge_index, batch', policy_net)
        
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in [32, 32]:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.append(layer_init(nn.Linear(last_layer_dim, 32)))
        self.policy_net = nn.Sequential(*policy_net)
        
        policy_comnet = [
            layer_init(nn.Linear(32 * 2, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, action_dim), gain=0.01)
        ]
        self.policy_comnet = nn.Sequential(*policy_comnet)
        # -------- init value network --------
        last_layer_dim = self.share_obs_dim
        value_net = []
        for current_layer_dim in [32, 32]:
            value_net.append(
                (layer_init(GCNConv(last_layer_dim, current_layer_dim)), 'x, edge_index -> x')
                )
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.extend(
            [
                (layer_init(nn.Linear(last_layer_dim, last_layer_dim)), 'x -> x'),
                nn.Tanh(),
                (global_add_pool, 'x, batch -> x'),
                (layer_init(nn.Linear(last_layer_dim, 32)), 'x -> x')
            ]
        )
        self.value_gnet = gnn.Sequential('x, edge_index, batch', value_net)
        
        last_layer_dim = share_dim
        value_net = []
        for current_layer_dim in [32, 32]:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 32)))
        self.value_net = nn.Sequential(*value_net)
        
        value_comnet = [
            layer_init(nn.Linear(32 * 2, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), gain=0.01)
        ]
        self.value_comnet = nn.Sequential(*value_comnet)
        
    def get_value(self, x, edge_index, share_state, value_batch):
        h1 = self.value_gnet(x, edge_index, value_batch).squeeze(dim=-2)
        h2 = self.value_net(share_state)
        value = self.value_comnet(torch.cat([h1, h2], dim=-1))
        return value

    def get_distribution(self, x, edge_index, state, mask, policy_batch):
        h1 = self.policy_gnet(x, edge_index, policy_batch)
        h2 = self.policy_net(state)
        if h1.shape != h2.shape:
            h1 = h1.squeeze(dim=-2)
        log_prob = self.policy_comnet(torch.cat([h1, h2], dim=-1))
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)
    
    def transpose(self, x):
        return torch.transpose(x, dim0=-1, dim1=-2)