from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
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
                (lambda xx: torch.transpose(xx, dim0=-1, dim1=-2), 'x -> x'),
                (global_add_pool, 'x, batch -> x'),
                (layer_init(nn.Linear(action_dim, 32)), 'x -> x')
            ]
        )
        self.policy_batch = torch.LongTensor([0 for _ in range(last_layer_dim)])
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
        self.value_batch = torch.LongTensor([0 for _ in range(self.num_pos)])
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
        
    def get_value(self, x, edge_index, share_state):
        h1 = self.value_gnet(x, edge_index, self.value_batch).squeeze(dim=-2)
        h2 = self.value_net(share_state)
        value = self.value_comnet(torch.cat([h1, h2], dim=-1))
        return value

    def get_distribution(self, x, edge_index, state, mask):
        h1 = self.policy_gnet(x, edge_index, self.policy_batch)
        h2 = self.policy_net(state)
        if h1.shape != h2.shape:
            h1 = h1.squeeze(dim=-2)
        log_prob = self.policy_comnet(torch.cat([h1, h2], dim=-1))
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)

class Network(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List, value_arch: List,
        args
        ):
        super(Network, self).__init__()
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

    def get_distribution(self, state):
        if self.action_list.dim() == state.dim():
            mask = (self.action_list > state[0]+0.05).long().bool()
        else:
            action_list = self.action_list.unsqueeze(0).repeat_interleave(state.shape[0], 0)
            state_soc = state[:, 0].unsqueeze(1).repeat_interleave(self.action_dim, 1)
            mask = (action_list > state_soc+0.05).long().bool()
        log_prob = self.policy_net(state)
        masked_logit = log_prob.masked_fill((~mask).bool(), -1e32)
        return Categorical(logits=masked_logit)

class PPOAgent(object):
    def __init__(
            self, 
            state_dim, share_dim, caction_dim, caction_list,
            obs_features_shape, global_features_shape, raction_dim, raction_list,
            edge_index, buffer, device, 
            args
        ):
        self.device = device
        # 充电相关
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.caction_dim = caction_dim
        self.caction_list = torch.Tensor(caction_list).to(self.device)
        # 路径相关
        self.obs_features_shape = obs_features_shape
        self.global_features_shape = global_features_shape
        self.raction_dim = raction_dim
        self.raction_list = torch.Tensor(raction_list).to(self.device)
        
        self.num_update = args.num_update
        self.k_epoch = args.k_epoch
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.lr = args.lr
        self.eps_clip = args.eps_clip
        self.grad_clip = args.max_grad_clip
        self.entropy_coef = args.entropy_coef
        
        self.ps = args.ps
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_index_shape = edge_index.shape
        
        self.charge_network = Network(
            state_dim, share_dim, caction_dim, self.caction_list, 
            args.policy_arch, args.value_arch,
            args
            ).to(self.device)
        self.charge_optimizer = torch.optim.Adam(self.charge_network.parameters(), lr=self.lr, eps=1e-5)

        self.route_network = GNet(
            obs_features_shape, global_features_shape, state_dim, share_dim,
            raction_dim, self.raction_list, 
            args
            ).to(self.device)
        self.route_network.policy_batch = self.route_network.policy_batch.to(self.device)
        self.route_network.value_batch = self.route_network.value_batch.to(self.device)
        self.route_optimizer = torch.optim.Adam(self.route_network.parameters(), lr=self.lr, eps=1e-5)

        self.rolloutBuffer = buffer

    def select_caction(self, state):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.charge_network.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_caction(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            dist = self.charge_network.get_distribution(state)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def select_raction(self, obs_feature, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, state=state, mask=mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_raction(self, obs_feature, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, state=state, mask=mask)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def train(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            obs_feature, global_cs_feature, rstate, share_rstate,\
                raction, raction_mask, rlog_prob, rreward, \
                    next_obs_feature, next_global_cs_feature, next_rstate, next_share_rstate,\
                        rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        
        with torch.no_grad():
            # there are N = num_env independent environments, cannot flatten state here
            # let "values" match the dimension of "done"
            # 充电部分
            cvalues = self.charge_network.get_value(share_state).squeeze(dim=-1)
            next_cvalues = self.charge_network.get_value(next_share_state).squeeze(dim=-1)
            cadvantage = torch.zeros_like(cvalues).to(self.device)
            cdelta = creward + self.gamma * (1 - cdone) * next_cvalues - cvalues
            cgae = 0
            # 路径部分
            rvalues = self.route_network.get_value(x=global_cs_feature, edge_index=self.edge_index, share_state=share_rstate).view(buffer_step, -1)
            next_rvalues = self.route_network.get_value(x=next_global_cs_feature, edge_index=self.edge_index, share_state=next_share_rstate).view(buffer_step, -1)
            radvantage = torch.zeros_like(rvalues).to(self.device)
            rdelta = rreward + self.gamma * (1 - rdone) * next_rvalues - rvalues
            rgae = 0
            for t in reversed(range(buffer_step)):
                cgae = cdelta[t] + self.gamma * self.gae_lambda * cgae * (1 - cdone[t])
                cadvantage[t] = cgae
                rgae = rdelta[t] + self.gamma * self.gae_lambda * rgae * (1 - rdone[t])
                radvantage[t] = rgae
            creturns = cadvantage + cvalues
            norm_cadv = (cadvantage - cadvantage.mean()) / (cadvantage.std() + 1e-8)
            rreturns = radvantage + rvalues
            norm_radv = (radvantage - radvantage.mean()) / (radvantage.std() + 1e-8)
            
        # -------- flatten vectorized environment --------
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network
        # 充电部分
        state = state.view(-1, self.state_dim)
        share_state = share_state.view(-1, self.share_dim)
        rstate = rstate.view(-1, self.state_dim)
        share_rstate = share_rstate.view(-1, self.share_dim)
        caction = caction.view(-1, 1)
        clog_prob = clog_prob.view(-1, 1)
        creturns = creturns.view(-1, 1)
        norm_cadv = norm_cadv.view(-1, 1)
        # 路径部分
        obs_feature = obs_feature.view(-1, self.obs_features_shape[0], self.obs_features_shape[1])
        global_cs_feature = global_cs_feature.view(-1, self.global_features_shape[0], self.global_features_shape[1])
        raction = raction.view(-1, 1)
        raction_mask = raction_mask.view(-1, self.raction_dim)
        rlog_prob = rlog_prob.view(-1, 1)
        rreturns = rreturns.view(-1, 1)
        norm_radv = norm_radv.view(-1, 1)
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                # 充电部分
                new_cdist = self.charge_network.get_distribution(state[index])
                new_clog_prob = new_cdist.log_prob(caction[index].squeeze()).unsqueeze(1)
                new_cvalues = self.charge_network.get_value(share_state[index]).view(self.mini_batch_size, -1)
                centropy = new_cdist.entropy()
                cratios = torch.exp(new_clog_prob - clog_prob[index])

                csurrogate1 = cratios * norm_cadv[index]
                csurrogate2 = torch.clamp(cratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_cadv[index]
                actor_closs = (-1 * torch.min(csurrogate1, csurrogate2)).mean()
                entropy_closs = (self.entropy_coef * centropy).mean()
                critic_closs = 0.5 * torch.nn.functional.mse_loss(new_cvalues, creturns[index])
                closs = actor_closs - entropy_closs + critic_closs
                # 路径部分
                new_rdist = self.route_network.get_distribution(x=obs_feature[index], edge_index=self.edge_index, mask=raction_mask[index], state=rstate)
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_rvalues = self.route_network.get_value(x=global_cs_feature[index], edge_index=self.edge_index, share_state=share_rstate).view(self.mini_batch_size, -1)
                rentropy = new_rdist.entropy()
                rratios = torch.exp(new_rlog_prob - rlog_prob[index])

                rsurrogate1 = rratios * norm_radv[index]
                rsurrogate2 = torch.clamp(rratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_radv[index]
                actor_rloss = (-1 * torch.min(rsurrogate1, rsurrogate2)).mean()
                entropy_rloss = (self.entropy_coef * rentropy).mean()
                critic_rloss = 0.5 * torch.nn.functional.mse_loss(new_rvalues, rreturns[index])
                rloss = actor_rloss - entropy_rloss + critic_rloss
                
                self.charge_optimizer.zero_grad()
                self.route_optimizer.zero_grad()
                closs.backward()
                rloss.backward()
                nn.utils.clip_grad_norm_(self.charge_network.parameters(), self.grad_clip) # type: ignore
                nn.utils.clip_grad_norm_(self.route_network.parameters(), self.grad_clip) # type: ignore
                self.charge_optimizer.step()
                self.route_optimizer.step()
                
        return actor_closs.item(), critic_closs.item(), entropy_closs.item(), \
            actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()

    def lr_decay(self, step):
        return self.lr
        factor = 1 - step / self.num_update
        lr = factor * self.lr
        for p in self.charge_optimizer.param_groups:
            p['lr'] = lr
        for p in self.route_optimizer.param_groups:
            p['lr'] = lr
        return lr

    def save(self, filename):
        torch.save(self.charge_network.state_dict(), "{}_c.pt".format(filename))
        torch.save(self.charge_optimizer.state_dict(), "{}_c_optimizer.pt".format(filename))
        torch.save(self.route_network.state_dict(), "{}_r.pt".format(filename))
        torch.save(self.route_optimizer.state_dict(), "{}_r_optimizer.pt".format(filename))

    def load(self, filename):
        self.charge_network.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.charge_optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        self.route_network.load_state_dict(torch.load("{}_r.pt".format(filename)))
        self.route_optimizer.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))
