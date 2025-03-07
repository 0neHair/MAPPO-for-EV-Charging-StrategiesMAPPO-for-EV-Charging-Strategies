from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
<<<<<<< HEAD
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
        state_shape, share_shape, action_dim, 
        action_list, 
        args
    ):
        super(GNet, self).__init__()
        self.state_dim = state_shape[1]
        self.action_dim = action_dim
        self.share_dim = share_shape[1]
        self.num_pos = state_shape[0]
        self.action_list = action_list
        # -------- init policy network --------
        last_layer_dim = self.state_dim
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
                (layer_init(nn.Linear(action_dim, action_dim), gain=0.01), 'x -> x')
            ]
        )
        self.policy_batch = torch.LongTensor([0 for _ in range(last_layer_dim)])
        self.policy_net = gnn.Sequential('x, edge_index, batch', policy_net)
        # -------- init value network --------
        last_layer_dim = self.share_dim
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
                (layer_init(nn.Linear(last_layer_dim, 1), gain=1.0), 'x -> x')
            ]
        )
        self.value_batch = torch.LongTensor([0 for _ in range(self.num_pos)])
        self.value_net = gnn.Sequential('x, edge_index, batch', value_net)
        
    def get_value(self, x, edge_index):
        value = self.value_net(x, edge_index, self.value_batch).squeeze(dim=-2)
        return value

    def get_distribution(self, x, edge_index, mask):
        log_prob = self.policy_net(x, edge_index, self.policy_batch).squeeze(dim=-2)
        masked_logit = log_prob.masked_fill((~mask.bool()), -1e32)
        return Categorical(logits=masked_logit)
=======


def layer_init(layer, gain=np.sqrt(2), bias=0.):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer

>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7

class Network(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List, value_arch: List,
<<<<<<< HEAD
        args
        ):
        super(Network, self).__init__()
=======
        agent_num, device
        ):
        super(Network, self).__init__()
        self.agent_num = agent_num
        self.device = device
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
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

<<<<<<< HEAD
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

=======
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        
        self.value_nets = []
        for _ in range(agent_num):
            last_layer_dim = share_dim
            value_net = []
            for current_layer_dim in value_arch:
                value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
                value_net.append(nn.Tanh())
                last_layer_dim = current_layer_dim
            value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))
            value_net = nn.Sequential(*value_net).to(device)
            self.value_nets.append(value_net)
        
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

<<<<<<< HEAD
=======
    # def get_critic_loss(self, share_state, returns):
    #     loss = 0
    #     for i in range(self.agent_num):
    #         new_value = self.value_nets[i](share_state[..., i, :])
    #         value_loss = 0.5 * torch.nn.functional.mse_loss(new_value, returns[..., i, :])
    #         loss += value_loss
    #     return loss
    
    # def get_value(self, share_state):
    #     value_list = []
    #     for i in range(self.agent_num):
            
    #         value = self.value_nets[i](share_state[..., i, :]).unsqueeze(dim=-2)
            
    #         # index = torch.LongTensor([i]).to(self.device)
    #         # input = torch.index_select(share_state, dim=-2, index=index)
    #         # value = self.value_nets[i](input)

    #         value_list.append(value)
    #     values = torch.cat(value_list, dim=2)
    #     return values

>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
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

<<<<<<< HEAD
class PPOAgent(object):
    def __init__(
            self, 
            state_dim, share_dim, caction_dim, caction_list,
            obs_features_shape, global_features_shape, raction_dim, raction_list,
            edge_index, buffer_list, device, 
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
=======
class PPOPSAgent(object):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list,
        buffer, device, agent_num,
        args
        ):
        self.device = device
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.action_dim = action_dim
        self.action_list = torch.Tensor(action_list).to(self.device)
        self.agent_num = agent_num
        
        self.max_steps = args.max_steps
        self.k_epoch = args.k_epoch
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size

>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.lr = args.lr
        self.eps_clip = args.eps_clip
        self.grad_clip = args.max_grad_clip
        self.entropy_coef = args.entropy_coef
        
<<<<<<< HEAD
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
            obs_features_shape, global_features_shape, 
            raction_dim, self.raction_list, 
            args
            ).to(self.device)
        self.route_network.policy_batch = self.route_network.policy_batch.to(self.device)
        self.route_network.value_batch = self.route_network.value_batch.to(self.device)
        self.route_optimizer = torch.optim.Adam(self.route_network.parameters(), lr=self.lr, eps=1e-5)

        self.rolloutBuffer_list = buffer_list

    def select_caction(self, state):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.charge_network.get_distribution(state)
=======
        self.network = Network(
            state_dim, share_dim, action_dim, self.action_list, 
            args.policy_arch, args.value_arch,
            agent_num, self.device
            )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)

        self.rolloutBuffer = buffer

    def select_action(self, state):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.network.get_distribution(state)
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

<<<<<<< HEAD
    def select_best_caction(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            dist = self.charge_network.get_distribution(state)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def select_raction(self, obs_feature, mask):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, mask=mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_raction(self, obs_feature, mask):
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, mask=mask)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def train(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            obs_feature, global_cs_feature, \
                raction, raction_mask, rlog_prob, rreward, \
                    next_obs_feature, next_global_cs_feature, \
                        rdone = self.rolloutBuffer_list[0].pull()
        buffer_step = self.rolloutBuffer_list[0].steps
        
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
            rvalues = self.route_network.get_value(x=global_cs_feature, edge_index=self.edge_index).view(buffer_step, -1)
            next_rvalues = self.route_network.get_value(x=next_global_cs_feature, edge_index=self.edge_index).view(buffer_step, -1)
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
                new_rdist = self.route_network.get_distribution(x=obs_feature[index], edge_index=self.edge_index, mask=raction_mask[index])
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_rvalues = self.route_network.get_value(x=global_cs_feature[index], edge_index=self.edge_index).view(self.mini_batch_size, -1)
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
=======
    def select_best_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            dist = self.network.get_distribution(state)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def train(self):
        state, share_state, action, log_prob, reward, next_state, next_share_state, done = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps

        with torch.no_grad():
            # there are N = num_env independent environments, cannot flatten state here
            # let "values" match the dimension of "done"
            values = self.network.get_value(share_state).squeeze(dim=-1)
            next_values = self.network.get_value(next_share_state).squeeze(dim=-1)
            advantage = torch.zeros_like(values).to(self.device)
            delta = reward + self.gamma * (1 - done) * next_values - values
            gae = 0
            for t in reversed(range(buffer_step)):
                gae = delta[t] + self.gamma * self.gae_lambda * gae * (1 - done[t])
                advantage[t] = gae
            returns = advantage + values
            norm_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
        # -------- flatten vectorized environment --------
        state = state.view(self.batch_size, -1, self.state_dim)
        share_state = share_state.view(self.batch_size, -1, self.share_dim)
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network
        action = action.view(self.batch_size, -1, 1)
        log_prob = log_prob.view(self.batch_size, -1, 1)
        returns = returns.view(self.batch_size, -1, 1)
        norm_adv = norm_adv.view(self.batch_size, -1, 1)
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                new_dist = self.network.get_distribution(state[index].view(-1, self.state_dim))
                new_log_prob = new_dist.log_prob(action[index].view(-1, 1).squeeze()).unsqueeze(1)
                # new_values = self.network.get_value(share_state[index].view(-1, self.share_dim)).view(self.mini_batch_size*self.agent_num, -1)
                
                entropy = new_dist.entropy()
                ratios = torch.exp(new_log_prob - log_prob[index].view(-1, 1))

                surrogate1 = ratios * norm_adv[index].view(-1, 1)
                surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_adv[index].view(-1, 1)
                actor_loss = (-1 * torch.min(surrogate1, surrogate2)).mean()
                entropy_loss = (self.entropy_coef * entropy).mean()
                
                new_values = self.network.get_value(share_state[index].view(-1, 1)).view(self.mini_batch_size*self.agent_num, -1)
                critic_loss = 0.5 * torch.nn.functional.mse_loss(new_values, returns[index].view(-1, 1))
                # critic_loss = self.network.get_critic_loss(share_state[index], returns[index])
                
                
                loss = actor_loss - entropy_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip) # type: ignore
                self.optimizer.step()
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def lr_decay(self, step):
        factor = 1 - step / self.max_steps
        lr = factor * self.lr
        for p in self.optimizer.param_groups:
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
            p['lr'] = lr
        return lr

    def save(self, filename):
<<<<<<< HEAD
        torch.save(self.charge_network.state_dict(), "{}_c.pt".format(filename))
        torch.save(self.charge_optimizer.state_dict(), "{}_c_optimizer.pt".format(filename))
        torch.save(self.route_network.state_dict(), "{}_r.pt".format(filename))
        torch.save(self.route_optimizer.state_dict(), "{}_r_optimizer.pt".format(filename))

    def load(self, filename):
        self.charge_network.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.charge_optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        self.route_network.load_state_dict(torch.load("{}_r.pt".format(filename)))
        self.route_optimizer.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))
=======
        torch.save(self.network.state_dict(), "{}.pt".format(filename))
        torch.save(self.optimizer.state_dict(), "{}_optimizer.pt".format(filename))

    def load(self, filename):
        self.network.load_state_dict(torch.load("{}.pt".format(filename)))
        self.optimizer.load_state_dict(torch.load("{}_optimizer.pt".format(filename)))
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
