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

<<<<<<< HEAD
class GNet(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List, value_arch: List,
        args
    ):
        super(GNet, self).__init__()
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
=======
class Network(nn.Module):
    def __init__(
            self, 
            state_dim, share_dim, 
            caction_dim, raction_dim, 
            caction_list, raction_list,
            policy_arch: List, value_arch: List
        ):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.caction_dim = caction_dim
        self.raction_dim = raction_dim
        self.share_dim = share_dim
        self.caction_list = caction_list
        self.raction_list = raction_list
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
        # -------- init policy network --------
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
<<<<<<< HEAD
        policy_net.append(layer_init(nn.Linear(last_layer_dim, action_dim), gain=0.01))
=======
        policy_net.append(layer_init(nn.Linear(last_layer_dim, raction_dim), gain=0.01))

        # charge_net = [
        #     layer_init(nn.Linear(last_layer_dim, caction_dim), gain=0.01),
        # ]
        # route_net = [
        #     layer_init(nn.Linear(last_layer_dim, raction_dim), gain=0.01),
        # ]

>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
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
<<<<<<< HEAD
=======
        # self.charge_net = nn.Sequential(*charge_net)
        # self.route_net = nn.Sequential(*route_net)
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7

    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

<<<<<<< HEAD
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
=======
    def get_distribution(self, state, state_mask):
        # hidden = self.policy_net(state)
        # if self.caction_list.dim() == state.dim():
        #     mask = (self.caction_list > state[0]+0.05).long().bool()
        # else:
        #     caction_list = self.caction_list.unsqueeze(0).repeat_interleave(state.shape[0], 0)
        #     state_soc = state[:, 0].unsqueeze(1).repeat_interleave(self.caction_dim, 1)
        #     mask = (caction_list > state_soc+0.05).long().bool()
        rlog_prob = self.policy_net(state)
        # clog_prob = self.charge_net(hidden)
        # rlog_prob = self.route_net(hidden)
        # masked_clogit = clog_prob.masked_fill((~mask).bool(), -1e32)
        masked_rlogit = rlog_prob.masked_fill((~state_mask.bool()), -1e32)
        return Categorical(logits=masked_rlogit)
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7

class PPOAgent(object):
    def __init__(
        self, 
        state_dim, share_dim, caction_dim, caction_list,
        obs_features_shape, global_features_shape, raction_dim, raction_list,
        edge_index, buffer, device, 
        args
        ):
        self.device = device
<<<<<<< HEAD
        # 充电相关
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.caction_dim = caction_dim
        self.caction_list = torch.Tensor(caction_list).to(self.device)
        # 路径相关
        # self.obs_features_shape = obs_features_shape
        # self.global_features_shape = global_features_shape
=======

        self.state_dim = state_dim
        self.share_dim = share_dim
        # 充电相关
        self.caction_dim = caction_dim
        self.caction_list = torch.Tensor(caction_list).to(self.device)
        # 路径相关
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
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
<<<<<<< HEAD
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_index_shape = edge_index.shape
        
        self.charge_network = Network(
            state_dim, share_dim, caction_dim, self.caction_list, 
            args.policy_arch, args.value_arch,
            args
            ).to(self.device)
        self.charge_optimizer = torch.optim.Adam(self.charge_network.parameters(), lr=self.lr, eps=1e-5)

        self.route_network = GNet(
            state_dim, share_dim, 
            raction_dim, self.raction_list, 
            args.policy_arch, args.value_arch,
            args
        ).to(self.device)
        # self.route_network.policy_batch = self.route_network.policy_batch.to(self.device)
        # self.route_network.value_batch = self.route_network.value_batch.to(self.device)
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
    
    def select_raction(self, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        # obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            dist = self.route_network.get_distribution(state=state, mask=mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_raction(self, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        with torch.no_grad():
            dist = self.route_network.get_distribution(state=state, mask=mask)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def train(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
=======
        
        self.network = Network(
            state_dim, share_dim, 
            caction_dim, raction_dim, 
            self.caction_list, self.raction_list,
            args.policy_arch, args.value_arch
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        self.rolloutBuffer = buffer

    def select_action(self, state, state_mask):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state_mask = torch.LongTensor(state_mask).to(self.device)
        with torch.no_grad():
            rdist = self.network.get_distribution(state, state_mask)
            caction = (state[0] < 0.15).int() * (len(self.caction_list)-1)
            raction = rdist.sample()
            rlog_prob = rdist.log_prob(raction)
        return caction.cpu().numpy().flatten(), raction.cpu().numpy().flatten(), rlog_prob.cpu().numpy().flatten()

    def select_best_action(self, state, state_mask):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state_mask = torch.LongTensor(state_mask).to(self.device)
        with torch.no_grad():
            rdist = self.network.get_distribution(state, state_mask)
            caction = (state[0] < 0.15).int() * len(self.caction_list)
            raction = rdist.probs.argmax() # type: ignore
            rlog_prob = rdist.log_prob(raction)
        return caction.cpu().numpy().flatten(), raction.cpu().numpy().flatten(), rlog_prob.cpu().numpy().flatten()

    def train(self):
        if self.ps:
            pass
        else:
            state, share_state, \
                caction, raction, raction_mask, log_prob, \
                    next_state, next_share_state, \
                        reward, done \
                            = self.rolloutBuffer.pull()
            buffer_step = self.rolloutBuffer.steps
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
        
        with torch.no_grad():
            # there are N = num_env independent environments, cannot flatten state here
            # let "values" match the dimension of "done"
<<<<<<< HEAD
            # 充电部分
            # 路径部分
            rvalues = self.route_network.get_value(share_rstate).view(buffer_step, -1)
            next_rvalues = self.route_network.get_value(next_share_rstate).view(buffer_step, -1)
            radvantage = torch.zeros_like(rvalues).to(self.device)
            rdelta = rreward + self.gamma * (1 - rdone) * next_rvalues - rvalues
            rgae = 0
            for t in reversed(range(buffer_step)):
                rgae = rdelta[t] + self.gamma * self.gae_lambda * rgae * (1 - rdone[t])
                radvantage[t] = rgae
            rreturns = radvantage + rvalues
            norm_radv = (radvantage - radvantage.mean()) / (radvantage.std() + 1e-8)
=======
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
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
            
        # -------- flatten vectorized environment --------
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network
        # 充电部分
        state = state.view(-1, self.state_dim)
        share_state = share_state.view(-1, self.share_dim)
<<<<<<< HEAD
        rstate = rstate.view(-1, self.state_dim)
        share_rstate = share_rstate.view(-1, self.share_dim)
        caction = caction.view(-1, 1)
        clog_prob = clog_prob.view(-1, 1)
        # 路径部分
        # obs_feature = obs_feature.view(-1, self.obs_features_shape[0], self.obs_features_shape[1])
        # global_cs_feature = global_cs_feature.view(-1, self.global_features_shape[0], self.global_features_shape[1])
        raction = raction.view(-1, 1)
        raction_mask = raction_mask.view(-1, self.raction_dim)
        rlog_prob = rlog_prob.view(-1, 1)
        rreturns = rreturns.view(-1, 1)
        norm_radv = norm_radv.view(-1, 1)
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                # 路径部分
                new_rdist = self.route_network.get_distribution(state=rstate[index], mask=raction_mask[index])
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_rvalues = self.route_network.get_value(share_rstate[index]).view(self.mini_batch_size, -1)
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
                rloss.backward()
                nn.utils.clip_grad_norm_(self.charge_network.parameters(), self.grad_clip) # type: ignore
                nn.utils.clip_grad_norm_(self.route_network.parameters(), self.grad_clip) # type: ignore
                self.charge_optimizer.step()
                self.route_optimizer.step()
                
        return actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()
=======

        caction = caction.view(-1, 1)
        raction = raction.view(-1, 1)
        log_prob = log_prob.view(-1, 1)
        raction_mask = raction_mask.view(-1, self.raction_dim)

        returns = returns.view(-1, 1)
        norm_adv = norm_adv.view(-1, 1)
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                new_rdist = self.network.get_distribution(
                        state=state[index], state_mask=raction_mask[index]
                    )
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_values = self.network.get_value(share_state[index]).view(self.mini_batch_size, -1)
                entropy = new_rdist.entropy()
                ratios = torch.exp(new_rlog_prob - log_prob[index])

                surrogate1 = ratios * norm_adv[index]
                surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_adv[index]
                actor_loss = (-1 * torch.min(surrogate1, surrogate2)).mean()
                entropy_loss = (self.entropy_coef * entropy).mean()
                critic_loss = 0.5 * torch.nn.functional.mse_loss(new_values, returns[index])
                loss = actor_loss - entropy_loss + critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                self.optimizer.step()
                
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7

    def lr_decay(self, step):
        return self.lr
        factor = 1 - step / self.num_update
        lr = factor * self.lr
<<<<<<< HEAD
        for p in self.charge_optimizer.param_groups:
            p['lr'] = lr
        for p in self.route_optimizer.param_groups:
=======
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
        torch.save(self.network.state_dict(), "{}_c.pt".format(filename))
        torch.save(self.optimizer.state_dict(), "{}_c_optimizer.pt".format(filename))

    def load(self, filename):
        self.network.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
>>>>>>> 8ce6d8f0c6ed187a6fd0eaae6b43825b53f771a7
