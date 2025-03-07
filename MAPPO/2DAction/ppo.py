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
        # -------- init policy network --------
        last_layer_dim = state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        # policy_net.append(layer_init(nn.Linear(last_layer_dim, action_dim), gain=0.01))

        charge_net = [
            layer_init(nn.Linear(last_layer_dim, caction_dim), gain=0.01),
        ]
        route_net = [
            layer_init(nn.Linear(last_layer_dim, raction_dim), gain=0.01),
        ]

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
        self.charge_net = nn.Sequential(*charge_net)
        self.route_net = nn.Sequential(*route_net)

    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

    def get_distribution(self, state, state_mask):
        if self.caction_list.dim() == state.dim():
            mask = (self.caction_list > state[0]+0.05).long().bool()
        else:
            caction_list = self.caction_list.unsqueeze(0).repeat_interleave(state.shape[0], 0)
            state_soc = state[:, 0].unsqueeze(1).repeat_interleave(self.caction_dim, 1)
            mask = (caction_list > state_soc+0.05).long().bool()
        hidden = self.policy_net(state)
        clog_prob = self.charge_net(hidden)
        rlog_prob = self.route_net(hidden)
        masked_clogit = clog_prob.masked_fill((~mask).bool(), -1e32)
        masked_rlogit = rlog_prob.masked_fill((~state_mask.bool()), -1e32)
        return Categorical(logits=masked_clogit), Categorical(logits=masked_rlogit)

class PPOAgent(object):
    def __init__(
        self, 
        state_dim, share_dim, caction_dim, caction_list,
        obs_features_shape, global_features_shape, raction_dim, raction_list,
        edge_index, buffer, device, 
        args
        ):
        self.device = device

        self.state_dim = state_dim
        self.share_dim = share_dim
        # 充电相关
        self.caction_dim = caction_dim
        self.caction_list = torch.Tensor(caction_list).to(self.device)
        # 路径相关
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
            cdist, rdist = self.network.get_distribution(state, state_mask)
            caction = cdist.sample()
            clog_prob = cdist.log_prob(caction)
            raction = rdist.sample()
            rlog_prob = rdist.log_prob(raction)
            log_prob = clog_prob + rlog_prob # TODO: right?
        return caction.cpu().numpy().flatten(), raction.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_action(self, state, state_mask):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state_mask = torch.LongTensor(state_mask).to(self.device)
        with torch.no_grad():
            cdist, rdist = self.network.get_distribution(state, state_mask)
            caction = cdist.probs.argmax() # type: ignore
            clog_prob = cdist.log_prob(caction)
            raction = rdist.probs.argmax() # type: ignore
            rlog_prob = rdist.log_prob(raction)
            log_prob = clog_prob + rlog_prob # TODO: right?
        return caction.cpu().numpy().flatten(), raction.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

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
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network
        # 充电部分
        state = state.view(-1, self.state_dim)
        share_state = share_state.view(-1, self.share_dim)

        caction = caction.view(-1, 1)
        raction = raction.view(-1, 1)
        log_prob = log_prob.view(-1, 1)
        raction_mask = raction_mask.view(-1, self.raction_dim)

        returns = returns.view(-1, 1)
        norm_adv = norm_adv.view(-1, 1)
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                new_cdist, new_rdist = self.network.get_distribution(
                        state=state[index], state_mask=raction_mask[index]
                    )
                new_clog_prob = new_cdist.log_prob(caction[index].squeeze()).unsqueeze(1)
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_log_prob = new_clog_prob + new_rlog_prob # TODO: right?
                new_values = self.network.get_value(share_state[index]).view(self.mini_batch_size, -1)
                entropy = new_cdist.entropy() + new_rdist.entropy() # TODO: right?
                ratios = torch.exp(new_log_prob - log_prob[index])

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

    def lr_decay(self, step):
        return self.lr
        factor = 1 - step / self.num_update
        lr = factor * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

    def save(self, filename):
        torch.save(self.network.state_dict(), "{}_c.pt".format(filename))
        torch.save(self.optimizer.state_dict(), "{}_c_optimizer.pt".format(filename))

    def load(self, filename):
        self.network.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
