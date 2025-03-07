from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler


def layer_init(layer, gain=np.sqrt(2), bias=0.):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class Network(nn.Module):
    def __init__(
        self, 
        state_dim, share_dim, action_dim, 
        action_list, policy_arch: List, value_arch: List,
        agent_num, device
        ):
        super(Network, self).__init__()
        self.agent_num = agent_num
        self.device = device
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
        
    def get_value(self, share_state):
        value = self.value_net(share_state)
        return value

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

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.lr = args.lr
        self.eps_clip = args.eps_clip
        self.grad_clip = args.max_grad_clip
        self.entropy_coef = args.entropy_coef
        
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
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

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
            p['lr'] = lr
        return lr

    def save(self, filename):
        torch.save(self.network.state_dict(), "{}.pt".format(filename))
        torch.save(self.optimizer.state_dict(), "{}_optimizer.pt".format(filename))

    def load(self, filename):
        self.network.load_state_dict(torch.load("{}.pt".format(filename)))
        self.optimizer.load_state_dict(torch.load("{}_optimizer.pt".format(filename)))
