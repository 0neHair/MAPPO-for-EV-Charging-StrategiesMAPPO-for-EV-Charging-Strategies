from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler
torch.nn.init.trunc_normal_

def layer_init(layer, gain=np.sqrt(2), bias=0.):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, shared_arch: List, policy_arch: List, value_arch: List):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # -------- init shared layers --------
        last_layer_dim = state_dim
        shared_net = []
        for current_layer_dim in shared_arch:
            shared_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            shared_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        # -------- init policy network --------
        last_layer_dim = shared_arch[-1] if shared_arch != [] else state_dim
        policy_net = []
        for current_layer_dim in policy_arch:
            policy_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            policy_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        policy_net.append(layer_init(nn.Linear(last_layer_dim, action_dim), gain=0.01))
        # policy_net.append(nn.Sigmoid())
        # -------- init value network --------
        last_layer_dim = shared_arch[-1] if shared_arch != [] else state_dim
        value_net = []
        for current_layer_dim in value_arch:
            value_net.append(layer_init(nn.Linear(last_layer_dim, current_layer_dim)))
            value_net.append(nn.Tanh())
            last_layer_dim = current_layer_dim
        value_net.append(layer_init(nn.Linear(last_layer_dim, 1), gain=1.0))
        # -------- init log action std --------
        self.log_action_std = nn.Parameter(torch.zeros(1, action_dim))

        self.shared_net = nn.Sequential(*shared_net) if shared_arch != [] else None
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def get_value(self, state):
        if self.shared_net is None:
            value = self.value_net(state)
        else:
            tmp = self.shared_net(state)
            value = self.value_net(tmp)
        return value

    def get_distribution(self, state):
        if self.shared_net is None:
            mean = self.policy_net(state)
        else:
            tmp = self.shared_net(state)
            mean = self.policy_net(tmp)
        # mean = torch.exp(mean)
        log_std = self.log_action_std.expand_as(mean.unsqueeze(0))
        std = torch.exp(log_std)
        return Normal(mean, std)


class PPOAgent(object):
    def __init__(self, state_dim, action_dim, shared_arch: List, policy_arch: List, value_arch: List, buffer, device,
                 max_steps, gamma, gae_lambda, k_epoch, lr, eps_clip, grad_clip, entropy_coef, batch_size,
                 mini_batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.k_epoch = k_epoch
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.eps_clip = eps_clip
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef

        self.device = device

        self.network = Network(state_dim, action_dim, shared_arch, policy_arch, value_arch).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)

        self.rolloutBuffer = buffer

    def select_action(self, state):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.network.get_distribution(state)
            action = dist.sample()
            action = torch.clamp(action, 0, 1-state[0].item())
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def select_best_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            dist = self.network.get_distribution(state)
            action = dist.mean
            log_prob = dist.log_prob(action).sum(1)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def get_log_prob(self, state, action):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.network.get_distribution(state)
            log_prob = dist.log_prob(action)
        return log_prob.cpu().numpy()

    def train(self):
        state, action, log_prob, reward, next_state, done = self.rolloutBuffer.pull()

        with torch.no_grad():
            # there are N = num_env independent environments, cannot flatten state here
            # let "values" match the dimension of "done"
            values = self.network.get_value(state).view(self.rolloutBuffer.steps, -1)
            next_values = self.network.get_value(next_state).view(self.rolloutBuffer.steps, -1)
            advantage = torch.zeros_like(values).to(self.device)
            delta = reward + self.gamma * (1 - done) * next_values - values
            gae = 0
            for t in reversed(range(self.rolloutBuffer.steps)):
                gae = delta[t] + self.gamma * self.gae_lambda * gae * (1 - done[t])
                advantage[t] = gae
            returns = advantage + values
            norm_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # -------- flatten vectorized environment --------
        state = state.view(-1, self.state_dim)
        action = action.view(-1, self.action_dim)
        log_prob = log_prob.view(-1, 1)
        returns = returns.view(-1, 1)
        norm_adv = norm_adv.view(-1, 1)
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                new_dist = self.network.get_distribution(state[index])
                new_log_prob = new_dist.log_prob(action[index])
                new_values = self.network.get_value(state[index]).view(self.mini_batch_size, -1)
                entropy = new_dist.entropy().sum(1)
                ratios = torch.exp(new_log_prob - log_prob[index])

                surrogate1 = ratios * norm_adv[index]
                surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_adv[index]
                actor_loss = (-1 * torch.min(surrogate1, surrogate2)).mean()
                entropy_loss = (self.entropy_coef * entropy).mean()
                critic_loss = 0.5 * torch.nn.functional.mse_loss(new_values, returns[index])
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
