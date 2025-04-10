import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .networks.FCN import cNetwork, rNetwork

class PPOAgent(object):
    def __init__(
            self, 
            state_dim, share_dim, caction_dim, caction_list,
            obs_features_shape, global_features_shape, raction_dim, raction_list,
            edge_index, buffer, device, 
            args
        ):
        self.mode = args.mode
        self.device = device
        # 充电相关
        self.state_dim = state_dim
        self.share_dim = share_dim
        self.caction_dim = caction_dim
        self.caction_list = torch.Tensor(caction_list).to(self.device)
        # 路径相关
        self.raction_dim = raction_dim
        self.raction_list = torch.Tensor(raction_list).to(self.device)
        self.obs_features_shape = obs_features_shape
        self.global_features_shape = global_features_shape
        
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_index_shape = edge_index.shape
        
        self.num_update = args.num_update
        self.k_epoch = args.k_epoch
        # self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.mini_rbatch_size = args.mini_rbatch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.lr = args.lr
        self.eps_clip = args.eps_clip
        self.grad_clip = args.max_grad_clip
        self.entropy_coef = args.entropy_coef
        self.rolloutBuffer = buffer

        if self.mode in ['GH', 'NGH', 'OC']:
            self.charge_network = torch.compile(cNetwork(
                    state_dim, share_dim, caction_dim, self.caction_list, 
                    args.policy_arch, args.value_arch,
                    args
                ).to(self.device))
            self.charge_optimizer = torch.optim.Adam(self.charge_network.parameters(), lr=self.lr, eps=1e-5)

        if self.mode in ['NGH', 'OR']:
            self.route_network = torch.compile(rNetwork(
                    state_dim, share_dim, 
                    raction_dim, self.raction_list, 
                    args.policy_arch, args.value_arch,
                    args
                ).to(self.device))
            self.route_optimizer = torch.optim.Adam(self.route_network.parameters(), lr=self.lr, eps=1e-5)

    def select_caction(self, state):
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
        rbuffer_step = self.rolloutBuffer.rsteps
        
        with torch.no_grad():
            if self.mode == 'NGH':
                # 充电部分
                cvalues = self.charge_network.get_value(share_state).squeeze(dim=-1)
                next_cvalues = self.charge_network.get_value(next_share_state).squeeze(dim=-1)
                cadvantage = torch.zeros_like(cvalues).to(self.device)
                cdelta = creward + self.gamma * (1 - cdone) * next_cvalues - cvalues
                cgae = 0
                # 路径部分
                rvalues = self.route_network.get_value(share_rstate).view(rbuffer_step, -1)
                next_rvalues = self.route_network.get_value(next_share_rstate).view(rbuffer_step, -1)
                radvantage = torch.zeros_like(rvalues).to(self.device)
                rdelta = rreward + self.gamma * (1 - rdone) * next_rvalues - rvalues
                rgae = 0
                for t in reversed(range(buffer_step)):
                    cgae = cdelta[t] + self.gamma * self.gae_lambda * cgae * (1 - cdone[t])
                    cadvantage[t] = cgae
                    if t < rbuffer_step:
                        rgae = rdelta[t] + self.gamma * self.gae_lambda * rgae * (1 - rdone[t])
                        radvantage[t] = rgae
                creturns = cadvantage + cvalues
                norm_cadv = (cadvantage - cadvantage.mean()) / (cadvantage.std() + 1e-8)
                rreturns = radvantage + rvalues
                norm_radv = (radvantage - radvantage.mean()) / (radvantage.std() + 1e-8)
            elif self.mode == 'OC':
                cvalues = self.charge_network.get_value(share_state).squeeze(dim=-1)
                next_cvalues = self.charge_network.get_value(next_share_state).squeeze(dim=-1)
                cadvantage = torch.zeros_like(cvalues).to(self.device)
                cdelta = creward + self.gamma * (1 - cdone) * next_cvalues - cvalues
                cgae = 0
                for t in reversed(range(buffer_step)):
                    cgae = cdelta[t] + self.gamma * self.gae_lambda * cgae * (1 - cdone[t])
                    cadvantage[t] = cgae

                creturns = cadvantage + cvalues
                norm_cadv = (cadvantage - cadvantage.mean()) / (cadvantage.std() + 1e-8)
            elif self.mode == 'OR':
                rvalues = self.route_network.get_value(share_rstate).view(rbuffer_step, -1)
                next_rvalues = self.route_network.get_value(next_share_rstate).view(rbuffer_step, -1)
                radvantage = torch.zeros_like(rvalues).to(self.device)
                rdelta = rreward + self.gamma * (1 - rdone) * next_rvalues - rvalues
                rgae = 0
                for t in reversed(range(rbuffer_step)):
                    rgae = rdelta[t] + self.gamma * self.gae_lambda * rgae * (1 - rdone[t])
                    radvantage[t] = rgae
                rreturns = radvantage + rvalues
                norm_radv = (radvantage - radvantage.mean()) / (radvantage.std() + 1e-8)
                
        # -------- flatten vectorized environment --------
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network)
        # 充电部分
        if self.mode in ['NGH', 'OC']:
            state = state.view(-1, self.state_dim)
            share_state = share_state.view(-1, self.share_dim)
            caction = caction.view(-1, 1)
            clog_prob = clog_prob.view(-1, 1)
            creturns = creturns.view(-1, 1)
            norm_cadv = norm_cadv.view(-1, 1)
            for _ in range(self.k_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), self.mini_batch_size, True):
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
                    
                    self.charge_optimizer.zero_grad()
                    closs.backward()
                    nn.utils.clip_grad_norm_(self.charge_network.parameters(), self.grad_clip) # type: ignore
                    self.charge_optimizer.step()
        # 路径部分
        if self.mode in ['NGH', 'OR']:
            rstate = rstate.view(-1, self.state_dim)
            share_rstate = share_rstate.view(-1, self.share_dim)
            raction = raction.view(-1, 1)
            raction_mask = raction_mask.view(-1, self.raction_dim)
            rlog_prob = rlog_prob.view(-1, 1)
            rreturns = rreturns.view(-1, 1)
            norm_radv = norm_radv.view(-1, 1)
            for _ in range(self.k_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), self.mini_rbatch_size, True):
                    new_rdist = self.route_network.get_distribution(state=rstate[index], mask=raction_mask[index])
                    new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                    new_rvalues = self.route_network.get_value(share_rstate[index]).view(self.mini_rbatch_size, -1)
                    rentropy = new_rdist.entropy()
                    rratios = torch.exp(new_rlog_prob - rlog_prob[index])

                    rsurrogate1 = rratios * norm_radv[index]
                    rsurrogate2 = torch.clamp(rratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_radv[index]
                    actor_rloss = (-1 * torch.min(rsurrogate1, rsurrogate2)).mean()
                    entropy_rloss = (self.entropy_coef * rentropy).mean()
                    critic_rloss = 0.5 * torch.nn.functional.mse_loss(new_rvalues, rreturns[index])
                    rloss = actor_rloss - entropy_rloss + critic_rloss
                    
                    self.route_optimizer.zero_grad()
                    rloss.backward()
                    nn.utils.clip_grad_norm_(self.route_network.parameters(), self.grad_clip) # type: ignore
                    self.route_optimizer.step()
        
        if self.mode == 'NGH':
            return actor_closs.item(), critic_closs.item(), entropy_closs.item(), \
                actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()
        elif self.mode == 'OC':
            return actor_closs.item(), critic_closs.item(), entropy_closs.item()
        elif self.mode == 'OR':
            return actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()

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
        if self.mode in ['GH', 'NGH', 'OC']:
            torch.save(self.charge_network.state_dict(), "{}_c.pt".format(filename))
            torch.save(self.charge_optimizer.state_dict(), "{}_c_optimizer.pt".format(filename))
        if self.mode in ['GH', 'NGH', 'OR']:
            torch.save(self.route_network.state_dict(), "{}_r.pt".format(filename))
            torch.save(self.route_optimizer.state_dict(), "{}_r_optimizer.pt".format(filename))

    def load(self, filename):
        if self.mode in ['GH', 'NGH', 'OC']:
            self.charge_network.load_state_dict(torch.load("{}_c.pt".format(filename)))
            self.charge_optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        if self.mode in ['GH', 'NGH', 'OR']:
            self.route_network.load_state_dict(torch.load("{}_r.pt".format(filename)))
            self.route_optimizer.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))
