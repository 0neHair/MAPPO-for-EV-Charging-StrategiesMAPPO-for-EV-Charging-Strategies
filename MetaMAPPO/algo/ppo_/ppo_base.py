import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .networks.FCN import *
import copy

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
            self.cPolicy = torch.compile(
                cActor(
                    state_dim, share_dim, caction_dim, self.caction_list, 
                    args.policy_arch,
                    args
                ).to(self.device))
            self.cpolicy_optim = torch.optim.Adam(self.cPolicy.parameters(), lr=self.lr, eps=1e-5)
            self.cValue = torch.compile(
                cCritic(
                    state_dim, share_dim, caction_dim, self.caction_list, 
                    args.value_arch,
                    args
                ).to(self.device))
            self.cvalue_optim = torch.optim.Adam(self.cValue.parameters(), lr=self.lr, eps=1e-5)

        if self.mode in ['NGH', 'OR']:
            self.rPolicy = torch.compile(
                rActor(
                    state_dim, share_dim, raction_dim, self.raction_list, 
                    args.policy_arch,
                    args
                ).to(self.device))
            self.rpolicy_optim = torch.optim.Adam(self.rPolicy.parameters(), lr=self.lr, eps=1e-5)
            self.rValue = torch.compile(
                rCritic(
                    state_dim, share_dim, raction_dim, self.raction_list, 
                    args.value_arch,
                    args
                ).to(self.device))
            self.rvalue_optim = torch.optim.Adam(self.rValue.parameters(), lr=self.lr, eps=1e-5)

    def select_caction(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.cPolicy.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_caction(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            dist = self.cPolicy.get_distribution(state)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def select_raction(self, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            dist = self.rPolicy.get_distribution(state=state, mask=mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_raction(self, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        with torch.no_grad():
            dist = self.rPolicy.get_distribution(state=state, mask=mask)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def train(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                next_raction_mask, rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        rbuffer_step = self.rolloutBuffer.rsteps
        
        with torch.no_grad():
            if self.mode == 'NGH':
                # 充电部分
                cvalues = self.cValue.get_value(share_state).squeeze(dim=-1)
                next_cvalues = self.cValue.get_value(next_share_state).squeeze(dim=-1)
                cadvantage = torch.zeros_like(cvalues).to(self.device)
                cdelta = creward + self.gamma * (1 - cdone) * next_cvalues - cvalues
                cgae = 0
                # 路径部分
                rvalues = self.rValue.get_value(share_rstate).view(rbuffer_step, -1)
                next_rvalues = self.rValue.get_value(next_share_rstate).view(rbuffer_step, -1)
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
                cvalues = self.cValue.get_value(share_state).squeeze(dim=-1)
                next_cvalues = self.cValue.get_value(next_share_state).squeeze(dim=-1)
                cadvantage = torch.zeros_like(cvalues).to(self.device)
                cdelta = creward + self.gamma * (1 - cdone) * next_cvalues - cvalues
                cgae = 0
                for t in reversed(range(buffer_step)):
                    cgae = cdelta[t] + self.gamma * self.gae_lambda * cgae * (1 - cdone[t])
                    cadvantage[t] = cgae

                creturns = cadvantage + cvalues
                norm_cadv = (cadvantage - cadvantage.mean()) / (cadvantage.std() + 1e-8)
            elif self.mode == 'OR':
                rvalues = self.rValue.get_value(share_rstate).view(rbuffer_step, -1)
                next_rvalues = self.rValue.get_value(next_share_rstate).view(rbuffer_step, -1)
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
                    new_cdist = self.cPolicy.get_distribution(state[index])
                    new_clog_prob = new_cdist.log_prob(caction[index].squeeze()).unsqueeze(1)
                    new_cvalues = self.cValue.get_value(share_state[index]).view(self.mini_batch_size, -1)
                    centropy = new_cdist.entropy()
                    cratios = torch.exp(new_clog_prob - clog_prob[index])

                    csurrogate1 = cratios * norm_cadv[index]
                    csurrogate2 = torch.clamp(cratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_cadv[index]
                    actor_closs = (-1 * torch.min(csurrogate1, csurrogate2)).mean()
                    entropy_closs = (self.entropy_coef * centropy).mean()
                    critic_closs = 0.5 * torch.nn.functional.mse_loss(new_cvalues, creturns[index])
                    closs = actor_closs - entropy_closs

                    self.cvalue_optim.zero_grad()
                    critic_closs.backward()
                    nn.utils.clip_grad_norm_(self.cValue.parameters(), self.grad_clip) # type: ignore
                    self.cvalue_optim.step()
                    self.cpolicy_optim.zero_grad()
                    closs.backward()
                    nn.utils.clip_grad_norm_(self.cPolicy.parameters(), self.grad_clip) # type: ignore
                    self.cpolicy_optim.step()
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
                    new_rdist = self.rPolicy.get_distribution(state=rstate[index], mask=raction_mask[index])
                    new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                    new_rvalues = self.rValue.get_value(share_rstate[index]).view(self.mini_rbatch_size, -1)
                    rentropy = new_rdist.entropy()
                    rratios = torch.exp(new_rlog_prob - rlog_prob[index])

                    rsurrogate1 = rratios * norm_radv[index]
                    rsurrogate2 = torch.clamp(rratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_radv[index]
                    actor_rloss = (-1 * torch.min(rsurrogate1, rsurrogate2)).mean()
                    entropy_rloss = (self.entropy_coef * rentropy).mean()
                    critic_rloss = 0.5 * torch.nn.functional.mse_loss(new_rvalues, rreturns[index])
                    rloss = actor_rloss - entropy_rloss

                    self.rvalue_optim.zero_grad()
                    critic_rloss.backward()
                    nn.utils.clip_grad_norm_(self.rValue.parameters(), self.grad_clip) # type: ignore
                    self.rvalue_optim.step()
                    self.rpolicy_optim.zero_grad()
                    rloss.backward()
                    nn.utils.clip_grad_norm_(self.rPolicy.parameters(), self.grad_clip) # type: ignore
                    self.rpolicy_optim.step()
        
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

    def save_theta(self):
        self.cp_theta = {name: param.clone().detach() for name, param in self.cPolicy.named_parameters()}
        self.rp_theta = {name: param.clone().detach() for name, param in self.rPolicy.named_parameters()}
        self.cv_theta = {name: param.clone().detach() for name, param in self.cValue.named_parameters()}
        self.rv_theta = {name: param.clone().detach() for name, param in self.rValue.named_parameters()}
        # self.cp_theta = {}
        # for name, new_param, in dict(self.cPolicy.named_parameters()).items():
        #     self.cp_theta[name] = new_param.clone()
        # self.rp_theta = {}
        # for name, new_param, in dict(self.rPolicy.named_parameters()).items():
        #     self.rp_theta[name] = new_param.clone()
        # self.cv_theta = {}
        # for name, new_param, in dict(self.cValue.named_parameters()).items():
        #     self.cv_theta[name] = new_param.clone()
        # self.rv_theta = {}
        # for name, new_param, in dict(self.rValue.named_parameters()).items():
        #     self.rv_theta[name] = new_param.clone()
    
    def load_theta(self, theta):
        cp_theta, rp_theta, cv_theta, rv_theta = theta
        update_module_params(self.cPolicy, cp_theta) #* 参数同步
        update_module_params(self.rPolicy, rp_theta) #* 参数同步
        update_module_params(self.cValue, cv_theta) #* 参数同步
        update_module_params(self.rValue, rv_theta) #* 参数同步
    
    def return_theta(self):
        return {name: param.clone().detach() for name, param in self.cPolicy.named_parameters()}, \
            {name: param.clone().detach() for name, param in self.rPolicy.named_parameters()}, \
            {name: param.clone().detach() for name, param in self.cValue.named_parameters()}, \
            {name: param.clone().detach() for name, param in self.rValue.named_parameters()}
    
    def save(self, filename):
        if self.mode in ['GH', 'NGH', 'OC']:
            torch.save(self.cPolicy.state_dict(), "{}_c.pt".format(filename))
        if self.mode in ['GH', 'NGH', 'OR']:
            torch.save(self.rPolicy.state_dict(), "{}_r.pt".format(filename))

    def load(self, filename):
        if self.mode in ['GH', 'NGH', 'OC']:
            self.charge_network.load_state_dict(torch.load("{}_c.pt".format(filename)))
            self.charge_optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        if self.mode in ['GH', 'NGH', 'OR']:
            self.route_network.load_state_dict(torch.load("{}_r.pt".format(filename)))
            self.route_optimizer.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))

# def update_module_params(module: torch.nn.Module, new_params: dict):
#     def update(module: torch.nn.Module, name, new_param):
#         del module._parameters[name]
#         setattr(module, name, new_param)
#         module._parameters[name] = new_param

#     named_module_dict = dict(module.named_modules())
#     for name, new_param, in new_params.items():
#         if "." in name:
#             module_name, param_name = name.rsplit(".", 1)  # policy.0.bias -> module_name: policy.0, param_name: bias
#             update(named_module_dict[module_name], param_name, new_param)
#         else:
#             update(module, name, new_param)

def update_module_params(module: torch.nn.Module, new_params: dict):
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(new_params[name])