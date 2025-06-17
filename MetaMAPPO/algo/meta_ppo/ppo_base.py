import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .networks.FCN import *
from .SGD import SGD
import copy

class PPOAgent(object):
    def __init__(
            self, 
            state_dim, share_dim, caction_dim, caction_list,
            obs_features_shape, global_features_shape, raction_dim, raction_list,
            edge_index, buffer, validbuffer, device, 
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
        self.adapt_lr = args.adapt_lr
        self.meta_lr = args.meta_lr
        self.eps_clip = args.eps_clip
        self.grad_clip = args.max_grad_clip
        self.entropy_coef = args.entropy_coef

        self.rolloutBuffer = buffer
        self.validBuffer = validbuffer

        self.cPolicy = torch.compile(cActor(
                state_dim, share_dim, caction_dim, self.caction_list, 
                args.policy_arch,
                args
            ).to(self.device))
        self.cpolicy_optim = torch.optim.Adam(self.cPolicy.parameters(), lr=self.meta_lr, eps=1e-5)
        self.cadapt_optim = SGD(self.cPolicy, lr=self.adapt_lr)
        self.cValue = torch.compile(cCritic(
                state_dim, share_dim, caction_dim, self.caction_list, 
                args.value_arch,
                args
            ).to(self.device))
        self.cvalue_optim = torch.optim.Adam(self.cValue.parameters(), lr=self.meta_lr, eps=1e-5)

        self.rPolicy = torch.compile(rActor(
                state_dim, share_dim, 
                raction_dim, self.raction_list, 
                args.policy_arch,
                args
            ).to(self.device))
        self.rpolicy_optim = torch.optim.Adam(self.rPolicy.parameters(), lr=self.meta_lr, eps=1e-5)
        self.radapt_optim = SGD(self.rPolicy, lr=self.adapt_lr)
        self.rValue = torch.compile(rCritic(
                state_dim, share_dim, 
                raction_dim, self.raction_list, 
                args.value_arch,
                args
            ).to(self.device))
        self.rvalue_optim = torch.optim.Adam(self.rValue.parameters(), lr=self.meta_lr, eps=1e-5)

        # self.cPolicy_valid = torch.compile(cActor(
        #         state_dim, share_dim, caction_dim, self.caction_list, 
        #         args.policy_arch,
        #         args
        #     ).to(self.device))
        # self.rPolicy_valid = torch.compile(rActor(
        #         state_dim, share_dim, 
        #         raction_dim, self.raction_list, 
        #         args.policy_arch,
        #         args
        #     ).to(self.device))

    def select_caction(self, state, valid: bool = False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if valid:
                dist = self.cPolicy.get_valid(state, self.new_ctheta)
                # dist = self.cPolicy_valid.get_distribution(state)
            else:
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
    
    def select_raction(self, state, mask, valid: bool = False):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            if valid:
                dist = self.rPolicy.get_valid(state, mask, self.new_rtheta)
                # dist = self.rPolicy_valid.get_distribution(state, mask)
            else:
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
    
    def train_value_function(self, task_id: int = 0, valid: bool = False):
        if valid:
            state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
                rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                    next_raction_mask, rdone = self.validBuffer[task_id].pull()
        else:
            state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
                rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                    next_raction_mask, rdone = self.rolloutBuffer[task_id].pull()
        buffer_step = self.rolloutBuffer[0].steps
        rbuffer_step = self.rolloutBuffer[0].rsteps
        with torch.no_grad():
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

        critic_closs = 0
        critic_rloss = 0
        if valid:
            self.validBuffer[task_id].push_adv(norm_cadv.cpu().numpy(), norm_radv.cpu().numpy())
            return critic_closs, critic_rloss
        else:
            self.rolloutBuffer[task_id].push_adv(norm_cadv.cpu().numpy(), norm_radv.cpu().numpy())

            state = state.view(-1, self.state_dim)
            share_state = share_state.view(-1, self.share_dim)
            creturns = creturns.view(-1, 1)
            for _ in range(self.k_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), self.mini_batch_size, True):
                    new_cvalues = self.cValue.get_value(share_state[index]).view(self.mini_batch_size, -1)
                    critic_closs = 0.5 * torch.nn.functional.mse_loss(new_cvalues, creturns[index])
                    
                    self.cvalue_optim.zero_grad()
                    critic_closs.backward()
                    nn.utils.clip_grad_norm_(self.cValue.parameters(), self.grad_clip) # type: ignore
                    self.cvalue_optim.step()

            rstate = rstate.view(-1, self.state_dim)
            share_rstate = share_rstate.view(-1, self.share_dim)
            rreturns = rreturns.view(-1, 1)
            for _ in range(self.k_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), self.mini_rbatch_size, True):
                    new_rvalues = self.rValue.get_value(share_rstate[index]).view(self.mini_rbatch_size, -1)
                    critic_rloss = 0.5 * torch.nn.functional.mse_loss(new_rvalues, rreturns[index])
                    
                    self.rvalue_optim.zero_grad()
                    critic_rloss.backward()
                    nn.utils.clip_grad_norm_(self.rValue.parameters(), self.grad_clip) # type: ignore
                    self.rvalue_optim.step()

            return critic_closs.item(), critic_rloss.item()

    def surrogate_loss(self, task_id, valid: bool = False):
        if valid:
            buffer = self.validBuffer
        else:
            buffer = self.rolloutBuffer

        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, norm_cadv, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                next_raction_mask, rdone, norm_radv  = buffer[task_id].pull_all()
        buffer_step = buffer[task_id].steps
        rbuffer_step = buffer[task_id].rsteps

        # 充电部分
        state = state.view(-1, self.state_dim)
        share_state = share_state.view(-1, self.share_dim)
        caction = caction.view(-1, 1)
        clog_prob = clog_prob.view(-1, 1)
        norm_cadv = norm_cadv.view(-1, 1)
        for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), self.mini_batch_size, True):
            if valid:
                new_cdist = self.cPolicy.get_valid(state[index], self.new_ctheta)
                # new_cdist = self.cPolicy_valid.get_distribution(state[index])
            else:
                new_cdist = self.cPolicy.get_distribution(state[index])
            new_clog_prob = new_cdist.log_prob(caction[index].squeeze()).unsqueeze(1)
            centropy = new_cdist.entropy()
            cratios = torch.exp(new_clog_prob - clog_prob[index])

            csurrogate1 = cratios * norm_cadv[index]
            csurrogate2 = torch.clamp(cratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_cadv[index]
            actor_closs = (-1 * torch.min(csurrogate1, csurrogate2)).mean()
            entropy_closs = (self.entropy_coef * centropy).mean()
        # 路径部分
        rstate = rstate.view(-1, self.state_dim)
        share_rstate = share_rstate.view(-1, self.share_dim)
        raction = raction.view(-1, 1)
        raction_mask = raction_mask.view(-1, self.raction_dim)
        rlog_prob = rlog_prob.view(-1, 1)
        norm_radv = norm_radv.view(-1, 1)
        for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), self.mini_rbatch_size, True):
            if valid:
                new_rdist = self.rPolicy.get_valid(state=rstate[index], mask=raction_mask[index], theta=self.new_rtheta)
                # new_rdist = self.rPolicy_valid.get_distribution(state=rstate[index], mask=raction_mask[index])
            else:
                new_rdist = self.rPolicy.get_distribution(state=rstate[index], mask=raction_mask[index])
            new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
            rentropy = new_rdist.entropy()
            rratios = torch.exp(new_rlog_prob - rlog_prob[index])

            rsurrogate1 = rratios * norm_radv[index]
            rsurrogate2 = torch.clamp(rratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_radv[index]
            actor_rloss = (-1 * torch.min(rsurrogate1, rsurrogate2)).mean()
            entropy_rloss = (self.entropy_coef * rentropy).mean()

        return actor_closs, entropy_closs, actor_rloss, entropy_rloss
    
    def adapt_policy(self, task_id, second_order=False):
        actor_closs, entropy_closs, actor_rloss, entropy_rloss = self.surrogate_loss(task_id=task_id)
        
        closs = actor_closs - entropy_closs
        self.cadapt_optim.zero_grad()
        cparam_grads = torch.autograd.grad(closs, list(self.cPolicy.parameters()), create_graph=second_order)
        self.new_ctheta = self.cadapt_optim.step(cparam_grads)
        # update_module_params(self.cPolicy_valid, self.new_ctheta) #* 参数同步

        rloss = actor_rloss - entropy_rloss
        self.radapt_optim.zero_grad()
        rparam_grads = torch.autograd.grad(rloss, list(self.rPolicy.parameters()), create_graph=second_order)
        self.new_rtheta = self.radapt_optim.step(rparam_grads)
        # update_module_params(self.rPolicy_valid, self.new_rtheta) #* 参数同步

        return actor_closs.item(), entropy_closs.item(), actor_rloss.item(), entropy_rloss.item()

    def lr_decay(self, step):
        return self.lr
        factor = 1 - step / self.num_update
        lr = factor * self.lr
        for p in self.charge_optimizer.param_groups:
            p['lr'] = lr
        for p in self.route_optimizer.param_groups:
            p['lr'] = lr
        return lr

    # def save_theta(self):
    #     self.ctheta = {name: param.clone().detach().requires_grad_(True) for name, param in self.cPolicy.named_parameters()}
    #     self.rtheta = {name: param.clone().detach().requires_grad_(True) for name, param in self.rPolicy.named_parameters()}

    #     # self.ctheta = [param.clone().detach() for param in self.cPolicy.parameters()]
    #     # self.rtheta = [param.clone().detach() for param in self.rPolicy.parameters()]
    #     # self.ctheta = copy.deepcopy(dict(self.cPolicy.named_parameters()))
    #     # self.rtheta = copy.deepcopy(dict(self.rPolicy.named_parameters()))

    # def restore_theta(self, second_order=False):
    #     update_module_params(self.cPolicy, self.ctheta, second_order) #* 参数同步
    #     update_module_params(self.rPolicy, self.rtheta, second_order) #* 参数同步

    def save(self, filename):
        torch.save(self.cPolicy.state_dict(), "{}_c.pt".format(filename))
        torch.save(self.rPolicy.state_dict(), "{}_r.pt".format(filename))
        torch.save(self.cValue.state_dict(), "{}_cv.pt".format(filename))
        torch.save(self.rValue.state_dict(), "{}_rv.pt".format(filename))
        
        torch.save(self.cpolicy_optim.state_dict(), "{}_c_optimizer.pt".format(filename))
        torch.save(self.rpolicy_optim.state_dict(), "{}_r_optimizer.pt".format(filename))
        
        torch.save(self.cvalue_optim.state_dict(), "{}_cv_optimizer.pt".format(filename))
        torch.save(self.rvalue_optim.state_dict(), "{}_rv_optimizer.pt".format(filename))
        
    def load(self, filename):
        self.cPolicy.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.rPolicy.load_state_dict(torch.load("{}_r.pt".format(filename)))
        self.cValue.load_state_dict(torch.load("{}_cv.pt".format(filename)))
        self.rValue.load_state_dict(torch.load("{}_rv.pt".format(filename)))
        
        self.cpolicy_optim.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        self.rpolicy_optim.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))
        self.cvalue_optim.load_state_dict(torch.load("{}_cv_optimizer.pt".format(filename)))
        self.rvalue_optim.load_state_dict(torch.load("{}_rv_optimizer.pt".format(filename)))

    def load_theta(self, filename):
        self.cPolicy.load_state_dict(torch.load("{}_c.pt".format(filename)))
        self.rPolicy.load_state_dict(torch.load("{}_r.pt".format(filename)))
