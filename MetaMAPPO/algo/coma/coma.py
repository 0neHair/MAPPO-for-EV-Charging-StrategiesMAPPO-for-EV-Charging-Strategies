import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .networks.FCN import cNetwork, rNetwork
import torch.nn.functional as F

class COMAAgent(object):
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
        self.alpha = 0.2
        self.tau = 1.0
        
        if self.mode in ['NGH', 'OC']:
            self.charge_network = torch.compile(cNetwork(
                    state_dim, share_dim, 
                    caction_dim, self.caction_list, 
                    args.policy_arch, args.value_arch,
                    args
                ).to(self.device))
            self.cactor_optimizer = torch.optim.Adam(
                list(self.charge_network.actor.parameters()), lr=self.lr, eps=1e-5
            )
            self.cq_optimizer = torch.optim.Adam(
                list(self.charge_network.qf.parameters()),
                lr=self.lr, eps=1e-5
            )

        if self.mode in ['NGH', 'OR']:
            self.route_network = torch.compile(rNetwork(
                    state_dim, share_dim, 
                    raction_dim, self.raction_list, 
                    args.policy_arch, args.value_arch,
                    args
                ).to(self.device))
            self.ractor_optimizer = torch.optim.Adam(
                list(self.route_network.actor.parameters()), lr=self.lr, eps=1e-5
            )
            self.rq_optimizer = torch.optim.Adam(
                list(self.route_network.qf.parameters()),
                lr=self.lr, eps=1e-5
            )

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
    
    def train_coma(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                next_raction_mask, rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        rbuffer_step = self.rolloutBuffer.rsteps

        # 充电部分
        cq_list = self.charge_network.qf(share_state)
        cq_value = gather...
        next_cq_target_list = self.charge_network.qf_target(next_share_state)
        next_cq_target_value = gather...
        creture = creward + (1 - cdone) * self.gamma * (next_cq_target_value)
        cqf_loss = F.mse_loss(creture, cq_value)
        
        self.cq_optimizer.zero_grad()
        cqf_loss.backward()
        self.cq_optimizer.step()

        with torch.no_grad():
            cq_target_list = self.charge_network.qf(share_state)
            cq_target_value = gather...
        
        cpi = self.charge_network.get_distribution(state).???
        baseline = torch.sum(cpi * cq_target_list, dim=1)
        advantage = cq_target_value - baseline
        log_pi
        actor_loss = - torch.mean(advantage * log_pi)
        
        self.cactor_optimizer.zero_grad()
        actor_closs.backward()
        self.cactor_optimizer.step()

        for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), buffer_step//2, True):
            state = state[index].view(-1, self.state_dim)
            next_state = next_state[index].view(-1, self.state_dim)
            share_state = share_state[index].view(-1, self.share_dim)
            next_share_state = next_share_state[index].view(-1, self.share_dim)
            caction = caction[index].view(-1, 1)
            creward = creward[index].view(-1, 1)
            cdone = cdone[index].view(-1, 1)
            with torch.no_grad():
                if self.mode == 'NGH':
                    next_dist, next_log_prob = self.charge_network.get_logits(next_state)
                    action_probs = next_dist.probs
                    qf1_next_target = self.charge_network.qf1_target(next_share_state)
                    qf2_next_target = self.charge_network.qf2_target(next_share_state)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1).view(-1, 1)
                    next_q_value = creward + (1 - cdone) * self.gamma * (min_qf_next_target)

            qf1_values = self.charge_network.qf1(share_state)
            qf2_values = self.charge_network.qf2(share_state)
            qf1_a_values = qf1_values.gather(1, caction.long())
            qf2_a_values = qf2_values.gather(1, caction.long())
            cqf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)
            
            self.cq_optimizer.zero_grad()
            cqf_loss.backward()
            self.cq_optimizer.step()

            dist, log_prob = self.charge_network.get_logits(state)
            action_probs = dist.probs
            with torch.no_grad():
                qf1_values = self.charge_network.qf1(share_state)
                qf2_values = self.charge_network.qf2(share_state)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_closs = (action_probs * ((self.alpha * log_prob) - min_qf_values)).mean()
            self.cactor_optimizer.zero_grad()
            actor_closs.backward()
            self.cactor_optimizer.step()
            break
        
        # 路径部分
        for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), rbuffer_step, True):
            rstate = rstate[index].view(-1, self.state_dim)
            raction_mask = raction_mask[index].view(-1, self.raction_dim)
            next_rstate = next_rstate[index].view(-1, self.state_dim)
            share_rstate = share_rstate[index].view(-1, self.share_dim)
            next_share_rstate = next_share_rstate[index].view(-1, self.share_dim)
            raction = raction[index].view(-1, 1)
            rreward = rreward[index].view(-1, 1)
            next_raction_mask = next_raction_mask[index].view(-1, self.raction_dim)
            
            rdone = rdone[index].view(-1, 1)
            with torch.no_grad():
                if self.mode == 'NGH':
                    next_dist, next_log_prob = self.route_network.get_logits(next_rstate, next_raction_mask)
                    action_probs = next_dist.probs
                    qf1_next_target = self.route_network.qf1_target(next_share_rstate)
                    qf2_next_target = self.route_network.qf2_target(next_share_rstate)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1).view(-1, 1)
                    next_q_value = rreward + (1 - rdone) * self.gamma * (min_qf_next_target)

            qf1_values = self.route_network.qf1(share_rstate)
            qf2_values = self.route_network.qf2(share_rstate)
            qf1_a_values = qf1_values.gather(1, raction.long())
            qf2_a_values = qf2_values.gather(1, raction.long())
            rqf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)
            
            self.rq_optimizer.zero_grad()
            rqf_loss.backward()
            self.rq_optimizer.step()

            dist, log_prob = self.route_network.get_logits(rstate, raction_mask)
            action_probs = dist.probs
            with torch.no_grad():
                qf1_values = self.route_network.qf1(share_rstate)
                qf2_values = self.route_network.qf2(share_rstate)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_rloss = (action_probs * ((self.alpha * log_prob) - min_qf_values)).mean()
            self.ractor_optimizer.zero_grad()
            actor_rloss.backward()
            self.ractor_optimizer.step()
            break
        
        return actor_closs.item(), cqf_loss.item(), actor_rloss.item(), rqf_loss.item()
    
    # def train_ppo(self):
    #     state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
    #         rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
    #             rdone = self.rolloutBuffer.pull()
    #     buffer_step = self.rolloutBuffer.steps
    #     rbuffer_step = self.rolloutBuffer.rsteps
        
    #     with torch.no_grad():
    #         if self.mode == 'NGH': 
    #             # 路径部分
    #             rvalues = self.route_network.get_value(share_rstate).view(rbuffer_step, -1)
    #             next_rvalues = self.route_network.get_value(next_share_rstate).view(rbuffer_step, -1)
    #             radvantage = torch.zeros_like(rvalues).to(self.device)
    #             rdelta = rreward + self.gamma * (1 - rdone) * next_rvalues - rvalues
    #             rgae = 0
    #             for t in reversed(range(buffer_step)):
    #                 if t < rbuffer_step:
    #                     rgae = rdelta[t] + self.gamma * self.gae_lambda * rgae * (1 - rdone[t])
    #                     radvantage[t] = rgae
    #             rreturns = radvantage + rvalues
    #             norm_radv = (radvantage - radvantage.mean()) / (radvantage.std() + 1e-8)
                         
    #     # 路径部分
    #     if self.mode in ['NGH', 'OR']:
    #         rstate = rstate.view(-1, self.state_dim)
    #         share_rstate = share_rstate.view(-1, self.share_dim)
    #         raction = raction.view(-1, 1)
    #         raction_mask = raction_mask.view(-1, self.raction_dim)
    #         rlog_prob = rlog_prob.view(-1, 1)
    #         rreturns = rreturns.view(-1, 1)
    #         norm_radv = norm_radv.view(-1, 1)
    #         for _ in range(self.k_epoch):
    #             for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), self.mini_rbatch_size, True):
    #                 new_rdist = self.route_network.get_distribution(state=rstate[index], mask=raction_mask[index])
    #                 new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
    #                 new_rvalues = self.route_network.get_value(share_rstate[index]).view(self.mini_rbatch_size, -1)
    #                 rentropy = new_rdist.entropy()
    #                 rratios = torch.exp(new_rlog_prob - rlog_prob[index])

    #                 rsurrogate1 = rratios * norm_radv[index]
    #                 rsurrogate2 = torch.clamp(rratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_radv[index]
    #                 actor_rloss = (-1 * torch.min(rsurrogate1, rsurrogate2)).mean()
    #                 entropy_rloss = (self.entropy_coef * rentropy).mean()
    #                 critic_rloss = 0.5 * torch.nn.functional.mse_loss(new_rvalues, rreturns[index])
    #                 rloss = actor_rloss - entropy_rloss + critic_rloss
                    
    #                 self.route_optimizer.zero_grad()
    #                 rloss.backward()
    #                 nn.utils.clip_grad_norm_(self.route_network.parameters(), self.grad_clip) # type: ignore
    #                 self.route_optimizer.step()
        
    #     if self.mode == 'NGH':
    #         return actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()
    #     elif self.mode == 'OC':
    #         return actor_closs.item(), critic_closs.item(), entropy_closs.item()
    #     elif self.mode == 'OR':
    #         return actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()

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
            torch.save(self.cactor_optimizer.state_dict(), "{}_cactor_optimizer.pt".format(filename))
            torch.save(self.cq_optimizer.state_dict(), "{}_cq_optimizer.pt".format(filename))
        if self.mode in ['GH', 'NGH', 'OR']:
            torch.save(self.route_network.state_dict(), "{}_c.pt".format(filename))
            torch.save(self.ractor_optimizer.state_dict(), "{}_cactor_optimizer.pt".format(filename))
            torch.save(self.rq_optimizer.state_dict(), "{}_cq_optimizer.pt".format(filename))
            
    def load(self, filename):
        if self.mode in ['GH', 'NGH', 'OC']:
            self.charge_network.load_state_dict(torch.load("{}_c.pt".format(filename)))
            self.charge_optimizer.load_state_dict(torch.load("{}_c_optimizer.pt".format(filename)))
        if self.mode in ['GH', 'NGH', 'OR']:
            self.route_network.load_state_dict(torch.load("{}_r.pt".format(filename)))
            self.route_optimizer.load_state_dict(torch.load("{}_r_optimizer.pt".format(filename)))

    def update_target_param(self):
        for param, target_param in zip(self.charge_network.qf1.parameters(), self.charge_network.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.charge_network.qf2.parameters(), self.charge_network.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.route_network.qf1.parameters(), self.route_network.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.route_network.qf2.parameters(), self.route_network.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)