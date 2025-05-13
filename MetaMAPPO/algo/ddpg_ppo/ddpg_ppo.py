import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .networks.FCN import cNetwork, rNetwork
import torch.nn.functional as F

class DDPG_PPOAgent(object):
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
        
        self.tau = 0.005

        if self.mode in ['GH', 'NGH', 'OC']:
            self.charge_network = torch.compile(cNetwork(
                    state_dim, share_dim, caction_dim, self.caction_list, 
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
            self.route_optimizer = torch.optim.Adam(self.route_network.parameters(), lr=self.lr, eps=1e-5)

    def select_caction(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.charge_network.get_distribution(state, train=True)
        return action.cpu().numpy().flatten(), 0

    def select_best_caction(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        with torch.no_grad():
            action = self.charge_network.get_distribution(state)
        return action.cpu().numpy().flatten(), 0
    
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
    
    def train_ddpg(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                next_raction_mask, rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        rbuffer_step = self.rolloutBuffer.rsteps
        # 充电部分
        for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), buffer_step, True):
            state = state[index].view(-1, self.state_dim)
            next_state = next_state[index].view(-1, self.state_dim)
            share_state = share_state[index].view(-1, self.share_dim)
            next_share_state = next_share_state[index].view(-1, self.share_dim)
            caction = caction[index].view(-1, 1)
            creward = creward[index].view(-1, 1)
            cdone = cdone[index].view(-1, 1)
            with torch.no_grad():
                next_cation = self.charge_network.actor_target(next_state)
                q_next_target = self.charge_network.qf_target(next_share_state, next_cation)
                next_q_value = creward + (1 - cdone) * self.gamma * (q_next_target)
                
            q_value = self.charge_network.qf(share_state, caction)
            qf_loss = F.mse_loss(q_value, next_q_value)            
            
            self.cq_optimizer.zero_grad()
            qf_loss.backward()
            self.cq_optimizer.step()

            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_closs = -self.charge_network.qf(share_state, self.charge_network.actor(state)).mean()
            self.cactor_optimizer.zero_grad()
            actor_closs.backward()
            self.cactor_optimizer.step()
            break
        return actor_closs, qf_loss
    
    def train_ppo(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            rstate, share_rstate, raction, raction_mask, rlog_prob, rreward, next_rstate, next_share_rstate,\
                next_raction_mask, rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        rbuffer_step = self.rolloutBuffer.rsteps
        
        with torch.no_grad():
            if self.mode in ['OR', 'NGH'] :
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
            torch.save(self.cactor_optimizer.state_dict(), "{}_cactor_optimizer.pt".format(filename))
            torch.save(self.cq_optimizer.state_dict(), "{}_cq_optimizer.pt".format(filename))
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

    def update_target_param(self):
        for param, target_param in zip(self.charge_network.qf.parameters(), self.charge_network.qf_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.charge_network.actor.parameters(), self.charge_network.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)