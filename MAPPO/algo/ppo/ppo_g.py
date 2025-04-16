import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from .ppo_base import PPOAgent
from .networks.GNN import GNet

class GPPOAgent(PPOAgent):
    def __init__(
            self, 
            state_dim, share_dim, caction_dim, caction_list,
            obs_features_shape, global_features_shape, raction_dim, raction_list,
            edge_index, buffer, device, 
            args
        ):
        super().__init__(
                state_dim, share_dim, caction_dim, caction_list,
                obs_features_shape, global_features_shape, raction_dim, raction_list,
                edge_index, buffer, device, 
                args
            )
        self.route_network = torch.compile(GNet(
                obs_features_shape, global_features_shape, state_dim, share_dim,
                raction_dim, self.raction_list, 
                args
            ).to(self.device))
        self.route_optimizer = torch.optim.Adam(self.route_network.parameters(), lr=self.lr, eps=1e-5)
        self.policy_batch = torch.LongTensor([0 for _ in range(32)]).to(self.device)
        self.value_batch = torch.LongTensor([0 for _ in range(obs_features_shape[0])]).to(self.device)

    def select_raction(self, obs_feature, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, state=state, mask=mask, policy_batch=self.policy_batch)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def select_best_raction(self, obs_feature, state, mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        obs_feature = torch.tensor(obs_feature, dtype=torch.float32).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        with torch.no_grad():
            dist = self.route_network.get_distribution(x=obs_feature, edge_index=self.edge_index, state=state, mask=mask, policy_batch=self.policy_batch)
            action = dist.probs.argmax() # type: ignore
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def train(self):
        state, share_state, caction, clog_prob, creward, next_state, next_share_state, cdone, \
            obs_feature, global_cs_feature, rstate, share_rstate,\
                raction, raction_mask, rlog_prob, rreward, \
                    next_obs_feature, next_global_cs_feature, next_rstate, next_share_rstate,\
                        next_raction_mask, rdone = self.rolloutBuffer.pull()
        buffer_step = self.rolloutBuffer.steps
        rbuffer_step = self.rolloutBuffer.rsteps
        
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
            rvalues = self.route_network.get_value(x=global_cs_feature, edge_index=self.edge_index, share_state=share_rstate, value_batch=self.value_batch).view(rbuffer_step, -1)
            next_rvalues = self.route_network.get_value(x=next_global_cs_feature, edge_index=self.edge_index, share_state=next_share_rstate, value_batch=self.value_batch).view(rbuffer_step, -1)
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
            
        # -------- flatten vectorized environment --------
        # note that this agent only supports the discrete action space, so the dimension of action in buffer is 1
        # the dimension of  action in buffer is different from the output dimension in policy network
        # 充电部分
        state = state.view(-1, self.state_dim)
        share_state = share_state.view(-1, self.share_dim)
        rstate = rstate.view(-1, self.state_dim)
        share_rstate = share_rstate.view(-1, self.share_dim)
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
            for index in BatchSampler(SubsetRandomSampler(range(buffer_step)), self.mini_batch_size, True):
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
                
                self.charge_optimizer.zero_grad()
                closs.backward()
                nn.utils.clip_grad_norm_(self.charge_network.parameters(), self.grad_clip) # type: ignore
                self.charge_optimizer.step()
                
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(rbuffer_step)), self.mini_rbatch_size, True):
                # 路径部分
                new_rdist = self.route_network.get_distribution(x=obs_feature[index], edge_index=self.edge_index, mask=raction_mask[index], state=rstate[index], policy_batch=self.policy_batch)
                new_rlog_prob = new_rdist.log_prob(raction[index].squeeze()).unsqueeze(1)
                new_rvalues = self.route_network.get_value(x=global_cs_feature[index], edge_index=self.edge_index, share_state=share_rstate[index], value_batch=self.value_batch).view(self.mini_rbatch_size, -1)
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
                
        return actor_closs.item(), critic_closs.item(), entropy_closs.item(), \
            actor_rloss.item(), critic_rloss.item(), entropy_rloss.item()
