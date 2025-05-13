'''
Author: CQZ
Date: 2025-04-10 19:43:38
Company: SEU
'''
import numpy as np
from env.Env_Base import Env_base

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_G(Env_base):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None, ps=False):
        super().__init__(scenario, seed, ps)

        self.sub_cs_feature = np.zeros((2, self.num_cs)) # Graph information
        # 观测
        # 充电站最低排队时间 + 编号
        self.obs_features_dim = 1 + 1
        # 全局
        # 充电站最低排队时间 + 所有智能体编号
        self.global_features_dim = self.cs_charger_waiting_time.shape[1] + self.agent_num

    def agents_arrange(self):
        self.onehot_pos *= 0
        for i, agent in enumerate(self.agents):
            # observation
            self.get_local_obs(i, agent)
            self.sub_cs_feature[0, :] = self.cs_waiting_time/100
            self.sub_cs_feature[1, :] = self.onehot_pos_          
            self.obs_feature_n[i] = self.sub_cs_feature.T
            self.onehot_pos[agent.target_pos][i] = 1

            # reward
            if i in self.activate_agent_ci:
                if self.activate_to_cact[self.activate_agent_ci.index(i)] == False:
                    self.creward_n[i] = self.get_reward(agent, c=True, isfinal=True)
                else:
                    self.creward_n[i] = self.get_reward(agent, c=True, isfinal=False)
            else:
                self.creward_n[i] = self.get_reward(agent, c=True, isfinal=False)
            self.rreward_n[i] = self.get_reward(agent, c=False)
            self.done_n[i] = self.get_done(agent)
        
        # shared observation
        self.get_share_obs()
        self.global_cs_feature[:, :-self.agent_num] = self.cs_charger_waiting_time / 100
        self.global_cs_feature[:, -self.agent_num:] = self.onehot_pos
    
