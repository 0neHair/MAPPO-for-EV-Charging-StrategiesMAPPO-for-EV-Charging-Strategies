'''
Author: CQZ
Date: 2025-04-14 21:52:11
Company: SEU
'''
import numpy as np
from env.Env_Base import Env_base

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_continuous(Env_base):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None, ps=False):
        super().__init__(scenario, seed, ps)
        
    def set_n_caction(self, agent, caction):
        if caction == 0:
            agent.set_caction_continuous(caction) # 将选择和充电时间输入给智能体
        else:
            waiting_time = self.cs_waiting_time[agent.target_pos]
            charging_time = int(self.get_charging_time(cur_SOC=agent.SOC, final_SOC=caction) * 100)

            agent.set_caction_continuous(caction, waiting_time, charging_time) # 将选择和充电时间输入给智能体
            
            self.cs_charger_waiting_time[agent.target_pos][self.cs_charger_min_id[agent.target_pos]] += charging_time
            self.cs_charger_min_id = np.argmin(self.cs_charger_waiting_time, axis=1)
            self.cs_waiting_time = self.cs_charger_waiting_time[
                    self.cs_charger_waiting_time_arange, 
                    self.cs_charger_min_id
                ]
        return caction