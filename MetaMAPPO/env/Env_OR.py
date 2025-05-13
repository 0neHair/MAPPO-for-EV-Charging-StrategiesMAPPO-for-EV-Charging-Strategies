'''
Author: CQZ
Date: 2025-04-10 19:43:38
Company: SEU
'''
import numpy as np
from env.Env_Base import Env_base

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_OR(Env_base):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None, ps=False):
        super().__init__(scenario, seed, ps)
            
        self.near_end = []
        for i in range(self.map_adj_index.shape[0]):
            if self.map_adj_index[i][-1] != 0:
                self.near_end.append(i)
        self.OD2edge = {} # O-D: 边编号
        for e in range(self.edge_index.shape[1]):
            O = int(self.edge_index[0][e])
            D = int(self.edge_index[1][e])
            index = str(O) + '-' + str(D)
            self.OD2edge[index] = e
            
    def set_n_caction(self, agent, caction):
        caction = 0
        if agent.target_pos in self.near_end:
            od = str(int(agent.target_pos)) + '-' + str(int(self.final_pos))
            consume_SOC = self.edge_attr[self.OD2edge[od]][0]  * agent.consume / agent.E_max
            if agent.SOC - consume_SOC < agent.SOC_exp:
                tSOC = consume_SOC + agent.SOC_exp + 0.01
                rtSOC = round(tSOC, 1)
                if rtSOC < tSOC:
                    rtSOC += 0.05
                caction = self.caction_list.index(rtSOC)
        else:
            if agent.SOC < 0.15:
                caction = len(self.caction_list)-1
    
        super().set_n_caction(agent, caction)
        return caction