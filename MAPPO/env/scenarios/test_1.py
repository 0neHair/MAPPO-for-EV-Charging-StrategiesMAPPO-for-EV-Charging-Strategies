'''
Author: CQZ
Date: 2024-03-31 18:02:01
Company: SEU
'''
import numpy as np
from env.EV_agent import EV_Agent

class Scenario():
    def __init__(self, frame, seed):
        # 设定随机数
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
        # 仿真参数
        self.frame = frame
        
        # 图
        # self.route = [40, 40, 40, 40] # km CS之间距离
        self.map_adj = [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ] # 邻接矩阵
        self.edge_index = [
            [0, 1, 2, 3, 2, 1],
            [1, 2, 3, 4, 4, 4]
        ] # 边编号
        self.edge_attr = [
            [40],
            [40],
            [40],
            [40],
            [60],
            [120]
        ] # 边特征，即边长度
    
        # CS节点初始化特征
        # self.cs_charger_waiting_time = [
        #     [2.0, 2.0, 2.0], 
        #     [0.0, 0.0, 2.0], 
        #     [3.0, 3.0, 3.0]
        #     ] # h 每个CS每个充电桩等待时间
        self.cs_charger_waiting_time = [
            [0.0], 
            [0.0], 
            [0.0]
            ] # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = [
            0, 
            0, 
            0
            ] # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = [0.0, 0.0, 0.0] # h 每个CS最小等待时间

        # assert len(self.route) == len(self.cs_charger_waiting_time)+1, "Error in map"
        # self.power = [30, 30, 30] # 功率
        # 动作
        self.caction_list = [i/100 for i in range(0, 105, 5)] # 动作列表
        self.caction_list[0] = 1.5 # 动作列表
        self.raction_list = [i for i in range(len(self.map_adj))] # 动作列表
        # 智能体
        self.agents = [] # 智能体列表
        agent = EV_Agent(
            id=0, frame=self.frame, 
            # route=self.route, 
            map_adj=self.map_adj,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            caction_list=self.caction_list, 
            raction_list=self.raction_list, 
            enter_time=0, 
            SOC_init=0.5, SOC_exp=0.5, 
            SOC90_penalty=0, SOC20_penalty=0, 
            consume=0.15, speed=100, E_max=60
            )
        self.agents.append(agent)

        