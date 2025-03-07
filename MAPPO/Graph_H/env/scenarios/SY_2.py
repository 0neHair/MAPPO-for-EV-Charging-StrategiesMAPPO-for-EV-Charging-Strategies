'''
Author: CQZ
Date: 2024-03-31 18:02:01
Company: SEU
'''
import numpy as np
import pandas as pd
from env.EV_agent import EV_Agent

class Scenario(): # Sioux-Falls transportation system
    def __init__(self, frame, seed):
        # 设定随机数
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
        # 仿真参数
        self.frame = frame
        
        # 图
        df_map = pd.read_csv("env/scenarios/SY.csv")
        O = []
        D = []
        length = []
        for i in range(df_map.shape[0]):
            O.append(df_map.iloc[i]['O'])
            D.append(df_map.iloc[i]['D'])
            length.append(df_map.iloc[i]['L'])
        self.edge_index = [
            O,
            D
        ] # 边编号
        self.edge_attr = []
        for l in length:
            self.edge_attr.append([l]) # 边特征，即边长度
        point_num = len(list(set(O + D)))
        self.map_adj = [
            [0 for _ in range(point_num)] for __ in range(point_num)
        ] # 邻接矩阵
        for i in range(len(self.edge_index[0])):
            self.map_adj[self.edge_index[0][i]][self.edge_index[1][i]] = 1
        # CS节点初始化特征
        self.cs_charger_waiting_time = [
            [0.0, 0.0] for _ in range(point_num)
            ] # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = [
            0 for _ in range(point_num)
            ] # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = [0.0 for _ in range(point_num)] # h 每个CS最小等待时间

        # assert len(self.route) == len(self.cs_charger_waiting_time)+1, "Error in map"
        # self.power = [30, 30, 30] # 功率
        # 动作
        self.caction_list = [i/100 for i in range(0, 105, 5)] # 动作列表
        self.caction_list[0] = 1.5 # 动作列表
        self.raction_list = [i for i in range(len(self.map_adj))] # 动作列表
        # 智能体
        self.agents = [] # 智能体列表
        total_vehicle_num = 30
        self.active_time_list = np.sort(np.round(np.random.uniform(0, 1, (total_vehicle_num)), 2))
        self.SOC_init_list = np.round(np.random.uniform(0.75, 0.95, (total_vehicle_num)), 2)
        self.SOC_exp_list = np.round(np.random.uniform(0.2, 0.5, (total_vehicle_num)), 2)
        for i in range(0, len(self.active_time_list)):
            # SOC_init = 0.8
            # SOC_exp = 0.4
            SOC_init = self.SOC_init_list[i]
            SOC_exp = self.SOC_exp_list[i]
            agent = EV_Agent(
                id=i, frame=self.frame, 
                # route=self.route, 
                map_adj=self.map_adj,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                caction_list=self.caction_list, 
                raction_list=self.raction_list, 
                enter_time=self.active_time_list[i], 
                SOC_init=SOC_init, SOC_exp=SOC_exp, 
                SOC90_penalty=0, SOC20_penalty=0, 
                consume=0.15, speed=100, E_max=60
                )
            self.agents.append(agent)

        