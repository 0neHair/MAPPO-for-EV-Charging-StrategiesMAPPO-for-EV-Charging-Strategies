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
        # 地图
        self.cs_charger_waiting_time = [
            [5.0, 5.0], 
            [0.0, 0.0], 
            [10.0, 10.0]
            ] # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = [
            0, 
            0, 
            0
            ] # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = [5.0, 0.0, 10.0] # h 每个CS等待时间
        self.route = [40, 40, 40, 40] # km CS之间距离
        assert len(self.route) == len(self.cs_charger_waiting_time)+1, "Error in map"
        # self.power = [30, 30, 30] # 功率
        # 动作
        self.action_list = [i/100 for i in range(0, 105, 5)] # 动作列表
        self.action_list[0] = 1.5 # 动作列表
        # 智能体
        self.agents = [] # 智能体列表
        agent = EV_Agent(
            id=0, frame=self.frame, 
            route=self.route, 
            action_list=self.action_list, 
            enter_time=0,
            SOC_init=0.5, SOC_exp=0.5, 
            SOC90_penalty=0, SOC20_penalty=0,
            consume=0.15, speed=100, E_max=60
            )
        self.agents.append(agent)
        
        agent = EV_Agent(
            id=1, frame=self.frame, 
            route=self.route, 
            action_list=self.action_list, 
            enter_time=0.40,
            SOC_init=0.5, SOC_exp=0.5, 
            SOC90_penalty=0, SOC20_penalty=0,
            consume=0.15, speed=100, E_max=60
            )
        self.agents.append(agent)
        