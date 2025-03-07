'''
Author: CQZ
Date: 2024-03-31 18:02:01
Company: SEU
'''
import numpy as np
from env.EV_agent import EV_Agent

class Scenario():
    def __init__(self, frame, seed=None):
        # 设定随机数
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
        # 仿真参数
        self.frame = frame
        # 地图
        # self.cs_waiting_time = [5.0, 0.0, 10.0, 0] # h 每个CS等待时间
        self.cs_charger_waiting_time = [
            list(np.round(np.random.uniform(0, 3, (3)), 2)), 
            list(np.round(np.random.uniform(0, 3, (3)), 2)), 
            list(np.round(np.random.uniform(0, 3, (3)), 2))
            ] # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = [] # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = []
        
        for i, cs in enumerate(self.cs_charger_waiting_time): # 充电站时间更新
            min_charger = min(self.cs_charger_waiting_time[i])
            min_charger_id = self.cs_charger_waiting_time[i].index(min_charger)
            self.cs_waiting_time.append(min_charger)
            self.cs_charger_min_id.append(min_charger_id)
        
        self.route = [40, 40, 80, 60] # km CS之间距离
        assert len(self.route) == len(self.cs_charger_waiting_time)+1, "Error in map"
        # self.power = [30, 30, 30] # 功率
        # 动作
        self.action_list = [i/100 for i in range(0, 105, 5)] # 动作列表
        self.action_list[0] = 1.5 # 动作列表
        # 智能体
        self.agents = [] # 智能体列表
        # total_vehicle_num = np.random.poisson(5) # 车辆生成服从泊松分布
        total_vehicle_num = 5
        active_time_list = np.sort(np.round(np.random.uniform(0, 1, (total_vehicle_num)), 2))
        for i in range(0, len(active_time_list)):
            SOC_init = np.random.uniform(0.2, 1)
            SOC_exp = np.random.uniform(0.3, 0.7)
            agent = EV_Agent(
                id=i, frame=self.frame, 
                route=self.route, 
                action_list=self.action_list, 
                enter_time=active_time_list[i],
                SOC_init=SOC_init, SOC_exp=SOC_exp, 
                SOC90_penalty=0, SOC20_penalty=0,
                consume=0.15, speed=100, E_max=60
                )
            self.agents.append(agent)
        
        