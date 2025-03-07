'''
Author: CQZ
Date: 2024-03-31 19:28:17
Company: SEU
'''
import numpy as np

class EV_Agent():
    def __init__(
        self, 
        id, frame, 
        map_adj,
        edge_index,
        edge_attr,
        caction_list, 
        raction_list, 
        enter_time,
        SOC_init=0.5, SOC_exp=0.5, 
        SOC90_penalty=0, SOC20_penalty=0,
        consume=0.15, speed=100, E_max=60
        ):
        self.id = id
        self.frame = frame #  帧
        
        # 图
        self.map_adj = np.array(map_adj)
        self.edge_index = np.array(edge_index)
        self.edge_attr = np.array(edge_attr)
        self.map_adj_index = self.map_adj * np.arange(1, self.map_adj.shape[0]+1) # 编号+1
        self.OD2edge = {} # O-D: 边编号
        for e in range(self.edge_index.shape[1]):
            O = int(self.edge_index[0][e])
            D = int(self.edge_index[1][e])
            index = str(O) + '-' + str(D)
            self.OD2edge[index] = e
            
        # 路径动作
        self.raction_list =  raction_list
        # 充电动作
        self.caction_list = caction_list # 动作序列
        self.final_pos = self.map_adj.shape[0]-1 # 默认最后一个点是终点
        
        self.target_pos = 0 # 前往的下一个节点
        self.on_the_way = False
        
        self.waiting_time = 0 # 等待时间还有多久
        self.charging_time = 0 # 充电时间还有多久
        self.waiting_charging_time = 0  # 充电等待时间总共还有多久
        self.time_to_next = -1 # 到下一个决策点还有多久
        self.action_type = []

        self.enter_time = enter_time # 进入时间
        self.total_reward = 0 # 记录总奖励
        self.total_waiting = 0
        self.total_charging = 0
        self.c_reward = 0 # 记录充电奖励
        self.r_reward = 0 # 记录路径奖励
        
        self.ev_state = 0 # 0-在路上，1-排队，2-充电
        self.current_position = 'P0'
        self.current_cs = 'P0'
        self.current_road = ''

        # 运行的指示器
        self.is_active = False # 判断是否进入环境
        self.is_charging = False # 判断是否在充电
        self.is_choosing = False # 判断是否在选择
        self.is_routing = 0 # 判断是否选择路径
        self.is_done = False # 判断是否完成状态迭代
        self.stop_update = False # 判断是否在环境中完成行程
        self.finish_trip = False # 判断是否走完行程
     
        self.SOC_init = SOC_init # 0.5
        self.SOC_exp = SOC_exp # 0.5
        self.SOC_min = 0.1
        self.SOC = self.SOC_init
        
        self.SOC90_penalty = SOC90_penalty # 0
        self.SOC20_penalty = SOC20_penalty # 0
        self.multi_times_penalty = 0.5 # 0
        self.unfinished_penalty = 30
        self.unexpected_penalty = 5
        
        self.travel_time_beta = 1 # 通行时间系数
        self.waiting_time_beta = 1 # 排队时间系数
        self.charging_time_beta = 1 # 充电时间系数
        self.fixed_charging_wasting_time = 0 # 充电固定损失时间
        
        self.consume = consume # kWh/km 每公里消耗电量 0.15  
        self.speed = speed # km/h 均匀车速 100
        self.E_max = E_max # kWh 总电量 60
        
        # # 计算理想值
        # self.total_distance = 0
        # for l in self.route:
        #     self.total_distance += l
        # self.ideal_charging_SOC = self.total_distance * self.consume / self.E_max + self.SOC_exp - self.SOC_init # 理想充电量
        # self.ideal_charging_time = self.ideal_charging_SOC / 0.4 # 理想充电时间
        # self.ideal_times = int(self.ideal_charging_SOC) + 1 # 理想充电次数
        
        # 数据存档
        self.time_memory = []
        self.reward_memory = []
        self.total_run_dis = 0
        
        self.total_route = []
        self.state_memory = []
        self.trip_memory = []
        self.pos_memory = []
        self.activity_memory = []
        self.SOC_memory = []
        self.action_choose_memory = []
        self.caction_memory = []
        self.raction_memory = []
        self.action_memory = []
        
        self.total_used_time = 0 # 累计被使用时间
        self.charging_ts = 0 # 累计充电次数
        self.SOC_charged = 0 # 累计充电量
    
    def reset(self):
        self.SOC = self.SOC_init
        
        self.target_pos = 0 # 前往的下一个CS
        self.on_the_way = False
        
        self.waiting_time = 0 # 等待时间还有多久
        self.charging_time = 0 # 充电时间还有多久
        self.waiting_charging_time = 0  # 充电等待时间总共还有多久
        self.time_to_next = -1 # 到下一个决策点还有多久
        
        self.raction_memory = []
        self.caction_memory = []
        self.action_memory = []
        self.action_type = []

        self.total_reward = 0 # 记录总奖励
        self.total_waiting = 0
        self.total_charging = 0
        self.c_reward = 0 # 记录充电奖励
        self.r_reward = 0 # 记录路径奖励  
        
        # 运行的指示器
        self.is_active = False # 判断是否进入环境
        self.is_charging = False # 判断是否在充电
        self.is_choosing = False # 判断是否在选择
        self.is_routing = 0 # 判断是否选择路径
        self.is_done = False # 判断是否完成状态迭代
        self.stop_update = False # 判断是否在环境中完成行程
        self.finish_trip = False # 判断是否走完行程

        # 数据存档
        self.time_memory = []
        self.reward_memory = []
        self.total_route = []
        self.total_run_dis = 0
        self.ev_state = 0 # 0-在路上，1-排队，2-充电
        self.current_position = 'P0'
        self.current_cs = 'P0'
        self.current_road = ''
        self.state_memory = []
        self.trip_memory = []
        self.pos_memory = []
        self.activity_memory = []
        self.SOC_memory = []
        self.action_choose_memory = []
        self.total_used_time = 0 # 累计被使用时间
        self.charging_ts = 0 # 累计充电次数
        self.SOC_charged = 0 # 累计充电量
        
    def activate(self):
        '''
        智能体启动，即智能体进入环境，并得到第一段路段的奖励
        '''
        self.is_active = True
        self.if_choose_route()
            
    def step(self, time):
        '''
        智能体自身状态转移，实时记录当前车状态
        
        先检查自身等待时间，若有则消耗，属于排队阶段
        然后检查自身充电时间，若有则消耗，属于充电阶段
        否则视为在路段上行驶，检查剩余距离，
            若有则消耗
            若无则检查是否到达决策点或终点，设置指示变量
        '''
        if self.is_charging == True:
            self.action_choose_memory.append(self.caction_memory[-1])
        else:
            self.action_choose_memory.append(-1)
            
        if self.stop_update == False and self.is_active == True:
            if self.is_charging: 
                self.current_position = self.current_cs
                self.waiting_time -= self.frame
                if round(self.waiting_time, 2) < 0:
                    self.waiting_time = 0
                    self.charging_time -= self.frame
                    self.ev_state = 2
                # 若排队充电时间尚未耗尽，则正在排队充电
                self.waiting_charging_time -= self.frame
                if round(self.waiting_charging_time, 2) <= 0:
                    self.waiting_charging_time = 0
                    self.charging_time = 0
                    # 若充电完成
                    self.is_charging = False # 状态变更
                    # self.current_pos += 1 # 更新坐标
                    self.ev_state = 0
                    # self.if_choose_route() # 看要不要进行路段选择
            else: 
                # 否则行驶
                self.current_position = self.current_road
                assert self.time_to_next > 0, "BUG exists"
                self.time_to_next -= self.frame
                self.total_run_dis += self.frame * self.speed
                if round(self.time_to_next, 2) <= 0:
                    # 若行程走完，说明到达检查点，更新坐标
                    self.time_to_next = 0
                    # self.current_pos += 1
                    if self.is_done: # 若已经判断出即将终止，到达终点或没电了
                        # 则以后不再更新状态
                        self.stop_update = True
                    else:
                        # 若不是终点，则成为下一个激活的智能体
                        self.is_choosing = True
        
        self.time_memory.append(time)
        self.trip_memory.append(self.total_run_dis)
        self.pos_memory.append(self.current_position)
        self.activity_memory.append(len(self.caction_memory) * self.is_charging)
        self.state_memory.append(self.ev_state)
        self.SOC_memory.append(self.SOC)
        # if self.is_charging == True:
        #     self.action_choose_memory.append(self.caction_memory[-1])
        # else:
        #     self.action_choose_memory.append(-1)
        
    def set_caction(self, caction=0, waiting_time=0.0, charging_time=0.0):
        '''
        设置充电动作，并执行
        智能体会根据CS状态获得瞬时的等待与充电时间奖励
        然后再根据接下来要行驶的路程获得行驶时间奖励
        
        奖励会预先获得，但不代表智能体已完成相应的活动
        '''
        # 记录动作
        self.caction_memory.append(caction)
        self.action_memory.append(caction)
        self.action_type.append('c')
        self.is_choosing = False # 不处于激活状态
        reward = 0
        
        if self.target_pos == 0:
            caction = 0
        
        if caction == 0: # 选择不充电
            pass
        else: # 选择充电，然后进行电量状态转移
            # 充电
            self.charging_time += charging_time
            self.waiting_time += waiting_time
            self.waiting_charging_time = waiting_time + charging_time + self.fixed_charging_wasting_time
            self.is_charging = True
            self.ev_state = 1
            
            self.total_waiting += waiting_time
            self.total_charging += charging_time
            self.total_used_time += (charging_time + waiting_time + self.fixed_charging_wasting_time)
            self.charging_ts += 1
            self.SOC_charged += self.caction_list[caction] - self.SOC
            
            # if self.charging_ts > self.ideal_times + 1:
            #     reward -= self.multi_times_penalty
            assert self.caction_list[caction] > self.SOC, "SOC > target_SOC"
            self.SOC = self.caction_list[caction] # 充电更新
            reward -= (waiting_time*self.waiting_time_beta + charging_time*self.charging_time_beta + self.fixed_charging_wasting_time)
            
        self.c_reward = reward
        self.r_reward = reward
        self.total_reward += reward

    def set_raction(self, raction=0, reset_record=False):
        '''
        设定路径动作，并完成位置转移和行程消耗计算
        智能体会在此函数判断终止
        
        奖励会预先获得，但不代表智能体已完成相应的活动
        '''
        reward = 0 # 记录接下来一系列操作的奖励
         # 记录动作
        self.raction_memory.append(raction)
        self.action_memory.append(raction)
        self.action_type.append('r')
        self.is_routing = 0
        
        if raction == 0:
            choose_set = self.get_choice_set()
            choose_set = choose_set[choose_set > 0] - 1
            next_pos = self.raction_list[choose_set[raction]]
            self.raction_memory[-1] = next_pos
            self.action_memory[-1] = next_pos
        else:
            next_pos = self.raction_list[raction]
        
        od = str(int(self.target_pos)) + '-' + str(int(next_pos))
        current_edge = self.OD2edge[od]
        dis_to_next = self.edge_attr[current_edge][0] # 路段距离
        self.time_to_next = dis_to_next / self.speed # 路段行驶时间
        self.total_used_time += self.time_to_next
        reward -= self.time_to_next*self.travel_time_beta
        self.current_cs = 'P'+str(self.target_pos)
        self.target_pos = next_pos
        self.current_road = od # 记录所在位置
        self.total_route.append(od)

        # 计算路段消耗
        consume_SOC = dis_to_next * self.consume / self.E_max # 行程消耗
        self.SOC -= consume_SOC
        if self.SOC < self.SOC_min: # 没电，没完成行程
            self.is_done = True
            reward -= self.unfinished_penalty
        else:
            if self.target_pos == self.final_pos: # 即将到达终点
                self.is_done = True
                self.finish_trip = True
                if self.SOC < self.SOC_exp:
                    reward -= self.unexpected_penalty
        # 更新奖励
        if reset_record: # 如果是决策点，重设奖励
            self.r_reward += reward
        else: # 否则累加
            self.r_reward = reward
        self.total_reward += reward
    
    def get_choice_set(self):
        choose_set = self.map_adj_index[self.target_pos].copy()
        return choose_set
    
    def if_choose_route(self):
        # 看要不要进行路段选择
        self.current_position = self.current_cs
        choose_set = self.get_choice_set()
        choose_set = choose_set[choose_set > 0] - 1
        if choose_set.shape[0] > 1:
            self.is_routing = choose_set.shape[0] # 设置选路标记
            self.is_choosing = True
        else:
            self.set_raction(raction=choose_set[0], reset_record=False)
    
    def get_choice_set_mask(self):
        choose_set_mask = self.map_adj[self.target_pos].copy()
        return choose_set_mask
    
    def get_penalty(self):
        # 给最后的充电奖励加上全局惩罚，即未完成行程或未达预期
        penalty = 0
        if self.SOC < self.SOC_min: # 没电，没完成行程
            penalty -= self.unfinished_penalty
        else:
            if self.SOC < self.SOC_exp: # 未达到预期
                penalty -= self.unexpected_penalty
        return penalty

        
    # def ideal_conditions(self):
    #     print('Ideal charging SOC: {}%   Ideal charging time: {}h   Ideal charging times: {}'.format(self.ideal_charging_SOC, self.ideal_charging_time, self.ideal_times))
        