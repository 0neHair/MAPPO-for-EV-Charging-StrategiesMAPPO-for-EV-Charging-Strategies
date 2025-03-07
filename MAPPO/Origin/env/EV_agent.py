'''
Author: CQZ
Date: 2024-03-31 19:28:17
Company: SEU
'''
class EV_Agent():
    def __init__(
            self, 
            id, frame, 
            route, 
            action_list, 
            enter_time,
            SOC_init=0.5, SOC_exp=0.5, 
            SOC90_penalty=0, SOC20_penalty=0,
            consume=0.15, speed=100, E_max=60
        ):
        self.id = id
        self.frame = frame #  帧
        self.route = route # 路径
        self.action_list = action_list # 动作序列
        self.total_pos = len(route)
        self.current_pos = 0 # 判断当前所在位置
        # 偶数编号是站点，//2是站点编号，奇数编号是路段
       
        self.SOC_init = SOC_init # 0.5
        self.SOC_exp = SOC_exp # 0.5
        self.SOC_min = 0.1
        self.SOC = self.SOC_init
        
        self.SOC90_penalty = SOC90_penalty # 0
        self.SOC20_penalty = SOC20_penalty # 0
        self.multi_times_penalty = 0.5 # 0
        self.unfinished_penalty = 30
        self.unexpected_penalty = 5
        
        self.fixed_charging_wasting_time = 0 # 充电固定损失时间
        self.consume = consume # kWh/km 每公里消耗电量 0.15  
        self.speed = speed # km/h 均匀车速 100
        self.E_max = E_max # kWh 总电量 60
        
        self.enter_time = enter_time # 进入时间
        self.total_reward = 0 # 记录总奖励
        self.total_waiting = 0
        self.total_charging = 0
        self.reward = 0 # 记录奖励        

        self.dis_to_next = -1 # 到下一个决策点还有多久
        self.action_memory = []
        
        # 运行的指示器
        self.is_active = False # 判断是否进入环境
        self.is_charging = False # 判断是否在充电
        self.is_choosing = False # 判断是否在选择
        self.is_done = False # 判断是否完成状态迭代
        self.stop_update = False # 判断是否在环境中完成行程
        self.finish_trip = False # 判断是否走完行程
    
        self.waiting_charging_time = 0
        
        # 计算理想值
        self.total_distance = 0
        for l in self.route:
            self.total_distance += l
        self.ideal_charging_SOC = self.total_distance * self.consume / self.E_max + self.SOC_exp - self.SOC_init # 理想充电量
        self.ideal_charging_time = self.ideal_charging_SOC / 0.4 # 理想充电时间
        self.ideal_times = int(self.ideal_charging_SOC) + 1 # 理想充电次数
        
        # 数据存档
        self.time_memory = []
        self.reward_memory = []
        self.total_run_dis = 0
        self.ev_state = 0 # 0-行驶，1-排队，2-充电
        self.state_memory = []
        self.trip_memory = []
        self.pos_memory = []
        self.activity_memory = []
        self.SOC_memory = []
        self.action_choose_memory = []
        self.waiting_time = 0 # 累计等待时间
        self.charging_time = 0 # 累计充电时间
        
        self.total_used_time = 0 # 累计被使用时间
        self.charging_ts = 0 # 累计充电次数
        self.SOC_charged = 0 # 累计充电量
    
    def reset(self):
        self.current_pos = 0 # 判断当前所在位置
        # 偶数编号是站点，//2是站点编号，奇数编号是路段
       
        self.SOC = self.SOC_init
        self.total_reward = 0 # 记录总奖励
        self.total_waiting = 0
        self.total_charging = 0
        self.reward = 0 # 记录奖励        

        self.dis_to_next = -1 # 到下一个决策点还有多久
        self.action_memory = []
        self.reward_memory = []
        
        self.is_active = False # 判断是否进入环境
        self.is_charging = False # 判断是否在充电
        self.is_choosing = False # 判断是否在选择
        self.is_done = False # 判断是否完成状态迭代
        self.stop_update = False # 判断是否在环境中完成行程
        self.finish_trip = False # 判断是否走完行程

        self.waiting_charging_time = 0

        # 数据存档
        self.time_memory
        self.reward_memory = []
        self.total_run_dis = 0
        self.ev_state = 0 # 0-行驶，1-排队，2-充电
        self.state_memory = []
        self.trip_memory = []
        self.pos_memory = []
        self.activity_memory = []
        self.SOC_memory = []
        self.action_choose_memory = []
        self.waiting_time = 0 # 累计等待时间
        self.charging_time = 0 # 累计充电时间
        
        self.total_used_time = 0 # 累计被使用时间
        self.charging_ts = 0 # 累计充电次数
        self.SOC_charged = 0 # 累计充电量
        
    def activate(self):
        self.is_active = True
        self.current_pos += 1
        self.dis_to_next = self.route[self.current_pos//2]
        # 行程消耗
        consume_SOC = self.dis_to_next * self.consume / self.E_max
        self.SOC -= consume_SOC

        if self.SOC < 0: # 没电，没完成行程
            self.is_done = True
            self.reward -= 10
        
    def step(self, time):
        if self.stop_update == False and self.is_active == True:
            if self.is_charging: 
                
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
                    self.current_pos += 1 # 更新坐标
                    self.ev_state = 0
            else: 
                # 否则行驶
                self.dis_to_next -= self.frame * self.speed
                self.total_run_dis += self.frame * self.speed
                if round(self.dis_to_next, 2) <= 0:
                    # 若行程走完，说明到达检查点，更新坐标
                    self.dis_to_next = 0
                    self.current_pos += 1
                    if self.is_done: # 若已经判断出即将终止，到达终点或没电了
                        # 则以后不再更新状态
                        self.stop_update = True
                    else:
                        # 若不是终点，则成为下一个激活的智能体
                        self.is_choosing = True
        
        self.time_memory.append(time)
        self.trip_memory.append(self.total_run_dis)
        self.pos_memory.append(self.current_pos)
        self.activity_memory.append(len(self.action_memory) * self.is_charging)
        self.state_memory.append(self.ev_state)
        self.SOC_memory.append(self.SOC)
        if self.is_charging == False:
            self.action_choose_memory.append(-1)
        else:
            self.action_choose_memory.append(self.action_memory[-1])
        
    def set_action(self, action=0, waiting_time=0.0, charging_time=0.0):
        self.action_memory.append(action) # 记录动作
        self.is_choosing = False # 不处于激活状态
        reward = 0
        
        if action == 0: # 选择不充电，进行状态转移
            # 记录下一段要行驶的距离
            self.dis_to_next = self.route[self.current_pos//2]
            self.current_pos += 1
            # 行程消耗
            consume_SOC = self.dis_to_next * self.consume / self.E_max
            self.SOC -= consume_SOC
            
            if self.SOC < self.SOC_min: # 没电，没完成行程
                self.is_done = True
                reward -= self.unfinished_penalty            
            else:
                if self.current_pos == self.total_pos*2 - 1: # 即将到达终点
                    self.is_done = True
                    self.finish_trip = True
                    if self.SOC < self.SOC_exp:
                        reward -= self.unexpected_penalty
        else: # 选择充电，然后进行状态转移
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
            self.SOC_charged += self.action_list[action] - self.SOC
            
            if self.charging_ts > self.ideal_times + 1:
                reward -= self.multi_times_penalty
            
            self.SOC = self.action_list[action] # 充电更新
            reward -= self.waiting_charging_time
            self.dis_to_next = self.route[self.current_pos//2]
            next_postion = self.current_pos + 1
            # 预测行程消耗
            consume_SOC = self.dis_to_next * self.consume / self.E_max
            self.SOC -= consume_SOC
            
            if self.SOC < self.SOC_min: # 没电，没完成行程
                self.is_done = True
                reward -= self.unfinished_penalty
            else:
                if next_postion == self.total_pos*2 - 1: # 到达终点
                    self.is_done = True
                    self.finish_trip = True
                    if self.SOC < self.SOC_exp:
                        reward -= self.unexpected_penalty
    
        self.reward = reward
        self.total_reward += reward

    def ideal_conditions(self):
        print('Ideal charging SOC: {}%   Ideal charging time: {}h   Ideal charging times: {}'.format(self.ideal_charging_SOC, self.ideal_charging_time, self.ideal_times))