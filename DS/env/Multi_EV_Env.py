'''
Author: CQZ
Date: 2024-03-31 18:05:02
Company: SEU
'''
# import gym
# from gym import spaces
# from gym.envs.registration import EnvSpec
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.EV_agent import EV_Agent

def charging_function(SOC):
    if SOC <= 0.8:
        return SOC / 0.4
    elif SOC <= 0.85:
        return 2 + (SOC - 0.8) / 0.25
    else:
        return 2.2 + (SOC -0.85) / 0.1875

def get_charging_time(cur_SOC, final_SOC):
    return charging_function(final_SOC) - charging_function(cur_SOC)

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Multi_EV_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None):
        # 设定随机数
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        # 仿真参数
        self.current_step = 0 # 步长即时间，0.05h一帧
        self.total_time = 0
        self.scenario = scenario
        self.frame = self.scenario.frame
        # 地图
        self.cs_charger_waiting_time = copy.deepcopy(self.scenario.cs_charger_waiting_time) # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = copy.deepcopy(self.scenario.cs_charger_min_id) # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = copy.deepcopy(self.scenario.cs_waiting_time)
        self.route = self.scenario.route.copy()
        self.num_position = len(self.route)
        self.num_cs = len(self.cs_charger_waiting_time)
        # self.power = self.scenario.power.copy()
        # 智能体
        self.agents = self.scenario.agents.copy()
        self.agent_num = len(self.agents)
        self.agents_active = self.agents.copy()
        
        # 动作
        self.action_list = self.scenario.action_list
        # 动作空间
        self.action_space = [spaces.Discrete(len(self.action_list)) for _ in range(self.agent_num)]  # type: ignore
        self.action_dim = len(self.action_list)
        self.state_name = ['agent_SOC', 'agent_pos', 'agent_usingtime', 'agent_charging_ts', 'agent_next_waiting', 'is_finish']
        self.state_dim = len(self.state_name) # 状态维度
        self.share_name = ['agent_SOC', 'agent_pos', 'agent_usingtime', 'agent_charging_ts', 'is_finish'] * self.agent_num + self.cs_waiting_time
        self.share_dim = len(self.share_name)
    
        # 状态空间
        self.observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, 
                shape=(self.state_dim,), dtype=np.float32
                ) for _ in range(self.agent_num) # type: ignore
            ]
        # 共享状态空间  
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, 
                shape=(self.share_dim,), dtype=np.float32
                ) for _ in range(self.agent_num) # type: ignore
            ]
        
        # 存档
        self.time_memory = []
        self.cs_waiting_cars_memory = []
        self.cs_charging_cars_memory = []
        self.cs_waiting_cars_num_memory = []
        self.cs_charging_cars_num_memory = []
        self.cs_waiting_time_memory = []
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            
    # step  this is  env.step()
    def step(self, action_n):
        activate_agent_i = [] # 记录正处于关键点的智能体id
        activate_to_act = [] # 处于关键点的智能体是否可以做选择，到达终点的不能
        
        obs_n = []
        act_n = []
        reward_n = []
        done_n = []
        info_n = []
        
        # 第一阶段：设置好智能体动作
        # 当动作为None时，说明到达检查地点仅终点
        # print('第一阶段：设置好智能体动作')
        for i, agent in enumerate(self.agents):
            if agent.is_choosing:
                self.set_n_action(agent, action_n[i]) # 设置动作
        
        # 第二阶段：让环境运行至有智能体到达检查地点
        # print('第二阶段：让环境运行至有智能体到达检查地点')
        agent_to_remove = []
        run_step = True
        while run_step and self.agents_active:
            self.total_time += self.frame
            for i, agent in enumerate(self.agents_active):
                if round(self.total_time, 2) >= agent.enter_time: # 到达EV进入时间则启动
                    if not agent.is_active:
                        agent.activate()
                agent.step(time=self.total_time)
                if agent.is_choosing: # 如果在检查点，记录并跳出
                    run_step = False
                    activate_agent_i.append(agent.id)
                    activate_to_act.append(True)
                if agent.stop_update: # 如果在终点，记录并跳出
                    run_step = False
                    activate_agent_i.append(agent.id)
                    activate_to_act.append(False)
                    agent_to_remove.append(agent)
            
            for i, cs in enumerate(self.cs_charger_waiting_time): # 充电站时间更新
                for j, charger in enumerate(cs):
                    if self.cs_charger_waiting_time[i][j] > 0:
                        self.cs_charger_waiting_time[i][j] -= self.frame
                        if round(self.cs_charger_waiting_time[i][j], 2) <= 0:
                            self.cs_charger_waiting_time[i][j] = 0
                            
                self.cs_waiting_time[i] = min(self.cs_charger_waiting_time[i])
                self.cs_charger_min_id[i] = self.cs_charger_waiting_time[i].index(self.cs_waiting_time[i])
                
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.cs_save_memory()
            
        for agent in agent_to_remove: # 将不再更新的智能体去掉
            self.agents_active.remove(agent)
        
        # 第三阶段：整理奖励和输出
        # 当智能体到达检查点后，其上一次状态选择的动作有了结果，因此输出上一次的动作和这一次的状态  
        # print('第三阶段：整理奖励和输出')  
        for i, agent in enumerate(self.agents):
            obs_n.append(self.get_obs(agent))
            act_n.append(self.get_act(agent))
            reward_n.append(self.get_reward(agent))
            done_n.append(self.get_done(agent))
            # info = {'individual_reward': self.get_reward(agent)}
            # env_info = self._get_info(agent)
            # if 'fail' in env_info.keys():
            #     info['fail'] = env_info['fail']
            # info_n.append(info)
        share_obs = self.get_share_state()
        
        self.current_step += 1
        return obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act

    def cs_save_memory(self):
        cs_waiting_cars = [[] for _ in range(self.num_cs)]
        cs_charging_cars = [[] for _ in range(self.num_cs)]
        cs_waiting_num_cars = [0 for _ in range(self.num_cs)]
        cs_charging_num_cars = [0 for _ in range(self.num_cs)]
        for i, agent in enumerate(self.agents):
            pos = agent.current_pos
            if pos % 2 == 0 and pos != 0 and agent.is_charging:
                cs_pos = pos//2 - 1
                if cs_pos < self.num_cs:
                    if agent.waiting_time == 0:
                        cs_charging_cars[cs_pos].append(agent.id)
                    else:
                        cs_waiting_cars[cs_pos].append(agent.id)
        for i in range(self.num_cs):
            cs_waiting_num_cars[i] = len(cs_waiting_cars[i])
            cs_charging_num_cars[i] = len(cs_charging_cars[i])
            
        self.time_memory.append(self.total_time)
        self.cs_waiting_cars_num_memory.append(copy.deepcopy(cs_waiting_num_cars))
        self.cs_charging_cars_num_memory.append(copy.deepcopy(cs_charging_num_cars))
        self.cs_waiting_cars_memory.append(copy.deepcopy(cs_waiting_cars))
        self.cs_charging_cars_memory.append(copy.deepcopy(cs_charging_cars))
        self.cs_waiting_time_memory.append(copy.deepcopy(self.cs_waiting_time))
    
    def reset(self):
        self.current_step = 0 # 步长即时间，0.05h一帧
        self.total_time = 0
        # 地图
        self.cs_charger_waiting_time = copy.deepcopy(self.scenario.cs_charger_waiting_time) # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = copy.deepcopy(self.scenario.cs_charger_min_id) # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = copy.deepcopy(self.scenario.cs_waiting_time)
        # self.power = self.scenario.power.copy()
        # 智能体
        for agent in self.agents:
            agent.reset()
        self.agents_active = self.agents.copy()
    
        # 存档
        self.time_memory = []
        self.cs_waiting_cars_memory = []
        self.cs_charging_cars_memory = []
        self.cs_waiting_time_memory = []    
        self.cs_waiting_cars_num_memory = []
        self.cs_charging_cars_num_memory = []
                    
    def get_obs(self, agent: EV_Agent):
        # 智能体观测量，即状态
        # 当前SOC、位置、当前检查点等待时间
        agent_SOC = agent.SOC
        agent_pos = agent.current_pos
        agent_usingtime = agent.total_used_time
        agent_charging_ts = agent.charging_ts
        agent_complete_trip = agent.finish_trip
        agent_next_waiting = 0
        cs_pos = int(agent.current_pos // 2) - 1
        if cs_pos != self.num_position-1:
            agent_next_waiting = self.cs_waiting_time[cs_pos]
        
        return np.array([agent_SOC, agent_pos, agent_usingtime, agent_charging_ts, agent_next_waiting, agent_complete_trip])
    
    def get_act(self, agent: EV_Agent):
        # 智能体上一个动作
        agent_act_memory = agent.action_memory.copy()
        agent_last_action = -1
        if len(agent_act_memory) > 0:
            agent_last_action = agent_act_memory[-1]
        return np.array([agent_last_action])
    
    def get_reward(self, agent: EV_Agent):
        # 智能体当前奖励
        agent_reward = agent.reward
        return np.array([agent_reward])
    
    def get_done(self, agent: EV_Agent):
        # 智能体当前奖励
        agent_done = agent.is_done
        return np.array([agent_done])


    def set_n_action(self, agent: EV_Agent, action):
        if action == 0:
            agent.set_action(action) # 将选择和充电时间输入给智能体
        else:
            postion = agent.current_pos
            sc_pos = postion//2-1
            agent_SOC = agent.SOC
            waiting_time = self.cs_waiting_time[sc_pos]
            act_SOC = action + agent_SOC
            charging_time = get_charging_time(cur_SOC=agent_SOC, final_SOC=act_SOC)

            agent.set_action(action, waiting_time, charging_time) # 将选择和充电时间输入给智能体
            
            min_charger_id = self.cs_charger_min_id[sc_pos]
            self.cs_charger_waiting_time[sc_pos][min_charger_id] += charging_time
            
            self.cs_waiting_time[sc_pos] = min(self.cs_charger_waiting_time[sc_pos])
            self.cs_charger_min_id[sc_pos] = self.cs_charger_waiting_time[sc_pos].index(self.cs_waiting_time[sc_pos])
    
    def get_policy(self):
        current_policy = []
        for agent in self.agents:
            current_policy.append(agent.action_memory)
        return current_policy
    
    def get_share_state(self):
        share_state = []
        for agent in self.agents:
            share_state.append(agent.SOC)
            share_state.append(agent.current_pos)
            share_state.append(agent.total_used_time)
            share_state.append(agent.charging_ts)
            share_state.append(agent.finish_trip)
        return np.array(share_state.copy() + self.cs_waiting_time.copy())

    def render(self):
        print('Time: {} h'.format(round(self.total_time, 2)))
        cs_list = [[] for _ in range(self.num_position+1)]
        link_list = [] # 每个-是10km
        for i in range(self.num_position):
            road = []
            distance = self.route[i]
            part_num = int(distance // 10)
            if part_num <= 0:
                part_num = 1
            for j in range(part_num):
                road.append([])
            link_list.append(road)
        
        for i, agent in enumerate(self.agents):
            agent_pos = agent.current_pos
            if agent_pos % 2 == 0:
                cs_pos = agent_pos//2
                cs_list[cs_pos].append(agent)
            else:
                link_pos = int(agent_pos//2)
                link_len = self.route[link_pos]
                real_pos = link_len - agent.dis_to_next
                part_num = int(real_pos // 10)
                link_list[link_pos][part_num].append(agent)

        for i in range(self.num_position):
            cs = cs_list[i]
            if i == 0:
                print('O:', end='')
            else:
                print('CS_{}:'.format(i-1), end='')
            if cs:
                print('[', end='')
                for j in range(len(cs)-1):
                    print('EV_{},'.format(cs[j].id), end='')
                print('EV_{}]'.format(cs[-1].id), end='')
            else:
                print('None', end='')
                
            link = link_list[i]
            for link_part in link:
                if link_part:
                    print('[', end='')
                    for j in range(len(link_part)-1):
                        print('EV_{},'.format(link_part[j].id), end='')
                    print('EV_{}]'.format(link_part[-1].id), end='')
                else:
                    print('-', end='')
        
        cs = cs_list[-1]
        print('D:', end='')
        if cs:
            print('[', end='')
            for j in range(len(cs)-1):
                print('EV_{},'.format(cs[j].id), end='')
            print('EV_{}]'.format(cs[-1].id))
        else:
            print('None')

        for agent in self.agents:
            print('EV_{}:'.format(agent.id))
            print(
                '\t SOC:{:.2f}%   Pos:{}   Reward:{:.2f}   Using time:{:.2f}h   Charing times:{}   Charing SOC:{:.2f}%'
                .format(agent.SOC, agent.current_pos, agent.total_reward, agent.total_used_time, agent.charging_ts, agent.SOC_charged)
                )
            print('\t Action_list: ', agent.action_memory)
        for i, cs in enumerate(self.cs_charger_waiting_time):
            print('CS_{}: '.format(i), end='')
            for charger in cs:
                print('{:.2f}\t'.format(charger), end='')
            print('')
        print('Global_inf: ', self.cs_waiting_time)