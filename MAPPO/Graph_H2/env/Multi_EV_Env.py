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

def adj2eindex(adj, map_shape):
    edge_index = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] == 1:
                edge_index.append([i, j])
    edge_index = np.array(edge_index).T
    edge_num = map_shape[1]
    fill_dim = edge_num - edge_index.shape[1]
    filled = np.zeros([2, fill_dim], dtype=int)
    edge_index = np.concatenate([edge_index, filled], axis=1)
    return edge_index
    
def adj2rea(adj):
    p = adj.copy()
    tmp = adj.copy()
    for _ in range(adj.shape[0]-1):
        tmp = np.dot(tmp, adj)
        p += tmp
    p += np.eye(p.shape[0], dtype=int)
    rea = np.where(p >= 1, 1, 0)
    return rea

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
    def __init__(self, scenario, seed=None, ps=False):
        # 设定随机数
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        self.ps = ps
        # 仿真参数
        self.total_time = 0
        self.scenario = scenario
        self.frame = self.scenario.frame
        # 地图
        self.map_adj = np.array(self.scenario.map_adj)
        self.map_adj_index = self.map_adj * np.arange(1, self.map_adj.shape[0]+1)
        self.edge_index = np.array(self.scenario.edge_index)
        self.edge_attr = np.array(self.scenario.edge_attr)
        self.final_pos = self.map_adj.shape[0]-1
        self.map_rea = adj2rea(self.map_adj) # 可达矩阵
        self.map_rea_index = self.map_rea * np.arange(1, self.map_rea.shape[0]+1)
    
        self.cs_charger_waiting_time = copy.deepcopy(self.scenario.cs_charger_waiting_time) # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = copy.deepcopy(self.scenario.cs_charger_min_id) # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = copy.deepcopy(self.scenario.cs_waiting_time)
        self.num_cs = len(self.cs_charger_waiting_time)
        assert self.num_cs == self.map_adj.shape[0]
        
        # self.power = self.scenario.power.copy()
        # 智能体
        self.agents = self.scenario.agents.copy()
        self.agent_num = len(self.agents)
        self.agents_active = self.agents.copy()
        
        # 充电动作
        self.caction_list = self.scenario.caction_list
        # 充电动作空间
        self.caction_dim = len(self.caction_list)
        self.caction_space = [spaces.Discrete(len(self.caction_list)) for _ in range(self.agent_num)]  # type: ignore
        # 路径动作
        self.raction_list = self.scenario.raction_list
        # 路径动作空间
        self.raction_dim = len(self.raction_list)
        self.raction_space = [spaces.Discrete(len(self.raction_list)) for _ in range(self.agent_num)]  # type: ignore
        
        # 电量状态设置
        # 观测
        self.state_name = [ 
                'agent_SOC', 'exp_SOC', 
                'agent_usingtime', 'agent_charging_ts', 'agent_next_waiting', 'is_finish'
                ] + [0 for _ in range(self.num_cs)] # 位置编码
        self.state_dim = len(self.state_name) # 状态维度
        if self.ps:
            self.state_dim += self.agent_num # 状态维度
        # 全局
        self.share_name = ([
            'agent_SOC', 'exp_SOC', 
            'agent_usingtime', 'agent_charging_ts', 'is_finish'
            ] + [0 for _ in range(self.num_cs)]) * self.agent_num + self.cs_waiting_time
        self.share_dim = len(self.share_name)
        # if self.args.ps:
        #     self.share_dim += self.agent_num # 状态维度
        
        # 路网状态设置
        self.map_shape = self.edge_index.shape # edge_index尺寸
        # 观测
        # 充电站最低排队时间 + 编号
        self.obs_features_dim = 1 + 1
        # 全局
        # 充电站最低排队时间 + 所有智能体编号
        self.global_features_dim = len(self.cs_charger_waiting_time[0]) + self.agent_num
        
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
        self.edge_state_memory = []
        self.edge_dic = {}
        for i in range(self.edge_index.shape[1]):
            o = self.edge_index[0][i]
            d = self.edge_index[1][i]
            self.edge_dic[str(o)+"-"+str(d)] = []
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            
    # step  this is  env.step()
    def step(self, action_n: tuple):
        caction_n = action_n[0]
        raction_n = action_n[1]
        activate_agent_ci = [] # 记录即将充电的智能体id
        activate_to_cact = [] # 处于关键点的智能体是否可以做选择，到达终点的不能
        activate_agent_ri = [] # 记录正处于关键点的智能体id
        activate_to_ract = [] # 处于关键点的智能体是否可以做选择，到达终点的不能
        
        # 第一阶段：设置好智能体动作
        # print('第一阶段：设置好智能体动作')
        run_step = True
        for i, agent in enumerate(self.agents):
            if agent.is_routing:
                self.set_n_raction(agent, raction_n[i]) # 设置动作
            if agent.is_choosing:
                self.set_n_caction(agent, caction_n[i]) # 设置动作
                if caction_n[i] == 0:
                    agent.if_choose_route()
                    if agent.is_routing != 0:
                        run_step = False
                        activate_agent_ri.append(agent.id)
                        activate_to_ract.append(agent.is_routing)
    
        # 第二阶段：让环境运行至有智能体到达检查地点
        # print('第二阶段：让环境运行至有智能体到达检查地点')
        agent_to_remove = []
        while run_step and self.agents_active:
            self.total_time += self.frame
            for i, agent in enumerate(self.agents_active):

                agent.step(time=self.total_time) # 智能体运行
            
                if round(self.total_time, 2) >= agent.enter_time and not agent.is_active: # 到达EV进入时间，则启动
                    agent.activate()
                if agent.is_routing: # 如果在分叉点，记录并跳出
                    run_step = False
                    activate_agent_ri.append(agent.id)
                    activate_to_ract.append(agent.is_routing)

                if agent.is_choosing: # 如果在CS，记录并跳出
                    run_step = False
                    activate_agent_ci.append(agent.id)
                    activate_to_cact.append(True)
                if agent.stop_update: # 如果在终点，记录并跳出
                    run_step = False
                    activate_agent_ri.append(agent.id)
                    activate_to_ract.append(0)
                    activate_agent_ci.append(agent.id)
                    activate_to_cact.append(False)
                    agent_to_remove.append(agent)
            self.update_cs_info(step=True)
            self.cs_save_memory()
        for agent in agent_to_remove: # 将不再更新的智能体去掉
            self.agents_active.remove(agent)
        
        # 第三阶段：整理奖励和输出
        # 当智能体到达检查点后，其上一次状态选择的动作有了结果，因此输出上一次的动作和这一次的状态  
        # print('第三阶段：整理奖励和输出')
        obs_n = []
        obs_feature_n = []
        obs_mask_n = []
        cact_n = []
        ract_n = []
        creward_n = []
        rreward_n = []
        done_n = []
        for i, agent in enumerate(self.agents):
            
            obs, sub_cs_feature, choice_set_mask = self.get_obs(agent)
            obs_n.append(obs)
            obs_feature_n.append(sub_cs_feature)
            obs_mask_n.append(choice_set_mask)
            
            cact_n.append(self.get_act(agent, c=True))
            ract_n.append(self.get_act(agent, c=False))
            
            if i in activate_agent_ci:
                if activate_to_cact[activate_agent_ci.index(i)] == False:
                    creward_n.append(self.get_reward(agent, c=True, isfinal=True))
                else:
                    creward_n.append(self.get_reward(agent, c=True, isfinal=False))
            else:
                creward_n.append(self.get_reward(agent, c=True, isfinal=False))
            
            rreward_n.append(self.get_reward(agent, c=False))
            done_n.append(self.get_done(agent))
            # info = {'individual_reward': self.get_reward(agent)}
            # env_info = self._get_info(agent)
            # if 'fail' in env_info.keys():
            #     info['fail'] = env_info['fail']
            # info_n.append(info)
        share_obs, global_cs_feature = self.get_share_state()
        
        return obs_n, obs_feature_n, obs_mask_n, \
            share_obs, global_cs_feature, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, activate_agent_ri, activate_to_ract

    def update_cs_info(self, step=False):
        # 充电站时间更新
        for i, cs in enumerate(self.cs_charger_waiting_time): 
            if step:
                for j, charger in enumerate(cs):
                    if self.cs_charger_waiting_time[i][j] > 0:
                        self.cs_charger_waiting_time[i][j] -= self.frame
                        if round(self.cs_charger_waiting_time[i][j], 2) <= 0:
                            self.cs_charger_waiting_time[i][j] = 0
            self.cs_waiting_time[i] = min(self.cs_charger_waiting_time[i])
            self.cs_charger_min_id[i] = self.cs_charger_waiting_time[i].index(self.cs_waiting_time[i])
    
    def cs_save_memory(self):
        cs_waiting_cars = [[] for _ in range(self.num_cs)]
        cs_charging_cars = [[] for _ in range(self.num_cs)]
        cs_waiting_num_cars = [0 for _ in range(self.num_cs)]
        cs_charging_num_cars = [0 for _ in range(self.num_cs)]
        edge_cars = copy.deepcopy(self.edge_dic) # 边信息

        for i, agent in enumerate(self.agents):
            if agent.is_charging:
                pos = agent.target_pos
                if agent.waiting_time == 0:
                    cs_charging_cars[pos].append(agent.id)
                else:
                    cs_waiting_cars[pos].append(agent.id)
            else:
                pos = agent.current_position
                if pos in edge_cars.keys():
                    edge_cars[pos].append(agent.id)
                
        for i in range(self.num_cs):
            cs_waiting_num_cars[i] = len(cs_waiting_cars[i])
            cs_charging_num_cars[i] = len(cs_charging_cars[i])
            
        self.time_memory.append(self.total_time)
        self.cs_waiting_cars_num_memory.append(copy.deepcopy(cs_waiting_num_cars))
        self.cs_charging_cars_num_memory.append(copy.deepcopy(cs_charging_num_cars))
        self.cs_waiting_cars_memory.append(copy.deepcopy(cs_waiting_cars))
        self.cs_charging_cars_memory.append(copy.deepcopy(cs_charging_cars))
        self.cs_waiting_time_memory.append(copy.deepcopy(self.cs_waiting_time))
        self.edge_state_memory.append(copy.deepcopy(edge_cars))
        
    def reset(self):
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
        self.edge_state_memory = []
                    
    def get_obs(self, agent: EV_Agent):
        # 智能体观测量，即状态
        # 当前SOC、位置、当前检查点等待时间
        agent_SOC = agent.SOC
        agent_exp_SOC = agent.SOC_exp
        agent_usingtime = agent.total_used_time
        agent_charging_ts = agent.charging_ts
        agent_complete_trip = agent.finish_trip
        agent_tpos = agent.target_pos
        if agent_tpos == self.final_pos:
            agent_next_waiting = 0
        else:
            agent_next_waiting = self.cs_waiting_time[agent_tpos]
            
        onehot_pos = [0 for _ in range(self.num_cs)]
        onehot_pos[agent_tpos] = 1
        cobs = [agent_SOC, agent_exp_SOC, agent_usingtime, agent_charging_ts, agent_next_waiting, agent_complete_trip] + onehot_pos
        
        # 计算当前能够到达的CS，进而获得还能去的CS的子图
        # reachable_cs = self.map_rea_index[agent_tpos]
        # reachable_cs = reachable_cs[reachable_cs != 0] - 1 # 节点编号
        # reachable_cs = reachable_cs[reachable_cs != self.final_pos] - 1 # cs编号
        # sub_map_adj = np.zeros_like(self.map_adj)
        # sub_map_adj[reachable_cs] = self.map_adj[reachable_cs]
        
        sub_cs_feature = np.array([self.cs_waiting_time, onehot_pos]).T
        # sub_edge_index = adj2eindex(sub_map_adj, map_shape=self.map_shape)
        choice_set_mask = agent.get_choice_set_mask()
        
        if self.ps:
            one_hot = [0] * self.agent_num
            one_hot[agent.id] = 1
            cobs =  cobs + one_hot
        
        return np.array(cobs), sub_cs_feature, choice_set_mask
    
    def get_act(self, agent: EV_Agent, c: bool):
        # 智能体上一个动作
        if c:
            agent_act_memory = agent.caction_memory.copy()
        else:
            agent_act_memory = agent.raction_memory.copy()
        agent_last_action = -1
        if len(agent_act_memory) > 0:
            agent_last_action = agent_act_memory[-1]
        return np.array([agent_last_action])
    
    def get_reward(self, agent: EV_Agent, c: bool, isfinal: bool = False):
        # 智能体当前奖励
        if c:
            agent_reward = agent.c_reward
            if isfinal:
                agent_reward += agent.get_penalty()
        else:
            agent_reward = agent.r_reward
        return np.array([agent_reward])
        
    def get_done(self, agent: EV_Agent):
        # 智能体当前奖励
        agent_done = agent.is_done
        return np.array([agent_done])

    def set_n_caction(self, agent: EV_Agent, caction):
        if caction == 0:
            agent.set_caction(caction) # 将选择和充电时间输入给智能体
        else:
            postion = agent.target_pos # EV当前所在CS
            agent_SOC = agent.SOC
            waiting_time = self.cs_waiting_time[postion]
            act_SOC = self.caction_list[caction]
            charging_time = get_charging_time(cur_SOC=agent_SOC, final_SOC=act_SOC)

            agent.set_caction(caction, waiting_time, charging_time) # 将选择和充电时间输入给智能体
            
            min_charger_id = self.cs_charger_min_id[postion]
            self.cs_charger_waiting_time[postion][min_charger_id] += charging_time
            
            self.cs_waiting_time[postion] = min(self.cs_charger_waiting_time[postion])
            self.cs_charger_min_id[postion] = self.cs_charger_waiting_time[postion].index(self.cs_waiting_time[postion])
            
    def set_n_raction(self, agent: EV_Agent, raction):
        agent.set_raction(raction, reset_record=True) 
    
    def get_policy(self):
        current_policy = []
        for agent in self.agents:
            current_policy.append(agent.action_memory)
        return current_policy
    
    def get_share_state(self):
        share_state = []
        onehot_pos = np.zeros([self.num_cs, self.agent_num])
        for i, agent in enumerate(self.agents):
            pos = agent.target_pos
            share_state.append(agent.SOC)
            share_state.append(agent.SOC_exp)
            share_state.append(agent.total_used_time)
            share_state.append(agent.charging_ts)
            share_state.append(agent.finish_trip)
            onehot_pos_ = [0 for _ in range(self.num_cs)]
            onehot_pos_[pos] = 1
            share_state.extend(onehot_pos_.copy())
            onehot_pos[pos][i] = 1
        share_state = np.array(share_state.copy())
        share_state = np.concatenate([share_state, self.cs_waiting_time])
        
        cs_feature = np.array(self.cs_charger_waiting_time)
        cs_feature = np.concatenate([cs_feature, onehot_pos], axis=1)
        
        return share_state, cs_feature

    def render(self):
        print('Time: {} h'.format(round(self.total_time, 2)))

        for agent in self.agents:
            if agent.is_active:
                print('EV_{}:'.format(agent.id))
                print(
                    '\t SOC:{:.3f}%   Pos:{}   Reward:{:.5f}   Using time:{:.3f}h   Charing times:{}   Charing SOC:{:.3f}%'
                    .format(agent.SOC, agent.current_position, agent.total_reward, agent.total_used_time, agent.charging_ts, agent.SOC_charged)
                    )
                print('\t Action_list: ', agent.action_memory)
                print('\t Action_type: ', agent.action_type)
                print('\t Route: ', agent.total_route)
        for i, cs in enumerate(self.cs_charger_waiting_time):
            print('CS_{}: '.format(i), end='')
            for charger in cs:
                print('{:.3f}\t'.format(charger), end='')
            print('')
        print('Global_inf: ', self.cs_waiting_time)