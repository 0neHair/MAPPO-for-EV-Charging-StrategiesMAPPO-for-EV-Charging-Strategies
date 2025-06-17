import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
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

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_base(gym.Env):
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
    
        self.cs_charger_waiting_time = np.array(self.scenario.cs_charger_waiting_time) # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = np.array(self.scenario.cs_charger_min_id) # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = np.array(self.scenario.cs_waiting_time)
        self.cs_charger_waiting_time_arange = np.arange(self.cs_charger_waiting_time.shape[0])
        self.num_cs = self.cs_charger_waiting_time.shape[0]
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
                'agent_next_waiting', 'agent_SOC', 'exp_SOC', 
                'agent_charging_ts',  'is_finish'
                ] + [0 for _ in range(self.num_cs)] # 位置编码
        self.state_dim = len(self.state_name) # 状态维度
        if self.ps:
            self.state_dim += self.agent_num # 状态维度
        self.one_hot_id = []
        for i in range(self.agent_num):
            one_hot_id = np.zeros((1, self.agent_num))
            one_hot_id[0][i] = 1
            self.one_hot_id.append(one_hot_id)
        
        # 全局
        self.share_name = ([
            'agent_SOC', 'exp_SOC', 
            'agent_charging_ts', 'is_finish'
            ] + [0 for _ in range(self.num_cs)]) * self.agent_num + [0] * self.cs_waiting_time.shape[0]
        self.share_dim = len(self.share_name)
        # if self.args.ps:
        #     self.share_dim += self.agent_num # 状态维度
        
        # 路网状态设置
        self.map_shape = self.edge_index.shape # edge_index尺寸

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
        
        # 观测
        # 充电站最低排队时间 + 编号
        self.obs_features_dim = 1 + 1
        # 全局
        # 充电站最低排队时间 + 所有智能体编号
        self.global_features_dim = self.cs_charger_waiting_time.shape[1] + self.agent_num
    
        # tmp 变量
        self.obs_n = np.zeros((self.agent_num, self.state_dim), dtype=np.float32)
        self.obs_mask_n = np.zeros((self.agent_num, self.raction_dim), dtype=np.float32)
        self.share_obs = np.zeros((self.share_dim), dtype=np.float32)
        self.done_n = np.zeros((self.agent_num, 1), dtype=np.float32)
        self.creward_n = np.zeros((self.agent_num, 1), dtype=np.float32)
        self.rreward_n = np.zeros((self.agent_num, 1), dtype=np.float32)
        self.cact_n = np.zeros((self.agent_num), dtype=np.float32)
        self.ract_n = np.zeros((self.agent_num), dtype=np.float32)
        
        self.obs_feature_n = np.zeros((self.agent_num, self.num_cs, self.obs_features_dim), dtype=np.float32)
        self.global_cs_feature = np.zeros((self.num_cs, self.global_features_dim), dtype=np.float32)
        self.onehot_pos = np.zeros([self.num_cs, self.agent_num])
        
        self.activate_agent_ci = []
        self.activate_to_cact = []
        self.activate_agent_ri = []
        self.activate_to_ract = []
        self.agent_to_remove = []
        self.run_step = True
        self.onehot_pos_ = np.zeros((self.num_cs))
        
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

        self.edge_list = []
        for l in range(len(self.edge_attr)):
            self.edge_list.append((
                self.edge_index[0][l],
                self.edge_index[1][l],
                {'length': self.edge_attr[l][0]}
            ))
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edge_list)
        
        for agent in self.agents:
            agent.route = self.shortest_way()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    
    def agents_step(self, train=True):
        self.agent_to_remove = []
        while self.run_step and self.agents_active:
            self.total_time += self.frame
            for i, agent in enumerate(self.agents_active):
                agent.step(time=self.total_time, train=train) # 智能体运行
                if self.total_time >= agent.enter_time and not agent.is_active: # 到达EV进入时间，则启动
                    agent.activate()
                if agent.is_routing: # 如果在分叉点，记录并跳出
                    self.run_step = False
                    self.activate_agent_ri.append(agent.id)
                    self.activate_to_ract.append(agent.is_routing)
                if agent.is_choosing: # 如果在CS，记录并跳出
                    self.run_step = False
                    self.activate_agent_ci.append(agent.id)
                    self.activate_to_cact.append(True)
                if agent.stop_update: # 如果在终点，记录并跳出
                    self.run_step = False
                    self.activate_agent_ri.append(agent.id)
                    self.activate_to_ract.append(0)
                    self.activate_agent_ci.append(agent.id)
                    self.activate_to_cact.append(False)
                    self.agent_to_remove.append(agent)
            self.update_cs_info(step=True)
            if not train:
                self.cs_save_memory()
        for agent in self.agent_to_remove: # 将不再更新的智能体去掉
            self.agents_active.remove(agent)

    def get_local_obs(self, i, agent):
        self.onehot_pos_ *= 0
        self.onehot_pos_[agent.target_pos] = 1        
        self.obs_n[i][0] = 0 if agent.target_pos == self.final_pos else self.cs_waiting_time[agent.target_pos] / 100
        self.obs_n[i][1] = agent.SOC
        self.obs_n[i][2] = agent.SOC_exp
        self.obs_n[i][3] = agent.charging_ts
        self.obs_n[i][4] = agent.finish_trip
        if self.ps:
            self.obs_n[i][-(self.num_cs+self.agent_num):-self.agent_num] = self.onehot_pos_
            self.obs_n[i][-self.agent_num:] = self.one_hot_id[agent.id] 
        else:
            self.obs_n[i][-self.num_cs:] = self.onehot_pos_
        self.obs_mask_n[i] = agent.get_choice_set_mask()

    def get_share_obs(self):
        if self.ps:
            self.share_obs[:-self.num_cs] = np.reshape(self.obs_n[:, 1:-self.agent_num], (-1, ))
        else:
            self.share_obs[:-self.num_cs] = np.reshape(self.obs_n[:, 1:], (-1, ))
        self.share_obs[-self.num_cs:] = self.cs_waiting_time / 100

    def agents_arrange(self):
        for i, agent in enumerate(self.agents):
            # Local observation
            self.get_local_obs(i, agent)

            # Reward
            if i in self.activate_agent_ci:
                if self.activate_to_cact[self.activate_agent_ci.index(i)] == False:
                    self.creward_n[i] = self.get_reward(agent, c=True, isfinal=True)
                else:
                    self.creward_n[i] = self.get_reward(agent, c=True, isfinal=False)
            else:
                self.creward_n[i] = self.get_reward(agent, c=True, isfinal=False)
            self.rreward_n[i] = self.get_reward(agent, c=False)
            self.done_n[i] = self.get_done(agent)
        
        # Shared observation
        self.get_share_obs()
        
    # step  this is  env.step()
    def step(self, action_n: tuple, train=True):
        caction_n = action_n[0]
        raction_n = action_n[1]
        self.activate_agent_ci = [] # 记录即将充电的智能体id
        self.activate_to_cact = [] # 处于关键点的智能体是否可以做选择，到达终点的不能
        self.activate_agent_ri = [] # 记录正处于关键点的智能体id
        self.activate_to_ract = [] # 处于关键点的智能体是否可以做选择，到达终点的不能
       
        # Phase 1: Set agents' actions
        self.run_step = True
        for i, agent in enumerate(self.agents):
            if agent.is_routing:
                self.set_n_raction(agent, raction_n[i]) # 设置动作
            if agent.is_choosing:
                caction_n[i] = self.set_n_caction(agent, caction_n[i]) # 设置动作
                if caction_n[i] == 0:
                    agent.if_choose_route()
                    if agent.is_routing != 0:
                        self.run_step = False
                        self.activate_agent_ri.append(agent.id)
                        self.activate_to_ract.append(agent.is_routing)
    
        # Phase 2: Step until agents need to choose actions
        self.agents_step(train=train)
            
        # Phase 3: Organize rewards and outputs
        self.agents_arrange()

        return self.obs_n, self.obs_feature_n, self.obs_mask_n, \
            self.share_obs, self.global_cs_feature, \
                self.done_n, self.creward_n, self.rreward_n, self.cact_n, self.ract_n, \
                    self.activate_agent_ci, self.activate_to_cact, \
                        self.activate_agent_ri, self.activate_to_ract

    def update_cs_info(self, step=False):
        # Update CS waitng time
        self.cs_charger_waiting_time -= self.frame
        self.cs_charger_waiting_time = np.maximum(self.cs_charger_waiting_time, 0)
        self.cs_charger_min_id = np.argmin(self.cs_charger_waiting_time, axis=1)
        self.cs_waiting_time = self.cs_charger_waiting_time[
                self.cs_charger_waiting_time_arange, 
                self.cs_charger_min_id
            ]
          
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
            
        self.time_memory.append(self.total_time / 100)
        self.cs_waiting_cars_num_memory.append(copy.deepcopy(cs_waiting_num_cars))
        self.cs_charging_cars_num_memory.append(copy.deepcopy(cs_charging_num_cars))
        self.cs_waiting_cars_memory.append(copy.deepcopy(cs_waiting_cars))
        self.cs_charging_cars_memory.append(copy.deepcopy(cs_charging_cars))
        self.cs_waiting_time_memory.append(copy.deepcopy(self.cs_waiting_time))
        self.edge_state_memory.append(copy.deepcopy(edge_cars))
        
    def reset(self):
        self.total_time = 0
        # 地图
        self.cs_charger_waiting_time = np.array(self.scenario.cs_charger_waiting_time) # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = np.array(self.scenario.cs_charger_min_id) # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = np.array(self.scenario.cs_waiting_time)# self.power = self.scenario.power.copy()
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
    
    def get_reward(self, agent: EV_Agent, c: bool, isfinal: bool = False):
        # Current reward
        if c:
            if isfinal:
                return agent.c_reward + agent.get_penalty()
            else:
                return agent.c_reward
        else:
            return agent.r_reward
        
    def get_done(self, agent: EV_Agent):
        return agent.is_done

    def set_n_caction(self, agent: EV_Agent, caction):
        if caction == 0:
            agent.set_caction(caction) # 将选择和充电时间输入给智能体
        else:
            waiting_time = self.cs_waiting_time[agent.target_pos]
            charging_time = int(self.get_charging_time(cur_SOC=agent.SOC, final_SOC=self.caction_list[caction]) * 100)

            agent.set_caction(caction, waiting_time, charging_time) # 将选择和充电时间输入给智能体
            
            self.cs_charger_waiting_time[agent.target_pos][self.cs_charger_min_id[agent.target_pos]] += charging_time
            self.cs_charger_min_id = np.argmin(self.cs_charger_waiting_time, axis=1)
            self.cs_waiting_time = self.cs_charger_waiting_time[
                    self.cs_charger_waiting_time_arange, 
                    self.cs_charger_min_id
                ]
        return caction
        
    def set_n_raction(self, agent: EV_Agent, raction):
        agent.set_raction(raction, reset_record=True)
        
    def weight_func(self, u, v, d):
        node_u_wt = self.cs_waiting_time[u] / 100
        edge_wt = d.get("length", 1)
        return node_u_wt + edge_wt / 100
    
    def shortest_way(self):
        node_path = nx.dijkstra_path(self.graph, 0, self.final_pos, weight=self.weight_func) # type: ignore
        path_dic = {}
        for p in range(len(node_path)-1): # type: ignore
            path_dic[node_path[p]] = node_path[p+1] # type: ignore
        return path_dic
    
    def render(self):
        print('Time: {} h'.format(self.total_time / 100))

        for agent in self.agents:
            if agent.is_active:
                print('EV_{}:'.format(agent.id))
                print(
                    '\t SOC:{:.3f}%   Pos:{}   Reward:{:.5f}   Using time:{:.3f}h   Charing times:{}   Charing SOC:{:.3f}%'
                    .format(agent.SOC, agent.current_position, agent.total_reward, agent.total_used_time / 100, agent.charging_ts, agent.SOC_charged)
                    )
                print('\t Action_list: ', agent.action_memory)
                print('\t Action_type: ', agent.action_type)
                print('\t Route: ', agent.total_route)
        for i in range(self.cs_charger_waiting_time.shape[0]):
            print('CS_{}: '.format(i), end='')
            for j in range(self.cs_charger_waiting_time.shape[1]):
                print('{:.3f}\t'.format(self.cs_charger_waiting_time[i][j]/100), end='')
            print('')
        print('Global_inf: ', self.cs_waiting_time)

    def charging_function(self, SOC):
        if SOC <= 0.8:
            return SOC / 0.4
        elif SOC <= 0.85:
            return 2 + (SOC - 0.8) / 0.25
        else:
            return 2.2 + (SOC -0.85) / 0.1875

    def get_charging_time(self, cur_SOC, final_SOC):
        return self.charging_function(final_SOC) - self.charging_function(cur_SOC)