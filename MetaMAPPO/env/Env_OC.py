import numpy as np
import networkx as nx
from env.Env_Base import Env_base

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_OC(Env_base):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None, ps=False):
        super().__init__(scenario, seed, ps)

        self.edge_list = []
        for l in range(len(self.edge_attr)):
            self.edge_list.append((
                self.edge_index[0][l],
                self.edge_index[1][l],
                {'length': self.edge_attr[l][0]}
            ))
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edge_list)
        
    def agents_step(self, train=True):
        self.agent_to_remove = []
        while self.run_step and self.agents_active:
            self.total_time += self.frame
            for i, agent in enumerate(self.agents_active):
                agent.step(time=self.total_time, train=train) # 智能体运行
                if self.total_time >= agent.enter_time and not agent.is_active: # 到达EV进入时间，则启动
                    agent.route = self.shortest_way()
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
      
    def set_n_raction(self, agent, raction):
        agent.set_raction(agent.route[agent.target_pos], reset_record=True)
        