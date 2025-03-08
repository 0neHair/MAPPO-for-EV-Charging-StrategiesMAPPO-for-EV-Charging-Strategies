'''
Author: CQZ
Date: 2024-12-02 15:38:02
Company: SEU
'''
import numpy as np
import pandas as pd
import os
import torch
from torch.distributions import Categorical

from env.EV_Sce_Env import EV_Sce_Env

def choice_route(prob, mask):
    prob = torch.tensor(prob, dtype=torch.float32)
    mask = torch.LongTensor(mask)
    masked_prob = prob.masked_fill(~mask.bool(), 0)
    dist = Categorical(probs=masked_prob)
    action = dist.probs.argmax() # type: ignore
    return action.numpy().flatten()

def run(cact_pi_seq, ract_pi_seq, env, agent_num, args):
    default_caction = np.zeros(agent_num) # 默认动作
    default_raction = np.zeros(agent_num) # 默认动作
    default_action = (default_caction, default_raction)
    
    # caction_seq = [[0, 13], [0, 13]]
    # ract_pi_seq = [
    #     [
    #         [0, 1, 0, 0, 0],
    #         [0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0]
    #     ],
    #     [
    #         [0, 1, 0, 0, 0],
    #         [0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0]
    #     ]
    # ]
        
    agents_total_reward = [0 for _ in range(agent_num)]
    env.reset()
    obs_n, obs_mask_n, share_obs, done_n, \
        reward_n, cact_n, ract_n, \
            activate_agent_i, activate_to_act \
                    = env.step(default_action)
    for i, agent_i in enumerate(activate_agent_i):
        agents_total_reward[agent_i] += reward_n[agent_i][0].copy()
    while 1:
        caction_n = np.array([-1 for _ in range(agent_num)])
        raction_n = np.array([-1 for _ in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                pos = int(obs_n[agent_i][-1])
                caction_n[agent_i] = cact_pi_seq[agent_i][pos]
                ract = choice_route(ract_pi_seq[agent_i][pos], obs_mask_n[agent_i])
                raction_n[agent_i] = ract[0]
                if env.caction_list[caction_n[agent_i]] <= obs_n[agent_i][0]:
                    caction_n[agent_i] = 0

        obs_n, obs_mask_n, share_obs, done_n, \
            reward_n, cact_n, ract_n, \
                activate_agent_i, activate_to_act \
                    = env.step((caction_n, raction_n)) # (caction_n, raction_n)
        
        #* 将被激活的智能体当前状态作为上一次动作的结果保存
        for i, agent_i in enumerate(activate_agent_i):
            agents_total_reward[agent_i] += reward_n[agent_i][0].copy()
        if env.agents_active == []: 
            break
    # print(agents_total_reward)
    total_reward = 0
    for i in range(agent_num):
        total_reward += agents_total_reward[i]

    if args.train == False:
        dir = 'output/{}_{}_GA'.format(args.sce_name, args.filename)
        if not os.path.exists(dir):
            os.mkdir(dir)
        # Map
        columns = ['time']
        for edge in env.edge_dic.keys():
            columns.append(edge)
        time_seq = env.time_memory
        df_route = pd.DataFrame(columns=columns)
        for i in range(len(time_seq)):
            df_route.loc[i, 'time'] = round(time_seq[i], 2)
            for edge in env.edge_dic.keys():
                cars_list = env.edge_state_memory[i][edge]
                if cars_list:
                    cars_list_str = str(cars_list[0])
                    for car_id in range(1, len(cars_list)):
                        cars_list_str += '-'+str(cars_list[car_id])
                else:
                    cars_list_str = ""
                df_route.loc[i, edge] = cars_list_str
        df_route.to_csv(dir+'/Map.csv', index=False)
        # CS
        columns = ['time']
        for i in range(1, env.num_cs-1):
            cs = 'CS{}'.format(i)
            columns.append(cs+'_waiting_num')
            columns.append(cs+'_charging_num')
            columns.append(cs+'_waiting_time')
        time_seq = env.time_memory
        df_cs = pd.DataFrame(columns=columns)
        for i in range(len(time_seq)):
            df_cs.loc[i, 'time'] = round(time_seq[i], 2)
            for j in range(env.num_cs):
                cs = 'CS{}'.format(j)
                df_cs.loc[i, cs+'_waiting_num'] = env.cs_waiting_cars_num_memory[i][j]
                df_cs.loc[i, cs+'_charging_num'] = env.cs_charging_cars_num_memory[i][j]
                df_cs.loc[i, cs+'_waiting_time'] = env.cs_waiting_time_memory[i][j]
        df_cs.to_csv(dir+'/CS.csv', index=False)
        # EV
        df_ev_g = pd.DataFrame(columns=['EV', 'Waiting_time', 'Charging_time', 'Total'])
        ccolumns = ['time', 'distance', 'position', 'state', 'cact_mode', 'SOC', 'action']
        rcolumns = ['num', 'edge']
        if not os.path.exists(dir + '/EV'):
            os.mkdir(dir + '/EV')
        for j, ev in enumerate(env.agents):
            df_ev_c = pd.DataFrame(columns=ccolumns)
            time_seq = ev.time_memory
            for i in range(len(time_seq)):
                if ev.action_choose_memory[i] != -1:
                    assert 'P' in ev.pos_memory[i], "BUG exists"
                    df_ev_c.loc[i, 'time'] = round(time_seq[i], 2)
                    df_ev_c.loc[i, 'distance'] = ev.trip_memory[i]
                    df_ev_c.loc[i, 'position'] = 'P' + str(int(ev.pos_memory[i][1:])-1)
                    df_ev_c.loc[i, 'state'] = ev.state_memory[i]
                    df_ev_c.loc[i, 'cact_mode'] = int(ev.activity_memory[i])
                    df_ev_c.loc[i, 'SOC'] = env.caction_list[ev.action_choose_memory[i]]
                    df_ev_c.loc[i, 'action'] = ev.action_choose_memory[i]
                else:
                    df_ev_c.loc[i, 'time'] = round(time_seq[i], 2)
                    df_ev_c.loc[i, 'distance'] = ev.trip_memory[i]
                    df_ev_c.loc[i, 'position'] = ev.pos_memory[i]
                    df_ev_c.loc[i, 'state'] = ev.state_memory[i]
                    df_ev_c.loc[i, 'cact_mode'] = int(ev.activity_memory[i])
                    df_ev_c.loc[i, 'SOC'] = ev.SOC_memory[i]
                    df_ev_c.loc[i, 'action'] = ev.action_choose_memory[i]
            df_ev_c.to_csv(dir + '/EV/EV{}.csv'.format(ev.id), index=False)
            df_ev_r = pd.DataFrame(columns=rcolumns)
            edge_i = 0
            for edge in ev.total_route:
                df_ev_r.loc[edge_i, 'num'] = edge_i
                df_ev_r.loc[edge_i, 'edge'] = edge
                edge_i += 1
            df_ev_r.to_csv(dir + '/EV/EV{}_route.csv'.format(ev.id), index=False)

            df_ev_g.loc[j, 'EV'] = ev.id
            df_ev_g.loc[j, 'Waiting_time'] = ev.total_waiting
            df_ev_g.loc[j, 'Charging_time'] = ev.total_charging
            df_ev_g.loc[j, 'Total'] = ev.total_used_time
            df_ev_g.loc[j, 'Reward'] = ev.total_reward
        df_ev_g.to_csv(dir+'/EV_g.csv', index=False)
        # print(total_reward)
        # env.render()
        env.close()

    return total_reward

if __name__ == "__main__":
    env = EV_Sce_Env(sce_name="test_2", seed=0, ps=False)
    ve_matrix = np.zeros((env.map_adj.shape[1], env.edge_index.shape[1]))
    for e in range(env.edge_index.shape[1]):
        ve_matrix[env.edge_index[0][e], e] = -1
        ve_matrix[env.edge_index[1][e], e] = 1
    caction_list = env.caction_list # 可选充电动作列表
    cact_dim = ve_matrix.shape[0]
    cat_max = len(caction_list)
    ract_dim = ve_matrix.shape[1]
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs # 充电站数量
    
    # section_p = [[点个数] * 点个数] * agent_num 每个点选择下一个点的概率，可以mask
    cact_pi_seq = [
            [0, 0, 13, 0, 0], 
            [0, 0, 13, 0, 0]
        ]
    # raction_seq = [[2, 4], [2, 4]]
    ract_pi_seq = [
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]
    ]
    
    total_reward = run(cact_pi_seq, ract_pi_seq, env, agent_num, None)
    print(total_reward)