'''
Author: CQZ
Date: 2024-11-20 15:49:53
Company: SEU
'''
import torch
import numpy as np
import pandas as pd
import os

def Evaluate(env, agents, args, mode, agent_num):
    default_caction = np.zeros(agent_num) # 默认动作
    default_raction = np.zeros(agent_num) # 默认动作
    default_action = (default_caction, default_raction)
        
    agents_total_reward = [0 for _ in range(agent_num)]
    env.reset()
    obs_n, obs_mask_n, \
        share_obs, \
            done_n, creward_n, rreward_n, cact_n, ract_n, \
                activate_agent_ci, activate_to_cact, \
                    activate_agent_ri, activate_to_ract \
                        = env.step(default_action)
    for i, agent_i in enumerate(activate_agent_ri):
        agents_total_reward[agent_i] += rreward_n[agent_i][0].copy()
    while 1:
        caction_n = np.array([-1 for _ in range(agent_num)])
        raction_n = np.array([-1 for _ in range(agent_num)])
        
        # 决策充电
        for i, agent_i in enumerate(activate_agent_ci):
            if activate_to_cact[i]:
                with torch.no_grad():
                    # Choose an action
                    if args.ps:
                        pass
                    else:
                        caction, clog_prob = agents[agent_i].select_best_caction(obs_n[agent_i])
                    caction_n[agent_i] = caction[0].copy()
        # 决策路径
        for i, agent_i in enumerate(activate_agent_ri):
            if activate_to_ract[i]:
                with torch.no_grad():
                    # Choose an action
                    if args.ps:
                        pass
                    else:
                        raction, rlog_prob = agents[agent_i].select_best_raction(
                            obs_n[agent_i], obs_mask_n[agent_i]
                        )
                    raction_n[agent_i] = raction[0].copy()
        
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract \
                            = env.step((caction_n, raction_n)) # (caction_n, raction_n)
        
        #* 将被激活的智能体当前状态作为上一次动作的结果保存
        for i, agent_i in enumerate(activate_agent_ri):
            # print("RR:", rreward_n)
            agents_total_reward[agent_i] += rreward_n[agent_i][0].copy()
        if env.agents_active == []: 
            break
        
    total_reward = 0
    for i in range(agent_num):
        total_reward += agents_total_reward[i]
        
    dir = 'output/{}_{}_{}'.format(args.sce_name, args.filename, mode)
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
            df_ev_c.loc[i, 'time'] = round(time_seq[i], 2)
            df_ev_c.loc[i, 'distance'] = ev.trip_memory[i]
            df_ev_c.loc[i, 'position'] = ev.pos_memory[i]
            df_ev_c.loc[i, 'state'] = ev.state_memory[i]
            df_ev_c.loc[i, 'act_mode'] = int(ev.activity_memory[i])
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
    
    print(
            'Total reward: {:.3f} \t Average reward: {:.3f}'.format(
                    total_reward, total_reward/agent_num
                )
        )
    print(total_reward)
    # env.render()
    env.close()

