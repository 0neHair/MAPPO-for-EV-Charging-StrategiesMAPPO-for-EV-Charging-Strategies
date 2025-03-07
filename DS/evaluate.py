"""
Created on  Mar 1 2021
@author: wangmeng
"""
import os
import argparse

import numpy as np
import pandas as pd
from env.EV_Sce_Env import EV_Sce_Env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="LS1_2P")
    parser.add_argument("--filename", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    return arguments

def policy_fix(env, actions, exp_charged_SOC):
    route = env.route
    new_actions = []
    for i, agent in enumerate(env.agents):
        agent_action = actions[i]
        exp_charged = exp_charged_SOC[i]
        new_agent_action = []
        for a in agent_action:
            if a < 0.1:
                new_agent_action.append(0)
            elif a > 1:
                new_agent_action.append(1)
            else:
                new_agent_action.append(a)
        agent_action = new_agent_action.copy()
        
        agent_SOC = agent.SOC 
        agent_SOC = agent_SOC - route[0]*agent.consume/agent.E_max
        new_agent_action = []
        for j, act in enumerate(agent_action):
            temp_soc = agent_SOC + act
            if temp_soc > 1.0:
                real_act = 1.0 - agent_SOC
                agent_SOC = 1.0
                agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                new_agent_action.append(real_act)
            else:
                temp_soc = temp_soc - route[j+1]*agent.consume/agent.E_max
                if temp_soc < 0:
                    if exp_charged >= 1:
                        real_act = 1.0 - agent_SOC
                        agent_SOC = 1.0
                        agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                        new_agent_action.append(real_act)
                        exp_charged -= real_act
                    else:
                        agent_SOC += exp_charged+0.01
                        agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                        new_agent_action.append(exp_charged+0.01)                
                else:
                    agent_SOC= agent_SOC + act
                    agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                    new_agent_action.append(act)

        new_actions.append(new_agent_action)
    return np.array(new_actions)

def train_policy(env, actions, args, agent_num, exp_charged_SOC):
    actions = policy_fix(env, actions, exp_charged_SOC)
    
    env.reset()
    agent_num = env.agent_num
    action_i = [0 for i in range(agent_num)]
    action_n = np.array([-1.0 for i in range(agent_num)])
    obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
    while True:
        action_n = np.array([-1.0 for i in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                action_n[agent_i] = actions[agent_i][action_i[agent_i]]
                action_i[agent_i] += 1
                
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
        if env.agents_active == []:
            break
    
    total_reward = 0
    for agent in env.agents:
        total_reward += agent.total_reward
        
    dir = 'DS/output/{}_{}'.format(args.sce_name, args.filename)
    if not os.path.exists(dir):
        os.mkdir(dir)
    # CS
    columns = ['time']
    for i in range(env.num_cs):
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
    columns = ['time', 'distance', 'position', 'state', 'act_mode', 'SOC', 'action']
    if not os.path.exists(dir + '/EV'):
        os.mkdir(dir + '/EV')
    for j, ev in enumerate(env.agents):
        df_ev = pd.DataFrame(columns=columns)
        time_seq = ev.time_memory
        for i in range(len(time_seq)):
            df_ev.loc[i, 'time'] = round(time_seq[i], 2)
            df_ev.loc[i, 'distance'] = ev.trip_memory[i]
            df_ev.loc[i, 'position'] = ev.pos_memory[i]
            df_ev.loc[i, 'state'] = ev.state_memory[i]
            df_ev.loc[i, 'act_mode'] = int(ev.activity_memory[i])
            df_ev.loc[i, 'SOC'] = ev.SOC_memory[i]
            df_ev.loc[i, 'action'] = ev.action_choose_memory[i]
        df_ev.to_csv(dir + '/EV/EV{}.csv'.format(ev.id), index=False)

        df_ev_g.loc[j, 'EV'] = ev.id
        df_ev_g.loc[j, 'Waiting_time'] = ev.total_waiting
        df_ev_g.loc[j, 'Charging_time'] = ev.total_charging
        df_ev_g.loc[j, 'Total'] = ev.total_used_time
        df_ev_g.loc[j, 'Reward'] = ev.total_reward
    df_ev_g.to_csv(dir+'/EV_g.csv', index=False)
        
    print(total_reward)
    # env.render()
    env.close()

def main():
    args = parse_args()
    env = EV_Sce_Env(args.sce_name, seed=args.seed)
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs
    
    # pol = pd.read_csv('policy.csv')
    # colums = []
    # for i in range(num_cs):
    #     colums.append('CS_{}'.format(i))
    # actions = np.array(pol[colums])
    actions = np.zeros((agent_num, num_cs))
    
    exp_charged_SOC = []
    for i in range(agent_num):
        exp_charged_SOC.append(env.agents[i].ideal_charging_SOC)
    
    train_policy(env, actions, args, agent_num, exp_charged_SOC)
    # writer.close()

if __name__ == '__main__':
    main()
