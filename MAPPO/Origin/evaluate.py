"""
Created on  Mar 1 2021
@author: wangmeng
"""
import os
import torch
import argparse
from ppo import PPOAgent
from buffer import RolloutBuffer
import gymnasium as gym
# import gym

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
from env.EV_Sce_Env import EV_Sce_Env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="LS1_2P")
    parser.add_argument("--filename", type=str, default="train")
    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=4) # 环境数
    parser.add_argument("--num_update", type=int, default=1500) # 最大更新轮次
    # parser.add_argument("--shared_arch", type=int, nargs='*', default=64)
    # parser.add_argument("--policy_arch", type=int, nargs="*", default=None)
    # parser.add_argument("--value_arch", type=int, nargs="*", default=None)

    # 共享网络层
    parser.add_argument("--shared_arch", type=list, default=[])  # [32, 32]
    parser.add_argument("--policy_arch", type=list, default=[32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32])

    parser.add_argument("--lr", type=float, default=2.5e-4) # 学习率
    parser.add_argument("--gamma", type=float, default=0.95) # 折减因子
    parser.add_argument("--gae_lambda", type=float, default=0.95) # ？？？
    parser.add_argument("--k_epoch", type=int, default=15) # 策略更新次数
    parser.add_argument("--eps_clip", type=float, default=0.2) # 裁剪参数
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01) # 熵正则
    parser.add_argument("--single_batch_size", type=int, default=60) # 单个buffer数据量
    parser.add_argument("--num_mini_batch", type=int, default=1) # 小batcch数量
    arguments = parser.parse_args()
    return arguments

def train_policy(env, agents, writer, args, mode, agent_num):
    default_action = np.zeros(agent_num)
        
    agents_total_reward = [0 for _ in range(agent_num)]
    env.reset()
    obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(default_action)

    while 1:
        action_n = np.array([-1 for i in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                with torch.no_grad():
                    action, log_prob = agents[agent_i].select_best_action(obs_n[agent_i])
                    if not args.ctde:
                        share_obs = obs_n[agent_i].copy()
                    action_n[agent_i] = action[0]
                    
        # last_info = info
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
        for i, agent_i in enumerate(activate_agent_i):
            if act_n[agent_i] == -1:
                pass
            else:
                if not args.ctde:
                    share_obs = obs_n[agent_i].copy()
                agents_total_reward[agent_i] += reward_n[agent_i][0]
        ########### 若无启动的智能体，说明环境运行结束 ###########
        if env.agents_active == []: 
            break
        
    total_reward = 0
    for i in range(agent_num):
        ep_running_reward = agents_total_reward[i]
        total_reward += ep_running_reward
        
    
    dir = 'output/{}_{}_{}'.format(args.sce_name, args.filename, mode)
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
    ############## Hyperparameters ##############
    args = parse_args()
    if args.randomize:
        seed = args.seed
        print("Random Seed: {}".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = EV_Sce_Env(args.sce_name, seed=args.seed)
    action_list = env.action_list # 可选动作列表
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs # 充电站数量
    
    args.train_times = int(args.single_batch_size // num_cs) # 单个环境运行次数
    args.batch_size = int(args.single_batch_size * args.num_env) # 总batch大小
    args.mini_batch_size = int(args.batch_size // args.num_mini_batch)
    args.train_steps = int(args.train_times * agent_num * (num_cs+1))
    args.max_steps = int(args.train_steps * args.num_update)
    
    state_dim = env.state_dim # type: ignore
    action_dim = env.action_dim # type: ignore
    state_shape = (state_dim, )
    mode = ''
    
    if args.ctde:
        print('Mode: MAPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'MAPPO'
        share_dim = env.share_dim
        share_shape = (share_dim, )
    else:
        print('Mode: IPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'IPPO'
        share_dim = env.state_dim
        share_shape = (state_dim, )
    #############################################
    
    writer = None
    # writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
    # writer.add_text("HyperParameters", 
    #                 "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    
    mappo = []
    for i in range(agent_num):
        buffer_list = [
            RolloutBuffer(
            num_env=1, steps=args.single_batch_size, 
            state_shape=state_shape, share_shape=share_shape, action_shape=(1, ), # type: ignore
            device=device)
            for _ in range(args.num_env)
        ]
        ppo = PPOAgent(
            state_dim=state_dim, share_dim=share_dim, action_dim=action_dim, action_list=action_list,
            shared_arch=args.shared_arch, policy_arch=args.policy_arch, value_arch=args.value_arch, 
            buffer=buffer_list, device=device, max_steps=args.max_steps, gamma=args.gamma, gae_lambda=args.gae_lambda, 
            k_epoch=args.k_epoch, lr=args.lr, eps_clip=args.eps_clip, grad_clip=args.max_grad_clip, 
            entropy_coef=args.entropy_coef, batch_size=args.batch_size, mini_batch_size=args.mini_batch_size)
        ppo.load("save/{}_{}_agent_{}_{}".format(args.sce_name, args.filename, i, mode))
        mappo.append(ppo)
    
    print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
    with torch.no_grad():
        train_policy(env, mappo, writer, args, mode, agent_num)
    # writer.close()


if __name__ == '__main__':
    main()
