import torch
import argparse
import gymnasium as gym
# import gym

import numpy as np
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
import os
from env.EV_Sce_Env import EV_Sce_Env
from env_wrappers import SubprocVecEnv, DummyVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="HY_2")
    parser.add_argument("--filename", type=str, default="T1")
    parser.add_argument("--train", type=bool, default=False)

    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--expert", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=1) # 环境数
    parser.add_argument("--num_update", type=int, default=500) # 最大更新轮次
    parser.add_argument("--save_freq", type=int, default=50) # 保存频率

    parser.add_argument("--ps", type=bool, default=False) # parameter sharing

    parser.add_argument("--policy_arch", type=list, default=[32, 32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32, 32])

    parser.add_argument("--lr", type=float, default=2.5e-4) # 学习率
    parser.add_argument("--gamma", type=float, default=0.95) # 折减因子
    parser.add_argument("--gae_lambda", type=float, default=0.95) # ？？？
    parser.add_argument("--k_epoch", type=int, default=15) # 策略更新次数
    parser.add_argument("--eps_clip", type=float, default=0.1) # 裁剪参数
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01) # 熵正则
    parser.add_argument("--single_batch_size", type=int, default=60) # 单个buffer数据量
    parser.add_argument("--num_mini_batch", type=int, default=1) # 小batcch数量
    arguments = parser.parse_args()
    return arguments

def make_train_env(args, agent_num):
    def get_env_fn():
        def init_env():
            env = EV_Sce_Env(sce_name=args.sce_name, seed=args.seed, ps=args.ps)
            return env
        return init_env
    if args.num_env == 1:
        return DummyVecEnv([get_env_fn()], agent_num)
    else:
        return SubprocVecEnv([get_env_fn() for _ in range(args.num_env)], agent_num)

def main():
    ############## Hyperparameters ##############
    args = parse_args()
    if args.randomize:
        seed = args.seed
        print("Random Seed: {}".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = EV_Sce_Env(sce_name=args.sce_name, seed=args.seed, ps=args.ps)
    caction_list = env.caction_list # 可选充电动作列表
    raction_list = env.raction_list # 可选路径动作列表
    
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs # 充电站数量
    
    args.train_times = int(args.single_batch_size * 2 // num_cs) # 单个环境运行次数
    args.batch_size = int(args.single_batch_size * args.num_env) # 总batch大小
    # args.batch_size = int(args.single_batch_size * args.num_env * agent_num) # 总batch大小

    args.mini_batch_size = int(args.batch_size // args.num_mini_batch)
    # args.train_steps = int(args.train_times * agent_num * (num_cs+1))
    # args.max_steps = int(args.train_steps * args.num_update)
    
    graph = env.map_adj
    caction_dim = env.caction_dim # type: ignore
    raction_dim = env.raction_dim # type: ignore
    raction_mask_dim = env.raction_dim # type: ignore
    raction_mask_shape = (raction_mask_dim, )
    state_dim = env.state_dim # type: ignore
    state_shape = (state_dim, )
    edge_index = env.edge_index # 原版地图
    obs_features_dim = env.obs_features_dim # 观测地图特征维度
    obs_features_shape = (graph.shape[0], obs_features_dim) # 观测地图特征尺寸
    mode = ''
    
    if args.ctde:
        print('Mode: MAPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'MAPPO'
        share_dim = env.share_dim
        share_shape = (share_dim, )
        global_features_dim = env.global_features_dim # 全局地图特征维度
        global_features_shape = (graph.shape[0], global_features_dim) # 全局地图特征尺寸
    else:
        print('Mode: IPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'IPPO'
        share_dim = env.state_dim
        share_shape = (state_dim, )
        global_features_dim = env.obs_features_dim
        global_features_shape = (graph.shape[0], obs_features_dim)
    #############################################
    if args.train:
        envs = make_train_env(args, agent_num)
        writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
        writer.add_text("HyperParameters", 
                        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    else:
        envs = env

    mappo = []
    if args.ps:
        from ppo_ps import PPOAgent
        from buffer import RolloutBuffer
        from train_ps import Train
        from evaluate import Evaluate
        
        buffer_list = []
        for i in range(agent_num):
            buffer = RolloutBuffer(
                steps=args.single_batch_size, num_env=args.num_env,
                state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                edge_index=edge_index,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, raction_shape=(1, ), # type: ignore
                raction_mask_shape=raction_mask_shape,
                device=device
                )
            buffer_list.append(buffer)
        ppo = PPOAgent(
            state_dim=state_dim, share_dim=share_dim, 
            caction_dim=caction_dim, caction_list=caction_list,
            obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
            raction_dim=raction_dim, raction_list=raction_list,
            edge_index=edge_index, buffer_list=buffer_list, device=device, args=args
            )
        if args.train: # train
            path = "save/{}_{}".format(args.sce_name, args.filename)
            if not os.path.exists(path):
                os.makedirs(path)
        else: # evaluate
            ppo.load("save/{}_{}/agent_{}".format(args.sce_name, args.filename, mode))
        mappo.append(ppo)
        print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
        if args.train: # train
            best_reward, best_step = Train(envs, mappo, writer, args, mode, agent_num)
            writer.close()
            print("best_reward:{}   best_step:{}".format(best_reward, best_step))
        else: # evaluate
            Evaluate(envs, mappo, args, mode, agent_num) 
    else:
        from ppo import PPOAgent
        from buffer import RolloutBuffer
        from train import Train
        from evaluate import Evaluate
        
        for i in range(agent_num):
            buffer = RolloutBuffer(
                steps=args.single_batch_size, num_env=args.num_env,
                state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                edge_index=edge_index,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, raction_shape=(1, ), # type: ignore
                raction_mask_shape=raction_mask_shape,
                device=device
                )
            ppo = PPOAgent(
                state_dim=state_dim, share_dim=share_dim, 
                caction_dim=caction_dim, caction_list=caction_list,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                raction_dim=raction_dim, raction_list=raction_list,
                edge_index=edge_index, buffer=buffer, device=device, args=args
                )
            if args.train: # train
                path = "save/{}_{}".format(args.sce_name, args.filename)
                if not os.path.exists(path):
                    os.makedirs(path)
            else: # evaluate
                ppo.load("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
            mappo.append(ppo)
        
        print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
        if args.train: # train
            best_reward, best_step = Train(envs, mappo, writer, args, mode, agent_num)
            writer.close()
            print("best_reward:{}   best_step:{}".format(best_reward, best_step))
        else: # evaluate
            Evaluate(envs, mappo, args, mode, agent_num)

if __name__ == '__main__':
    main()
