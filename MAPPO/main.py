import torch
import argparse
import gymnasium as gym
# import gym

import numpy as np
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
import os

from env.Sce_Env import Sce_Env
from env_wrappers import SubprocVecEnv, DummyVecEnv
from trainer.evaluate import Evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="SY_45")
    parser.add_argument("--filename", type=str, default="T2")
    parser.add_argument("--mode", type=str, default="OC", help='Choose one from [GH, NGH, OC, OR]')
    parser.add_argument("--algo", type=str, default="MAPPO", help='Choose one from [MAPPO, MASAC, MADDPG]')
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--continuous", type=bool, default=False)

    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--expert", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=1) # 环境数
    parser.add_argument("--num_update", type=int, default=100) # 最大更新轮次
    parser.add_argument("--save_freq", type=int, default=50) # 保存频率

    parser.add_argument("--ps", type=bool, default=False) # parameter sharing

    parser.add_argument("--policy_arch", type=list, default=[32, 32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32, 32])

    parser.add_argument("--lr", type=float, default=2.5e-4) # 学习率
    parser.add_argument("--gamma", type=float, default=0.95) # 折减因子
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--k_epoch", type=int, default=10) # 策略更新次数
    parser.add_argument("--eps_clip", type=float, default=0.1) # 裁剪参数
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01) # 熵正则
    parser.add_argument("--single_batch_size", type=int, default=60) # 单个buffer数据量
    parser.add_argument("--single_rbatch_size", type=int, default=30) # 单个buffer数据量
    parser.add_argument("--num_mini_batch", type=int, default=1) # 小batcch数量
    arguments = parser.parse_args()
    return arguments

def make_train_env(args, agent_num):
    def get_env_fn():
        def init_env():
            env = Sce_Env(args)
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

    env = Sce_Env(args)
    caction_list = env.caction_list # 可选充电动作列表
    raction_list = env.raction_list # 可选路径动作列表
    
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs # 充电站数量
    
    # args.train_times = int(args.single_batch_size * 2 // num_cs) # 单个环境运行次数
    args.batch_size = int(args.single_batch_size * args.num_env) # 总batch大小
    args.rbatch_size = int(args.single_rbatch_size * args.num_env) # 总batch大小
    # args.batch_size = int(args.single_batch_size * args.num_env * agent_num) # 总batch大小

    args.mini_batch_size = int(args.batch_size // args.num_mini_batch)
    args.mini_rbatch_size = int(args.rbatch_size // args.num_mini_batch)
    # args.train_steps = int(args.train_times * agent_num * (num_cs+1))
    # args.max_steps = int(args.train_steps * args.num_update)
    mode = args.mode

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
    mode = args.mode
    
    ############## CTDE or not ##############
    if args.ctde:
        print('Mode: {}   Agent_num: {}   CS_num: {}   Env_num: {}'.format(mode, agent_num, num_cs, args.num_env))
        share_dim = env.share_dim
        share_shape = (share_dim, )
        global_features_dim = env.global_features_dim # 全局地图特征维度
        global_features_shape = (graph.shape[0], global_features_dim) # 全局地图特征尺寸
    else:
        print('Mode: {}   Agent_num: {}   CS_num: {}   Env_num: {}'.format(mode+'_I', agent_num, num_cs, args.num_env))
        share_dim = env.state_dim
        share_shape = (state_dim, )
        global_features_dim = env.obs_features_dim
        global_features_shape = (graph.shape[0], obs_features_dim)
    if args.train:
        envs = make_train_env(args, agent_num)
        writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
        writer.add_text("HyperParameters", 
                        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    else:
        envs = env

    ############## Agents ##############
    agents = []
    if args.ps:
        from algo.ppo_ps.ppo_base import PPOAgentPS as Agent
        from trainer.train_ps import Train
        from buffer.buffer_ps import RolloutBufferPS as Buffer

        buffer = Buffer(
                steps=args.single_batch_size, rsteps=args.single_rbatch_size, num_env=args.num_env,
                state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                edge_index=edge_index,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                raction_shape=(1, ), # type: ignore
                raction_mask_shape=raction_mask_shape,
                agent_num=agent_num,
                device=device
            )
        agents = Agent(
                state_dim=state_dim, share_dim=share_dim, 
                caction_dim=caction_dim, caction_list=caction_list,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                raction_dim=raction_dim, raction_list=raction_list,
                edge_index=edge_index, buffer=buffer, device=device, 
                args=args
            )
        if args.train:
            path = "save/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, args.algo)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            agents.load("save/{}_{}_{}_{}/agents_{}".format(args.sce_name, args.filename, mode, args.algo, mode))
        print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
        if args.train: # train
            best_reward, best_step = Train(envs, agents, writer, args, mode, agent_num)
            writer.close()
            print("best_reward:{}   best_step:{}".format(best_reward, best_step))
        else: # evaluate
            Evaluate(envs, agents, args, mode, agent_num)
    else:
        if mode in ['GH']:
            assert args.algo == 'MAPPO', "Error"
            from algo.ppo.ppo_g import GPPOAgent as Agent
            from buffer.buffer_g import G_RolloutBuffer as Buffer
        else:
            if args.algo == 'MADDPG':
                assert args.continuous == True, "Continuous"
                from algo.ddpg_ppo.ddpg_ppo import DDPG_PPOAgent as Agent
                from trainer.train_ddpg import Train
            if args.algo == 'MASAC':
                from algo.sac.sac import SACAgent as Agent
                from trainer.train_sac import Train
            if args.algo == 'MAPPO':
                from algo.ppo.ppo_base import PPOAgent as Agent
                from trainer.train import Train
            from buffer.buffer_base import RolloutBuffer as Buffer
        
        for i in range(agent_num):
            buffer = Buffer(
                    steps=args.single_batch_size, rsteps=args.single_rbatch_size, num_env=args.num_env,
                    state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                    edge_index=edge_index,
                    obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                    raction_shape=(1, ), # type: ignore
                    raction_mask_shape=raction_mask_shape,
                    device=device
                )
            agent = Agent(
                    state_dim=state_dim, share_dim=share_dim, 
                    caction_dim=caction_dim, caction_list=caction_list,
                    obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                    raction_dim=raction_dim, raction_list=raction_list,
                    edge_index=edge_index, buffer=buffer, device=device, args=args
                )
            if args.train:
                path = "save/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, args.algo)
                if not os.path.exists(path):
                    os.makedirs(path)
            else:
                agent.load("save/{}_{}_{}_{}/agent_{}_{}".format(args.sce_name, args.filename, mode, args.algo, i, mode))
            agents.append(agent)
        
        print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
        if args.train: # train
            best_reward, best_step = Train(envs, agents, writer, args, mode, agent_num)
            writer.close()
            print("best_reward:{}   best_step:{}".format(best_reward, best_step))
        else: # evaluate
            Evaluate(envs, agents, args, mode, agent_num)

if __name__ == '__main__':
    main()
