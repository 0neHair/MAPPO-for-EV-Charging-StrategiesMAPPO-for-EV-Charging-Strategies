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
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="SY_2")
    task_list = ['SY_2', 'SY_2p']
    parser.add_argument("--task_list", type=list, default=task_list)
    parser.add_argument("--filename", type=str, default="test")
    parser.add_argument("--mode", type=str, default="NGH", help='Choose one from [GH, NGH, OC, OR]')
    parser.add_argument("--algo", type=str, default="MAPPO", help='Choose one from [MAPPO, MASAC, MADDPG]')
    
    parser.add_argument("--meta_algo", type=str, default="MAML", help='Choose one from [MAML, Reptile, Adapt]')
    parser.add_argument("--adapt_from", type=str, default="MAML", help='Choose one from [MAML, Reptile]')
    
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--continuous", type=bool, default=False)

    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--expert", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=1) # 环境数
    parser.add_argument("--num_update", type=int, default=3000) # 最大更新轮次
    parser.add_argument("--save_freq", type=int, default=50) # 保存频率

    parser.add_argument("--ps", type=bool, default=False) # parameter sharing

    parser.add_argument("--policy_arch", type=list, default=[32, 32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32, 32])

    parser.add_argument("--adapt_num", type=int, default=1)
    parser.add_argument("--adapt_lr", type=float, default=1e-1) # 学习率
    parser.add_argument("--meta_lr", type=float, default=2.5e-4) # 学习率
    parser.add_argument("--gamma", type=float, default=0.95) # 折减因子
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--k_epoch", type=int, default=10) # 策略更新次数
    parser.add_argument("--eps_clip", type=float, default=0.05) # 裁剪参数
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01) # 熵正则
    parser.add_argument("--single_batch_size", type=int, default=40) # 单个buffer数据量
    parser.add_argument("--single_rbatch_size", type=int, default=20) # 单个buffer数据量
    parser.add_argument("--num_mini_batch", type=int, default=1) # 小batcch数量
    arguments = parser.parse_args()
    return arguments

def make_train_env(sce_name, args, agent_num):
    def get_env_fn():
        def init_env():
            env = Sce_Env(sce_name, args)
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

    env = Sce_Env(args.sce_name, args)
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
    args.lr = args.meta_lr
    
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
    task_name_list = args.task_list
    
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
        envs = make_train_env(args.sce_name, args, agent_num)
        task_list = []
        for task_name in task_name_list:
            task_list.append(make_train_env(task_name, args, agent_num))
        writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
        writer.add_text("HyperParameters", 
                        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    
    else:
        # envs = env
        pass

    ############## Agents ##############
    agents = []
    # if mode in ['GH']:
    #     assert args.algo == 'MAPPO', "Error"
    #     from algo.ppo.ppo_g import GPPOAgent as Agent
    #     from buffer.buffer_g import G_RolloutBuffer as Buffer
    # else:
    #     from algo.meta_ppo.ppo_base import PPOAgent as Agent
    #     from meta_trainer.train import Meta_Trainer
    #     from buffer.buffer_meta import Meta_RolloutBuffer as Buffer
    
    
    if args.meta_algo == 'MAML':
        from algo.meta_ppo.ppo_base import PPOAgent as Agent
        from meta_trainer.train import Meta_Trainer
        from buffer.buffer_meta import Meta_RolloutBuffer as Buffer
        
        for i in range(agent_num):
            buffer_list = []
            validbuffer_list = []
            for t in range(len(task_name_list)):
                buffer_list.append(Buffer(
                        steps=args.single_batch_size, rsteps=args.single_rbatch_size, num_env=args.num_env,
                        state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                        edge_index=edge_index,
                        obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                        raction_shape=(1, ), # type: ignore
                        raction_mask_shape=raction_mask_shape,
                        device=device
                    ))
                validbuffer_list.append(Buffer(
                        steps=args.single_batch_size, rsteps=args.single_rbatch_size, num_env=args.num_env,
                        state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                        edge_index=edge_index,
                        obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                        raction_shape=(1, ), # type: ignore
                        raction_mask_shape=raction_mask_shape,
                        device=device
                    ))
            agent = Agent(
                    state_dim=state_dim, share_dim=share_dim, 
                    caction_dim=caction_dim, caction_list=caction_list,
                    obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                    raction_dim=raction_dim, raction_list=raction_list,
                    edge_index=edge_index, buffer=buffer_list, validbuffer=validbuffer_list, device=device, args=args
                )

            args.path = "save/{}_{}_{}".format(args.filename, mode, args.meta_algo)
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            # agent.load(args.path + "/agent_{}_{}_{}".format(i, 'meta', args.mode))
            agents.append(agent)
    
    elif args.meta_algo == 'Reptile':
        from algo.ppo_.ppo_base import PPOAgent as Agent
        from meta_trainer.train_reptile import Meta_Trainer
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

            args.path = "save/{}_{}_{}".format(args.filename, mode, args.meta_algo)
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            # agent.load(args.path + "/agent_{}_{}_{}".format(i, 'meta', args.mode))
            agents.append(agent)
            
    elif args.meta_algo == 'Adapt':
        from algo.ppo_.ppo_base import PPOAgent as Agent
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
            #TODO: load agent
            agent.load("save/{}_{}_{}/agent_{}_{}_{}".format(args.filename, mode, args.adapt_from, i, 'meta', mode))
            args.path = "save/{}_{}_{}".format(args.filename, mode, args.meta_algo+'-'+args.adapt_from)
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            agents.append(agent)
    
    print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.meta_lr, args.gamma))
    
    if args.meta_algo == 'Adapt':
        if args.train: # train
            best_reward, best_step = Train(envs, agents, writer, args, mode, agent_num)
            writer.close()
            print("best_reward:{}   best_step:{}".format(best_reward, best_step))
        else: # evaluate
            Evaluate(envs, agents, args, mode, agent_num)
    else:
        trainer = Meta_Trainer(
            args,
            task_list, agents, writer,
            adapt_num=args.adapt_num, 
            device=device,
        )
        trainer.train(epochs=args.num_update)
        writer.close()

if __name__ == '__main__':
    main()
