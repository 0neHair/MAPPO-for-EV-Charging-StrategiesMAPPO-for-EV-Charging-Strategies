'''
Author: CQZ
Date: 2024-04-11 22:51:15
Company: SEU
'''
'''
方案2：
    EV在CS选择充到某个目标电量，该值一定大于当前电力
    设置当目标电量大于1时，视为不充电
'''
import torch
import argparse
from ppo import PPOAgent
from buffer import RolloutBuffer
import gymnasium as gym
# import gym

import numpy as np
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
from env.EV_Sce_Env import EV_Sce_Env
from env_wrappers import SubprocVecEnv, DummyVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="LS1_2P")
    parser.add_argument("--filename", type=str, default="Train")
    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--expert", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=6) # 环境数
    parser.add_argument("--num_update", type=int, default=3000) # 最大更新轮次
    parser.add_argument("--save_freq", type=int, default=50) # 保存频率
    # parser.add_argument("--shared_arch", type=int, nargs='*', default=64)
    # parser.add_argument("--policy_arch", type=int, nargs="*", default=None)
    # parser.add_argument("--value_arch", type=int, nargs="*", default=None)

    # 共享网络层
    parser.add_argument("--shared_arch", type=list, default=[])  # [32, 32]
    parser.add_argument("--policy_arch", type=list, default=[32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32])

    parser.add_argument("--lr", type=float, default=4e-4) # 学习率
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

def train_policy(envs, agents, writer, args, mode, agent_num):
    current_step = 0 # 总步数

    start_time = time.time()
    # agents_best_reward = [-10000 for _ in range(agent_num)] # 每个智能体的最佳奖励
    total_best_reward = -10000 # 最佳总奖励
    global_total_reward = 0 # 当前总奖励存档
    best_step = 0 # 最佳总奖励对应轮次
    log_interval = 10
    
    default_action = [np.zeros(agent_num) for _ in range(args.num_env)] # 默认动作
    # current_policy = []
    run_times = [0 for _ in range(args.num_env)] # 每个环境的运行次数

    for i_episode in range(1, args.num_update + 1):
        #^ 学习率递减
        for agent in agents:
            lr = agent.lr_decay(current_step)
        # expert_percent = 1 - current_step / args.demo_step
        writer.add_scalar("Global/lr", lr, i_episode-1)
        #* 环境初始化
        agents_total_reward = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        envs.reset()
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = envs.step(default_action)
        ########### 采样循环 ###########
        for t in range(args.train_steps):
            action_n = np.array([[-1 for i in range(agent_num)] for j in range(args.num_env)])
            #* 为已经激活的智能体选择动作，并记下当前的状态
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_i[e]):
                    if activate_to_act[e][i]:
                        with torch.no_grad():
                            action, log_prob = agents[agent_i].select_action(obs_n[e][agent_i])
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer_list[e].push_last_state(
                                obs_n[e][agent_i], 
                                share_obs_, 
                                action, 
                                log_prob
                                )
                            action_n[e][agent_i] = action[0].copy()
            #* 环境运行，直到有智能体被激活
            # last_info = info
            obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = envs.step(action_n)
            current_step += 1
            #* 将被激活的智能体当前状态作为上一次动作的结果保存
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_i[e]):
                    if act_n[e][agent_i] != -1:
                        if not args.ctde:
                            share_obs_ = obs_n[e][agent_i].copy()
                        else:
                            share_obs_ = share_obs[e].copy()
                        agents[agent_i].rolloutBuffer_list[e].push(
                            reward=reward_n[e][agent_i],
                            next_state=obs_n[e][agent_i], 
                            next_share_state=share_obs_, 
                            done=done_n[e][agent_i]
                            )
                        agents_total_reward[e][agent_i] += reward_n[e][agent_i][0].copy()
            #* 若没有可启动的智能体，说明环境运行结束，重启
            is_finished = envs.is_finished()
            if is_finished != []:
                # current_policy = env.get_policy().copy()
                obs_n_, share_obs_, reward_n_, done_n_, info_n_, act_n_, activate_agent_i_, activate_to_act_ = envs.reset_process(is_finished)
                for i, e in enumerate(is_finished):
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    reward_n[e] = reward_n_[i]
                    done_n[e] = done_n_[i]
                    info_n[e] = info_n_[i]
                    act_n[e] = act_n_[i]
                    activate_agent_i[e] = activate_agent_i_[i]
                    activate_to_act[e] = activate_to_act_[i]

                    # 计算该环境总奖励
                    total_reward = 0 
                    for i in range(agent_num):
                        total_reward += agents_total_reward[e][i]
                    # writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, run_times[e])
                    writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, i_episode)
                    # 统计总体奖励
                    if total_reward > total_best_reward:
                        total_best_reward = total_reward
                    # writer.add_scalar("Global/total_reward", total_reward, sum(run_times))
                    # writer.add_scalar("Global/total_best_reward", total_best_reward, sum(run_times))
                    writer.add_scalar("Global/total_reward", total_reward, i_episode)
                    writer.add_scalar("Global/total_best_reward", total_best_reward, i_episode)
                    best_step = i_episode

                    agents_total_reward[e] = np.array([0.0 for _ in range(agent_num)])
                    run_times[e] += 1
                    global_total_reward = total_reward

                # avg_length += 1
        # print(i_episode, i_episode)
        if i_episode % log_interval == 0:
            print('Episode {} \t Total reward: {:.3f} \t Total best reward: {:.3f}'.format(i_episode, global_total_reward, total_best_reward))
        
        # 更新网络
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        for i, agent in enumerate(agents):
            actor_loss, critic_loss, entropy_loss = agent.train()
            # eval_reward, eval_tts = eval_policy(agent, times=1)
            # writer.add_scalar("Eval/returns", eval_reward, current_step)
            # writer.add_scalar("Eval/tts", eval_tts, current_step)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            
            writer.add_scalar("Loss/agent_{}_actor_loss".format(i), actor_loss, i_episode)
            writer.add_scalar("Loss/agent_{}_critic_loss".format(i), critic_loss, i_episode)
            writer.add_scalar("Loss/agent_{}_entropy_loss".format(i), entropy_loss, i_episode)

            if i_episode % args.save_freq == 0:
                agent.save("save/{}_{}_agent_{}_{}".format(args.sce_name, args.filename, i, mode))
                
        writer.add_scalar("Global_loss/actor_loss", total_actor_loss, i_episode)
        writer.add_scalar("Global_loss/critic_loss", total_critic_loss, i_episode)
        writer.add_scalar("Global_loss/entropy_loss", total_entropy_loss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

def make_train_env(args, agent_num):
    def get_env_fn():
        def init_env():
            env = EV_Sce_Env(args.sce_name, seed=args.seed)
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

    envs = make_train_env(args, agent_num)
    writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
    writer.add_text("HyperParameters", 
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    
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
        # ppo.load("save/{}_{}_agent_{}_{}".format(args.sce_name, args.filename, i, mode))
        mappo.append(ppo)
        
    print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))

    best_reward, best_step = train_policy(envs, mappo, writer, args, mode, agent_num)
    writer.close()
    print("best_reward:{}   best_step:{}".format(best_reward, best_step))
    print()

if __name__ == '__main__':
    main()
