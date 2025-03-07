'''
方案2：
    EV在CS选择充到某个目标电量，该值一定大于当前电力
    设置当目标电量大于1时，视为不充电
    
    加入路径问题，采用图卷积网络处理路径信息
'''
import torch
import numpy as np
import time

def Train(envs, agents, writer, args, mode, agent_num):
    current_step = 0 # 总步数

    start_time = time.time()
    # agents_best_reward = [-10000 for _ in range(agent_num)] # 每个智能体的最佳奖励
    total_best_reward = -10000 # 最佳总奖励
    global_total_reward = 0 # 当前总奖励存档
    best_step = 0 # 最佳总奖励对应轮次
    log_interval = 10
    
    default_caction = np.zeros([args.num_env, agent_num]) # 默认动作
    default_raction = np.zeros([args.num_env, agent_num]) # 默认动作
    default_action = (default_caction, default_raction)
    # current_policy = []
    run_times = [0 for _ in range(args.num_env)] # 每个环境的运行次数

    for i_episode in range(1, args.num_update + 1):
        ########### 学习率递减 ###########
        if args.ps:
            lr = agents[0].lr_decay(i_episode)
        else:
            for agent in agents:
                lr = agent.lr_decay(i_episode)
        # expert_percent = 1 - current_step / args.demo_step
        writer.add_scalar("Global/lr", lr, i_episode-1)
        ########### 环境初始化 ###########
        agents_total_reward = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        envs.reset()
        obs_n, obs_mask_n, share_obs, done_n, \
            reward_n, cact_n, ract_n, \
                activate_agent_i, activate_to_act = envs.step(default_action)
        for e in range(args.num_env):
            for i, agent_i in enumerate(activate_agent_i[e]):
                agents_total_reward[e][agent_i] += reward_n[e][agent_i][0].copy()
        active_to_push = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        buffer_times = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        ########### 采样循环 ###########
        while (buffer_times>=args.single_batch_size).sum() < agent_num * args.num_env:
            caction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            raction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            #* 为已经激活的智能体选择动作，并记下当前的状态
            for e in range(args.num_env):
                # 决策充电和路径
                for i, agent_i in enumerate(activate_agent_i[e]):
                    if activate_to_act[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                pass
                            else:
                                caction, log_prob = agents[agent_i].select_action(
                                        obs_n[e][agent_i]
                                    )
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()  
                                # Push
                                agents[agent_i].rolloutBuffer.push_last_state(
                                    state=obs_n[e][agent_i], 
                                    share_state=share_obs_, 
                                    caction=caction, 
                                    log_prob=log_prob,
                                    env_id=e
                                    )
                            active_to_push[e][agent_i] = True
                            caction_n[e][agent_i] = caction[0].copy()

            #* 环境运行，直到有智能体被激活
            # last_info = info
            obs_n, obs_mask_n, share_obs, done_n, \
                reward_n, cact_n, ract_n, \
                    activate_agent_i, activate_to_act = envs.step((caction_n, raction_n)) # (caction_n, raction_n)
            current_step += 1
            #* 将被激活的智能体当前状态作为上一次动作的结果保存
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_i[e]):
                    # print("RR:", rreward_n)
                    agents_total_reward[e][agent_i] += reward_n[e][agent_i][0].copy()
                    if active_to_push[e][agent_i]:
                        if args.ps:
                            pass
                        else:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer.push(
                                reward=reward_n[e][agent_i],
                                next_state=obs_n[e][agent_i], 
                                next_share_state=share_obs_, 
                                done=done_n[e][agent_i],
                                env_id=e
                                )
                            buffer_times[e][agent_i] += 1
            #* 若没有可启动的智能体，说明环境运行结束，重启
            is_finished = envs.is_finished()
            if is_finished != []:
                # current_policy = env.get_policy().copy()
                # print(agents_total_reward)
                # 环境重启
                obs_n_, obs_mask_n_, share_obs_, done_n_, \
                    reward_n_, cact_n_, ract_n_, \
                        activate_agent_i_, activate_to_act_ \
                            = envs.reset_process(is_finished)
                # 奖励整理
                for i, e in enumerate(is_finished):
                    # 计算该环境总奖励
                    total_reward = 0 
                    for j in range(agent_num):
                        total_reward += agents_total_reward[e][j]
                        writer.add_scalar("Single_Env/reward_{}_agent_{}".format(e, j), agents_total_reward[e][j], i_episode)
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

                    agents_total_reward[e] = np.array([0.0 for _ in range(agent_num)]).copy()
                    run_times[e] += 1
                    global_total_reward = total_reward
                    
                    # 变量重启
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    reward_n[e] = reward_n_[i]
                    cact_n[e] = cact_n_[i]
                    activate_agent_i[e] = activate_agent_i_[i]
                    activate_to_act[e] = activate_to_act_[i]
                    obs_mask_n[e] = obs_mask_n_[i]
                    ract_n[e] = ract_n_[i]
                    done_n[e] = done_n_[i]
                    active_to_push[e] = [False for _ in range(agent_num)]
                    for e in range(args.num_env):
                        for i, agent_i in enumerate(activate_agent_i[e]):
                            agents_total_reward[e][agent_i] += reward_n[e][agent_i][0].copy()
                # avg_length += 1
        # print(i_episode, i_episode)
        if i_episode % log_interval == 0:
            print(
                    'Episode {} \t Total reward: {:.3f} \t Average reward: {:.3f} \t Total best reward: {:.3f} \t Average best reward: {:.3f}'.format(
                            i_episode, global_total_reward, global_total_reward/agent_num, total_best_reward, total_best_reward/agent_num
                        )
                )
        
        # 更新网络
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        if args.ps:
            pass
        else:
            for i, agent in enumerate(agents):
                actor_loss, critic_loss, entropy_loss = agent.train()
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy_loss
                writer.add_scalar("Loss/agent_{}_actor_loss".format(i), actor_loss, i_episode)
                writer.add_scalar("Loss/agent_{}_critic_loss".format(i), critic_loss, i_episode)
                writer.add_scalar("Loss/agent_{}_entropy_loss".format(i), entropy_loss, i_episode)
                if i_episode % args.save_freq == 0:
                    agent.save("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
                
        writer.add_scalar("Global_loss/actor_loss", total_actor_loss, i_episode)
        writer.add_scalar("Global_loss/critic_loss", total_critic_loss, i_episode)
        writer.add_scalar("Global_loss/entropy_loss", total_entropy_loss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

