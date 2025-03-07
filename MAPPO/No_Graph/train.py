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
        # 学习率递减
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
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract \
                            = envs.step(default_action)
        for e in range(args.num_env):
            for i, agent_i in enumerate(activate_agent_ri[e]):
                agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0].copy()
        active_to_cpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        active_to_rpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        buffer_times = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        ########### 采样循环 ###########
        while (buffer_times>=args.single_batch_size).sum() < agent_num * args.num_env:
            caction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            raction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            #* 为已经激活的智能体选择动作，并记下当前的状态
            for e in range(args.num_env):
                # 决策充电
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if activate_to_cact[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                pass
                            else:
                                caction, clog_prob = agents[agent_i].select_caction(obs_n[e][agent_i])
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()  
                                # Push
                                agents[agent_i].rolloutBuffer.cpush_last_state(
                                    state=obs_n[e][agent_i], 
                                    share_state=share_obs_, 
                                    action=caction, 
                                    log_prob=clog_prob,
                                    env_id=e
                                    )
                            active_to_cpush[e][agent_i] = True
                            caction_n[e][agent_i] = caction[0].copy()
                # 决策路径
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if activate_to_ract[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                pass
                            else:
                                raction, rlog_prob = agents[agent_i].select_raction(
                                    obs_n[e][agent_i], obs_mask_n[e][agent_i]
                                )
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()  
                                # Push
                                agents[agent_i].rolloutBuffer.rpush_last_state(
                                    rstate=obs_n[e][agent_i], 
                                    share_rstate=share_obs_,
                                    action=raction, 
                                    action_mask=obs_mask_n[e][agent_i], 
                                    log_prob=rlog_prob,
                                    env_id=e
                                    )
                            active_to_rpush[e][agent_i] = True
                            raction_n[e][agent_i] = raction[0].copy()
            #* 环境运行，直到有智能体被激活
            # last_info = info
            obs_n, obs_mask_n, \
                share_obs, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, \
                            activate_agent_ri, activate_to_ract \
                                = envs.step((caction_n, raction_n)) # (caction_n, raction_n)
            current_step += 1
            #* 将被激活的智能体当前状态作为上一次动作的结果保存
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    # print("CR:", creward_n)
                    if active_to_cpush[e][agent_i]:
                        if args.ps:
                            pass
                        else:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer.cpush(
                                reward=creward_n[e][agent_i],
                                next_state=obs_n[e][agent_i], 
                                next_share_state=share_obs_, 
                                done=done_n[e][agent_i],
                                env_id=e
                                )
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    # print("RR:", rreward_n)
                    if active_to_rpush[e][agent_i]:
                        agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0].copy()
                        if args.ps:
                            pass
                        else:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer.rpush(
                                reward=rreward_n[e][agent_i],
                                next_rstate=obs_n[e][agent_i], 
                                next_share_rstate=share_obs_, 
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
                obs_n_, obs_mask_n_, \
                    share_obs_, \
                        done_n_, creward_n_, rreward_n_, cact_n_, ract_n_, \
                            activate_agent_ci_, activate_to_cact_, \
                                activate_agent_ri_, activate_to_ract_ \
                                    = envs.reset_process(is_finished)
                # 奖励整理
                for i, e in enumerate(is_finished):
                    # 计算该环境总奖励
                    total_reward = 0 
                    for j in range(agent_num):
                        total_reward += agents_total_reward[e][j]
                        writer.add_scalar("Single_Env/reward_{}_agent_{}".format(e, j), agents_total_reward[e][j], i_episode)
                    writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, i_episode)
                    # 统计总体奖励
                    if total_reward > total_best_reward:
                        total_best_reward = total_reward
                    writer.add_scalar("Global/total_reward", total_reward, i_episode)
                    writer.add_scalar("Global/total_best_reward", total_best_reward, i_episode)
                    best_step = i_episode

                    agents_total_reward[e] = np.array([0.0 for _ in range(agent_num)]).copy()
                    run_times[e] += 1
                    global_total_reward = total_reward
                    
                    # 变量重启
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    creward_n[e] = creward_n_[i]
                    cact_n[e] = cact_n_[i]
                    activate_agent_ci[e] = activate_agent_ci_[i]
                    activate_to_cact[e] = activate_to_cact_[i]
                    obs_mask_n[e] = obs_mask_n_[i]
                    rreward_n[e] = rreward_n_[i]
                    ract_n[e] = ract_n_[i]
                    activate_agent_ri[e] = activate_agent_ri_[i]
                    activate_to_ract[e] = activate_to_ract_[i]
                    done_n[e] = done_n_[i]
                    active_to_cpush[e] = [False for _ in range(agent_num)]
                    active_to_rpush[e] = [False for _ in range(agent_num)]
                    for e in range(args.num_env):
                        for i, agent_i in enumerate(activate_agent_ri[e]):
                            agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0].copy()
                # avg_length += 1
        # print(i_episode, i_episode)
        if i_episode % log_interval == 0:
            print(
                    'Episode {} \t Total reward: {:.3f} \t Average reward: {:.3f} \t Total best reward: {:.3f} \t Average best reward: {:.3f}'.format(
                            i_episode, global_total_reward, global_total_reward/agent_num, total_best_reward, total_best_reward/agent_num
                        )
                )
            
        # 更新网络
        total_actor_closs = 0
        total_critic_closs = 0
        total_entropy_closs = 0
        total_actor_rloss = 0
        total_critic_rloss = 0
        total_entropy_rloss = 0
        
        if args.ps:
            pass
            # actor_loss, critic_loss, entropy_loss = agents[0].train()
            # total_actor_loss += actor_loss
            # total_critic_loss += critic_loss
            # total_entropy_loss += entropy_loss
            # writer.add_scalar("Loss/agent_ps_actor_loss", actor_loss, i_episode)
            # writer.add_scalar("Loss/agent_ps_critic_loss", critic_loss, i_episode)
            # writer.add_scalar("Loss/agent_ps_entropy_loss", entropy_loss, i_episode)
            # if i_episode % args.save_freq == 0:
            #     agents[0].save("save/{}_{}/agent_ps_{}".format(args.sce_name, args.filename, mode))
        else:
            for i, agent in enumerate(agents):
                actor_closs, critic_closs, entropy_closs, \
                    actor_rloss, critic_rloss, entropy_rloss= agent.train()
                total_actor_closs += actor_closs
                total_critic_closs += critic_closs
                total_entropy_closs += entropy_closs
                total_actor_rloss += actor_rloss
                total_critic_rloss += critic_rloss
                total_entropy_rloss += entropy_rloss
                # writer.add_scalar("Loss/agent_{}_actor_closs".format(i), actor_closs, i_episode)
                # writer.add_scalar("Loss/agent_{}_critic_closs".format(i), critic_closs, i_episode)
                # writer.add_scalar("Loss/agent_{}_entropy_closs".format(i), entropy_closs, i_episode)
                # writer.add_scalar("Loss/agent_{}_actor_rloss".format(i), actor_rloss, i_episode)
                # writer.add_scalar("Loss/agent_{}_critic_rloss".format(i), critic_rloss, i_episode)
                # writer.add_scalar("Loss/agent_{}_entropy_rloss".format(i), entropy_rloss, i_episode)
                if i_episode % args.save_freq == 0:
                    agent.save("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
                
        writer.add_scalar("Global_loss/actor_closs", total_actor_closs, i_episode)
        writer.add_scalar("Global_loss/critic_closs", total_critic_closs, i_episode)
        writer.add_scalar("Global_loss/entropy_closs", total_entropy_closs, i_episode)
        writer.add_scalar("Global_loss/actor_rloss", total_actor_rloss, i_episode)
        writer.add_scalar("Global_loss/critic_rloss", total_critic_rloss, i_episode)
        writer.add_scalar("Global_loss/entropy_rloss", total_entropy_rloss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

