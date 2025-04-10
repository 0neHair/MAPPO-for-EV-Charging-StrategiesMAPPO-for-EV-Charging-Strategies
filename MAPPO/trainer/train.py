import numpy as np
import time
from .utils import *

def Train(envs, agents, writer, args, mode, agent_num):
    current_step = 0 # 总步数

    start_time = time.time()
    total_best_reward = -10000 # 最佳总奖励
    global_total_reward = 0 # 当前总奖励存档
    best_step = 0 # 最佳总奖励对应轮次
    log_interval = 10
    
    default_caction = np.zeros([args.num_env, agent_num])
    default_raction = np.zeros([args.num_env, agent_num])
    default_action = (default_caction, default_raction)
    # run_times = [0 for _ in range(args.num_env)] # 每个环境的运行次数

    caction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
    raction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
    for i_episode in range(1, args.num_update + 1):
        sample_time = time.time()
        # 学习率递减
        if args.ps:
            lr = agents[0].lr_decay(i_episode)
        else:
            for agent in agents:
                lr = agent.lr_decay(i_episode)
        writer.add_scalar("Global/lr", lr, i_episode-1)
        
        ########### Initialize env ###########
        agents_total_reward = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        envs.reset()
        obs_n, obs_feature_n, obs_mask_n, \
            share_obs, global_cs_feature, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract \
                            = envs.step(default_action)
        for e in range(args.num_env):
            for i, agent_i in enumerate(activate_agent_ri[e]):
                agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]
        active_to_cpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        active_to_rpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        
        buffer_times = np.zeros((args.num_env, agent_num))
        rbuffer_times = np.zeros((args.num_env, agent_num))
        if mode in ['OC']:
            rbuffer_times += (args.single_rbatch_size+1)
        if mode in ['OR']:
            buffer_times += (args.single_batch_size+1)
            
        ########### Sample ###########
        while ((buffer_times>=args.single_batch_size).sum() < agent_num * args.num_env) or \
            ((rbuffer_times>=args.single_rbatch_size).sum() < agent_num * args.num_env):
            #* Select actions for activated agents and record their current states
            for e in range(args.num_env):
                # Charging decision
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if activate_to_cact[e][i]:
                        active_to_cpush[e][agent_i] = True
                        if mode not in ['OR']:
                            caction_n[e][agent_i] = choose_cation(
                                args, e, agents[agent_i], 
                                obs_n[e][agent_i], 
                                share_obs[e]
                            )
                # Routing decision
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if activate_to_ract[e][i]:
                        active_to_rpush[e][agent_i] = True
                        if mode not in ['OC']:
                            raction_n[e][agent_i] = choose_raction(
                                args, e, agents[agent_i], 
                                obs_feature_n[e][agent_i], obs_n[e][agent_i], obs_mask_n[e][agent_i],
                                global_cs_feature[e], share_obs[e]
                            )
            
            #* Run env until agent is activated
            obs_n, obs_feature_n, obs_mask_n, \
                share_obs, global_cs_feature, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, \
                            activate_agent_ri, activate_to_ract \
                                = envs.step((caction_n, raction_n))
            current_step += 1

            #* Save current states of activated agents as the results of last actions
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if active_to_cpush[e][agent_i]:
                        if mode not in ['OR']:
                            push_last_c(
                                args, e, agents[agent_i],
                                creward_n[e][agent_i], 
                                obs_n[e][agent_i],
                                share_obs[e],
                                done_n[e][agent_i]
                            )
                            buffer_times[e][agent_i] += 1
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if active_to_rpush[e][agent_i]:
                        agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]
                        if mode not in ['OC']:
                            push_last_r(
                                args, e, agents[agent_i],
                                rreward_n[e][agent_i], 
                                obs_feature_n[e][agent_i], obs_n[e][agent_i],
                                global_cs_feature[e], share_obs[e],
                                done_n[e][agent_i]
                            )
                            rbuffer_times[e][agent_i] += 1

            #* If no agent can be activated，reset env
            is_finished = envs.is_finished()
            if is_finished != []:
                obs_n_, obs_feature_n_, obs_mask_n_, \
                    share_obs_, global_cs_feature_, \
                        done_n_, creward_n_, rreward_n_, cact_n_, ract_n_, \
                            activate_agent_ci_, activate_to_cact_, \
                                activate_agent_ri_, activate_to_ract_ \
                                    = envs.reset_process(is_finished) # Reset
                for i, e in enumerate(is_finished):
                    total_reward = 0 
                    for j in range(agent_num):
                        total_reward += agents_total_reward[e][j]
                        writer.add_scalar("Single_Env/reward_{}_agent_{}".format(e, j), agents_total_reward[e][j], i_episode)
                    writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, i_episode)
                    
                    if total_reward > total_best_reward: # Calclulate total reward
                        total_best_reward = total_reward
                    writer.add_scalar("Global/total_reward", total_reward, i_episode)
                    writer.add_scalar("Global/total_best_reward", total_best_reward, i_episode)
                    best_step = i_episode

                    agents_total_reward[e] *= 0
                    # run_times[e] += 1
                    global_total_reward = total_reward
                    
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    creward_n[e] = creward_n_[i]
                    cact_n[e] = cact_n_[i]
                    activate_agent_ci[e] = activate_agent_ci_[i]
                    activate_to_cact[e] = activate_to_cact_[i]
                    obs_feature_n[e] = obs_feature_n_[i]
                    obs_mask_n[e] = obs_mask_n_[i]
                    global_cs_feature[e] = global_cs_feature_[i]
                    rreward_n[e] = rreward_n_[i]
                    ract_n[e] = ract_n_[i]
                    activate_agent_ri[e] = activate_agent_ri_[i]
                    activate_to_ract[e] = activate_to_ract_[i]
                    done_n[e] = done_n_[i]
                    active_to_cpush[e] = [False for _ in range(agent_num)]
                    active_to_rpush[e] = [False for _ in range(agent_num)]
                    for e in range(args.num_env):
                        for i, agent_i in enumerate(activate_agent_ri[e]):
                            agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]

        if i_episode % log_interval == 0:
            print(
                    'Episode {} \t Total reward: {:.3f} \t Average reward: {:.3f} \t Total best reward: {:.3f} \t Average best reward: {:.3f}'.format(
                            i_episode, global_total_reward, global_total_reward/agent_num, total_best_reward, total_best_reward/agent_num
                        )
                )
        
        print("Sampling: ", time.time()-sample_time)
        ########### Train network ###########
        train_time = time.time()
        total_actor_closs = 0
        total_critic_closs = 0
        total_entropy_closs = 0
        total_actor_rloss = 0
        total_critic_rloss = 0
        total_entropy_rloss = 0
        
        if args.ps:
            pass
        else:
            for i, agent in enumerate(agents):
                if mode in ['GH', 'NGH']:
                    actor_closs, critic_closs, entropy_closs, \
                        actor_rloss, critic_rloss, entropy_rloss = agent.train()
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
                elif mode in ['OC']:
                    actor_closs, critic_closs, entropy_closs = agent.train()
                    total_actor_closs += actor_closs
                    total_critic_closs += critic_closs
                    total_entropy_closs += entropy_closs
                    # writer.add_scalar("Loss/agent_{}_actor_closs".format(i), actor_closs, i_episode)
                    # writer.add_scalar("Loss/agent_{}_critic_closs".format(i), critic_closs, i_episode)
                    # writer.add_scalar("Loss/agent_{}_entropy_closs".format(i), entropy_closs, i_episode)
                elif mode in ['OR']:
                    actor_rloss, critic_rloss, entropy_rloss= agent.train()
                    total_actor_rloss += actor_rloss
                    total_critic_rloss += critic_rloss
                    total_entropy_rloss += entropy_rloss
                    # writer.add_scalar("Loss/agent_{}_actor_rloss".format(i), actor_rloss, i_episode)
                    # writer.add_scalar("Loss/agent_{}_critic_rloss".format(i), critic_rloss, i_episode)
                    # writer.add_scalar("Loss/agent_{}_entropy_rloss".format(i), entropy_rloss, i_episode)
                if i_episode % args.save_freq == 0:
                    agent.save("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
        if mode in ['GH', 'NGH', 'OC']:
            writer.add_scalar("Global_loss/actor_closs", total_actor_closs, i_episode)
            writer.add_scalar("Global_loss/critic_closs", total_critic_closs, i_episode)
            writer.add_scalar("Global_loss/entropy_closs", total_entropy_closs, i_episode)
        if mode in ['GH', 'NGH', 'OR']:
            writer.add_scalar("Global_loss/actor_rloss", total_actor_rloss, i_episode)
            writer.add_scalar("Global_loss/critic_rloss", total_critic_rloss, i_episode)
            writer.add_scalar("Global_loss/entropy_rloss", total_entropy_rloss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
        print("Traim: ", time.time()-train_time)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

