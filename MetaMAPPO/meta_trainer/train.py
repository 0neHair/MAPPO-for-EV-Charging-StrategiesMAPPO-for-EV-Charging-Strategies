import numpy as np
import time
from .utils_meta import *

class Meta_Trainer:
    def __init__(
            self, args,

            task_list, 
            agents, 
            writer,

            adapt_num: int = 1, 
            device="cpu",
        ):
        self.tasks = task_list
        self.agents = agents
        self.agent_num = len(agents)
        self.args = args
        self.num_env = args.num_env
        self.mode = args.mode
        self.writer = writer

        self.caction_n = np.array([[-1 for _ in range(self.agent_num)] for __ in range(args.num_env)])
        self.raction_n = np.array([[-1 for _ in range(self.agent_num)] for __ in range(args.num_env)])
        self.default_caction = np.zeros([args.num_env, self.agent_num])
        self.default_raction = np.zeros([args.num_env, self.agent_num])
        self.default_action = (self.default_caction, self.default_raction)
        self.agents_total_reward = np.array([[0.0 for _ in range(self.agent_num)] for __ in range(args.num_env)])
        self.global_total_reward = -99999999
        self.total_best_reward = -99999999

        # self.ctheta = dict(self.agents[0].cPolicy.named_parameters())
        # self.rtheta = dict(self.agents[0].rPolicy.named_parameters())

        self.adapt_num = adapt_num
        self.device = torch.device(device)
        self.iter = 0

    def adapt_individual(self, task_id):
        for i, agent in enumerate(self.agents):
            agent.adapt_policy(task_id, second_order=False)

    def train_value_fuction(self, epoch, task_id, valid: bool = False):
        total_critic_closs = 0
        total_critic_rloss = 0
        for i, agent in enumerate(self.agents):
            cvloss, rvloss = agent.train_value_function(task_id=task_id, valid=valid)
            if not valid:
                self.writer.add_scalar("Local_loss/{}_critic_closs_task_{}".format(i, task_id), cvloss, epoch)
                self.writer.add_scalar("Local_loss/{}_critic_rloss_task_{}".format(i, task_id), rvloss, epoch)
            total_critic_closs += cvloss
            total_critic_rloss += rvloss
        if not valid:
            self.writer.add_scalar("Global_loss/critic_closs_task_{}".format(task_id), total_critic_closs, epoch)
            self.writer.add_scalar("Global_loss/critic_rloss_task_{}".format(task_id), total_critic_rloss, epoch)
            # print(total_critic_closs, total_critic_rloss)

    def train(self, epochs: int):
        start_time = time.time()
        for epoch in range(1, epochs+1):
            
            for agent in self.agents:
                agent.save_theta()
                
            for  t, task in enumerate(self.tasks):
                t1 = time.time()
                self.sample(epoch=epoch, task_id=t, valid=False) #* 传统强化学习跑一遍，取样轨迹
                # 训练value和actor
                self.train_value_fuction(epoch=epoch, task_id=t, valid=False)
                self.adapt_individual(task_id=t)
                print('T1: ', time.time()-t1)

                t2 = time.time()
                self.sample(epoch=epoch, task_id=t, valid=True) #* 新更新后的策略参数，取一串轨迹用于验证
                self.train_value_fuction(epoch=epoch, task_id=t, valid=True) #* 计算adv
                for agent in self.agents:
                    agent.restore_theta()
                print('T2: ', time.time()-t2)

            t3 = time.time()
            total_cactor_loss, total_centropy, total_ractor_loss, total_rentropy = 0, 0, 0, 0
            for i, agent in enumerate(self.agents):
                cactor_loss, centropy, ractor_loss, rentropy = self.step(agent_id=i, epoch=epoch)
                total_cactor_loss += cactor_loss
                total_centropy += centropy
                total_ractor_loss += ractor_loss
                total_rentropy += rentropy
                if epoch % 50 == 0:
                    agent.save("save/{}_{}_{}_{}/agent_{}_{}_{}".format(self.args.sce_name, self.args.filename, self.mode, self.args.meta_algo, i, 'meta', self.mode))

            self.writer.add_scalar("Global_loss/actor_closs", total_cactor_loss, epoch)
            self.writer.add_scalar("Global_loss/entropy_closs", total_centropy, epoch)
            self.writer.add_scalar("Global_loss/actor_rloss", total_ractor_loss, epoch)
            self.writer.add_scalar("Global_loss/entropy_rloss", total_rentropy, epoch)
            print('T3: ', time.time()-t3)

            self.writer.add_scalar("Global/time_step", (self.iter + 1) / (time.time() - start_time), epoch)
            self.iter += 1
            if epoch % 50 == 0:
                print(
                        'Episode {} \t Total reward: {:.3f} \t Average reward: {:.3f} \t Total best reward: {:.3f} \t Average best reward: {:.3f}'.format(
                                epoch, self.global_total_reward, self.global_total_reward/self.agent_num, self.total_best_reward, self.total_best_reward/self.agent_num
                            )
                    )
                
    def meta_loss(self, agent_id, second_order: bool):
        closs, centropy, rloss, rentropy = [], [], [], []
        self.agents[agent_id].save_theta()
        for  t, task in enumerate(self.tasks):
            self.agents[agent_id].adapt_policy(t, second_order=second_order)
            ctask_loss, ctask_entropy, rtask_loss, rtask_entropy = self.agents[agent_id].surrogate_loss(t, valid=True) #* 验证轨迹算loss

            closs.append(ctask_loss)
            centropy.append(ctask_entropy)
            rloss.append(rtask_loss)
            rentropy.append(rtask_entropy)
            self.agents[agent_id].restore_theta(second_order=second_order)

        return torch.stack(closs), torch.stack(centropy), torch.stack(rloss), torch.stack(rentropy)

    def step(self, agent_id, epoch):
        cactor_loss, centropy, ractor_loss, rentropy = self.meta_loss(agent_id, second_order=True)
        closs = cactor_loss - centropy
        rloss = ractor_loss - rentropy
        mean_closs, mean_centropy, mean_rloss, mean_rentropy = closs.mean(), centropy.mean(), rloss.mean(), rentropy.mean()
        self.agents[agent_id].cpolicy_optim.zero_grad() #* 元损失更新
        mean_closs.backward()
        self.agents[agent_id].cpolicy_optim.step()
        self.agents[agent_id].rpolicy_optim.zero_grad() #* 元损失更新
        mean_rloss.backward()
        self.agents[agent_id].rpolicy_optim.step()
        cactor_loss = cactor_loss.mean().item()
        ractor_loss = ractor_loss.mean().item()
        centropy = mean_centropy.item()
        rentropy = mean_rentropy.item()
        self.writer.add_scalar("Local_loss/{}_actor_closs".format(agent_id), cactor_loss, epoch)
        self.writer.add_scalar("Local_loss/{}_entropy_closs".format(agent_id), mean_centropy, epoch)
        self.writer.add_scalar("Local_loss/{}_actor_rloss".format(agent_id), ractor_loss, epoch)
        self.writer.add_scalar("Local_loss/{}_entropy_rloss".format(agent_id), mean_rentropy, epoch)
        return cactor_loss, centropy, ractor_loss, rentropy
    
    def sample(self, epoch, task_id, valid: bool = False):
        env = self.tasks[task_id]
        self.agents_total_reward *= 0
        env.reset()
        obs_n, obs_feature_n, obs_mask_n, \
            share_obs, global_cs_feature, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract \
                            = env.step(self.default_action)
        for e in range(self.num_env):
            for i, agent_i in enumerate(activate_agent_ri[e]):
                self.agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]
        active_to_cpush = [[False for _ in range(self.agent_num)] for __ in range(self.num_env)]
        active_to_rpush = [[False for _ in range(self.agent_num)] for __ in range(self.num_env)]
        buffer_times = np.zeros((self.num_env, self.agent_num))
        rbuffer_times = np.zeros((self.num_env, self.agent_num))
        while ((buffer_times>=self.args.single_batch_size).sum() < self.agent_num * self.num_env) or \
            ((rbuffer_times>=self.args.single_rbatch_size).sum() < self.agent_num * self.num_env):
            #* Select actions for activated agents and record their current states
            for e in range(self.num_env):
                # Charging decision
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if activate_to_cact[e][i]:
                        active_to_cpush[e][agent_i] = True
                        self.caction_n[e][agent_i] = choose_cation(
                                self.args, e, self.agents[agent_i], 
                                obs_n[e][agent_i], 
                                share_obs[e], best=False, valid=valid, task_id=task_id
                            )
                # Routing decision
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if activate_to_ract[e][i]:
                        active_to_rpush[e][agent_i] = True
                        self.raction_n[e][agent_i] = choose_raction(
                            self.args, e, self.agents[agent_i], 
                            obs_feature_n[e][agent_i], obs_n[e][agent_i], obs_mask_n[e][agent_i],
                            global_cs_feature[e], share_obs[e], best=False, valid=valid, task_id=task_id
                        )
            
            #* Run env until agent is activated
            obs_n, obs_feature_n, obs_mask_n, \
                share_obs, global_cs_feature, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, \
                            activate_agent_ri, activate_to_ract \
                                = env.step((self.caction_n, self.raction_n))

            #* Save current states of activated agents as the results of last actions
            for e in range(self.num_env):
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if active_to_cpush[e][agent_i]:
                        push_last_c(
                            self.args, e, self.agents[agent_i],
                            creward_n[e][agent_i], 
                            obs_n[e][agent_i], share_obs[e],
                            done_n[e][agent_i], 
                            valid, task_id
                        )
                        buffer_times[e][agent_i] += 1
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if active_to_rpush[e][agent_i]:
                        self.agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]
                        push_last_r(
                            self.args, e, self.agents[agent_i],
                            rreward_n[e][agent_i], 
                            obs_feature_n[e][agent_i], obs_n[e][agent_i],
                            global_cs_feature[e], share_obs[e],
                            obs_mask_n[e][agent_i],
                            done_n[e][agent_i], 
                            valid, task_id
                        )
                        rbuffer_times[e][agent_i] += 1

            #* If no agent can be activated，reset env
            is_finished = env.is_finished()
            if is_finished != []:
                obs_n_, obs_feature_n_, obs_mask_n_, \
                    share_obs_, global_cs_feature_, \
                        done_n_, creward_n_, rreward_n_, cact_n_, ract_n_, \
                            activate_agent_ci_, activate_to_cact_, \
                                activate_agent_ri_, activate_to_ract_ \
                                    = env.reset_process(is_finished) # Reset
                for i, e in enumerate(is_finished):
                    total_reward = 0 
                    for j in range(self.agent_num):
                        total_reward += self.agents_total_reward[e][j]
                        if not valid:
                            self.writer.add_scalar("Single_Env/reward_{}_agent_{}_task_{}".format(e, j, task_id), self.agents_total_reward[e][j], epoch)

                    if total_reward > self.total_best_reward: # Calclulate total reward
                        self.total_best_reward = total_reward
                    if not valid:
                        self.writer.add_scalar("Single_Env/reward_{}_task_{}".format(e, task_id), total_reward, epoch)
                        self.writer.add_scalar("Global/total_reward_task_{}".format(task_id), total_reward, epoch)
                        self.writer.add_scalar("Global/total_best_reward_{}".format(task_id), self.total_best_reward, epoch)
                    
                    self.agents_total_reward[e] *= 0
                    # run_times[e] += 1
                    self.global_total_reward = total_reward
                    
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
                    active_to_cpush[e] = [False for _ in range(self.agent_num)]
                    active_to_rpush[e] = [False for _ in range(self.agent_num)]
                    for e in range(self.num_env):
                        for i, agent_i in enumerate(activate_agent_ri[e]):
                            self.agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0]
