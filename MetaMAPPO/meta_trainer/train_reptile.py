import numpy as np
import time
from .utils import *

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
        self.task_num = len(task_list)
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

        self.log_interval = 10
        self.save_freq = args.save_freq

    def update_theta(self, theta, old_theta):
        torch.set_grad_enabled(False)
        new_theta = {name: old_theta[name] +  0.1 * (theta[name] - old_theta[name]) for name in theta}
        torch.set_grad_enabled(True)
        for name in new_theta:
            new_theta[name].requires_grad = True
        return new_theta
        
    def adapt(self, epoch):
        for i, agent in enumerate(self.agents):
            cp_theta, rp_theta, cv_theta, rv_theta = agent.return_theta()
            new_cp_theta = self.update_theta(cp_theta, agent.cp_theta)
            new_rp_theta = self.update_theta(rp_theta, agent.rp_theta)
            new_cv_theta = self.update_theta(cv_theta, agent.cv_theta)
            new_rv_theta = self.update_theta(rv_theta, agent.rv_theta)
            agent.load_theta((new_cp_theta, new_rp_theta, new_cv_theta, new_rv_theta))
        
    def train_ppo(self, epoch, task_id):
        total_critic_closs = 0
        total_critic_rloss = 0
        total_actor_closs = 0
        total_actor_rloss = 0
        total_entropy_closs = 0
        total_entropy_rloss = 0
        for i, agent in enumerate(self.agents):
            actor_closs, critic_closs, entropy_closs, \
                actor_rloss, critic_rloss, entropy_rloss = agent.train()
            total_actor_closs += actor_closs
            total_critic_closs += critic_closs
            total_entropy_closs += entropy_closs
            total_actor_rloss += actor_rloss
            total_critic_rloss += critic_rloss
            total_entropy_rloss += entropy_rloss

        self.writer.add_scalar("Global_loss/critic_closs_task_{}".format(task_id), total_critic_closs, epoch)
        self.writer.add_scalar("Global_loss/critic_rloss_task_{}".format(task_id), total_critic_rloss, epoch)
        # self.writer.add_scalar("Global_loss/actor_closs_task_{}".format(task_id), total_actor_closs, epoch)
        # self.writer.add_scalar("Global_loss/entropy_closs_task_{}".format(task_id), total_entropy_closs, epoch)
        # self.writer.add_scalar("Global_loss/actor_rloss_task_{}".format(task_id), total_actor_rloss, epoch)
        # self.writer.add_scalar("Global_loss/entropy_rloss_task_{}".format(task_id), total_entropy_rloss, epoch)
        return total_actor_closs, total_entropy_closs, total_actor_rloss, total_entropy_rloss

    def save_theta(self):
        for agent in self.agents:
            agent.save_theta()
    
    def train(self, epochs: int):
        start_time = time.time()
        for epoch in range(1, epochs+1):
            total_actor_closs = 0
            total_entropy_closs = 0
            total_actor_rloss = 0
            total_entropy_rloss = 0
            for  t, task in enumerate(self.tasks):
                self.save_theta()
                
                # t1 = time.time()
                self.sample(epoch=epoch, task_id=t) #* 新更新后的策略参数，取一串轨迹用于验证
                # print(time.time()-t1)

                # t2 = time.time()
                actor_closs, entropy_closs, actor_rloss, entropy_rloss = self.train_ppo(epoch=epoch, task_id=t) #* 计算adv
                total_actor_closs += actor_closs
                total_entropy_closs += entropy_closs
                total_actor_rloss += actor_rloss
                total_entropy_rloss += entropy_rloss
                # print(time.time()-t2)
                
                # t3 = time.time()
                self.adapt(epoch=epoch)
                # print(time.time()-t3)
                
            self.writer.add_scalar("Global_loss/actor_closs", total_actor_closs/self.task_num, epoch)
            self.writer.add_scalar("Global_loss/entropy_closs", total_entropy_closs/self.task_num, epoch)
            self.writer.add_scalar("Global_loss/actor_rloss", total_actor_rloss/self.task_num, epoch)
            self.writer.add_scalar("Global_loss/entropy_rloss", total_entropy_rloss/self.task_num, epoch)
            if epoch % self.log_interval == 0:
                print(
                        'Episode {} \t Total reward: {:.3f} \t Average reward: {:.3f} \t Total best reward: {:.3f} \t Average best reward: {:.3f}'.format(
                                epoch, self.global_total_reward, self.global_total_reward/self.agent_num, self.total_best_reward, self.total_best_reward/self.agent_num
                            )
                    )
            if epoch % self.save_interval == 0:
                for i, agent in enumerate(self.agents):
                    agent.save(self.args.path + "/agent_{}_{}_{}".format(i, 'meta', self.mode))
            self.writer.add_scalar("Global/time_step", (self.iter + 1) / (time.time() - start_time), epoch)
            self.iter += 1
            
    def sample(self, epoch, task_id):
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
                                share_obs[e], best=False
                            )
                # Routing decision
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if activate_to_ract[e][i]:
                        active_to_rpush[e][agent_i] = True
                        self.raction_n[e][agent_i] = choose_raction(
                                self.args, e, self.agents[agent_i], 
                                obs_feature_n[e][agent_i], obs_n[e][agent_i], obs_mask_n[e][agent_i],
                                global_cs_feature[e], share_obs[e], best=False
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
                            done_n[e][agent_i]
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
                            done_n[e][agent_i]
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
                        self.writer.add_scalar("Single_Env/reward_{}_agent_{}_task_{}".format(e, j, task_id), self.agents_total_reward[e][j], epoch)

                    if total_reward > self.total_best_reward: # Calclulate total reward
                        self.total_best_reward = total_reward
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
