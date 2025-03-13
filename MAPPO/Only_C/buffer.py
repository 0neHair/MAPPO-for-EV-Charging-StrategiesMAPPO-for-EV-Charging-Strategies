'''
Author: CQZ
Date: 2024-09-17 16:11:38
Company: SEU
'''
import torch
import numpy as np


class RolloutBuffer(object):
    def __init__(
        self, 
        steps: int, num_env: int, 
        state_shape: tuple, share_shape: tuple, caction_shape: tuple, 
        edge_index, 
        obs_features_shape: tuple, global_features_shape: tuple, raction_shape: tuple,
        raction_mask_shape: tuple,
        device
        ):
        self.steps = steps
        self.device = device
        self.edge_index = edge_index
        map_shape = edge_index.shape
        # 充电相关
        self.state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.caction = np.zeros((steps, num_env) + caction_shape, dtype=np.float32)
        self.clog_prob = np.zeros((steps, num_env) + caction_shape, dtype=np.float32)
        self.next_state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.next_share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.creward = np.zeros((steps, num_env), dtype=np.float32)
        self.cdone = np.zeros((steps, num_env), dtype=np.float32)
        # 路径相关
        self.rstate = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.share_rstate = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.raction = np.zeros((steps, num_env) + raction_shape, dtype=np.float32)
        self.raction_mask = np.zeros((steps, num_env) + raction_mask_shape, dtype=np.float32)
        self.rlog_prob = np.zeros((steps, num_env) + raction_shape, dtype=np.float32)
        self.next_rstate = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.next_share_rstate = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.rreward = np.zeros((steps, num_env), dtype=np.float32)
        self.rdone = np.zeros((steps, num_env), dtype=np.float32)

        # self.obs_map = np.zeros((steps, num_env) + map_shape, dtype=np.float32)
        # for i in range(steps):
        #     for j in range(num_env):
        #         self.obs_map[i][j] = self.edge_index.copy()
        # self.global_cs_map = self.obs_map.copy()
        # self.next_obs_map = self.obs_map.copy()
        # self.next_global_cs_map = self.obs_map.copy()
        
        self.cptr = [0 for _ in range(num_env)]
        self.rptr = [0 for _ in range(num_env)]

    def cpush(self, reward, next_state, next_share_state, done, env_id):
        cptr = self.cptr[env_id]
        self.creward[cptr][env_id] = reward
        self.next_state[cptr][env_id] = next_state
        self.next_share_state[cptr][env_id] = next_share_state
        self.cdone[cptr][env_id] = done

        self.cptr[env_id] = (cptr + 1) % self.steps

    def cpush_last_state(self, state, share_state, action, log_prob, env_id):
        cptr = self.cptr[env_id]
        self.state[cptr][env_id] = state
        self.share_state[cptr][env_id] = share_state 
        self.caction[cptr][env_id] = action
        self.clog_prob[cptr][env_id] = log_prob

    def rpush(self, reward, next_rstate, next_share_rstate, done, env_id):
        rptr = self.rptr[env_id]
        self.rreward[rptr][env_id] = reward
        self.next_rstate[rptr][env_id] = next_rstate
        self.next_share_rstate[rptr][env_id] = next_share_rstate
        self.rdone[rptr][env_id] = done

        self.rptr[env_id] = (rptr + 1) % self.steps

    def rpush_last_state(self, rstate, share_rstate, action, action_mask, log_prob, env_id):
        rptr = self.rptr[env_id]
        self.rstate[rptr][env_id] = rstate
        self.share_rstate[rptr][env_id] = share_rstate
        self.raction[rptr][env_id] = action
        self.raction_mask[rptr][env_id] = action_mask
        self.rlog_prob[rptr][env_id] = log_prob
    
    def pull(self):
        return (
            torch.tensor(self.state, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.caction, dtype=torch.float32).to(self.device),
            torch.tensor(self.clog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.creward, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.cdone, dtype=torch.float32).to(self.device),
            
            # torch.tensor(self.obs_feature, dtype=torch.float32).to(self.device),
            # torch.tensor(self.global_cs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.raction, dtype=torch.float32).to(self.device),
            torch.tensor(self.raction_mask, dtype=torch.float32).to(self.device),
            torch.tensor(self.rlog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.rreward, dtype=torch.float32).to(self.device),
            # torch.tensor(self.next_obs_feature, dtype=torch.float32).to(self.device),
            # torch.tensor(self.next_global_cs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.rdone, dtype=torch.float32).to(self.device)
        )
    # @property
    # def full(self):
    #     return self.ptr == 0

# class SharedRolloutBuffer(object):
#     def __init__(
#         self, 
#         steps: int, num_env: int, 
#         state_shape: tuple, share_shape: tuple, action_shape: tuple, 
#         agent_num: int,
#         device
#         ):
#         self.steps = steps
#         self.agent_num = agent_num
#         self.device = device

#         self.state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
#         self.share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
#         self.action = np.zeros((steps, num_env, agent_num) + action_shape, dtype=np.float32)
#         self.log_prob = np.zeros((steps, num_env, agent_num) + action_shape, dtype=np.float32)
#         self.next_state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
#         self.next_share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
#         self.reward = np.zeros((steps, num_env, agent_num), dtype=np.float32)
#         self.done = np.zeros((steps, num_env, agent_num), dtype=np.float32)

#         self.ptr = 0

#     def push(self, reward, next_state, next_share_state, done, env_id, agent_id):
#         self.reward[self.ptr][env_id][agent_id] = reward
#         self.next_state[self.ptr][env_id][agent_id] = next_state
#         self.next_share_state[self.ptr][env_id][agent_id] = next_share_state
#         self.done[self.ptr][env_id][agent_id] = done

#         self.ptr = (self.ptr + 1) % self.steps

#     def push_last_state(self, state, share_state, action, log_prob, env_id, agent_id):
#         self.state[self.ptr][env_id][agent_id] = state
#         self.share_state[self.ptr][env_id][agent_id] = share_state 
#         self.action[self.ptr][env_id][agent_id] = action
#         self.log_prob[self.ptr][env_id][agent_id] = log_prob
    
#     def pull(self):
#         return (
#             torch.tensor(self.state, dtype=torch.float32).to(self.device),
#             torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
#             torch.tensor(self.action, dtype=torch.float32).to(self.device),
#             torch.tensor(self.log_prob, dtype=torch.float32).to(self.device),
#             torch.tensor(self.reward, dtype=torch.float32).to(self.device),
#             torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
#             torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
#             torch.tensor(self.done, dtype=torch.float32).to(self.device)
#         )

#     @property
#     def full(self):
#         return self.ptr == 0
