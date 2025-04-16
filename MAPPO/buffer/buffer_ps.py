
import torch
import numpy as np


class RolloutBufferPS(object):
    def __init__(
            self, 
            steps: int, rsteps: int, num_env: int, 
            state_shape: tuple, share_shape: tuple, caction_shape: tuple, 
            edge_index, 
            obs_features_shape: tuple, global_features_shape: tuple, 
            raction_shape: tuple,
            raction_mask_shape: tuple,
            agent_num: int,
            device
        ):
        self.steps = steps
        self.rsteps = rsteps
        self.device = device
        self.edge_index = edge_index
        self.agent_num = agent_num
        # map_shape = edge_index.shape
        # 充电相关
        self.state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.caction = np.zeros((steps, num_env, agent_num) + caction_shape, dtype=np.float32)
        self.clog_prob = np.zeros((steps, num_env, agent_num) + caction_shape, dtype=np.float32)
        self.next_state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.next_share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.creward = np.zeros((steps, num_env, agent_num), dtype=np.float32)
        self.cdone = np.zeros((steps, num_env, agent_num), dtype=np.float32)
        # 路径相关
        self.rstate = np.zeros((self.rsteps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.share_rstate = np.zeros((self.rsteps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.obs_feature = np.zeros((self.rsteps, num_env, agent_num) + obs_features_shape, dtype=np.float32)
        self.global_cs_feature = np.zeros((self.rsteps, num_env, agent_num) + global_features_shape, dtype=np.float32)
        self.raction = np.zeros((self.rsteps, num_env, agent_num) + raction_shape, dtype=np.float32)
        self.raction_mask = np.zeros((self.rsteps, num_env, agent_num) + raction_mask_shape, dtype=np.float32)
        self.next_raction_mask = np.zeros((self.rsteps, num_env, agent_num) + raction_mask_shape, dtype=np.float32)
        self.rlog_prob = np.zeros((self.rsteps, num_env, agent_num) + raction_shape, dtype=np.float32)
        self.next_rstate = np.zeros((self.rsteps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.next_share_rstate = np.zeros((self.rsteps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.next_obs_feature = np.zeros((self.rsteps, num_env, agent_num) + obs_features_shape, dtype=np.float32)
        self.next_global_cs_feature = np.zeros((self.rsteps, num_env, agent_num) + global_features_shape, dtype=np.float32)
        self.rreward = np.zeros((self.rsteps, num_env, agent_num), dtype=np.float32)
        self.rdone = np.zeros((self.rsteps, num_env, agent_num), dtype=np.float32)
        
        self.cptr = [[0 for _ in range(agent_num)] for __ in range(num_env)]
        self.rptr = [[0 for _ in range(agent_num)] for __ in range(num_env)]

    def cpush(self, reward, next_state, next_share_state, done, env_id, agent_id):
        cptr = self.cptr[env_id][agent_id]
        self.creward[cptr][env_id][agent_id] = reward
        self.next_state[cptr][env_id][agent_id] = next_state
        self.next_share_state[cptr][env_id][agent_id] = next_share_state
        self.cdone[cptr][env_id][agent_id] = done

        self.cptr[env_id][agent_id] = (cptr + 1) % self.steps

    def cpush_last_state(self, state, share_state, action, log_prob, env_id, agent_id):
        cptr = self.cptr[env_id][agent_id]
        self.state[cptr][env_id][agent_id] = state
        self.share_state[cptr][env_id][agent_id] = share_state 
        self.caction[cptr][env_id][agent_id] = action
        self.clog_prob[cptr][env_id][agent_id] = log_prob

    def rpush(self, reward, next_obs_feature, next_global_cs_feature, next_rstate, next_share_rstate, next_action_mask, done, env_id, agent_id):
        rptr = self.rptr[env_id][agent_id]
        self.rreward[rptr][env_id][agent_id] = reward
        self.next_rstate[rptr][env_id][agent_id] = next_rstate
        self.next_share_rstate[rptr][env_id][agent_id] = next_share_rstate
        self.next_obs_feature[rptr][env_id][agent_id] = next_obs_feature
        self.next_global_cs_feature[rptr][env_id][agent_id] = next_global_cs_feature
        self.raction_mask[rptr][env_id][agent_id] = next_action_mask
        self.rdone[rptr][env_id][agent_id] = done

        self.rptr[env_id][agent_id] = (rptr + 1) % self.rsteps

    def rpush_last_state(self, obs_feature, global_cs_feature, rstate, share_rstate, action, action_mask, log_prob, env_id, agent_id):
        rptr = self.rptr[env_id][agent_id]
        self.obs_feature[rptr][env_id][agent_id] = obs_feature
        self.global_cs_feature[rptr][env_id][agent_id] = global_cs_feature
        self.rstate[rptr][env_id][agent_id] = rstate
        self.share_rstate[rptr][env_id][agent_id] = share_rstate 
        self.raction[rptr][env_id][agent_id] = action
        self.raction_mask[rptr][env_id][agent_id] = action_mask
        self.rlog_prob[rptr][env_id][agent_id] = log_prob
    
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
            
            torch.tensor(self.rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_rstate, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.raction, dtype=torch.float32).to(self.device),
            torch.tensor(self.raction_mask, dtype=torch.float32).to(self.device),
            torch.tensor(self.rlog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.rreward, dtype=torch.float32).to(self.device),

            torch.tensor(self.next_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_raction_mask, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.rdone, dtype=torch.float32).to(self.device)
        )

