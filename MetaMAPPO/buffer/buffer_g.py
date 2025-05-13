'''
Author: CQZ
Date: 2025-04-10 20:35:29
Company: SEU
'''
import torch
import numpy as np
from .buffer_base import RolloutBuffer

class G_RolloutBuffer(RolloutBuffer):
    def __init__(
            self, 
            steps: int, rsteps: int, num_env: int, 
            state_shape: tuple, share_shape: tuple, caction_shape: tuple, 
            edge_index, 
            obs_features_shape: tuple, global_features_shape: tuple, 
            raction_shape: tuple,
            raction_mask_shape: tuple,
            device
        ):
        super().__init__(
            steps, rsteps, num_env, 
            state_shape, share_shape, caction_shape, 
            edge_index, 
            obs_features_shape, global_features_shape, 
            raction_shape,
            raction_mask_shape,
            device
        )

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
            
            torch.tensor(self.obs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.global_cs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_rstate, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.raction, dtype=torch.float32).to(self.device),
            torch.tensor(self.raction_mask, dtype=torch.float32).to(self.device),
            torch.tensor(self.rlog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.rreward, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.next_obs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_global_cs_feature, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_raction_mask, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.rdone, dtype=torch.float32).to(self.device)
        )

