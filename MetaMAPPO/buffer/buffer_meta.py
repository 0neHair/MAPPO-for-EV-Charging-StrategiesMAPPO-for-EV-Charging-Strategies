
import torch
import numpy as np
from .buffer_base import RolloutBuffer

class Meta_RolloutBuffer(RolloutBuffer):
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
        # 充电相关
        self.norm_cadv = np.zeros((self.steps, num_env), dtype=np.float32)
        # 路径相关
        self.norm_radv = np.zeros((self.rsteps, num_env), dtype=np.float32)

    def push_adv(self, norm_cadv, norm_radv):
        self.norm_cadv[:, :] = norm_cadv
        self.norm_radv[:, :] = norm_radv

    def pull_all(self):
        return (
            torch.tensor(self.state, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.caction, dtype=torch.float32).to(self.device),
            torch.tensor(self.clog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.creward, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.cdone, dtype=torch.float32).to(self.device),
            torch.tensor(self.norm_cadv, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_rstate, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.raction, dtype=torch.float32).to(self.device),
            torch.tensor(self.raction_mask, dtype=torch.float32).to(self.device),
            torch.tensor(self.rlog_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.rreward, dtype=torch.float32).to(self.device),

            torch.tensor(self.next_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_rstate, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_raction_mask, dtype=torch.float32).to(self.device),
            
            torch.tensor(self.rdone, dtype=torch.float32).to(self.device),
            torch.tensor(self.norm_radv, dtype=torch.float32).to(self.device)
        )