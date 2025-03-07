import torch
import numpy as np


class RolloutBuffer(object):
    def __init__(self, num_env: int, steps: int, state_shape: tuple, share_shape: tuple, action_shape: tuple, device):
        self.steps = steps
        self.device = device

        self.state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.action = np.zeros((steps, num_env) + action_shape, dtype=np.float32)
        self.log_prob = np.zeros((steps, num_env) + action_shape, dtype=np.float32)
        self.next_state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.next_share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.reward = np.zeros((steps, num_env), dtype=np.float32)
        self.done = np.zeros((steps, num_env), dtype=np.float32)

        self.ptr = 0

    def push(self, reward, next_state, next_share_state, done):
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.next_share_state[self.ptr] = next_share_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.steps

    def push_last_state(self, state, share_state, action, log_prob):
        self.state[self.ptr] = state
        self.share_state[self.ptr] = share_state 
        self.action[self.ptr] = action
        self.log_prob[self.ptr] = log_prob
    
    def pull(self):
        return (
            torch.tensor(self.state, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.action, dtype=torch.float32).to(self.device),
            torch.tensor(self.log_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.reward, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.done, dtype=torch.float32).to(self.device)
        )

    @property
    def full(self):
        return self.ptr == 0
