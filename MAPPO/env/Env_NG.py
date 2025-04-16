'''
Author: CQZ
Date: 2025-04-10 19:43:38
Company: SEU
'''
import numpy as np
from env.Env_Base import Env_base

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Env_NG(Env_base):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, scenario, seed=None, ps=False):
        super().__init__(scenario, seed, ps)
