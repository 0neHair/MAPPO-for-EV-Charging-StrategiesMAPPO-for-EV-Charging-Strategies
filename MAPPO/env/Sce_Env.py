'''
Author: CQZ
Date: 2024-04-03 12:27:22
Company: SEU
'''
from env.scenarios import load

def Sce_Env(sce_name, seed=0, ps=False, mode=None):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see Multi_EV_Env.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    # load scenario from script
    # scenario = load(sce_name + ".py").Scenario(frame=0.01, seed=seed)
    scenario = load(sce_name + ".py").Scenario(frame=1, seed=seed) # frame / 100
    if mode == "GH":
        from env.Env_G import Env_G as Env
    elif mode == 'NGH':
        from env.Env_NG import Env_NG as Env
    elif mode == 'OC':
        from env.Env_OC import Env_OC as Env
    elif mode == 'OR':
        from env.Env_OR import Env_OR as Env
    # create multiagent environment
    env = Env(scenario=scenario, seed=seed, ps=ps)

    return env
