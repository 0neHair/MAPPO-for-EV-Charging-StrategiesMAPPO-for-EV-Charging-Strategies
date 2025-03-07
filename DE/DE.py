import numpy as np
import pandas as pd
import random

from env.Multi_EV_Env import Multi_EV_Env
from env.EV_Sce_Env import EV_Sce_Env

def evolve(xs, cr, env):
    # 变异
    vs = []
    N = len(xs[0][0])
    for _ in range(len(xs)):
        x = xs.pop(0)
        r1, r2, r3 = random.sample(xs, 3)
        xs.append(x)
        vs.append(r1 + random.random()*(r2-r3))

    # 交叉
    us = []
    for i in range(len(xs)):
        us.append(vs[i] if random.random() < cr else xs[i])
        for j in range(len(xs[i])):
            k = random.randint(0, N-1)
            us[i][j][k] = vs[i][j][k]

    # 选择
    xNext = []
    best_reward = -9999999999999
    best_action = None
    for x, u in zip(xs, us):
        x_, total_reward_x = policy_test(env, x)
        u_, total_reward_u = policy_test(env, u)
        
        xNext.append(x_ if total_reward_x > total_reward_u else u_)
        
        if total_reward_x > best_reward:
            best_reward = total_reward_x
            best_action = x_
        if total_reward_u > best_reward:
            best_reward = total_reward_u
            best_action = u_
    return xNext, best_reward, best_action

def policy_test(env: Multi_EV_Env, actions):
    new_actions = policy_fix(env, actions)
    
    env.reset()
    agent_num = env.agent_num
    action_i = [0 for i in range(agent_num)]
    action_n = np.array([-1.0 for i in range(agent_num)])
    obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
    while True:
        action_n = np.array([-1.0 for i in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                action_n[agent_i] = new_actions[agent_i][action_i[agent_i]]
                action_i[agent_i] += 1
                
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
        if env.agents_active == []:
            break
    
    total_reward = 0
    for agent in env.agents:
        total_reward += agent.total_reward
    env.reset()
    return new_actions, total_reward

def policy_fix(env: Multi_EV_Env, actions):
    route = env.route
    new_actions = []
    for i, agent in enumerate(env.agents):
        agent_action = actions[i]
        new_agent_action = []
        for a in agent_action:
            if a < 0.1:
                new_agent_action.append(0)
            elif a > 1:
                new_agent_action.append(1)
            else:
                new_agent_action.append(a)
        agent_action = new_agent_action.copy()
        back_up = False # 备用方案
        
        agent_SOC = agent.SOC 
        agent_SOC = agent_SOC - route[0]*agent.consume/agent.E_max
        new_agent_action = []
        for j, act in enumerate(agent_action):
            temp_soc = agent_SOC + act
            if temp_soc > 1.0:
                real_act = 1.0 - agent_SOC
                agent_SOC = 1.0
                agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                new_agent_action.append(real_act)
            else:
                temp_soc = temp_soc - route[j+1]*agent.consume/agent.E_max
                if temp_soc < 0.1:
                    real_act = 1.0 - agent_SOC
                    if real_act > 1.0:
                        back_up = True
                        break
                    agent_SOC = 1.0
                    agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                    new_agent_action.append(real_act)
                else:
                    agent_SOC= agent_SOC + act
                    agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                    new_agent_action.append(act)
                    
        if back_up:
            temp_act = []
            agent_SOC = agent.SOC 
            agent_SOC = agent_SOC - route[0]*agent.consume/agent.E_max
            for j, act in enumerate(agent_action):
                real_act = 1.0 - agent_SOC
                agent_SOC = 1.0
                agent_SOC = agent_SOC - route[j+1]*agent.consume/agent.E_max
                temp_act.append(real_act)
            new_agent_action = temp_act.copy()

        new_actions.append(new_agent_action)
    return np.array(new_actions)

def policy_init(env: Multi_EV_Env, N):
    total_actions = []
    for _ in range(N):
        actions = []
        for i, agent in enumerate(env.agents):
            agent_act = [random.random() for _ in range(env.num_position-1)]
            actions.append(agent_act.copy())
        total_actions.append(np.array(actions.copy()))
    return total_actions

N = 50
cr = 0.5
nIter = 500
scenario = 'LS1_4P'

np.random.seed(0)
random.seed(0)

env = EV_Sce_Env(scenario, seed=0)
xs = policy_init(env, N)

global_best_reward = -np.inf
global_best_action = None
i = 0
while i <= 3000:
    xs, best_reward, best_action = evolve(xs, cr, env)
    if best_reward > global_best_reward:
        global_best_reward = best_reward
        global_best_action = best_action

        actions = np.array(global_best_action).round(3)
        columns = []
        for j in range(actions.shape[1]):
            columns.append('CS_'+str(j))

        df = pd.DataFrame(actions, columns=columns)
        df['EV'] = range(actions.shape[0])
        df.to_csv('policy_{}.csv'.format(scenario), index=False)

    if i % 10 == 0:
        print('Iter: {} Reward: {} Best Reward: {}'.format(i, best_reward, global_best_reward))

    i += 1
fs = []
xs_new = []
for x in xs:
    act, reward = policy_test(env, x)
    fs.append(reward)
    xs_new.append(act)
xBest = xs_new[np.argmax(fs)]
rBest = np.max(fs)
print("当前最优结果为{}，参数为:".format(rBest))
print(xBest)

if rBest > global_best_reward:
    global_best_reward = rBest
    global_best_action = xBest
    
print("当前最优结果为{}，参数为:".format(global_best_reward))
print(global_best_action)

actions = np.array(global_best_action).round(3)
columns = []
for i in range(actions.shape[1]):
    columns.append('CS_'+str(i))

df = pd.DataFrame(actions, columns=columns)
df['EV'] = range(actions.shape[0])

df.to_csv('DS/policy_{}.csv'.format(scenario), index=False)
