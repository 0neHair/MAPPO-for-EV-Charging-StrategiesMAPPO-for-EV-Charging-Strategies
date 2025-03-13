'''
Author: CQZ
Date: 2024-12-01 11:00:41
Company: SEU
'''
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GA import GA_tqdm

from env.EV_Sce_Env import EV_Sce_Env
from run_env import run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="SY_4")
    parser.add_argument("--filename", type=str, default="T1")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--size_pop", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--prob_mut", type=float, default=0.05)
    
    arguments = parser.parse_args()
    return arguments

args = parse_args()
if args.randomize:
    seed = args.seed
    print("Random Seed: {}".format(seed))
    np.random.seed(seed)

env = EV_Sce_Env(sce_name=args.sce_name, seed=args.seed, ps=False)

pos_num = env.map_adj.shape[0]

caction_list = env.caction_list # 可选充电动作列表
# cact_dim = ve_matrix.shape[0]
cat_max = len(caction_list)-1
# ract_dim = ve_matrix.shape[1]
agent_num = env.agent_num # 智能体数量
num_cs = env.num_cs # 充电站数量

def obj_func(x):
    # print(x)
    x = np.array(x)
    cact_pi = []
    ract_pi = []
    i = 0
    for _ in range(agent_num):
        cact_pi.append(x[i:i+pos_num])
        i += pos_num
    for _ in range(agent_num):
        ract_pi.append(x[i:i+pos_num*pos_num].reshape([pos_num, pos_num]))
        i += pos_num*pos_num
    
    reward = run(cact_pi, ract_pi, env, agent_num, args)
    return -reward

# cact_pi = [[0, 13], [0, 13]] * agent_num 在每个点上的充电动作编号，整数
# ract_pi = [
#     [
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0]
#     ],
#     [
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0]
#     ]
# ] * agent_num 在节点选择路径的策略，0~1，会在运行环境时根据拓扑进行掩码，以保证满足路径约束

if args.train:
    n_dim = pos_num*agent_num+pos_num*pos_num*agent_num
    ga = GA_tqdm(
            func=obj_func, 
            n_dim=n_dim, 
            lb=[0]*(pos_num*agent_num) + [1e-5]*(pos_num*pos_num*agent_num), # type: ignore
            ub=[cat_max]*(pos_num*agent_num) + [1]*(pos_num*pos_num*agent_num), # type: ignore
            size_pop=args.size_pop, # 种群数量
            max_iter=args.max_iter, # 迭代次数
            prob_mut=args.prob_mut, # 变异系数
            precision=[1]*(pos_num*agent_num) + [0.0001]*(pos_num*pos_num*agent_num), # 精度，1则为整数 # type: ignore
            # constraint_ueq=constraint
        )

    best_x, best_y = ga.run()
    # print('Best_X: ', best_x)
    print('best reward: {:.5f} \t best average reward: {:.5f}'.format(best_y[0], best_y[0]/agent_num))
    act_pd = pd.DataFrame(best_x, columns=["action"])
    act_pd.to_csv('action/{}_{}.csv'.format(args.sce_name, args.filename), index=False)

    y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y_history.index, y_history.values, '.', color='red')
    # ax[0].set_ylim(0, 50)
    ax[1].plot(y_history.index, y_history.min(axis=1).cummin(), color='blue')
    # ax[1].set_ylim(0, 50)
    plt.savefig('logs/{}_{}.png'.format(args.sce_name, args.filename), dpi=300)
    y_history.to_csv('logs/{}_{}.csv'.format(args.sce_name, args.filename), index=False)
else:
    act_pd = pd.read_csv('action/{}_{}.csv'.format(args.sce_name, args.filename))
    act = np.array(act_pd['action'])

    total_reward = obj_func(act)
    print('Reward: {:.5f} \t Average Reward: {:.5f}'.format(total_reward, total_reward/agent_num))

