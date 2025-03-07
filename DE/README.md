<!--
 * @Author: CQZ
 * @Date: 2024-09-13 21:27:24
 * @Company: SEU
-->
# Differential Evolution Algorithm (DE)

采用差分进化算法，得到最小化总通行时间的EV联合策略。

1. 运行`DE.py`进行训练，得到的策略保存在`policy_<scenario_name>.csv`中
2. 运行`evaluate.py`进行策略评价

注意：为了方便建模，此处策略为EV到达CS时选择充多少电。
