<!--
 * @Author: CQZ
 * @Date: 2024-09-13 16:03:44
 * @Company: SEU
-->
# Environment

* `Multi_EV_Env.py`: 环境运行主程序，包括状态-动作转移、交互和场景可视化等

* `EV_agent.py`: 智能体建模，记录智能体状态、奖励等信息

* `EV_Sce_Env.py`: 场景导入程序，场景文件保存在`scenarios`中

* `test.ipynb`: 环境测试

# Scenarios

* `test`: 简单测试场景，包括
  * `test_1.py`
  * `test_2.py`
  * `test_3.py`

* `LS1`: 连云港-上海高速路径1，包括
  * `LS1_2P.py`: 每个充电站有2个充电桩
  * `LS1_3P.py`: 每个充电站有3个充电桩
  * `LS1_4P.py`: 每个充电站有4个充电桩

* `LS2`: 连云港-上海高速路径2
