# TSPLIB TSP Solver Framework

本项目提供一个基于 `tsplib95` 的 TSP 求解器代码框架，支持遗传算法（GA）、
蚁群算法（ACO）和离散交换序列粒子群算法（PSO）。算法通过同一个
`TSPInstance` 接口读取 TSPLIB 的边权，因此坐标距离（EUC_2D、GEO、ATT 等）
和显式距离矩阵均由 `tsplib95` 按文件定义解释。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 浏览数据集

```powershell
python -m tsp_solver list
python -m tsp_solver list --dataset-dir path\to\tsplib-data
```

默认数据目录为项目根目录下的 `dataset\`。只有 `.tsp` 文件作为问题实例；
例如 `.opt.tour` 文件不会被当成实例载入。

## 运行算法

```powershell
python -m tsp_solver solve dataset\berlin52.tsp --algorithm ga
python -m tsp_solver solve dataset\berlin52.tsp --algorithm aco
python -m tsp_solver solve dataset\berlin52.tsp --algorithm pso
```

命令输出 JSON 格式的算法名称、闭合路径、路径长度和迭代次数。算法参数不从
命令行传入；需要调整参数时，直接修改对应算法文件中的配置默认值：
`ga.py` 的 `GAConfig`、`aco.py` 的 `ACOConfig`、`pso.py` 的 `PSOConfig`。
该命令会实际执行所选算法；这里仅提供求解框架，不预先运行或承诺数据集的
具体解或最优性。

## Python API

```python
from tsp_solver import load_instance
from tsp_solver.algorithms import GeneticAlgorithm

instance = load_instance("dataset/berlin52.tsp")
result = GeneticAlgorithm().solve(instance)
print(result.distance, result.closed_tour)
```

也可以用 `discover_instances("dataset")` 递归获取所有 `.tsp` 实例路径。

## 算法模块

- `tsp_solver/instance.py`：TSPLIB 文件载入、节点集合和 TSPLIB 边权访问。
- `tsp_solver/algorithms/ga.py`：排列编码、锦标赛选择、顺序交叉、交换变异。
- `tsp_solver/algorithms/aco.py`：基于信息素和距离启发因子的路径构造及更新。
- `tsp_solver/algorithms/pso.py`：排列粒子和由交换操作构成的离散速度。
- `tsp_solver/result.py`：统一结果数据结构。

内存和运行成本随实例规模、算法参数显著增加。当前 ACO 实现维护稠密信息素
矩阵，适合小型到中型实例；面对超大实例，应先实现稀疏候选边/邻居策略，并
单独评估内存和运行时间。

## 测试

```powershell
python -m unittest discover -s tests
```