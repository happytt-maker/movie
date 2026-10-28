# legacy/big_data —— 旧模型（movielens_mf.pt）的训练脚本

这个目录存放**前一个项目**（Big_Data）里训练评分矩阵分解的原始脚本，用来交代
`models/movielens_mf.pt` 的来历，保证"旧模型热启动"这一步可追溯、可复现。

## 文件

| 文件 | 说明 |
|---|---|
| `Big_Data.py` | 评分矩阵分解（PyTorch）的训练与推荐脚本，`models/movielens_mf.pt` 就是它训练出来的 |

## 与原始脚本的差异

**只改了两处默认路径**，训练逻辑、模型结构、超参数、随机种子完全未动：

```python
# 原始版本
DEFAULT_DATA_DIR = r"E:\ml-32m\ml-32m"
DEFAULT_MODEL_PATH = "movielens_mf.pt"

# 本项目版本（相对项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "ml-32m")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "movielens_mf.pt")
```

## 与原项目的关系

`movielens_mf.pt` 是**评分预测**模型，不是排序模型：

| 项 | 值 |
|---|---|
| 结构 | 用户 embedding（200,948 × 64）+ 电影 embedding（84,432 × 64）+ 用户/电影偏置 + 全局偏置 |
| 训练目标 | 评分回归（MSE），`r̂ = <p_u, q_i> + b_u + b_i + μ` |
| 划分 | **随机 8:1:1**（`--seed 42`，`np.random.default_rng` 打乱后切分） |
| 结果 | 训练 4 轮，验证集 RMSE 0.7640 |

## 为什么本项目不直接用它打分

任务书要求"按每位用户的时间戳前 80% / 中 10% / 后 10%"划分。旧模型用的是**随机划分**，
测试期评分在训练时已经出现过，直接拿来预测就属于数据泄漏（老师明确"存在数据泄漏的结果不计入
核心指标成绩"）。

所以本项目只继承它的 embedding 与偏置作为初始化，换成 BPR 成对排序目标、在时间划分上重新训练，
见 `src/ranking/bpr.py`：

```bash
# 热启动（默认读 models/movielens_mf.pt）
python -m src.ranking.bpr --mode all --epochs 4 --model-name bpr_mf

# 对照组：随机初始化
python -m src.ranking.bpr --mode all --epochs 4 --model-name bpr_scratch --no-init
```

对照结果：热启动 NDCG@10 = 0.8562，随机初始化 0.8028，**相对提升 6.6%**。

## 如需重新训练旧模型

```bash
cd legacy/big_data
python Big_Data.py --epochs 4 --embedding-dim 64 --seed 42
# 产物写入 models/movielens_mf.pt（会覆盖现有文件，注意先备份）
```

> 提醒：重新训练得到的是评分回归模型，验证指标是 RMSE，与本项目的 NDCG 不可直接比较；
> 它在本项目中的唯一用途是提供 BPR 的初始 embedding。