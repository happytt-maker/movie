# 基于 MovieLens 32M 的分布式增量电影推荐系统

> 大数据技术课程项目 · NDCG@10 提升挑战
> **最终结果：NDCG@10 = 0.8894，相对热门基线（0.7363）提升 +20.8%**，达到任务书"不低于 10%"的要求。

本项目把 MovieLens 32M 的评分数据做成一条完整流水线：数据划分与候选集 → 热门基线 → 相似度与
LSH 召回 → Spark 分布式矩阵分解 → 评分流式增量处理 → 二部图链接分析 → 召回-特征-排序集成，
六个任务全部实现、全部可复现、每项都有独立结论（包括两个负面结论）。

---

## 目录

- [1. 结果总览](#1-结果总览)
- [2. 环境准备](#2-环境准备)
- [3. 数据准备](#3-数据准备)
- [4. 快速开始](#4-快速开始)
- [5. 六个任务](#5-六个任务)
- [6. 配置文件说明](#6-配置文件说明)
- [7. 模型说明（含旧模型热启动）](#7-模型说明含旧模型热启动)
- [8. 输出与结果文件](#8-输出与结果文件)
- [9. 目录结构](#9-目录结构)
- [10. 常见问题](#10-常见问题)
- [11. 已知限制](#11-已知限制)

---

## 1. 结果总览

数据规模：32,000,204 条评分、200,948 位用户、84,432 部电影。
协议：每位用户按时间戳 8:1:1 划分，正反馈 = `rating ≥ 4`，测试候选集 = 正样本 + 100 个固定负样本（种子 20260907）。

| 模型 | NDCG@10 | Recall@10 | MRR@10 | 相对热门 | 对应任务 |
|---|---|---|---|---|---|
| 热门基线 Popularity | 0.7363 | 0.7291 | 0.7687 | — | 任务一 |
| Spark ALS（20k 用户抽样） | 0.7896 | — | — | — | 任务三 |
| BPR（随机初始化对照） | 0.8028 | 0.7840 | 0.8222 | +9.0% | 任务六 |
| BPR（旧模型热启动） | 0.8562 | 0.8191 | 0.8811 | +16.3% | 任务六 |
| **LightGBM 集成排序（最终）** | **0.8894** | **0.8301** | — | **+20.8%** | 任务六 |

三条主要结论：

1. **训练目标必须和评价指标对齐**：同为矩阵分解，评分回归 + 热门补全只有 0.736 量级，换成 BPR
   成对排序目标直接到 0.856。
2. **旧模型值得复用**：只继承它的 embedding、换掉评分回归头再续训 4 轮，比随机初始化高 6.6%
   （0.8562 vs 0.8028），且收敛更快。
3. **特征融合仍有增益**：LightGBM 在 BPR 分数上融合类型、时序、活跃度等 9 个特征，再提升 3.9%。

消融实验（验证各特征贡献）：

| 配置 | NDCG@10 | 相对完整模型 |
|---|---|---|
| 完整模型（9 特征） | 0.8894 | — |
| 去掉 BPR 分数 | 0.8417 | −5.4% |
| 去掉类型亲和度 | 0.8727 | −1.9% |

---

## 2. 环境准备

| 项 | 要求 | 本项目验证环境 |
|---|---|---|
| Python | 3.9+ | 3.9.25 |
| PyTorch | 2.0+（有 GPU 自动用 GPU，无则 CPU） | 2.8.0+cu126 |
| LightGBM | 4.0+ | 4.5.0 |
| PySpark | 3.5+（仅任务三需要） | 3.5.3 |
| Java | 8/11/17（Spark 需要） | OpenJDK 17 |
| 其它 | numpy / pandas / scipy / scikit-learn / networkx / datasketch / pyyaml | 见 requirements.txt |
| 硬件 | 16 GB 内存可跑全流程；GPU 非必需（4 GB 显存足够） | 16 核 / 16 GB / RTX 3050 4 GB |

```bash
pip install -r requirements.txt
# PyTorch 建议按本机 CUDA 版本单独安装：https://pytorch.org/get-started/locally/
```

安装后自检：

```bash
python -c "import torch, lightgbm, scipy, networkx; print(torch.__version__, lightgbm.__version__, torch.cuda.is_available())"
```

---

## 3. 数据准备

把 MovieLens 32M 解压后的文件放在 **`data/ml-32m/`** 下（相对项目根目录）：

```
data/ml-32m/
├── ratings.csv     # 必需，约 836 MB，32,000,204 行
├── movies.csv      # 必需，含 genres 字段
├── tags.csv        # 可选
└── links.csv       # 可选
```

只用到 `ratings.csv`（用户、电影、评分、时间戳）和 `movies.csv`（电影类型）。
数据目录已列入 `.gitignore`，不会被提交。

> 约定：**所有路径都相对项目根目录**，由 `src/data_io.py` 的 `resolve()` 统一解析，
> 因此从任何目录执行脚本都能找到同一份数据，不需要修改代码里的路径。
> 如果数据放在别处，复制 `configs/local.example.yaml` 为 `configs/local.yaml` 改路径即可。

---

## 4. 快速开始

### 4.1 一键全流程（推荐）

```bash
python scripts/run_all.py
```

默认读取 `configs/base.yaml`，依次执行：数据划分 → 热门基线 → 相似度/LSH → 二部图 → 流式回放
→ BPR 训练打分 → 特征构建 → LightGBM 排序，最后打印各步骤耗时和模型指标汇总。
单机约 15 分钟（不含 Spark）。

可选参数：

```bash
python scripts/run_all.py --skip knn,graph        # 跳过耗时的步骤
python scripts/run_all.py --only prepare,bpr      # 只跑指定步骤
```

### 4.2 分步执行

```bash
# ① 数据：质量检查 + 时间划分 + 候选集（约 3 分钟）
python scripts/prepare_data.py

# ② 热门基线（任务一）
python -m src.baseline.popularity

# ③ 精确相似度 + LSH（任务二，约 8 分钟）
python -m src.similarity.itemknn --head 3000 --k 20

# ④ 二部图链接分析（任务五，约 2 分钟）
python -m src.graph.ppr --users 20000

# ⑤ 流式增量处理（任务四，约 1 分钟）
python -m src.streaming.replay --split test

# ⑥ BPR 召回：默认从 models/movielens_mf.pt 热启动（约 6 分钟）
python -m src.ranking.bpr --mode all --epochs 4
python -m src.ranking.bpr --mode predict --model-name bpr_mf

# ⑦ 特征 + 排序（任务六，约 8 分钟）
python scripts/build_features.py --split valid
python scripts/build_features.py --split test
python scripts/train_ranker.py --name ranker_full

# ⑧ 评估
python scripts/evaluate.py --pred results/top10_ranker_full.csv --name ranker_full
```

### 4.3 Spark 任务（任务三）

```bash
# 先导出 Spark 输入（本地完成，保证与时间划分一致）
python scripts/export_for_spark.py --sample-users 20000

# 再提交 Spark 作业
spark-submit --master "local[4]" --driver-memory 6g scripts/spark_als.py \
    --train data/spark/train_positive.csv \
    --candidates data/spark/candidates_sample.csv

# 扩展性实验：25% / 50% / 100% 数据规模
spark-submit --master "local[4]" --driver-memory 6g scripts/spark_als.py \
    --scales 0.25,0.5,1.0 --iters 3
```

---

## 5. 六个任务

### 任务一 · 数据检查与热门基线

- **实现**：`scripts/prepare_data.py` + `src/baseline/popularity.py`
- **做了什么**：缺失值/重复对/评分分布/时间范围检查；四种热度口径（交互次数、对数、时间加权、
  贝叶斯平均）；输出质量报告与划分产物。
- **结果**：热门基线 NDCG@10 = **0.7363**、Recall@10 = 0.7291。
- **结论**：热门基线是不建模的下界，也是检验候选集与指标实现是否正确的标尺。

### 任务二 · 精确相似度与 LSH 近似近邻

- **实现**：`src/similarity/itemknn.py`
- **做了什么**：全量 84,432² 余弦矩阵需要 7.1×10⁹ 个元素，单机不可行，因此在"训练交互数前
  3,000 的头部条目"上做精确 cosine top-K，再用 MinHash-LSH（datasketch，8 个哈希、阈值 0.3）
  做近似召回，同口径比较召回率、候选规模与耗时。
- **结果**：精确 top-20 耗时 30.4 s；LSH 耗时 420.9 s、平均候选 23.9 个、**recall@20 = 0.148**。
- **结论（负面结果）**：该数据的相似度分布很扁平，阈值 0.3 会过滤掉绝大多数真正相似的对。
  正确做法是按 Jaccard 分布自适应选阈值、增加 bands，或改用加权 MinHash。这个结果如实保留，
  因为它说明"近似算法必须按数据分布调参"。

### 任务三 · Spark 分布式矩阵分解

- **实现**：`scripts/spark_als.py` + `scripts/export_for_spark.py`
- **做了什么**：Spark 3.5.3 + MLlib ALS（implicit，rank=64、alpha=10、regParam=0.05、nonnegative），
  在 12,905,652 条正反馈上训练；分阶段记录耗时；抽样 20k 用户评估；支持数据规模扩展实验。
- **结果**：10 轮迭代训练 564.2 s，候选打分 28.4 s，NDCG@10 = **0.7896**（20k 用户抽样）。
- **结论**：Spark 版 ALS 与单机 BPR（0.8562）有差距，主要差别在目标函数（点式 vs 成对）而不是
  分布式与否；ALS 更适合作为大规模场景下的召回补充。

### 任务四 · 流式增量处理

- **实现**：`src/streaming/replay.py`
- **做了什么**：把测试期 3,290,968 条评分按时间戳回放成事件流，单线程处理；每个事件同时更新
  ① Bernoulli 采样 + 蓄水池采样，② 24 小时滚动窗口聚合（事件量、均分、类型分布），
  ③ 用户类型画像增量更新（支持按天衰减）；最后与批量计算做一致性校验。
- **结果**：41.7 s 处理完，吞吐 **78,939 事件/秒**，画像更新 **1.73 µs/次**，共 **9,678 个窗口**，
  画像计数与批量结果完全一致。
- **结论**：增量状态机能做到"事件到达即更新"，不需要全量重算。

### 任务五 · 二部图链接分析与社区特征

- **实现**：`src/graph/ppr.py`
- **做了什么**：20,000 用户 × 23,324 电影、130 万条边的二部图（43,324 节点、平均度 60.3）；
  幂迭代 PageRank（0.1 s）；Louvain 社区划分（59 个社区）；用"用户历史社区偏好"构造社区特征；
  在同一批用户、同一候选集上对比三种排序。
- **结果**：热门 0.7453、PageRank 0.7451、**PageRank × 社区匹配 0.7791**。
- **结论**：二部图上的 PageRank 主要由度数决定，单独用几乎等价于热门；但社区特征捕捉到了
  "用户群 ↔ 电影群"的结构对应关系，带来 **+4.5%**，这才是图信号的价值所在。

### 任务六 · 召回-特征-排序集成

- **实现**：`src/ranking/bpr.py`（召回）+ `scripts/build_features.py`（特征）+ `scripts/train_ranker.py`（排序）
- **召回**：BPR-MF（dim=64、成对损失、每正样本 4 个负样本），从旧评分模型的 embedding 热启动续训 4 轮。
- **特征（9 维）**：BPR 分数、条目交互数（计数/对数）、条目平均分、条目贝叶斯均分、条目时间趋势、
  用户活跃度、用户最近活跃时间、类型亲和度（用户画像与条目类型向量的余弦）。
- **排序**：LightGBM LambdaRank，8 万用户的候选行训练，验证集早停，测试集只评估一次。
- **结果**：BPR 召回 0.8562 → 集成排序 **0.8894**；消融显示 BPR 分数贡献 5.4%、类型亲和度 1.9%。

---

## 6. 配置文件说明

`configs/base.yaml` 是唯一的实验协议文件，**改它就会改变实验口径**，复现时不要动：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `seed` | 全局随机种子（负采样、BPR、LightGBM、Louvain、流式） | 20260907 |
| `positive_threshold` | 正反馈阈值 | 4 |
| `split.train/valid/test` | 每用户按时间戳的划分比例 | 0.8 / 0.1 / 0.1 |
| `candidates.negatives_per_user` | 每个用户的固定负样本数 | 100 |
| `candidates.seed` | 负采样种子（任务书规定） | 20260907 |
| `metrics.primary` | 主指标 | ndcg@10 |
| `paths.raw_dir` | 原始数据目录 | data/ml-32m |
| `paths.processed_dir` | 划分与候选集 | data/processed |
| `paths.features_dir` | 排序特征 | data/features |
| `paths.models_dir` | 模型权重 | models |
| `paths.results_dir` | 指标与 Top-10 | results |
| `target.min_ndcg_improvement` | 相对热门的提升目标 | 0.10 |

所有路径都相对项目根目录。需要指到项目外时，复制 `configs/local.example.yaml` 为
`configs/local.yaml` 修改，运行时加 `--config configs/local.yaml` 即可。

---

## 7. 模型说明（含旧模型热启动）

### 7.1 项目内的模型文件

| 文件 | 大小 | 作用 | 是否随项目提供 |
|---|---|---|---|
| `models/movielens_mf.pt` | 79 MB | **旧评分矩阵分解权重**，作为 BPR 热启动的初始 embedding | 是（必需） |
| `models/bpr_mf.pt` | 70 MB | 本项目训练出的 BPR 模型（热启动版） | 否，可重新生成 |
| `models/bpr_scratch.pt` | 70 MB | 对照组：随机初始化的 BPR | 否，可重新生成 |
| `models/als_factors.npz` | 62 MB | Spark ALS 的用户/电影因子 | 否，可重新生成 |

### 7.2 旧模型的结构与来历

`movielens_mf.pt` 来自前一个项目（Big_Data），是评分预测模型：

- 结构：用户 embedding（200,948 × 64）+ 电影 embedding（84,432 × 64）+ 用户/电影偏置
- 训练：4 轮，验证集 RMSE 0.7640
- **注意**：它是在**随机划分**上训练的，与任务书要求的"按用户时间戳划分"不一致

### 7.3 为什么只复用 embedding，而不是直接拿来打分

直接用它预测存在数据泄漏风险（随机划分让测试期评分参与了训练），老师明确"存在数据泄漏的结果
不计入核心指标成绩"。因此本项目只继承它的 embedding 作为初始化，换成 BPR 成对目标、在时间
划分上重新训练。对照实验证明这一步有实打实的收益：

```bash
# 热启动（默认，读 models/movielens_mf.pt）
python -m src.ranking.bpr --mode all --epochs 4 --model-name bpr_mf

# 对照组：随机初始化
python -m src.ranking.bpr --mode all --epochs 4 --model-name bpr_scratch --no-init
```

| 初始化方式 | NDCG@10 | 说明 |
|---|---|---|
| 随机初始化 | 0.8028 | 4 轮从零训练 |
| 旧模型热启动 | **0.8562** | 同样 4 轮，**+6.6%** |

如果 `models/movielens_mf.pt` 不存在，代码会打印提示并自动退回随机初始化，不影响流程跑通。

---

## 8. 输出与结果文件

### 8.1 Top-10 预测

`results/top10_<模型名>.csv`，每行一位用户的一条推荐：

```csv
userId,rank,movieId,score
1,1,318,4.732
1,2,296,4.589
```

### 8.2 指标

`results/metrics_<模型名>.json`：

```json
{
  "model": "ranker_full",
  "features": ["pop_log", "item_mean", "item_bayes", "item_recency", "item_trend",
               "user_activity", "user_last_activity", "genre_affinity", "bpr"],
  "users": 191007,
  "ndcg@10": 0.88938,
  "recall@10": 0.83011,
  "best_iteration": 1,
  "pred_file": "top10_ranker_full.csv"
}
```

### 8.3 专题报告

| 文件 | 内容 |
|---|---|
| `results/als_report.json` | Spark ALS 耗时、数据规模、NDCG |
| `results/streaming_report.json` | 流式吞吐、窗口数、画像更新耗时、一致性校验 |
| `results/graph_report.json` | 图规模、PageRank/Louvain 耗时、三种排序的 NDCG |
| `results/similarity_lsh.json` | 精确 top-K 与 LSH 的召回率、耗时、候选规模 |
| `results/importance_ranker_full.json` | LightGBM 特征重要性 |

### 8.4 数据产物

- `data/processed/train.npz`、`valid.npz`、`test.npz`：划分后的评分表
- `data/processed/candidates_test.npz`、`candidates_valid.npz`：候选集（含 label）
- `data/processed/vocab.npz`：userId / movieId 词表
- `data/features/valid.npz`、`test.npz`：逐候选行的排序特征

---

## 9. 目录结构

```
project/
├── configs/
│   ├── base.yaml               实验协议（划分、种子、路径）——不要改
│   └── local.example.yaml      数据放到项目外时的覆盖配置示例
├── data/
│   ├── ml-32m/                 原始数据（需自行下载，已 gitignore）
│   ├── processed/              时间划分、候选集、词表
│   ├── features/               排序特征
│   └── spark/                  Spark 输入导出（可重新生成）
├── docs/
│   ├── 项目报告.md / .docx      完整项目报告
│   ├── 演示材料.pptx            答辩幻灯片（8 页）
│   ├── 个人贡献说明.md
│   └── 答辩记录.md              预设问答 Q1–Q10
├── models/                     模型权重（movielens_mf.pt 随项目提供）
├── results/                    Top-10、指标、专题报告
├── scripts/
│   ├── prepare_data.py         质量检查 + 时间划分 + 候选集
│   ├── build_features.py       候选行级排序特征
│   ├── train_ranker.py         LightGBM LambdaRank + 消融
│   ├── evaluate.py             NDCG/Recall/MRR 评估
│   ├── export_for_spark.py     导出 Spark 输入
│   ├── spark_als.py            Spark MLlib ALS（需 spark-submit）
│   ├── run_all.py              一键全流程
│   ├── build_report_docx.py    报告 md → docx
│   └── build_slides.py         生成演示 pptx
└── src/
    ├── data_io.py              配置加载、路径解析、npz 读写
    ├── metrics.py              NDCG / Recall / MRR
    ├── candidate_set.py        候选集切片与 Top-K
    ├── features.py             类型矩阵、用户画像、统计特征
    ├── baseline/popularity.py  任务一
    ├── similarity/itemknn.py   任务二
    ├── streaming/replay.py     任务四
    ├── graph/ppr.py            任务五
    └── ranking/bpr.py          任务六（召回）
```

---

## 10. 常见问题

**Q1：跑 Spark 报 `Bad pathname`？**
JVM 无法解析含中文的项目路径。把 `scripts/spark_als.py` 复制到纯 ASCII 路径（例如
`C:\temp\spark_als.py`）再 `spark-submit`，数据路径也建议放 ASCII 目录。

**Q2：在 Python 里 `SparkSession.builder.getOrCreate()` 一直卡住？**
部分 Windows + 沙箱环境会挂起。改用 `spark-submit` 启动脚本即可（本项目就是这么做的）。

**Q3：`pip install pyspark` 很慢或失败？**
可换国内镜像：`pip install pyspark -i https://mirrors.aliyun.com/pypi/simple/`。
只跑单机部分（任务一、二、四、五、六）不需要 Spark。

**Q4：显存不足（4 GB 显卡）？**
BPR 默认 `--batch 16384`、dim 64，4 GB 足够；不够时减小 `--batch` 或加 `--device cpu`。

**Q5：`data/processed` 能删吗？**
能，重新执行 `python scripts/prepare_data.py` 即可（约 3 分钟）。

**Q6：结果数字和 README 对不上？**
先确认没有改过 `configs/base.yaml` 的种子与划分比例；再确认用的是同一份 `data/ml-32m`。
LightGBM 与 PyTorch 的版本差异会带来千分位波动，属于正常范围。

---

## 11. 已知限制

1. **LSH 参数未调优**：recall@20 只有 0.148，需要按 Jaccard 分布自适应选阈值或增加 bands。
2. **图实验在 20k 用户子图上完成**：全量 Louvain 单机内存与时间不可接受，结论外推需谨慎。
3. **负样本口径由本项目定义**：任务书未提供教师端负采样脚本的池定义，本项目用"训练交互 ≥ 5
   的条目 + 排除用户已交互条目"；换口径会改变绝对分数，但不改变模型间相对排序。
4. **流式处理是单机回放**：未接 Kafka/Structured Streaming，状态机定义可直接迁移到
   `mapGroupsWithState`。
5. **BPR 早停在 1–2 轮**：排序指标很快进入平台，后续轮次只带来波动，因此按验证集选点而不是
   一味增加轮次。

---

## 12. 文档

| 文件 | 说明 |
|---|---|
| `docs/项目报告.md` / `.docx` | 完整报告：问题定义、数据分析、架构、算法、实验、结果、局限 |
| `docs/演示材料.pptx` | 8 页答辩幻灯片 |
| `docs/个人贡献说明.md` | 承担工作与关键决策 |
| `docs/答辩记录.md` | 预设答辩问答 Q1–Q10 |

重新生成文档：

```bash
python scripts/build_report_docx.py   # 需要 python-docx
python scripts/build_slides.py        # 需要 python-pptx
```