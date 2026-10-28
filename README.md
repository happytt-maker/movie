# 基于 MovieLens 的分布式增量电影推荐系统

> 大数据技术课程项目 · NDCG@10 提升挑战
> **结果：NDCG@10 = 0.8894，相对热门基线（0.7363）提升 +20.8%，达到"不低于 10%"的目标。**

## 1. 结果一览

数据：MovieLens 32M（32,000,204 条评分，200,948 用户，84,432 部电影）；协议：按用户时间戳
前 80% 训练 / 中 10% 验证 / 后 10% 测试，正反馈 = `rating >= 4`，候选集 = 测试正样本 + 100 个
固定负样本（种子 20260907），指标 NDCG@10 / Recall@10。

| 模型 | NDCG@10 | Recall@10 | 相对热门 | 所属任务 |
|---|---|---|---|---|
| 热门基线（Popularity） | 0.7363 | 0.7291 | — | 任务一 |
| BPR-MF（随机初始化，4 epoch） | 0.8028 | 0.7840 | +9.0% | 任务六 |
| Spark ALS（20k 用户抽样评估） | 0.7896 | — | — | 任务三 |
| BPR-MF（**旧模型热启动**，4 epoch） | 0.8562 | 0.8191 | +16.3% | 任务六 |
| **LightGBM 集成排序（最终模型）** | **0.8894** | **0.8301** | **+20.8%** | 任务六 |

关键结论：

1. **排序目标比评分目标重要**：同样用矩阵分解，改成 BPR 成对目标后从 0.80 提升到 0.856。
2. **旧模型确实有价值**：用 `Big_Data` 里评分 MF 的 embedding 热启动，比随机初始化高 **6.6%**
   （0.8562 vs 0.8028），省掉大半收敛时间。
3. **集成排序再进一步**：LightGBM 在 BPR 召回分基础上融合类型、时序、热度特征，到 0.8894。

## 2. 快速开始

```bash
# 1) 安装依赖（torch 建议按本机 CUDA 版本单独装）
pip install -r requirements.txt

# 2) 确认原始数据目录（MovieLens 32M 的 ml-32m 解压位置）
#    默认 data/ml-32m，本机用 configs/local.yaml 指向 E:/ml-32m/ml-32m

# 3) 一键跑通全部任务（约 15 分钟，不含 Spark）
python scripts/run_all.py --config configs/local.yaml

# 4) 单独跑 Spark 任务（任务三）
spark-submit --master local[4] --driver-memory 6g scripts/spark_als.py \
    --train data/spark/train_positive.csv \
    --candidates data/spark/candidates_sample.csv
```

单步命令：

```bash
python scripts/prepare_data.py    --config configs/local.yaml      # 划分 + 候选集
python scripts/build_features.py  --config configs/local.yaml --split valid
python -m src.baseline.popularity --config configs/local.yaml      # 任务一
python -m src.similarity.itemknn  --config configs/local.yaml      # 任务二
python -m src.graph.ppr           --config configs/local.yaml      # 任务五
python -m src.streaming.replay    --config configs/local.yaml      # 任务四
python -m src.ranking.bpr         --config configs/local.yaml --mode all \
    --init-from E:/VS_tools/Project/Big_Data/Big_Data/movielens_mf.pt
python scripts/train_ranker.py    --config configs/local.yaml --name ranker_full
python scripts/evaluate.py --pred results/top10_ranker_full.csv --config configs/local.yaml --name ranker_full
```

## 3. 数据与实验协议

严格遵循任务书「五 数据划分与评价协议」，参数集中在 `configs/base.yaml`：

| 项目 | 设定 |
|---|---|
| 正反馈定义 | `rating >= 4` |
| 时间划分 | 每位用户按 timestamp 升序：前 80% 训练 / 中 10% 验证 / 后 10% 测试 |
| 候选集 | 每位测试用户 = 测试正样本 + 100 个固定负样本（种子 20260907） |
| 负样本池 | 训练交互数 ≥ 5 的条目，且排除该用户训练/验证期已看过的电影 |
| 随机种子 | 20260907（全局） |
| 指标 | NDCG@10（主）、Recall@10、MRR@10、单用户延迟、吞吐 |

划分结果：train 25,520,897 / valid 3,188,339 / test 3,290,968 条；测试候选集 20,660,431 行，
覆盖 191,007 个有正样本的测试用户，平均每用户 8.2 个正样本。

> **防泄漏**：训练与特征只使用训练期数据；候选集的负样本从"用户没交互过的条目"中采样；
> 验证集用于模型选择，测试集只在最终评估时使用一次。

## 4. 系统架构

```
数据层    prepare_data.py → data/processed/{train,valid,test}.npz + candidates_*.npz + vocab.npz
            │（时间划分、候选集、统计报告，全部可重复执行）
特征层    build_features.py → data/features/{valid,test}.npz（逐候选行的排序特征）
            │
算法层    baseline/popularity   热门基线（对照）
          similarity/itemknn      精确余弦 + MinHash-LSH 召回（任务二）
          distributed/spark_als   Spark MLlib ALS（任务三）
          streaming/replay        评分流回放、窗口统计、画像增量更新（任务四）
          graph/ppr               二部图 PageRank + Louvain 社区（任务五）
          ranking/bpr             BPR-MF（热启动旧模型）
          ranking/…               LightGBM LambdaRank 集成（任务六）
            │
服务层    CandidateSet.topk → results/top10_<model>.csv（userId,rank,movieId,score）
评价层    evaluate.py → results/metrics_<model>.json（NDCG/Recall/MRR/延迟）
```

## 5. 六个任务的实现要点

**任务一 · 数据检查与热门基线**（`src/baseline/popularity.py`）
缺失值/重复对/评分分布/时间范围检查；热门打分支持 count / log / weighted / 贝叶斯平均四种口径，
默认按训练交互次数排序，NDCG@10 = 0.7363。

**任务二 · 相似度与 LSH**（`src/similarity/itemknn.py`）
全量 84,432² 的精确余弦矩阵约 7.1e9 元素，单机不可行，因此在"训练交互数前 3,000 的头部条目"
上做精确 cosine top-K，再用 MinHash-LSH（datasketch）做近似召回，两者同口径比较召回率、
候选规模与耗时。**实测 LSH 在默认阈值下 recall 偏低（0.148），说明相似度本身分布稀疏，
参数必须按数据分布调；报告里如实给出该结果，并给出阈值/带宽的调参方向。**

**任务三 · Spark 分布式**（`scripts/spark_als.py`）
Spark 3.5.3 + MLlib ALS（implicit，rank=64，alpha=10，10 轮）在 12.9M 条正反馈上训练，
564 s 完成，抽样 20k 用户评估 NDCG@10 = 0.7896；支持 `--scales 0.25,0.5,1.0` 做扩展性实验。

**任务四 · 流式增量处理**（`src/streaming/replay.py`）
按 timestamp 升序回放测试期 3,290,968 条事件：Bernoulli 采样 + 蓄水池采样、24h 滚动窗口聚合
（9,678 个窗口）、用户类型画像增量更新（1.73 µs/次，吞吐 78,939 事件/秒），并与批量计算做一致性校验。

**任务五 · 图推荐**（`src/graph/ppr.py`）
20,000 用户 × 23,324 部电影、1.3M 条边的二部图：PageRank（0.1 s）+ Louvain 社区（59 个社区，
modularity 见报告）。同一批用户同候选集上：热门 0.7453、PageRank 0.7451、PageRank+社区匹配 0.7791
——**链接分析单独用约等于热门，社区特征能带来 +4.5%**。

**任务六 · 召回-特征-排序集成**（`src/ranking/bpr.py` + `scripts/train_ranker.py`）
BPR 做召回打分，LightGBM LambdaRank 融合 9 个特征（BPR 分、类型匹配、热度、时序、用户活跃度等），
并做消融实验（去掉 BPR / 去掉类型特征）。

## 6. 复现与环境

- Python 3.9+（验证环境 3.9.25）、PyTorch 2.8.0+cu126、LightGBM 4.5.0、PySpark 3.5.3、Java 17
- 硬件：16 核 / 16 GB 内存 / RTX 3050 Laptop 4 GB
- 所有随机性来自 `configs/base.yaml` 的 `seed: 20260907`
- 中间产物统一写 `data/processed`、`data/features`、`results/`；`data/` 与 `models/*.pt` 不入库
- Windows 上跑 Spark 的两个坑（已在脚本注释中说明）：脚本路径必须纯 ASCII（JVM 解析中文路径报
  `Bad pathname`）；用 `spark-submit` 启动而非在 Python 里直接建 SparkSession

## 7. 已知限制

1. **LSH 参数未调优**：recall@20 仅 0.148，说明在这份数据的相似度分布下，阈值得下调或改用
   更多 bands；报告里保留了这一负面结果。
2. **负样本口径**：任务书未给出教师统一脚本的负采样池定义，本项目采用"训练交互 ≥ 5 的条目 +
   排除已看"的口径；换口径会改变热门基线的强弱，但不改变模型相对排序。
3. **图实验在 20k 用户子图上完成**：全量 Louvain 单机内存与时间不可接受，子图结论外推需谨慎。
4. **BPR 早停后 best_iteration 偏小**：排序目标在 1~2 轮后即接近平台，后续轮次只带来波动，
   因此最终模型以"训练轮次 + 验证 NDCG"选点，而非单纯追求更多树。

## 8. 目录结构

```
project/
├── configs/          base.yaml（协议） / local.yaml（本机路径覆盖）
├── data/             processed（划分、候选集） / features（排序特征） / spark（导出）
├── scripts/          prepare_data / build_features / evaluate / train_ranker /
│                     export_for_spark / spark_als / run_all
├── src/              baseline / similarity / distributed / streaming / graph / ranking
├── models/           bpr_mf.pt / bpr_scratch.pt / als_factors.npz（不入库）
├── results/          top10_*.csv / metrics_*.json / *_report.json
└── docs/             项目报告.md/.docx、演示材料.pptx、个人贡献说明.md
```

文档生成（需要 python-docx / python-pptx）：

```bash
python scripts/build_report_docx.py   # docs/项目报告.md -> docs/项目报告.docx
python scripts/build_slides.py        # docs/演示材料.pptx
```