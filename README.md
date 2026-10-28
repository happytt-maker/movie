# 基于 MovieLens 的分布式增量电影推荐系统

> 大数据技术课程项目 · NDCG@10 提升挑战
> 目标：Top 10 推荐结果的 NDCG@10 相对热门推荐基线提升 **不低于 10%**

## 1. 项目简介

本项目在教师统一的数据划分、候选集与评价脚本上，构建一个**可运行、可评价、可复现**的分布式增量电影推荐系统，覆盖：

- 大规模行为数据清洗与分布式处理（Spark）
- 相似度计算与 LSH 近似候选召回
- 基于时间戳的流式增量处理（窗口统计、用户画像更新）
- 用户–电影二部图上的链接分析 / 随机游走
- 召回 + 特征 + 排序的系统集成与消融实验

## 2. 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.10+ | 建议 3.11 |
| Java | 17 | PySpark 必需 |
| PySpark | 3.5+ | 分布式计算 |
| 内存 | ≥ 16 GB | MovieLens 32M 全量处理 |
| 磁盘 | ≥ 10 GB | 原始数据 + 中间结果 |

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 数据准备

### 3.1 下载数据

从 GroupLens 官方下载 **MovieLens 32M**：

<https://grouplens.org/datasets/movielens/32m/>

> ⚠️ **原始数据不纳入版本控制**。`data/` 已在 `.gitignore` 中忽略，且任务书要求「不提交未经许可重新发布的 MovieLens 原始数据」。

### 3.2 目录结构

```
data/
└── ml-32m/
    ├── ratings.csv     # userId, movieId, rating, timestamp
    ├── movies.csv      # movieId, title, genres
    ├── tags.csv        # userId, movieId, tag, timestamp
    └── links.csv       # movieId, imdbId, tmdbId
```

### 3.3 生成数据划分

```bash
python scripts/prepare_data.py --config configs/base.yaml
```

此脚本按任务书协议产出训练 / 验证 / 测试集，并生成候选集。

## 4. 实验协议

严格遵循任务书「五 数据划分与评价协议」，配置见 `configs/base.yaml`：

| 项目 | 设定 |
|---|---|
| 正反馈定义 | `rating >= 4` |
| 时间划分 | 每位用户按时间戳排序：前 80% 训练 / 中间 10% 验证 / 最后 10% 测试 |
| 候选集 | 每个测试用户 = 测试正样本 + 100 个固定负样本 |
| 随机种子 | **20260907**（全局统一） |
| 核心指标 | NDCG@10（同时记录 Recall@10、吞吐量、单用户推荐延迟） |

> **防泄漏约束**：训练过程不得读取测试标签；任何时序特征只能使用推荐时刻之前的数据。

## 5. 项目结构

```
movie/
├── configs/
│   └── base.yaml            # 实验协议配置（种子、划分比例、负采样数）
├── data/                    # 数据目录（.gitignore 忽略，不入库）
│   ├── ml-32m/              # 原始数据（需自行下载）
│   └── processed/           # 划分后的训练/验证/测试集
├── scripts/
│   ├── prepare_data.py      # 数据检查、统计与划分
│   └── evaluate.py          # NDCG@10 / Recall@10 评价
├── src/
│   ├── baseline/            # 任务一：热门推荐基线
│   ├── similarity/          # 任务二：精确相似度 + LSH 近似召回
│   ├── distributed/         # 任务三：Spark 分布式推荐
│   ├── streaming/           # 任务四：流式增量处理
│   ├── graph/               # 任务五：二部图链接分析
│   └── ranking/             # 任务六：召回-特征-排序集成
├── results/                 # 推荐结果与指标（入库，便于对照）
└── docs/                    # 报告、演示材料
```

## 6. 任务进度

| 任务 | 模块 | 状态 |
|---|---|---|
| 任务一 | 数据检查与基线 | ⬜ 待完成 |
| 任务二 | 相似度与召回 | ⬜ 待完成 |
| 任务三 | 分布式推荐（Spark） | ⬜ 待完成 |
| 任务四 | 流式增量处理 | ⬜ 待完成 |
| 任务五 | 图推荐实验 | ⬜ 待完成 |
| 任务六 | 系统集成与消融 | ⬜ 待完成 |

## 7. 实验结果

| 模型 | NDCG@10 | Recall@10 | 平均延迟 (s) | 备注 |
|---|---|---|---|---|
| 热门基线（Popularity） | — | — | — | 对照基准 |
| 改进模型 | — | — | — | 相对基线 +X% |

> 表格在实验完成后填写。**消融实验**需说明性能提升具体来自哪个算法或系统设计。

## 8. 复现说明

为保证教师可用提交代码重新运行出相同结果：

- 固定随机种子 `20260907`（见 `configs/base.yaml`）
- 依赖版本锁定于 `requirements.txt`
- 数据版本：MovieLens 32M（ml-32m）
- 所有实验参数集中在 `configs/` 下，不硬编码
- 运行日志与指标统一输出到 `results/`

## 9. 参考与致谢

- MovieLens 数据集：F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS.
- 引用开源代码 / 模型 / 外部资料时，在源码中以注释标注来源。
