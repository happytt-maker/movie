# 结果目录

本目录用于存放**可提交**的实验结果，便于与教师统一脚本对照。

典型内容：

| 文件 | 说明 |
|---|---|
| `top10_<model>.csv` | 统一格式的 Top 10 推荐结果 |
| `metrics_<model>.json` | NDCG@10 / Recall@10 / 延迟等指标 |
| `ablation.csv` | 消融实验结果 |

> `.gitignore` 中的 `*.csv` 规则对本目录做了例外处理（`!results/**/*.csv`），
> 因此结果 CSV 会被正常提交；而 `data/` 下的原始数据不会。

提交前请确认结果文件**不含**原始 MovieLens 明细数据，仅含推荐输出与指标。
