#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一评价脚本：NDCG@10 / Recall@10 / MRR@10。

对应任务书「五 数据划分与评价协议」第 5 条：正反馈记 1（rating >= 4），
在候选集上按推荐顺序计算指标，并记录单用户推荐延迟。

用法：
    python scripts/evaluate.py --pred results/top10_popularity.csv --name popularity
    python scripts/evaluate.py --pred results/top10_itemknn.csv --name itemknn

预测文件格式（统一）：userId,rank,movieId[,score]，rank 从 1 开始。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, load_table, resolve  # noqa: E402
from src.metrics import evaluate_predictions  # noqa: E402


def load_truth(candidates_path: Path) -> dict[int, set[int]]:
    table = load_table(candidates_path)
    uid = table["userId"]
    mid = table["movieId"]
    positive = table["label"] == 1
    order = np.argsort(uid[positive], kind="stable")
    u_sorted = uid[positive][order]
    m_sorted = mid[positive][order]
    counts = np.bincount(u_sorted.astype(np.int64))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    truth = {}
    for u, start, count in zip(np.flatnonzero(counts), starts[counts > 0], counts[counts > 0]):
        truth[int(u)] = set(m_sorted[start:start + count].tolist())
    return truth


def load_predictions(pred_path: Path, k: int) -> dict[int, list[int]]:
    df = pd.read_csv(pred_path, dtype={"userId": "int64", "rank": "int32", "movieId": "int64"})
    if not {"userId", "rank", "movieId"}.issubset(df.columns):
        raise ValueError(f"{pred_path} 需要 userId,rank,movieId 三列，实际为 {list(df.columns)}")
    df = df[df["rank"] <= k].sort_values(["userId", "rank"], kind="stable")
    pred = {}
    for u, group in df.groupby("userId", sort=False):
        pred[int(u)] = group["movieId"].tolist()
    return pred


def main() -> int:
    parser = argparse.ArgumentParser(description="NDCG@10 / Recall@10 评价")
    parser.add_argument("--pred", required=True, help="Top-10 推荐结果 CSV")
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--candidates", default=None, help="候选集文件（默认取 test）")
    parser.add_argument("--name", default=None, help="结果文件名中的模型名")
    parser.add_argument("--latency-ms", type=float, default=None, help="单用户推荐延迟（毫秒）")
    parser.add_argument("--throughput", type=float, default=None, help="每秒处理用户数")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    k = 10
    pred_path = resolve(args.pred)
    cand_path = resolve(args.candidates) if args.candidates else resolve(cfg["paths"]["processed_dir"]) / "candidates_test.npz"
    out_dir = resolve(cfg["paths"]["results_dir"])
    name = args.name or pred_path.stem.replace("top10_", "")

    truth = load_truth(cand_path)
    pred = load_predictions(pred_path, k)
    metrics = evaluate_predictions(pred, truth, k)

    covered = {m for items in pred.values() for m in items}
    metrics.update({
        "model": name,
        "pred_file": f"{pred_path.parent.name}/{pred_path.name}",
        "candidates_file": f"{cand_path.parent.name}/{cand_path.name}",
        "k": k,
        "seed": cfg["seed"],
        "positive_threshold": cfg["positive_threshold"],
        "distinct_items_recommended": len(covered),
        "avg_recommendations_per_user": round(sum(len(v) for v in pred.values()) / len(pred), 3),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    })
    if args.latency_ms is not None:
        metrics["latency_ms_per_user"] = round(args.latency_ms, 3)
    if args.throughput is not None:
        metrics["throughput_users_per_sec"] = round(args.throughput, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(f"模型：{name}")
    print(f"用户数：{metrics['users']:,}")
    print(f"NDCG@10   = {metrics['ndcg@10']:.4f}")
    print(f"Recall@10 = {metrics['recall@10']:.4f}")
    print(f"MRR@10    = {metrics['mrr@10']:.4f}")
    print(f"结果写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())