# -*- coding: utf-8 -*-
"""任务一：热门推荐基线（Popularity Baseline）。

打分 = 训练集中该电影的交互次数（可切换为加权分或贝叶斯平均），
只在候选集内排序，输出 Top-10。作为 NDCG@10 提升的对照基准。

用法：
    python -m src.baseline.popularity --config configs/local.yaml
    python -m src.baseline.popularity --config configs/local.yaml --score weighted
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..candidate_set import CandidateSet, write_top10
from ..data_io import load_config, load_table, resolve


def item_statistics(train: dict[str, np.ndarray], item_count: int, movie_ids: np.ndarray) -> dict[str, np.ndarray]:
    """按电影统计交互次数、评分和、平均分。"""
    lookup = np.full(int(movie_ids.max()) + 1, -1, dtype=np.int32)
    lookup[movie_ids] = np.arange(item_count, dtype=np.int32)
    codes = lookup[train["movieId"]]
    if (codes < 0).any():
        raise ValueError("训练集出现 vocab 之外的 movieId")

    counts = np.bincount(codes, minlength=item_count).astype(np.int64)
    rating_sum = np.bincount(codes, weights=train["rating"].astype(np.float64), minlength=item_count)
    mean = np.divide(rating_sum, counts, out=np.zeros_like(rating_sum), where=counts > 0)
    return {"count": counts, "rating_sum": rating_sum, "mean": mean}


def score_items(stats: dict[str, np.ndarray], mode: str = "count", global_mean: float = 3.5,
                prior_count: float = 50.0) -> np.ndarray:
    """把统计量转成排序分。"""
    counts = stats["count"].astype(np.float64)
    if mode == "count":
        return counts.astype(np.float32)
    if mode == "log_count":
        return np.log1p(counts).astype(np.float32)
    if mode == "weighted":
        # 交互次数 × 平均分，兼顾热度与口碑
        return (counts * stats["mean"]).astype(np.float32)
    if mode == "bayes":
        # 贝叶斯平均：样本少的电影向全局均分收缩
        mean = (stats["rating_sum"] + prior_count * global_mean) / (counts + prior_count)
        return mean.astype(np.float32)
    raise ValueError(f"未知打分方式：{mode}")


def run(config: str, score_mode: str = "count", split: str = "test", out: str | None = None) -> dict:
    cfg = load_config(config)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])

    t0 = time.perf_counter()
    train = load_table(processed / "train.npz")
    vocab = load_table(processed / "vocab.npz")
    movie_ids = vocab["movieId"]
    stats = item_statistics(train, len(movie_ids), movie_ids)
    item_score = score_items(stats, score_mode, global_mean=float(train["rating"].mean()))

    cand = CandidateSet(processed / f"candidates_{split}.npz", processed / "vocab.npz")
    scores = item_score[cand.item_codes].astype(np.float32)
    prepare_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    columns = cand.topk(scores, k=10)
    rank_seconds = time.perf_counter() - t1

    out_path = Path(out) if out else results / f"top10_popularity_{score_mode}.csv" if score_mode != "count" else results / "top10_popularity.csv"
    write_top10(out_path, columns)

    n_users = len(np.unique(columns["userId"]))
    metrics = {
        "model": f"popularity_{score_mode}",
        "score_mode": score_mode,
        "split": split,
        "users": int(n_users),
        "prepare_seconds": round(prepare_seconds, 2),
        "ranking_seconds": round(rank_seconds, 2),
        "latency_ms_per_user": round(rank_seconds / max(n_users, 1) * 1000, 4),
        "throughput_users_per_sec": round(n_users / max(rank_seconds, 1e-9), 1),
        "pred_file": str(out_path),
    }
    print(f"[popularity] 打分方式={score_mode} 用户={n_users:,} "
          f"排序耗时={rank_seconds:.1f}s 单用户延迟={metrics['latency_ms_per_user']:.3f}ms")
    print(f"[popularity] 结果写入 {out_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="热门推荐基线")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--score", default="count", choices=["count", "log_count", "weighted", "bayes"])
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.config, args.score, args.split, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())