#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出 Spark 任务需要的数据（本地 pandas 完成，保证与划分协议一致）。

产物：
    data/spark/train_positive.csv    训练集正反馈（rating >= 阈值）
    data/spark/candidates_sample.csv 抽样测试用户的候选集（含 label）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, load_table, resolve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 Spark 输入数据")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sample-users", type=int, default=20000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    out_dir = resolve("data/spark")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_table(processed / "train.npz")
    mask = train["rating"] >= cfg["positive_threshold"]
    train_df = pd.DataFrame({
        "userId": train["userId"][mask].astype(np.int32),
        "movieId": train["movieId"][mask].astype(np.int32),
        "rating": train["rating"][mask].astype(np.float32),
    })
    train_df.to_csv(out_dir / "train_positive.csv", index=False)
    print(f"训练正反馈：{len(train_df):,} 行 -> train_positive.csv")

    cand = load_table(processed / "candidates_test.npz")
    users = np.unique(cand["userId"])
    rng = np.random.default_rng(int(cfg["seed"]))
    sample = np.sort(rng.choice(users, min(args.sample_users, len(users)), replace=False))
    lookup = np.zeros(int(users.max()) + 1, dtype=bool)
    lookup[sample] = True
    keep = lookup[cand["userId"]]
    sample_df = pd.DataFrame({
        "userId": cand["userId"][keep].astype(np.int32),
        "movieId": cand["movieId"][keep].astype(np.int32),
        "label": cand["label"][keep].astype(np.int8),
    })
    sample_df.to_csv(out_dir / "candidates_sample.csv", index=False)
    print(f"候选抽样：{len(sample_df):,} 行 / {len(sample):,} 用户 -> candidates_sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())