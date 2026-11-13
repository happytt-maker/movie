#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务三补交付：用 Spark ALS 的因子在测试候选集上生成 Top10。

背景：
    scripts/spark_als.py 训练完只写报告/因子，没有独立的推荐结果文件；
    本脚本读取 models/als_factors.npz，在候选集内点积打分并取 Top10，
    输出格式与 top10_ranker_full.csv 一致，可直接交给 scripts/evaluate.py。

用法：
    python scripts/export_top10_als.py --config configs/base.yaml
    python scripts/evaluate.py --pred results/top10_als.csv --name als
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_set import CandidateSet, write_top10  # noqa: E402
from src.data_io import load_config, load_table, resolve  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ALS 因子 -> 测试候选集 Top10")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--factors", default=None, help="默认 models/als_factors.npz")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    models_dir = resolve(cfg["paths"].get("models_dir", "models"))
    results = resolve(cfg["paths"]["results_dir"])

    factor_path = Path(args.factors) if args.factors else models_dir / "als_factors.npz"
    if not factor_path.exists():
        raise SystemExit(f"找不到 ALS 因子：{factor_path}（先跑 scripts/spark_als.py）")
    table = np.load(factor_path)
    user_ids, item_ids = table["user_ids"], table["item_ids"]
    user_f, item_f = table["user"], table["item"]
    log(f"因子：users {user_f.shape}，items {item_f.shape}")

    u_index = {int(u): i for i, u in enumerate(user_ids)}
    i_index = {int(m): i for i, m in enumerate(item_ids)}

    cand = CandidateSet(processed / "candidates_test.npz", processed / "vocab.npz")
    uid, mid = cand.uid, cand.mid
    u_pos = np.fromiter((u_index.get(int(u), -1) for u in uid), dtype=np.int32, count=len(uid))
    m_pos = np.fromiter((i_index.get(int(m), -1) for m in mid), dtype=np.int32, count=len(mid))
    covered = (u_pos >= 0) & (m_pos >= 0)
    log(f"候选 {len(uid):,} 行，因子可覆盖 {int(covered.sum()):,} 行（{covered.mean():.2%}）")

    scores = np.full(len(uid), -np.inf, dtype=np.float32)
    idx = np.flatnonzero(covered)
    chunk = 2_000_000
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        scores[part] = np.einsum("ij,ij->i", user_f[u_pos[part]], item_f[m_pos[part]])

    # 逐用户取 Top-k（只在有覆盖的候选行里取）
    cols = cand.topk(np.where(covered, scores, -np.inf), k=args.k)
    keep = np.isfinite(cols["score"])
    cols = {k: v[keep] for k, v in cols.items()}

    out_path = Path(args.out) if args.out else results / "top10_als.csv"
    write_top10(out_path, {**cols, "score": np.nan_to_num(cols["score"], nan=0.0)})
    users = int(len(np.unique(cols["userId"])))
    summary = {
        "model": "als",
        "method": "Spark MLlib ALS（隐式反馈，rank=64）因子点积，在测试候选集内取 Top10",
        "factors": str(factor_path),
        "users": users,
        "rows": int(len(cols["userId"])),
        "k": args.k,
        "candidate_coverage": round(float(covered.mean()), 4),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "pred_file": out_path.name,
        "note": "ALS 只覆盖训练中出现过的用户/电影；未覆盖的行不参与排序，因此用户数可能少于 191,007",
    }
    (results / "similarity_top10_als.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"输出 {out_path}（{users:,} 用户 / {len(cols['userId']):,} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
