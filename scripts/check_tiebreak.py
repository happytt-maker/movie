#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对工具：量化「同分如何打破」对 NDCG@10 的影响。

背景：排序模型早停在 best_iteration=1，全表只有几十个不同分值；候选集行序又是
「正样本在前、负样本在后」，于是稳定排序会把正样本排到前面，把分数抬高。

本脚本用同一份模型输出，只改同分规则，给出三种口径的 NDCG：
    row     —— 稳定排序（≈ 现有 CandidateSet.topk 的行为）
    random  —— 随机打破
    bpr     —— 同分按 BPR 分数（业务上更合理）

用法：
    python scripts/check_tiebreak.py --config configs/base.yaml
    python scripts/check_tiebreak.py --model ranker_no_temporal --limit 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_set import CandidateSet  # noqa: E402
from src.data_io import load_config, resolve  # noqa: E402
from src.metrics import ndcg_at_k  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="同分规则对 NDCG 的影响")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="ranker_full")
    parser.add_argument("--limit", type=int, default=0, help="只算前 N 个用户（0=全量）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import lightgbm as lgb

    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    feature_dir = resolve(cfg["paths"].get("features_dir", "data/features"))
    models = resolve(cfg["paths"].get("models_dir", "models"))

    meta = json.loads((models / f"{args.model}.meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(models / f"{args.model}.txt"))
    cand = CandidateSet(processed / "candidates_test.npz", processed / "vocab.npz")
    test = np.load(feature_dir / "test.npz")

    scores = booster.predict(
        np.column_stack([test[n] for n in meta["features"]]).astype(np.float32),
        num_iteration=meta["best_iteration"],
    ).astype(np.float32)
    bpr = test["bpr"]
    n_values = int(np.unique(scores).size)

    rng = np.random.default_rng(args.seed)
    users = np.flatnonzero(cand.counts)
    if args.limit:
        users = users[: args.limit]

    res = {"row": [], "random": [], "bpr": []}
    for u in users:
        sl = cand.user_slice(int(u))
        seg, bs = scores[sl], bpr[sl]
        truth = set(cand.mid[sl][cand.label[sl] == 1].tolist())
        if not truth:
            continue
        i_row = np.argsort(-seg, kind="stable")[:10]
        i_rand = np.argsort(-(seg + rng.random(len(seg), dtype=np.float32) * 1e-6), kind="stable")[:10]
        i_bpr = np.lexsort((-bs, -seg))[:10]
        res["row"].append(ndcg_at_k(cand.mid[sl][i_row].tolist(), truth, 10))
        res["random"].append(ndcg_at_k(cand.mid[sl][i_rand].tolist(), truth, 10))
        res["bpr"].append(ndcg_at_k(cand.mid[sl][i_bpr].tolist(), truth, 10))

    summary = {
        "model": args.model,
        "users": len(res["row"]),
        "distinct_score_values": n_values,
        "ndcg_row_order": round(float(np.mean(res["row"])), 5),
        "ndcg_random_tiebreak": round(float(np.mean(res["random"])), 5),
        "ndcg_bpr_tiebreak": round(float(np.mean(res["bpr"])), 5),
        "gap_row_minus_random": round(float(np.mean(res["row"]) - np.mean(res["random"])), 5),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    out = Path(args.out) if args.out else resolve(cfg["paths"]["results_dir"]) / "tiebreak_audit.json"
    out.write_text(text, encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
