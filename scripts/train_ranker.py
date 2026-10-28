#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务六：召回-特征-排序集成（LightGBM LambdaRank）与消融实验。

训练集用 valid 候选集（含标签），测试集用 test 候选集；
两者都由 prepare_data.py 按同一协议生成，用户级切分不重叠。

用法：
    python scripts/train_ranker.py --config configs/base.yaml --name ranker_full
    python scripts/train_ranker.py --config configs/base.yaml --name ranker_no_bpr --exclude bpr
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
from src.metrics import ndcg_at_k  # noqa: E402

RESERVED = {"label", "userId", "movieId"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feature_names(npz, exclude: set[str]) -> list[str]:
    names = [k for k in npz.files if k not in RESERVED]
    return [n for n in names if n not in exclude]


def main() -> int:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description="LightGBM 排序模型")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--name", default="ranker_full")
    parser.add_argument("--exclude", default="", help="逗号分隔的排除特征，用于消融")
    parser.add_argument("--train-users", type=int, default=80000)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-chunk", type=int, default=1000000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"]) if args.seed is None else args.seed
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])
    feature_dir = resolve(cfg["paths"].get("features_dir", "data/features"))

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    t0 = time.perf_counter()

    train_npz = np.load(feature_dir / "valid.npz")
    names = feature_names(train_npz, exclude)
    log(f"特征：{names}")

    uid = train_npz["userId"]
    label = train_npz["label"]
    uniq_users = np.unique(uid)
    rng = np.random.default_rng(seed)
    if args.train_users < len(uniq_users):
        keep = np.sort(rng.choice(uniq_users, args.train_users, replace=False))
    else:
        keep = uniq_users
    lookup = np.zeros(int(uniq_users.max()) + 1, dtype=bool)
    lookup[keep] = True
    mask = lookup[uid]

    X = np.column_stack([train_npz[n][mask] for n in names]).astype(np.float32)
    y = label[mask].astype(np.int8)
    groups_all = np.bincount(uid[mask])
    user_all = uid[mask]
    log(f"训练矩阵 {X.shape}，正样本率 {y.mean():.4f}，用时 {time.perf_counter()-t0:.0f}s")

    # 按用户切出 10% 做 early stopping，避免与 test 混用
    users_train = np.flatnonzero(groups_all > 0)
    is_valid_user = (users_train % 10 == 0)
    valid_users = set(users_train[is_valid_user].tolist())
    valid_mask = np.isin(user_all, np.fromiter(valid_users, dtype=np.int64))
    train_mask = ~valid_mask

    def build(mask_rows, label_arr):
        sub_u = user_all[mask_rows]
        groups = np.bincount(sub_u)
        groups = groups[groups > 0]
        return X[mask_rows], label_arr[mask_rows], groups

    X_tr, y_tr, g_tr = build(train_mask, y)
    X_va, y_va, g_va = build(valid_mask, y)
    log(f"训练 {X_tr.shape[0]:,} 行 / {len(g_tr):,} 用户；验证 {X_va.shape[0]:,} 行 / {len(g_va):,} 用户")

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=names, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, feature_name=names, free_raw_data=True)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": args.lr,
        "num_leaves": args.leaves,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 30,
        "num_threads": 12,
        "seed": seed,
        "verbose": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=args.rounds,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    log(f"训练完成，best_iteration={booster.best_iteration}，用时 {time.perf_counter()-t0:.0f}s")

    # ---- 测试集打分 ----
    test_npz = np.load(feature_dir / "test.npz")
    cand = CandidateSet(processed / "candidates_test.npz", processed / "vocab.npz")
    scores = np.empty(cand.rows, dtype=np.float32)
    n = cand.rows
    for start in range(0, n, args.eval_chunk):
        end = min(start + args.eval_chunk, n)
        block = np.column_stack([test_npz[k][start:end] for k in names]).astype(np.float32)
        scores[start:end] = booster.predict(block, num_iteration=booster.best_iteration)
    log(f"测试集打分完成（{n:,} 行）")

    cols = cand.topk(scores, k=10)
    out_csv = results / f"top10_{args.name}.csv"
    write_top10(out_csv, cols)

    truth = {}
    for u in np.flatnonzero(cand.counts):
        sl = cand.user_slice(int(u))
        pos = cand.mid[sl][cand.label[sl] == 1]
        truth[int(cand.user_ids[u - 1])] = set(pos.tolist())

    by_user = {}
    order = np.argsort(-scores, kind="stable")
    # 逐用户算 NDCG（与 topk 一致）
    ndcgs, recalls = [], []
    for u in np.flatnonzero(cand.counts):
        sl = cand.user_slice(int(u))
        seg = scores[sl]
        take = min(10, len(seg))
        idx = np.argpartition(-seg, take - 1)[:take]
        idx = idx[np.argsort(-seg[idx], kind="stable")]
        top = cand.mid[sl][idx]
        rel = truth[int(cand.user_ids[u - 1])]
        ndcgs.append(ndcg_at_k(top.tolist(), rel, 10))
        recalls.append(len(set(top.tolist()) & rel) / max(len(rel), 1))

    metrics = {
        "model": args.name,
        "features": names,
        "excluded": sorted(exclude),
        "users": int(len(ndcgs)),
        "ndcg@10": round(float(np.mean(ndcgs)), 5),
        "recall@10": round(float(np.mean(recalls)), 5),
        "train_rows": int(X_tr.shape[0]),
        "best_iteration": int(booster.best_iteration or 0),
        "pred_file": out_csv.name,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    out_json = results / f"metrics_{args.name}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    log(f"NDCG@10={metrics['ndcg@10']:.4f} Recall@10={metrics['recall@10']:.4f}")
    log(f"结果：{out_csv} / {out_json}")

    importance = sorted(zip(names, booster.feature_importance("gain")), key=lambda x: -x[1])
    imp_path = results / f"importance_{args.name}.json"
    with open(imp_path, "w", encoding="utf-8") as fh:
        json.dump([{"feature": k, "gain": float(v)} for k, v in importance], fh, ensure_ascii=False, indent=2)
    for k, v in importance:
        log(f"  {k:18s} gain={v:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())