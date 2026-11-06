#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对"任意候选集文件"生成 Top10 —— 教师测试集对接入口（契约第 4.1 节）。

输入：CSV，至少含 userId、movieId 两列（label 可有可无，我们不读）。
输出：CSV，userId,rank,movieId,score —— 与 results/top10_ranker_full.csv 完全同格式。

处理链路与训练时一致（scripts/build_features.py + scripts/train_ranker.py）：
    候选行 -> 条目统计特征 / 用户统计特征 / 类型亲和度 / BPR 分数 -> LightGBM 排序 -> Top10

用法：
    python scripts/predict.py --input teacher_test.csv --out results/top10_teacher.csv
    python scripts/predict.py --input teacher_test.csv --filter-seen      # 排除训练集已交互
    python scripts/predict.py --input teacher_test.csv --model ranker_full --topk 10
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
from src.features import (  # noqa: E402
    bpr_scores,
    item_stat_features,
    movie_genre_matrix,
    user_genre_profile,
    user_stat_features,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_candidates(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"userId": "int64", "movieId": "int64"})
    missing = {"userId", "movieId"} - set(df.columns)
    if missing:
        raise SystemExit(f"输入缺少列：{sorted(missing)}（需要 userId,movieId[,label]）")
    return df


def topk_by_user(user_ids: np.ndarray, scores: np.ndarray, movie_ids: np.ndarray, k: int) -> dict:
    """按 userId 分组取前 k 个（userId 已排序时可用分段）。"""
    order = np.argsort(user_ids, kind="stable")
    u_sorted = user_ids[order]
    s_sorted = scores[order]
    m_sorted = movie_ids[order]
    # 通用分段（不假设 userId 连续）
    uniq, starts = np.unique(u_sorted, return_index=True)
    ends = np.append(starts[1:], len(u_sorted))

    out_u, out_rank, out_m, out_s = [], [], [], []
    for uid, start, end in zip(uniq, starts, ends):
        seg = s_sorted[start:end]
        n = end - start
        take = min(k, n)
        idx = np.argpartition(-seg, take - 1)[:take] if take < n else np.arange(n)
        idx = idx[np.argsort(-seg[idx], kind="stable")]
        out_u.append(np.full(take, uid, dtype=np.int64))
        out_rank.append(np.arange(1, take + 1, dtype=np.int8))
        out_m.append(m_sorted[start:end][idx])
        out_s.append(seg[idx])
    return {
        "userId": np.concatenate(out_u) if out_u else np.empty(0, dtype=np.int64),
        "rank": np.concatenate(out_rank) if out_rank else np.empty(0, dtype=np.int8),
        "movieId": np.concatenate(out_m) if out_m else np.empty(0, dtype=np.int64),
        "score": np.concatenate(out_s) if out_s else np.empty(0, dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="任意候选集 -> Top10（教师测试集对接）")
    parser.add_argument("--input", required=True, help="候选集 CSV（userId,movieId[,label]）")
    parser.add_argument("--out", default=None, help="输出 CSV（默认 results/top10_custom.csv）")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="ranker_full", help="models/<name>.txt + .meta.json")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--filter-seen", action="store_true", help="排除训练集已交互的电影")
    parser.add_argument("--summary", default=None, help="统计信息 JSON 输出路径")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    raw = resolve(cfg["paths"]["raw_dir"])
    models_dir = resolve(cfg["paths"].get("models_dir", "models"))
    results = resolve(cfg["paths"]["results_dir"])

    df = load_candidates(args.input)
    log(f"读入候选 {len(df):,} 行 / {df['userId'].nunique():,} 用户")

    vocab = load_table(processed / "vocab.npz")
    train = load_table(processed / "train.npz")
    n_users, n_items = len(vocab["userId"]), len(vocab["movieId"])

    m_lookup = np.full(int(vocab["movieId"].max()) + 1, -1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(n_items, dtype=np.int32)
    u_max = int(vocab["userId"].max())

    movie_ids = df["movieId"].to_numpy(np.int64)
    user_ids = df["userId"].to_numpy(np.int64)
    item_codes = m_lookup[np.clip(movie_ids, 0, len(m_lookup) - 1)]
    unknown_movie = int((item_codes < 0).sum())
    unknown_user = int(((user_ids < 1) | (user_ids > u_max)).sum())
    keep = (item_codes >= 0) & (user_ids >= 1) & (user_ids <= u_max)
    if not keep.all():
        log(f"丢弃不可打分行：未知 movieId {unknown_movie:,}，未知 userId {unknown_user:,}")
        df = df.loc[keep].reset_index(drop=True)
        movie_ids = movie_ids[keep]
        user_ids = user_ids[keep]
        item_codes = item_codes[keep]
    if len(df) == 0:
        raise SystemExit("过滤后没有可打分的候选行")

    user_codes = (user_ids - 1).astype(np.int64)

    # ---- 特征（与 build_features.py 完全一致）----
    train_u = train["userId"].astype(np.int64) - 1
    train_m = m_lookup[train["movieId"].astype(np.int64)]
    global_mean = float(train["rating"].mean())

    feats: dict[str, np.ndarray] = {}
    for name, arr in item_stat_features(train, train_m, n_items, global_mean).items():
        feats[name] = arr[item_codes].astype(np.float32)
    for name, arr in user_stat_features(train, n_users).items():
        feats[name] = arr[user_codes].astype(np.float32)
    log("统计类特征完成")

    genre, _ = movie_genre_matrix(raw / "movies.csv", vocab["movieId"])
    profile = user_genre_profile(train, train_u, train_m, genre, n_users, cfg["positive_threshold"])
    item_genre = genre[item_codes]
    item_norm = np.linalg.norm(item_genre, axis=1)
    item_norm[item_norm == 0] = 1.0
    feats["genre_affinity"] = ((profile[user_codes] * item_genre).sum(axis=1) / item_norm).astype(np.float32)
    log("类型亲和度完成")

    bpr_path = models_dir / "bpr_mf.pt"
    if bpr_path.exists():
        feats["bpr"] = bpr_scores(bpr_path, user_codes.astype(np.int32), item_codes.astype(np.int32)).astype(np.float32)
        log("BPR 分数完成")

    # ---- 排序 ----
    import lightgbm as lgb
    meta_path = models_dir / f"{args.model}.meta.json"
    model_path = models_dir / f"{args.model}.txt"
    if not model_path.exists() or not meta_path.exists():
        raise SystemExit(f"找不到模型 {model_path} 或 {meta_path}，请先跑 scripts/train_ranker.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [n for n in meta["features"] if n not in feats]
    if missing:
        raise SystemExit(f"缺少模型需要的特征：{missing}")
    X = np.column_stack([feats[n] for n in meta["features"]]).astype(np.float32)
    booster = lgb.Booster(model_file=str(model_path))
    scores = booster.predict(X, num_iteration=meta["best_iteration"])
    log(f"打分完成（{len(scores):,} 行），模型 {args.model}@iter{meta['best_iteration']}")

    # ---- 可选：排除训练集已交互 ----
    if args.filter_seen:
        key_train = np.sort(train_u * n_items + train_m)
        key_cand = user_codes * n_items + item_codes
        pos = np.searchsorted(key_train, key_cand)
        pos = np.clip(pos, 0, len(key_train) - 1)
        seen = key_train[pos] == key_cand
        log(f"排除训练集已交互：{int(seen.sum()):,} 行")
        keep2 = ~seen
        user_ids, movie_ids, scores = user_ids[keep2], movie_ids[keep2], scores[keep2]
        if len(scores) == 0:
            raise SystemExit("过滤后没有剩余候选")

    cols = topk_by_user(user_ids, scores.astype(np.float32), movie_ids, args.topk)

    out_path = Path(args.out) if args.out else results / "top10_custom.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "userId": cols["userId"],
        "rank": cols["rank"],
        "movieId": cols["movieId"],
        "score": cols["score"],
    }).to_csv(out_path, index=False, float_format="%.6f")

    summary = {
        "model": args.model,
        "input": str(args.input),
        "output": str(out_path),
        "input_rows": int(len(df)),
        "users": int(len(np.unique(cols["userId"]))),
        "output_rows": int(len(cols["userId"])),
        "topk": args.topk,
        "filter_seen": bool(args.filter_seen),
        "unknown_movie_rows": unknown_movie,
        "unknown_user_rows": unknown_user,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"输出 {out_path}（{summary['users']:,} 用户 / {summary['output_rows']:,} 行，用时 {summary['elapsed_seconds']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
