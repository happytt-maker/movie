#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成候选集上的排序特征（任务六）。

用法：
    python scripts/build_features.py --config configs/base.yaml --split valid
    python scripts/build_features.py --config configs/base.yaml --split test

产物：data/features/{split}.npz
    每列一个与候选集等长的 float32 数组，另有 label / userId / movieId 便于校验。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_set import CandidateSet  # noqa: E402
from src.data_io import load_config, load_table, resolve, save_table  # noqa: E402
from src.features import (  # noqa: E402
    bpr_scores,
    item_stat_features,
    movie_genre_matrix,
    user_genre_profile,
    user_stat_features,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def user_item_codes(table: dict, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    u_lookup = np.zeros(int(vocab["userId"].max()) + 1, dtype=np.int32)
    u_lookup[vocab["userId"]] = np.arange(len(vocab["userId"]), dtype=np.int32)
    m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(len(vocab["movieId"]), dtype=np.int32)
    return u_lookup[table["userId"]], m_lookup[table["movieId"]]


def genre_affinity(cand: CandidateSet, profile: np.ndarray, genre: np.ndarray) -> np.ndarray:
    """候选行上：用户类型画像与电影类型的余弦相似度。"""
    out = np.empty(cand.rows, dtype=np.float32)
    item_genre = genre[cand.item_codes]
    item_norm = np.linalg.norm(item_genre, axis=1)
    item_norm[item_norm == 0] = 1.0
    chunk = 2_000_000
    for start in range(0, cand.rows, chunk):
        end = min(start + chunk, cand.rows)
        users = cand.uid[start:end] - 1
        dots = (profile[users] * item_genre[start:end]).sum(axis=1)
        out[start:end] = dots / item_norm[start:end]
    return out


def item_wise_codes(cand: CandidateSet, table: dict, vocab: dict, n_items: int) -> np.ndarray:
    m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(n_items, dtype=np.int32)
    return m_lookup[table["movieId"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="候选集特征生成")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    models = resolve(cfg["paths"].get("models_dir", "models"))
    feature_dir = resolve(cfg["paths"].get("features_dir", "data/features")) if not args.out_dir else Path(args.out_dir)

    vocab = load_table(processed / "vocab.npz")
    n_users, n_items = len(vocab["userId"]), len(vocab["movieId"])
    train = load_table(processed / "train.npz")
    train_u, train_m = user_item_codes(train, vocab)
    global_mean = float(train["rating"].mean())

    cand = CandidateSet(processed / f"candidates_{args.split}.npz", processed / "vocab.npz")
    log(f"候选集 {cand.rows:,} 行，用户 {int((cand.counts > 0).sum()):,}")

    feats: dict[str, np.ndarray] = {}
    item_stats = item_stat_features(train, train_m, n_items, global_mean)
    user_stats = user_stat_features(train, n_users)
    # 条目/用户统计量是"每个条目/每个用户"一个值，这里展开到候选集的每一行
    for name, arr in item_stats.items():
        feats[name] = arr[cand.item_codes].astype(np.float32)
    for name, arr in user_stats.items():
        feats[name] = arr[cand.uid - 1].astype(np.float32)
    log(f"统计类特征完成：{sorted(feats)}")

    genre, genre_names = movie_genre_matrix(resolve(cfg["paths"]["raw_dir"]) / "movies.csv", vocab["movieId"])
    profile = user_genre_profile(train, train_u, train_m, genre, n_users, cfg["positive_threshold"])
    feats["genre_affinity"] = genre_affinity(cand, profile, genre)
    log(f"类型画像特征完成（{len(genre_names)} 个类型）")

    bpr_path = models / "bpr_mf.pt"
    if bpr_path.exists():
        feats["bpr"] = bpr_scores(bpr_path, (cand.uid - 1).astype(np.int32), cand.item_codes.astype(np.int32))
        log("BPR 打分特征完成")
    else:
        log(f"缺少 {bpr_path}，跳过 bpr 特征")

    # 扩展特征（任务二/三/五产出后自动启用）
    for name, filename in (("knn", "knn_scores"), ("als", "als_scores"), ("ppr", "ppr_scores")):
        path = feature_dir / f"{args.split}_{filename}.npy"
        if path.exists():
            arr = np.load(path)
            if len(arr) == cand.rows:
                feats[name] = arr.astype(np.float32)
                log(f"合并扩展特征 {name}")
            else:
                log(f"{path.name} 长度不符（{len(arr)} vs {cand.rows}），已跳过")

    # 逐列存盘，避免一次性构造超大二维数组
    feature_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: arr.astype(np.float32) for name, arr in feats.items()}
    payload["label"] = cand.label.astype(np.int8)
    payload["userId"] = cand.uid.astype(np.int32)
    payload["movieId"] = cand.mid.astype(np.int32)
    out_path = feature_dir / f"{args.split}.npz"
    save_table(out_path, allow_unequal=False, **payload)
    log(f"已写入 {out_path}（{len(payload)} 列，{cand.rows:,} 行，用时 {time.perf_counter()-t0:.0f}s）")

    for name in sorted(feats):
        arr = feats[name]
        log(f"  {name:18s} mean={arr.mean():.4f} std={arr.std():.4f} nan={int(np.isnan(arr).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())