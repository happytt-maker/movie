#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据检查、时间划分与候选集生成（任务书「五 数据划分与评价协议」）。

协议：
  1. 正反馈：rating >= 4
  2. 时间划分：每位用户按 timestamp 升序，前 80% 训练 / 中间 10% 验证 / 最后 10% 测试
  3. 候选集：每位测试用户 = 测试正样本 + 100 个固定负样本（种子 20260907）
  4. 防泄漏：训练侧不读取测试标签；时序特征只用推荐时刻之前的数据

用法：
    python scripts/prepare_data.py --config configs/local.yaml

产物（默认 data/processed/）：
    train.npz / valid.npz / test.npz    userId,movieId,rating,timestamp
    vocab.npz                           userIds / movieIds 的编码表
    candidates_test.npz                  userId,movieId,label（1 正 / 0 负）
    candidates_valid.npz
    manifest.json                       每张表的行列数与 dtype
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

from src.data_io import load_config, resolve, save_table  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_raw(raw_dir: Path) -> pd.DataFrame:
    ratings_path = raw_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"找不到评分文件：{ratings_path}")
    dtype = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    df = pd.read_csv(ratings_path, dtype=dtype)
    log(f"读取 ratings.csv：{len(df):,} 行")
    return df


def quality_report(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicated_user_item": int(df.duplicated(subset=["userId", "movieId"]).sum()),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean()),
        "users": int(df["userId"].nunique()),
        "movies": int(df["movieId"].nunique()),
        "timestamp_min": int(df["timestamp"].min()),
        "timestamp_max": int(df["timestamp"].max()),
        "rating_hist": {str(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()},
    }
    report["positives_ratio"] = float((df["rating"] >= 4).mean())
    log(f"质量检查：缺失 {sum(report['missing'].values())}，重复对 {report['duplicated_user_item']:,}")
    return report


def time_split(df: pd.DataFrame, ratios=(0.8, 0.1, 0.9)) -> tuple[np.ndarray, dict]:
    """按 (userId, timestamp) 排序后切分，返回逐行划分标记（0 训练 / 1 验证 / 2 测试）。"""
    order = np.lexsort((df["timestamp"].to_numpy(np.int64), df["userId"].to_numpy(np.int64)))
    user_sorted = df["userId"].to_numpy(np.int64)[order]

    counts = np.bincount(user_sorted)
    if counts[0] != 0:
        raise ValueError("userId 不是从 1 开始的连续整数，需要额外的编码映射")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(user_sorted), dtype=np.int64) - starts[user_sorted]
    n_per_user = counts[user_sorted]

    train_end = np.maximum(np.floor(n_per_user * ratios[0]).astype(np.int64), 1)
    valid_end = np.maximum(np.floor(n_per_user * ratios[2]).astype(np.int64), train_end)

    split_code = np.where(rank < train_end, 0, np.where(rank < valid_end, 1, 2)).astype(np.int8)
    masks = np.empty(len(df), dtype=np.int8)
    masks[order] = split_code

    stats = {
        "users": int(len(counts) - 1),
        "train_rows": int((masks == 0).sum()),
        "valid_rows": int((masks == 1).sum()),
        "test_rows": int((masks == 2).sum()),
        "min_interactions_per_user": int(counts[1:][counts[1:] > 0].min()) if len(counts) > 1 else 0,
    }
    stats["train_ratio"] = round(stats["train_rows"] / len(df), 6)
    log(f"时间划分：train {stats['train_rows']:,} / valid {stats['valid_rows']:,} / test {stats['test_rows']:,}")
    return masks, stats


def generate_candidates(
    pos_u: np.ndarray,
    pos_m: np.ndarray,
    seen_u: np.ndarray,
    seen_m: np.ndarray,
    eligible: np.ndarray,
    user_count: int,
    n_neg: int,
    seed: int,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """为每个含正样本的用户构造「正样本 + n_neg 个负样本」。

    pos_u/pos_m  : 该划分下的正样本（编码，0 基）
    seen_u/seen_m: 训练期已交互对（编码），负样本必须避开，保证候选集不泄漏
    eligible     : 可作为负样本的 item 编码（训练交互数达标的条目）
    """
    order = np.argsort(pos_u, kind="stable")
    pos_u_sorted, pos_m_sorted = pos_u[order], pos_m[order]
    pos_counts = np.bincount(pos_u_sorted, minlength=user_count).astype(np.int64)
    pos_starts = np.concatenate([[0], np.cumsum(pos_counts)[:-1]])

    order_seen = np.argsort(seen_u, kind="stable")
    seen_u_sorted, seen_m_sorted = seen_u[order_seen], seen_m[order_seen]
    seen_counts = np.bincount(seen_u_sorted, minlength=user_count).astype(np.int64)
    seen_starts = np.concatenate([[0], np.cumsum(seen_counts)[:-1]])

    targets = np.flatnonzero(pos_counts > 0)
    log(f"候选集：{len(targets):,} 个用户有正样本，负样本池 {len(eligible):,} 部电影")

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(eligible), size=(len(targets), n_neg + 32), dtype=np.int32)

    pos_slices = [pos_m_sorted[pos_starts[u]: pos_starts[u] + pos_counts[u]] for u in targets]
    pos_out_u = np.repeat(targets.astype(np.int32), pos_counts[targets])
    pos_out_m = np.concatenate(pos_slices)

    neg_u = np.empty(len(targets) * n_neg, dtype=np.int32)
    neg_m = np.empty(len(targets) * n_neg, dtype=np.int32)

    fallback = 0
    for i, u in enumerate(targets):
        seen = seen_m_sorted[seen_starts[u]: seen_starts[u] + seen_counts[u]]
        seen_set = seen
        pos = pos_slices[i]
        cand = np.unique(eligible[draw[i]])
        cand = cand[~np.isin(cand, seen_set)]
        cand = cand[~np.isin(cand, pos)]
        if cand.size < n_neg:  # 兜底：该用户看过的片子过多时逐个补抽
            fallback += 1
            chosen = [int(x) for x in cand]
            chosen_set = set(chosen)
            attempts = 0
            while len(chosen) < n_neg and attempts < 200:
                extra = eligible[rng.integers(0, len(eligible), size=n_neg * 2, dtype=np.int32)]
                for m in extra:
                    m = int(m)
                    if m in chosen_set or m in pos:
                        continue
                    chosen_set.add(m)
                    chosen.append(m)
                    if len(chosen) == n_neg:
                        break
                attempts += 1
            cand = np.asarray(chosen[:n_neg], dtype=np.int32)
        neg_m[i * n_neg:(i + 1) * n_neg] = cand[:n_neg]
        neg_u[i * n_neg:(i + 1) * n_neg] = u

    if fallback:
        log(f"  注意：{fallback} 个用户触发兜底补抽")

    return {
        "userId": np.concatenate([user_ids[pos_out_u].astype(np.int32), user_ids[neg_u].astype(np.int32)]),
        "movieId": np.concatenate([movie_ids[pos_out_m].astype(np.int32), movie_ids[neg_m].astype(np.int32)]),
        "label": np.concatenate([np.ones(len(pos_out_u), dtype=np.int8), np.zeros(len(neg_u), dtype=np.int8)]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MovieLens 数据检查与时间划分")
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--raw-dir", default=None, help="覆盖配置中的原始数据目录")
    parser.add_argument("--out-dir", default=None, help="覆盖配置中的 processed 目录")
    parser.add_argument("--skip-candidates", action="store_true", help="只做划分，不生成候选集")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve(cfg["paths"]["raw_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    n_neg = int(cfg["candidates"]["negatives_per_user"])
    neg_seed = int(cfg["candidates"].get("seed", seed))
    min_item = int(cfg["candidates"].get("min_item_interactions", 5))

    df = load_raw(raw_dir)
    report = quality_report(df)
    if any(report["missing"].values()):
        raise ValueError(f"原始数据存在缺失值：{report['missing']}")

    masks, split_stats = time_split(df)
    report.update({
        "split": split_stats,
        "seed": seed,
        "protocol": {
            "positive_threshold": int(cfg["positive_threshold"]),
            "negatives_per_user": n_neg,
            "candidate_seed": neg_seed,
            "min_item_interactions": min_item,
        },
    })

    user_codes, user_ids = pd.factorize(df["userId"], sort=True)
    movie_codes, movie_ids = pd.factorize(df["movieId"], sort=True)
    user_codes = np.asarray(user_codes, dtype=np.int32)
    movie_codes = np.asarray(movie_codes, dtype=np.int32)

    base_cols = {
        "userId": df["userId"].to_numpy(np.int32),
        "movieId": df["movieId"].to_numpy(np.int32),
        "rating": df["rating"].to_numpy(np.float32),
        "timestamp": df["timestamp"].to_numpy(np.int64),
    }
    for name, code in (("train", 0), ("valid", 1), ("test", 2)):
        sel = masks == code
        save_table(out_dir / f"{name}.npz", **{k: v[sel] for k, v in base_cols.items()})
        log(f"已写入 {name}.npz（{int(sel.sum()):,} 行）")
    save_table(out_dir / "vocab.npz", allow_unequal=True,
               userId=user_ids.to_numpy(np.int64), movieId=movie_ids.to_numpy(np.int64))

    item_counts = None
    if not args.skip_candidates:
        train_sel = masks == 0
        seen_u = user_codes[train_sel]
        seen_m = movie_codes[train_sel]
        item_counts = np.bincount(seen_m, minlength=len(movie_ids))
        eligible = np.flatnonzero(item_counts >= min_item).astype(np.int32)

        threshold = int(cfg["positive_threshold"])
        ratings = df["rating"].to_numpy()
        for name, code in (("test", 2), ("valid", 1)):
            sel = (masks == code) & (ratings >= threshold)
            cand = generate_candidates(
                pos_u=user_codes[sel], pos_m=movie_codes[sel],
                seen_u=seen_u, seen_m=seen_m, eligible=eligible,
                user_count=len(user_ids), n_neg=n_neg, seed=neg_seed,
                user_ids=user_ids.to_numpy(np.int64), movie_ids=movie_ids.to_numpy(np.int64),
            )
            save_table(out_dir / f"candidates_{name}.npz", **cand)
            users_with_pos = len(np.unique(cand["userId"]))
            log(f"已写入 candidates_{name}.npz（{len(cand['label']):,} 行，{users_with_pos:,} 用户，"
                f"正样本 {int(cand['label'].sum()):,}）")
            report[f"candidates_{name}"] = {
                "rows": int(len(cand["label"])),
                "users": int(users_with_pos),
                "positives": int(cand["label"].sum()),
                "avg_positives_per_user": round(float(cand["label"].sum() / max(users_with_pos, 1)), 3),
            }

    report["items_eligible_as_negative"] = int(len(eligible)) if not args.skip_candidates else None
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    with open(out_dir / "data_stats.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"完成，用时 {report['elapsed_seconds']} s，统计写入 {out_dir / 'data_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())