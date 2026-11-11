# -*- coding: utf-8 -*-
"""数据集审计：结构、分布、半边隔离、时间关系、可预测性上界。

用法： python scripts/audit.py
产物： reports/audit.json 与若干控制台摘要
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ML_USER_MAX = 200948
ML_MOVIE_MAX = 87585


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    dtype = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    df = pd.read_csv(DATA / "ratings.csv", dtype=dtype)
    log(f"ratings.csv 载入 {len(df):,} 行，用时 {time.time() - t0:.1f}s")

    rep: dict = {}
    rep["rows"] = int(len(df))
    rep["columns"] = list(df.columns)
    rep["missing"] = {c: int(df[c].isna().sum()) for c in df.columns}
    rep["duplicated_pairs"] = int(df.duplicated(["userId", "movieId"]).sum())
    rep["users"] = int(df.userId.nunique())
    rep["movies"] = int(df.movieId.nunique())
    rep["rating_min"] = float(df.rating.min())
    rep["rating_max"] = float(df.rating.max())
    rep["rating_mean"] = float(df.rating.mean())
    rep["rating_std"] = float(df.rating.std())
    rep["rating_values"] = sorted(float(v) for v in df.rating.unique())
    rep["timestamp_min"] = int(df.timestamp.min())
    rep["timestamp_max"] = int(df.timestamp.max())
    rep["time_span_days"] = round((df.timestamp.max() - df.timestamp.min()) / 86400, 1)
    rep["density"] = round(len(df) / (rep["users"] * rep["movies"]), 5)

    # ---- 半边划分 ----
    df["half"] = np.where(df.userId <= ML_USER_MAX, "ml", "synthetic")
    half_stats = {}
    for h, g in df.groupby("half"):
        half_stats[h] = {
            "rows": int(len(g)),
            "users": int(g.userId.nunique()),
            "movies": int(g.movieId.nunique()),
            "rating_mean": round(float(g.rating.mean()), 4),
            "rating_std": round(float(g.rating.std()), 4),
            "rating_min": float(g.rating.min()),
            "rating_max": float(g.rating.max()),
            "unique_rating_values": sorted(float(v) for v in g.rating.unique()),
            "t_min": pd.to_datetime(int(g.timestamp.min()), unit="s").strftime("%Y-%m-%d"),
            "t_max": pd.to_datetime(int(g.timestamp.max()), unit="s").strftime("%Y-%m-%d"),
            "ratings_per_user_median": float(g.groupby("userId").size().median()),
            "ratings_per_movie_median": float(g.groupby("movieId").size().median()),
        }
    rep["half_stats"] = half_stats

    # ---- 半边是否隔离 ----
    u_ml = df.userId <= ML_USER_MAX
    m_ml = df.movieId <= ML_MOVIE_MAX
    rep["cross_half_rows"] = int((u_ml != m_ml).sum())

    # ---- 评分分布（整体 / 分半）----
    overall = df.rating.value_counts().sort_index()
    rep["rating_hist_overall"] = {str(k): int(v) for k, v in overall.items()}
    rep["rating_hist_by_half"] = {
        h: {str(k): int(v) for k, v in g.rating.value_counts().sort_index().items()}
        for h, g in df.groupby("half")
    }

    # ---- 测试集校验 ----
    tp = pd.read_csv(DATA / "test_pairs.csv")
    rep["test"] = {
        "rows": int(len(tp)),
        "users": int(tp.userId.nunique()),
        "movies": int(tp.movieId.nunique()),
        "duplicated_pairs": int(tp.duplicated(["userId", "movieId"]).sum()),
        "t_min": pd.to_datetime(int(tp.timestamp.min()), unit="s").strftime("%Y-%m-%d %H:%M"),
        "t_max": pd.to_datetime(int(tp.timestamp.max()), unit="s").strftime("%Y-%m-%d %H:%M"),
        "rows_user_in_train": int(tp.userId.isin(df.userId.unique()).sum()),
        "rows_movie_in_train": int(tp.movieId.isin(df.movieId.unique()).sum()),
    }
    # 测试对是否出现在训练里（同 user+movie）；用 int64 复合键 + 有序查找，避免建巨型 set
    def pair_key(u, m):
        return u.astype(np.int64) * 1_000_000 + m.astype(np.int64)

    train_keys = np.sort(pair_key(df.userId.to_numpy(), df.movieId.to_numpy()))
    test_keys = pair_key(tp.userId.to_numpy(), tp.movieId.to_numpy())
    pos = np.searchsorted(train_keys, test_keys)
    pos = np.clip(pos, 0, len(train_keys) - 1)
    rep["test"]["pairs_already_in_train"] = int((train_keys[pos] == test_keys).sum())
    del train_keys

    # 每个测试用户的训练历史条数
    hist = tp[["userId"]].merge(df.groupby("userId").size().rename("n").reset_index(), on="userId", how="left")
    rep["test"]["train_history_per_test_user"] = {
        "min": int(hist.n.min()), "p25": float(hist.n.quantile(.25)),
        "median": float(hist.n.median()), "p75": float(hist.n.quantile(.75)), "max": int(hist.n.max()),
    }

    # ---- 测试是否晚于该用户全部训练评分 ----
    last_train = df.groupby("userId")["timestamp"].max().rename("last_ts").reset_index()
    chk = tp.merge(last_train, on="userId", how="left")
    rep["test"]["rows_after_user_last_train"] = int((chk.timestamp >= chk.last_ts).sum())
    rep["test"]["rows_before_user_last_train"] = int((chk.timestamp < chk.last_ts).sum())

    # ---- 分半测试集 ----
    rep["test"]["by_half"] = {
        "ml": int((tp.userId <= ML_USER_MAX).sum()),
        "synthetic": int((tp.userId > ML_USER_MAX).sum()),
    }

    # ---- 可预测性上界：全局均值基线 ----
    mu = float(df.rating.mean())
    # 用每用户留最后 5% 做快速验证
    df_sorted = df.sort_values(["userId", "timestamp"], kind="stable")
    rank_from_end = df_sorted.groupby("userId").cumcount(ascending=False)
    n_per_user = df_sorted.groupby("userId")["userId"].transform("size")
    hold = rank_from_end < np.maximum(2, (n_per_user * 0.05).astype(int))
    va, tr = df_sorted[hold], df_sorted[~hold]
    rep["holdout"] = {"rows": int(len(va)), "train_rows": int(len(tr))}

    def rmse(pred, y):
        return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))

    base = {}
    base["global_mean"] = rmse(np.full(len(va), tr.rating.mean()), va.rating)
    ub = tr.groupby("userId").rating.mean()
    base["user_mean"] = rmse(va.userId.map(ub).fillna(tr.rating.mean()), va.rating)
    ib = tr.groupby("movieId").rating.mean()
    base["item_mean"] = rmse(va.movieId.map(ib).fillna(tr.rating.mean()), va.rating)
    # 偏置式：μ + b_u + b_i，交替最小二乘（各 3 轮），用 bincount 做分组求和
    mu = float(tr.rating.mean())
    nu_max, nm_max = int(df.userId.max()), int(df.movieId.max())
    y = tr.rating.to_numpy(np.float64)
    uid = tr.userId.to_numpy(np.int64)
    mid = tr.movieId.to_numpy(np.int64)
    cnt_u = np.bincount(uid, minlength=nu_max + 1).astype(np.float64)
    cnt_m = np.bincount(mid, minlength=nm_max + 1).astype(np.float64)
    bu = np.zeros(nu_max + 1)
    bi = np.zeros(nm_max + 1)
    for _ in range(3):
        resid = y - mu - bi[mid]
        bu = np.bincount(uid, weights=resid, minlength=nu_max + 1) / (cnt_u + 12.0)
        resid = y - mu - bu[uid]
        bi = np.bincount(mid, weights=resid, minlength=nm_max + 1) / (cnt_m + 8.0)
    pred_bias = mu + bu[va.userId.to_numpy(np.int64)] + bi[va.movieId.to_numpy(np.int64)]
    base["bias_als"] = rmse(pred_bias, va.rating)
    base["bias_als_clipped"] = rmse(np.clip(pred_bias, 0.5, 5.0), va.rating)
    rep["baselines_quick"] = {k: round(v, 5) for k, v in base.items()}

    rep["elapsed_seconds"] = round(time.time() - t0, 1)
    (REPORTS / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印摘要 ----
    print("\n" + "=" * 62)
    print(f"总评分 {rep['rows']:,} | 用户 {rep['users']:,} | 电影 {rep['movies']:,}")
    print(f"评分 {rep['rating_min']}~{rep['rating_max']} 均值 {rep['rating_mean']:.4f} 标准差 {rep['rating_std']:.4f}")
    print(f"时间 {pd.to_datetime(rep['timestamp_min'], unit='s').date()} → {pd.to_datetime(rep['timestamp_max'], unit='s').date()}")
    print(f"重复 (user,movie) 对 {rep['duplicated_pairs']:,} | 跨半边行数 {rep['cross_half_rows']:,}")
    print("-" * 62)
    for h, s in half_stats.items():
        print(f"{h:10} 行 {s['rows']:>10,} | 用户 {s['users']:>7,} | 电影 {s['movies']:>7,} "
              f"| 均分 {s['rating_mean']:.3f} 标准差 {s['rating_std']:.3f}")
        print(f"{'':10} 每人中位 {s['ratings_per_user_median']:.0f} 条 | 每片中位 {s['ratings_per_movie_median']:.0f} 条 "
              f"| {s['t_min']} → {s['t_max']}")
        print(f"{'':10} 评分取值 {s['unique_rating_values'][:12]}{' ...' if len(s['unique_rating_values']) > 12 else ''}")
    print("-" * 62)
    t = rep["test"]
    print(f"测试集 {t['rows']:,} 行 | 用户 {t['users']:,} | 电影 {t['movies']:,} | 重复对 {t['duplicated_pairs']}")
    print(f"  覆盖：用户全在训练内？{t['rows_user_in_train'] == t['rows']} | 电影全在训练内？{t['rows_movie_in_train'] == t['rows']}（按行计）")
    print(f"  与训练重复的 pair 数 {t['pairs_already_in_train']}")
    print(f"  训练历史条数 min/p25/中位/p75/max = {t['train_history_per_test_user']}")
    print(f"  时间关系：晚于该用户最后一条训练记录 {t['rows_after_user_last_train']:,} 行，"
          f"早于 {t['rows_before_user_last_train']:,} 行")
    print(f"  分半：ML {t['by_half']['ml']:,} | 合成 {t['by_half']['synthetic']:,}")
    print("-" * 62)
    print(f"快速基线（用户最后 5% 留出，{rep['holdout']['rows']:,} 行）：")
    for k, v in rep["baselines_quick"].items():
        print(f"  {k:22} RMSE {v:.5f}")
    print("=" * 62)
    log(f"审计完成 {rep['elapsed_seconds']}s → reports/audit.json")


if __name__ == "__main__":
    main()
