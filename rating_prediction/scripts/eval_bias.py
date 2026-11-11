# -*- coding: utf-8 -*-
"""偏置基线：μ + b_u + b_i，在本地验证集上与 MF 同口径评估。

同时做一轮超参搜索（b_u / b_i 的收缩强度），因为不同收缩对 RMSE 影响可观。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cutoff", "tail"], default="cutoff",
                    help="cutoff=固定日历截断；tail=时间尾部（对齐老师测试集口径）")
    ap.add_argument("--grid", action="store_true", help="是否重跑收缩强度网格（较慢）")
    args = ap.parse_args()
    suffix = "" if args.split == "cutoff" else "_tail"

    t0 = time.time()
    df = C.load_ratings()
    n_users, n_movies = int(df.userId.max()) + 1, int(df.movieId.max()) + 1
    if args.split == "tail":
        train_mask, valid_mask = C.tail_split(df)
    else:
        train_mask, valid_mask = C.temporal_split(df)
    info = C.describe_split(df, train_mask, valid_mask)
    C.log(f"训练 {int(train_mask.sum()):,} / 留出 {int(valid_mask.sum()):,}")

    uid = df.userId.to_numpy(np.int32)[train_mask]
    mid = df.movieId.to_numpy(np.int32)[train_mask]
    y = df.rating.to_numpy(np.float32)[train_mask]
    vu = df.userId.to_numpy(np.int32)[valid_mask]
    vm = df.movieId.to_numpy(np.int32)[valid_mask]
    vy = df.rating.to_numpy(np.float32)[valid_mask]
    half = df.half.to_numpy(np.int8)[valid_mask]

    if args.grid:
        best = None
        rows = []
        for lam_u in (2, 5, 10, 20, 50):
            for lam_i in (2, 5, 10, 20, 50):
                m = C.BiasModel(n_users, n_movies, lam_u=lam_u, lam_i=lam_i).fit(uid, mid, y, rounds=5)
                p = np.clip(m.predict(vu, vm), C.RATING_MIN, C.RATING_MAX)
                r = C.rmse(p, vy)
                rows.append({"lam_u": lam_u, "lam_i": lam_i, "rmse": r})
                if best is None or r < best["rmse"]:
                    best = {"lam_u": lam_u, "lam_i": lam_i, "rmse": r}
    else:
        # 复用 cutoff 口径上搜出的最优收缩，避免重复搜索
        rows = []
        best = {"lam_u": 50, "lam_i": 2, "rmse": float("nan")}

    C.log(f"收缩 lam_u={best['lam_u']} lam_i={best['lam_i']} RMSE={best['rmse']:.5f}")

    m = C.BiasModel(n_users, n_movies, lam_u=best["lam_u"], lam_i=best["lam_i"]).fit(uid, mid, y, rounds=8)
    m.save(C.MODELS / "bias_full.npz")
    p_raw = m.predict(vu, vm)
    p = np.clip(p_raw, C.RATING_MIN, C.RATING_MAX)

    rec = {
        "tag": "bias" + suffix, "mode": "valid", "rmse": C.rmse(p, vy), "mae": C.mae(p, vy),
        "rmse_unclipped": C.rmse(p_raw, vy),
        "rmse_ml": C.rmse(p[half == 0], vy[half == 0]),
        "rmse_synthetic": C.rmse(p[half == 1], vy[half == 1]),
        "best_shrinkage": {"lam_u": best["lam_u"], "lam_i": best["lam_i"]},
        "grid": rows, "split": info, "elapsed_seconds": round(time.time() - t0, 1),
    }
    (C.REPORTS / f"metrics_bias{suffix}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    C.append_experiment({"model": "bias_als" + suffix, **{k: rec[k] for k in
                        ("rmse", "mae", "rmse_ml", "rmse_synthetic", "elapsed_seconds")}})
    np.savez(C.MODELS / f"valid_pred_bias{suffix}.npz", uid=vu, mid=vm, y=vy, pred=p, half=half)
    C.log(f"偏置基线 RMSE {rec['rmse']:.5f} (真实 {rec['rmse_ml']:.5f} / 合成 {rec['rmse_synthetic']:.5f}) "
          f"MAE {rec['mae']:.5f}")


if __name__ == "__main__":
    main()
