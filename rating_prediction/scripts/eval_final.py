# -*- coding: utf-8 -*-
"""一次算齐报告需要的所有数字：基线阶梯、分段误差、半区差异，写 reports/final_eval.json。

用法：python scripts/eval_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

TAGS = ["V1", "T0", "V3"]          # 提交配置：κ=0、6 轮、三个 seed
GATE_ABLATION = {                  # 稀疏历史模拟集上的门控消融（同 seed=2）
    "kappa=0": 0.92358, "kappa=10": 0.92801, "kappa=100": 0.93949,
}
SPARSE_RMSE = 0.92433              # 稀疏群上与提交配置同时刻（第 6 轮）的 RMSE
SPARSE_SHARE = 65_168 / 100_000    # 测试集里稀疏群占比


def main() -> None:
    df = C.load_ratings()
    n_users, n_movies = int(df.userId.max()) + 1, int(df.movieId.max()) + 1
    trm, vam = C.tail_split(df)

    uid = df.userId.to_numpy(np.int32)[trm]
    mid = df.movieId.to_numpy(np.int32)[trm]
    y = df.rating.to_numpy(np.float32)[trm]

    vu = df.userId.to_numpy(np.int32)[vam]
    vm = df.movieId.to_numpy(np.int32)[vam]
    vy = df.rating.to_numpy(np.float32)[vam]
    half = df.half.to_numpy(np.int8)[vam].astype(int)

    cnt_u = np.bincount(uid.astype(np.int64), minlength=n_users).astype(np.float64)
    cnt_m = np.bincount(mid.astype(np.int64), minlength=n_movies).astype(np.float64)

    out: dict = {"split": C.describe_split(df, trm, vam)}

    # ---------- 基线阶梯 ----------
    mu = float(y.mean())
    ladder = {"全局均值": C.rmse(np.full(len(vy), mu), vy)}

    u_sum = np.bincount(uid.astype(np.int64), weights=y.astype(np.float64), minlength=n_users)
    i_sum = np.bincount(mid.astype(np.int64), weights=y.astype(np.float64), minlength=n_movies)
    u_mean = np.divide(u_sum, cnt_u, out=np.full(n_users, mu), where=cnt_u > 0)
    i_mean = np.divide(i_sum, cnt_m, out=np.full(n_movies, mu), where=cnt_m > 0)
    ladder["用户均值"] = C.rmse(np.clip(u_mean[vu], 0.5, 5.0), vy)
    ladder["电影均值"] = C.rmse(np.clip(i_mean[vm], 0.5, 5.0), vy)

    zb = np.load(C.MODELS / "valid_pred_bias_tail.npz")
    ladder["偏置 ALS"] = C.rmse(zb["pred"], zb["y"])

    per_model = {}
    mats = []
    for t in TAGS:
        z = np.load(C.MODELS / f"valid_pred_{t}.npz")
        p = z["pred"].astype(np.float64)
        mats.append(p)
        per_model[t] = {
            "rmse": C.rmse(np.clip(p, 0.5, 5.0), vy), "mae": C.mae(p, vy),
            "rmse_ml": C.rmse(p[half == 0], vy[half == 0]),
            "rmse_syn": C.rmse(p[half == 1], vy[half == 1]),
        }
    # 三 seed 等权平均 —— 与提交文件完全一致（不做权重优化，避免在验证集上再拟合一次）
    P = np.vstack(mats)
    w = np.ones(len(TAGS)) / len(TAGS)
    ens = np.clip(w @ P, 0.5, 5.0)
    per_model["ensemble"] = {
        "rmse": C.rmse(ens, vy), "mae": C.mae(ens, vy),
        "rmse_ml": C.rmse(ens[half == 0], vy[half == 0]),
        "rmse_syn": C.rmse(ens[half == 1], vy[half == 1]),
        "weights": dict(zip(TAGS, np.round(w, 4).tolist())),
    }
    ladder["Bias-MF（单 seed 最优）"] = min(v["rmse"] for k, v in per_model.items() if k != "ensemble")
    ladder["三 seed 等权融合"] = per_model["ensemble"]["rmse"]

    out["ladder"] = ladder
    out["models"] = per_model

    # ---------- 预期测试集 RMSE：按测试集的真实构成加权 ----------
    dense = per_model["ensemble"]["rmse"]
    mse_mix = SPARSE_SHARE * SPARSE_RMSE ** 2 + (1 - SPARSE_SHARE) * dense ** 2
    out["test_expectation"] = {
        "sparse_share": SPARSE_SHARE, "sparse_rmse": SPARSE_RMSE,
        "dense_rmse": dense, "predicted_test_rmse": float(np.sqrt(mse_mix)),
        "note": "稀疏群来自口径 C 的模拟实验；密集群来自口径 B 的三 seed 融合",
    }
    out["gate_ablation"] = GATE_ABLATION

    # ---------- 分段误差（融合模型）----------
    err2 = (vy - ens) ** 2
    n_u_row = cnt_u[vu]
    seg_u = []
    for a, b in [(0, 6), (6, 10), (10, 30), (30, 100), (100, 300), (300, 10 ** 9)]:
        m = (n_u_row >= a) & (n_u_row < b)
        if m.sum() == 0:
            continue
        seg_u.append({"bucket": f"{a}-{b if b < 10**9 else 'inf'}", "rows": int(m.sum()),
                      "share_pct": round(100 * m.mean(), 2),
                      "rmse": C.rmse(ens[m], vy[m]),
                      "err_share_pct": round(100 * err2[m].sum() / err2.sum(), 2)})
    out["segment_by_user_history"] = seg_u

    n_m_row = cnt_m[vm]
    seg_m = []
    for a, b in [(1, 10), (10, 50), (50, 200), (200, 1000), (1000, 10 ** 9)]:
        m = (n_m_row >= a) & (n_m_row < b)
        if m.sum() == 0:
            continue
        seg_m.append({"bucket": f"{a}-{b if b < 10**9 else 'inf'}", "rows": int(m.sum()),
                      "share_pct": round(100 * m.mean(), 2),
                      "rmse": C.rmse(ens[m], vy[m]),
                      "err_share_pct": round(100 * err2[m].sum() / err2.sum(), 2)})
    out["segment_by_movie_popularity"] = seg_m

    # ---------- 半区差异 ----------
    out["by_half"] = {
        "ml": {"rows": int((half == 0).sum()), "rmse": C.rmse(ens[half == 0], vy[half == 0]),
               "y_mean": float(vy[half == 0].mean()), "pred_mean": float(ens[half == 0].mean())},
        "syn": {"rows": int((half == 1).sum()), "rmse": C.rmse(ens[half == 1], vy[half == 1]),
                "y_mean": float(vy[half == 1].mean()), "pred_mean": float(ens[half == 1].mean())},
    }

    # ---------- 残差与校准 ----------
    resid = vy - ens
    out["residual"] = {"mean": float(resid.mean()), "std": float(resid.std()),
                       "p1": float(np.percentile(resid, 1)), "p99": float(np.percentile(resid, 99)),
                       "skew": float(pd.Series(resid).skew())}
    best = min(((C.rmse(np.clip(ens + d, 0.5, 5.0), vy), d) for d in np.arange(-0.08, 0.081, 0.005)))
    out["offset_probe"] = {"best_offset": float(best[1]), "best_rmse": float(best[0]),
                           "rmse_at_zero": C.rmse(ens, vy)}

    # ---------- 极值截断检查 ----------
    out["clip"] = {"pred_at_0.5": int((ens <= 0.5).sum()), "pred_at_5.0": int((ens >= 5.0).sum()),
                   "pred_mean": float(ens.mean()), "pred_std": float(ens.std())}

    (C.REPORTS / "final_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    C.log("阶梯：" + "  ".join(f"{k}={v:.5f}" for k, v in ladder.items()))
    C.log(f"融合权重 {per_model['ensemble']['weights']}")
    C.log(f"残差 均值 {resid.mean():+.4f} 标准差 {resid.std():.4f} | 最优偏移 {best[1]:+.3f} → {best[0]:.5f}")
    C.log("已写 reports/final_eval.json")


if __name__ == "__main__":
    main()
