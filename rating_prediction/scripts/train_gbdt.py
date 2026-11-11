# -*- coding: utf-8 -*-
"""特征构建 + 梯度提升树（残差修正层）。

思路：MF 已经抓住了「用户 × 条目」的主要结构，这里让树去学它的残差里还能被
统计特征解释的部分——用户/条目活跃度、条目评分分散度、类型亲和度、时间新鲜度。

注意：所有统计量只用「拟合集」计算（验证时为训练段，提交时为全量），
避免把留出期的信息带进特征。

用法：
    python scripts/train_gbdt.py --mf a_d32l1e-4k10 --tag gbdt
    python scripts/train_gbdt.py --final --mf a_d32l1e-4k10 --tag gbdt_final
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor

import common as C


def load_genres() -> tuple[np.ndarray, list[str]]:
    mv = pd.read_csv(C.DATA / "movies.csv", dtype={"movieId": "int32", "genres": "string"})
    names = sorted({g for s in mv.genres.fillna("") for g in str(s).split("|") if g and g != "(no genres listed)"})
    idx = {g: k for k, g in enumerate(names)}
    n_movies = int(mv.movieId.max()) + 1
    G = np.zeros((n_movies, len(names)), dtype=np.float32)
    for mid, s in zip(mv.movieId.to_numpy(), mv.genres.fillna("").astype(str).to_numpy()):
        for g in s.split("|"):
            if g in idx:
                G[mid, idx[g]] = 1.0
    return G, names


class Stats:
    """一次遍历算好用户/条目统计量。"""

    def __init__(self, df: pd.DataFrame, mask: np.ndarray):
        u = df.userId.to_numpy(np.int64)
        m = df.movieId.to_numpy(np.int64)
        y = df.rating.to_numpy(np.float64)
        ts = df.timestamp.to_numpy(np.int64)
        um = u[mask]; mm = m[mask]; ym = y[mask]; tm = ts[mask]
        self.nu = int(df.userId.max()) + 1
        self.nm = int(df.movieId.max()) + 1
        self.mu = float(ym.mean())

        self.u_cnt = np.bincount(um, minlength=self.nu).astype(np.float32)
        self.i_cnt = np.bincount(mm, minlength=self.nm).astype(np.float32)
        self.u_sum = np.bincount(um, weights=ym, minlength=self.nu)
        self.i_sum = np.bincount(mm, weights=ym, minlength=self.nm)
        self.u_sq = np.bincount(um, weights=ym * ym, minlength=self.nu)
        self.i_sq = np.bincount(mm, weights=ym * ym, minlength=self.nm)
        self.u_mean = np.zeros(self.nu, dtype=np.float32)
        self.i_mean = np.zeros(self.nm, dtype=np.float32)
        ok_u = self.u_cnt > 0
        ok_i = self.i_cnt > 0
        self.u_mean[ok_u] = (self.u_sum[ok_u] / self.u_cnt[ok_u]).astype(np.float32)
        self.i_mean[ok_i] = (self.i_sum[ok_i] / self.i_cnt[ok_i]).astype(np.float32)
        self.u_std = np.sqrt(np.maximum(self.u_sq / np.maximum(self.u_cnt, 1) - self.u_mean ** 2, 0)).astype(np.float32)
        self.i_std = np.sqrt(np.maximum(self.i_sq / np.maximum(self.i_cnt, 1) - self.i_mean ** 2, 0)).astype(np.float32)
        self.u_last = np.zeros(self.nu, dtype=np.int64)
        self.i_last = np.zeros(self.nm, dtype=np.int64)
        np.maximum.at(self.u_last, um, tm)
        np.maximum.at(self.i_last, mm, tm)
        self.i_first = np.full(self.nm, tm.max(), dtype=np.int64)
        np.minimum.at(self.i_first, mm, tm)
        self.t_ref = int(tm.max())

        # 类型亲和度：用户画像（看过片子的类型加权平均）× 条目类型向量 的余弦
        self.G, self.genre_names = load_genres()
        R = sparse.csr_matrix((ym.astype(np.float32), (um, mm)), shape=(self.nu, self.nm))
        profile = R @ self.G                       # nu × n_genre
        norm_u = np.sqrt((profile ** 2).sum(1)) + 1e-8
        norm_i = np.sqrt((self.G ** 2).sum(1)) + 1e-8
        self.u_profile = (profile / norm_u[:, None]).astype(np.float32)
        self.g_norm = norm_i.astype(np.float32)

    def build(self, u: np.ndarray, m: np.ndarray, ts: np.ndarray, mf_pred: np.ndarray,
              half: np.ndarray, chunk: int = 1_000_000) -> tuple[np.ndarray, list[str]]:
        """分块组装特征。31.5M × 20 类型的中间矩阵若一次性展开会吃掉 2.5 GB，必须分块。"""
        n = len(u)
        names = ["mf_pred", "u_mean", "i_mean", "log_u_cnt", "log_i_cnt", "i_std", "u_std",
                 "u_recency_d", "i_recency_d", "i_age_d", "genre_aff", "half"]
        X = np.empty((n, len(names)), dtype=np.float32)
        days = 86400.0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            uu, mm, tt = u[s:e], m[s:e], ts[s:e]
            aff = (self.u_profile[uu] * self.G[mm]).sum(1) / self.g_norm[mm]
            X[s:e, 0] = mf_pred[s:e]
            X[s:e, 1] = self.u_mean[uu]
            X[s:e, 2] = self.i_mean[mm]
            X[s:e, 3] = np.log1p(self.u_cnt[uu])
            X[s:e, 4] = np.log1p(self.i_cnt[mm])
            X[s:e, 5] = self.i_std[mm]
            X[s:e, 6] = self.u_std[uu]
            X[s:e, 7] = (self.t_ref - self.u_last[uu]) / days
            X[s:e, 8] = (self.t_ref - self.i_last[mm]) / days
            X[s:e, 9] = (self.t_ref - self.i_first[mm]) / days
            X[s:e, 10] = aff
            X[s:e, 11] = half[s:e].astype(np.float32)
        return X, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf", required=True, help="作为特征的 MF tag")
    ap.add_argument("--tag", default="gbdt")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--max-rows", type=int, default=12_000_000, help="训练行数上限（随机抽样）")
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--leaves", type=int, default=63)
    args = ap.parse_args()

    t0 = time.time()
    df = C.load_ratings()
    half = df.half.to_numpy(np.int8)

    if args.final:
        fit_mask = np.ones(len(df), dtype=bool)
        pred_mask = np.zeros(len(df), dtype=bool)
    else:
        fit_mask, pred_mask = C.temporal_split(df)

    C.log("构建统计量…")
    st = Stats(df, fit_mask)
    C.log(f"  完成（{time.time()-t0:.0f}s）")

    # ---- 拟合集特征 ----
    idx = np.flatnonzero(fit_mask)
    rng = np.random.default_rng(2026)
    if len(idx) > args.max_rows:
        idx = np.sort(rng.choice(idx, size=args.max_rows, replace=False))
    C.log(f"拟合集抽样 {len(idx):,} 行")

    if args.final:
        mf_all = np.load(C.MODELS / f"pred_test_{args.mf}.npy")
    mf_fit = np.load(C.MODELS / f"valid_pred_{args.mf}.npz")
    # valid_pred 只覆盖留出集；拟合集上的 MF 预测需要用模型重算 → 这里用偏置近似会失真，
    # 因此改为：拟合集用 MF 在训练段上的预测（由 train_mf 另存），否则退化为只用统计特征。
    mf_fit_path = C.MODELS / f"fit_pred_{args.mf}.npz"
    if mf_fit_path.exists():
        z = np.load(mf_fit_path)
        mf_on_fit = z["pred"]
        assert len(mf_on_fit) == len(df), "fit_pred 长度需与全表一致（未选中的行置 0）"
        X_fit, names = st.build(df.userId.to_numpy()[idx], df.movieId.to_numpy()[idx],
                                df.timestamp.to_numpy()[idx], mf_on_fit[idx], half[idx])
    else:
        raise FileNotFoundError(f"缺少 {mf_fit_path}：需要先用 train_mf --save-fit-pred 生成")

    y_fit = df.rating.to_numpy(np.float32)[idx]
    # 在 MF 残差上拟合：让树只学「MF 还差多少」，而不是去近似一个恒等映射。
    # 直接以评分为目标时，树的输出是分段常数，会引入约 0.02 的离散化误差（实测 0.8750 vs 0.8539）。
    base_fit = X_fit[:, 0].astype(np.float64)
    resid = y_fit.astype(np.float64) - base_fit
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter, learning_rate=args.lr, max_leaf_nodes=args.leaves,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=25, random_state=2026, verbose=1,
    )
    C.log("训练 GBDT（残差目标）…")
    model.fit(X_fit, resid.astype(np.float32))
    C.log(f"  完成 {time.time()-t0:.0f}s，迭代 {model.n_iter_}")

    if args.final:
        tp = C.load_test_pairs()
        X_te, _ = st.build(tp.userId.to_numpy(), tp.movieId.to_numpy(),
                           tp.timestamp.to_numpy(), mf_all, tp.half.to_numpy())
        p = np.clip(mf_all.astype(np.float64) + model.predict(X_te), C.RATING_MIN, C.RATING_MAX)
        np.save(C.MODELS / f"pred_test_{args.tag}.npy", p.astype(np.float32))
        C.log(f"测试集预测 均值 {p.mean():.4f} 标准差 {p.std():.4f}")
    else:
        vu = df.userId.to_numpy()[pred_mask]
        vm = df.movieId.to_numpy()[pred_mask]
        vt = df.timestamp.to_numpy()[pred_mask]
        vy = df.rating.to_numpy(np.float32)[pred_mask]
        vh = half[pred_mask]
        X_va, _ = st.build(vu, vm, vt, mf_fit["pred"].astype(np.float32), vh)
        base_va = mf_fit["pred"].astype(np.float64)
        p = np.clip(base_va + model.predict(X_va), C.RATING_MIN, C.RATING_MAX)
        rec = {
            "tag": args.tag, "mf_feature": args.mf,
            "rmse": C.rmse(p, vy), "mae": C.mae(p, vy),
            "rmse_ml": C.rmse(p[vh == 0], vy[vh == 0]),
            "rmse_synthetic": C.rmse(p[vh == 1], vy[vh == 1]),
            "fit_rows": int(len(idx)), "n_iter": int(model.n_iter_),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        np.savez(C.MODELS / f"valid_pred_{args.tag}.npz", uid=vu, mid=vm, y=vy, pred=p, half=vh)
        (C.REPORTS / f"metrics_{args.tag}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        C.append_experiment({"model": args.tag, **{k: rec[k] for k in
                            ("rmse", "mae", "rmse_ml", "rmse_synthetic", "elapsed_seconds")}})
        C.log(f"GBDT RMSE {rec['rmse']:.5f} (真实 {rec['rmse_ml']:.5f} / 合成 {rec['rmse_synthetic']:.5f})")

    imp = None
    try:
        from sklearn.inspection import permutation_importance  # noqa: F401
    except Exception:
        pass


if __name__ == "__main__":
    main()
