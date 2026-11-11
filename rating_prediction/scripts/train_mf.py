# -*- coding: utf-8 -*-
"""带偏置的矩阵分解（Bias-MF），显式反馈 + MSE。

预测式：  r̂ = μ + b_u + b_i + p_u · q_i

设计要点
  - 偏置初值由 ALS 偏置解给出（比零初始化收敛快很多），因子零初始化
  - 因子矩阵用稀疏梯度 + SparseAdam：40 万用户 × 64 维，若用稠密梯度每步要写 100 MB，
    稀疏梯度只更新 batch 内出现过的行
  - 因子上的 L2 用「batch 内显式惩罚」实现（SparseAdam 不支持 weight_decay）
  - 可选时间衰减权重：越近的评分权重越高（测试集是时间外推，近期行为更有信息量）
  - 每轮在留出集上评估，保留最优轮次；预测统一裁剪到 [0.5, 5.0]

用法：
    python scripts/train_mf.py --dim 64 --epochs 15 --seed 1 --tag mf64
    python scripts/train_mf.py --final --dim 64 --epochs 15 --seed 1 --tag mf64_final
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SparseAdam

import common as C


class BiasMF(nn.Module):
    """带置信门控的 Bias-MF。

    gate_u = n_u / (n_u + kappa)：用户历史越短，因子项被压得越狠。
    本数据里留出用户的历史中位数只有 6 条，不加门控时 64 维因子会把 6 条记录记死，
    验证集 RMSE 反而比纯偏置更差（实测 0.9306 vs 0.9240）。kappa=0 退化为普通 MF。
    """

    def __init__(self, n_users: int, n_movies: int, dim: int, mu: float,
                 bu_init: np.ndarray | None = None, bi_init: np.ndarray | None = None,
                 kappa: float = 0.0, user_counts: np.ndarray | None = None):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim, sparse=True)
        self.item_emb = nn.Embedding(n_movies, dim, sparse=True)
        self.user_bias = nn.Embedding(n_users, 1, sparse=True)
        self.item_bias = nn.Embedding(n_movies, 1, sparse=True)
        self.global_bias = nn.Parameter(torch.tensor([mu], dtype=torch.float32))

        # 因子必须小随机初始化：若零初始化，p·q 恒为 0 且梯度恒为 0，因子永远不会更新，
        # 模型会退化成纯偏置模型（实测 dim=16/32、kappa=0/10 结果逐位相同，就是这个原因）
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        if bu_init is not None:
            with torch.no_grad():
                self.user_bias.weight.copy_(torch.from_numpy(bu_init).view(-1, 1))
        if bi_init is not None:
            with torch.no_grad():
                self.item_bias.weight.copy_(torch.from_numpy(bi_init).view(-1, 1))

        if kappa > 0 and user_counts is not None:
            g = user_counts.astype(np.float32) / (user_counts.astype(np.float32) + kappa)
        else:
            g = np.ones(n_users, dtype=np.float32)
        self.register_buffer("gate", torch.from_numpy(g))

    def forward(self, u, i):
        p = self.user_emb(u)
        q = self.item_emb(i)
        return (self.global_bias
                + self.user_bias(u).squeeze(1)
                + self.item_bias(i).squeeze(1)
                + self.gate[u] * (p * q).sum(1))


@torch.no_grad()
def predict(model: BiasMF, u: np.ndarray, i: np.ndarray, device, batch: int = 262144) -> np.ndarray:
    model.eval()
    out = np.empty(len(u), dtype=np.float32)
    for s in range(0, len(u), batch):
        e = min(s + batch, len(u))
        uu = torch.from_numpy(u[s:e].astype(np.int64)).to(device, non_blocking=True)
        ii = torch.from_numpy(i[s:e].astype(np.int64)).to(device, non_blocking=True)
        out[s:e] = model(uu, ii).float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=2e-4, help="因子的 batch 内 L2 系数")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="用户历史门控强度，gate=n/(n+kappa)，0 表示不门控")
    ap.add_argument("--lam-u", type=float, default=50.0, help="用户偏置收缩强度")
    ap.add_argument("--lam-i", type=float, default=2.0, help="条目偏置收缩强度")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-half-life", type=float, default=0.0,
                    help="时间衰减半衰期（天），0 表示不加权")
    ap.add_argument("--final", action="store_true", help="用全量数据训练并预测 test_pairs")
    ap.add_argument("--split", choices=["cutoff", "tail", "sparse"], default="cutoff",
                    help="验证切分口径：cutoff=固定日历截断（间隔中位 120 天）；"
                         "tail=时间尾部 10 万条（对齐老师测试集，间隔中位 0.1 天）；"
                         "sparse=稀疏历史模拟（1,043 位用户只留 5 条历史，对齐测试集最重的群体）")
    ap.add_argument("--half", choices=["all", "ml", "syn"], default="all",
                    help="只用某半边训练（两个半边 ID 空间互相隔离，可独立建模）")
    ap.add_argument("--tag", default="mf")
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.log(f"device={device} dim={args.dim} epochs={args.epochs} seed={args.seed} "
          f"final={args.final} half={args.half} kappa={args.kappa} lr={args.lr} l2={args.l2}")

    t0 = time.time()
    df = C.load_ratings()
    n_users = int(df.userId.max()) + 1
    n_movies = int(df.movieId.max()) + 1

    if args.final:
        train_mask = np.ones(len(df), dtype=bool)
        valid_mask = np.zeros(len(df), dtype=bool)
        split_info = {"mode": "final", "train_rows": int(len(df))}
    else:
        if args.split == "tail":
            train_mask, valid_mask = C.tail_split(df)
        elif args.split == "sparse":
            train_mask, valid_mask = C.sparse_sim_split(df)
        else:
            train_mask, valid_mask = C.temporal_split(df)
        split_info = C.describe_split(df, train_mask, valid_mask)

    half_code = df.half.to_numpy(np.int8)
    if args.half != "all":
        want = 0 if args.half == "ml" else 1
        train_mask &= half_code == want
        if not args.final:
            valid_mask &= half_code == want
    C.log(f"训练行 {int(train_mask.sum()):,} / 留出行 {int(valid_mask.sum()):,}  (载入 {time.time()-t0:.0f}s)")
    if not args.final:
        C.log(f"  留出明细 {json.dumps(split_info['valid_by_half'])} 每人中位 {split_info['valid_per_user_median']:.0f}")

    uid = df.userId.to_numpy(np.int32)[train_mask]
    mid = df.movieId.to_numpy(np.int32)[train_mask]
    y = df.rating.to_numpy(np.float32)[train_mask]
    ts = df.timestamp.to_numpy(np.int64)[train_mask]

    # ---- 偏置初值：在训练段上拟合 ALS 偏置（收缩强度用网格搜出的最优值）----
    bias = C.BiasModel(n_users, n_movies, lam_u=args.lam_u, lam_i=args.lam_i).fit(
        uid, mid, y, rounds=8)
    C.log(f"偏置初值完成 μ={bias.mu:.4f} (lam_u={args.lam_u} lam_i={args.lam_i})")

    # 用户历史条数（训练段），用于因子门控
    user_counts = np.bincount(uid.astype(np.int64), minlength=n_users).astype(np.float32)

    # ---- 时间权重 ----
    w = None
    if args.time_half_life > 0:
        age_days = (ts.max() - ts) / 86400.0
        w = np.power(0.5, age_days / args.time_half_life).astype(np.float32)
        C.log(f"时间权重：半衰期 {args.time_half_life:.0f} 天，最旧样本权重 {w.min():.4f}")

    model = BiasMF(n_users, n_movies, args.dim, bias.mu, bias.bu, bias.bi,
                   kappa=args.kappa, user_counts=user_counts).to(device)
    opt_sparse = SparseAdam(list(model.user_emb.parameters()) + list(model.item_emb.parameters())
                            + list(model.user_bias.parameters()) + list(model.item_bias.parameters()),
                            lr=args.lr)
    opt_dense = AdamW([model.global_bias], lr=args.lr, weight_decay=0.0)

    n = len(y)
    steps = (n + args.batch_size - 1) // args.batch_size
    C.log(f"每轮 {steps:,} 步（batch={args.batch_size}）")

    valid_u = valid_mid = valid_y = None
    if not args.final:
        valid_u = df.userId.to_numpy(np.int32)[valid_mask]
        valid_mid = df.movieId.to_numpy(np.int32)[valid_mask]
        valid_y = df.rating.to_numpy(np.float32)[valid_mask]

    best = {"rmse": float("inf"), "epoch": 0, "state": None}
    rng = np.random.default_rng(args.seed)
    y_t = torch.from_numpy(y)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n)
        te = time.time()
        total_loss = 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            uu = torch.from_numpy(uid[idx].astype(np.int64)).to(device, non_blocking=True)
            ii = torch.from_numpy(mid[idx].astype(np.int64)).to(device, non_blocking=True)
            yy = y_t[idx].to(device, non_blocking=True)

            pred = model(uu, ii)
            err = pred - yy
            if w is None:
                loss = (err * err).mean()
            else:
                ww = torch.from_numpy(w[idx]).to(device, non_blocking=True)
                loss = (ww * err * err).sum() / ww.sum()
            if args.l2 > 0:
                p = model.user_emb(uu)
                q = model.item_emb(ii)
                loss = loss + args.l2 * ((p * p).sum() + (q * q).sum()) / len(idx)

            opt_sparse.zero_grad(set_to_none=True)
            opt_dense.zero_grad(set_to_none=True)
            loss.backward()
            opt_sparse.step()
            opt_dense.step()
            total_loss += float(loss.detach()) * len(idx)

        msg = f"epoch {epoch:02d}/{args.epochs} train_MSE {total_loss/n:.4f} ({time.time()-te:.0f}s)"
        if not args.final:
            pv = np.clip(predict(model, valid_u, valid_mid, device), C.RATING_MIN, C.RATING_MAX)
            v = C.rmse(pv, valid_y)
            msg += f" | valid_RMSE {v:.5f}"
            if v < best["rmse"] - 1e-5:
                best = {"rmse": v, "epoch": epoch,
                        "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}
                msg += "  *"
        C.log(msg)

    if not args.final:
        model.load_state_dict(best["state"])
        C.log(f"回载最优 epoch={best['epoch']} valid_RMSE={best['rmse']:.5f}")

    # ---- 保存 / 预测 ----
    tag = args.tag
    if args.final:
        tp = C.load_test_pairs()
        p = predict(model, tp.userId.to_numpy(), tp.movieId.to_numpy(), device)
        p_clip = np.clip(p, C.RATING_MIN, C.RATING_MAX)
        np.save(C.MODELS / f"pred_test_{tag}.npy", p_clip)
        meta = {"tag": tag, "mode": "final", "dim": args.dim, "epochs": args.epochs,
                "seed": args.seed, "l2": args.l2, "time_half_life": args.time_half_life,
                "rows": int(len(tp)), "pred_mean": float(p_clip.mean()),
                "pred_std": float(p_clip.std()), "split": split_info}
        (C.REPORTS / f"pred_test_{tag}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        C.log(f"测试集预测完成 均值 {p_clip.mean():.4f} 标准差 {p_clip.std():.4f} → pred_test_{tag}.npy")
    else:
        pv_raw = predict(model, valid_u, valid_mid, device)
        pv = np.clip(pv_raw, C.RATING_MIN, C.RATING_MAX)
        overall = C.rmse(pv, valid_y)
        half = df.half.to_numpy(np.int8)[valid_mask]
        rec = {
            "tag": tag, "mode": "valid", "dim": args.dim, "epochs": args.epochs,
            "epoch_used": best["epoch"], "seed": args.seed, "l2": args.l2,
            "time_half_life": args.time_half_life,
            "rmse": overall, "mae": C.mae(pv, valid_y),
            "rmse_unclipped": C.rmse(pv_raw, valid_y),
            "rmse_ml": C.rmse(pv[half == 0], valid_y[half == 0]),
            "rmse_synthetic": C.rmse(pv[half == 1], valid_y[half == 1]),
            "split": split_info, "elapsed_seconds": round(time.time() - t0, 1),
        }
        np.savez(C.MODELS / f"valid_pred_{tag}.npz", uid=valid_u, mid=valid_mid,
                 y=valid_y, pred=pv, half=half)
        # 全表预测：供 GBDT 在「拟合集」上使用 MF 输出作为特征
        p_all = np.clip(predict(model, df.userId.to_numpy(), df.movieId.to_numpy(), device),
                        C.RATING_MIN, C.RATING_MAX)
        np.savez(C.MODELS / f"fit_pred_{tag}.npz", pred=p_all)
        C.log(f"已保存全表预测 fit_pred_{tag}.npz（供 GBDT 特征使用）")
        (C.REPORTS / f"metrics_{tag}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        C.append_experiment({"model": f"biasmf_dim{args.dim}_{tag}", **{k: rec[k] for k in
                            ("rmse", "mae", "rmse_ml", "rmse_synthetic", "epoch_used", "elapsed_seconds")}})
        C.log(f"留出集 RMSE {overall:.5f} (真实 {rec['rmse_ml']:.5f} / 合成 {rec['rmse_synthetic']:.5f}) "
              f"MAE {rec['mae']:.5f}")

    torch.save({"state": model.state_dict(), "args": vars(args),
                "n_users": n_users, "n_movies": n_movies,
                "best_rmse": None if args.final else best["rmse"]},
               C.MODELS / f"mf_{tag}.pt")


if __name__ == "__main__":
    main()
