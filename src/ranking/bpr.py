# -*- coding: utf-8 -*-
"""BPR 矩阵分解（Bayesian Personalized Ranking）。

相对"预测评分"的 MF，这里直接优化"正样本得分高于负样本"的成对目标，
与 NDCG@10 的排序语义一致；这也是旧模型（rating MF）在本项目里的主要改造点。

支持从 Big_Data 的 movielens_mf.pt 热启动：两份词表完全一致
（同一份 ratings.csv 做 factorize(sort=True)），embedding 形状可直接对齐。

用法：
    python -m src.ranking.bpr --config configs/local.yaml --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..candidate_set import CandidateSet, write_top10
from ..data_io import load_config, load_table, resolve
from ..metrics import ndcg_at_k

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def score(self, u, i):
        return (self.user(u) * self.item(i)).sum(dim=1)

    def bpr_loss(self, u, i, j, reg: float = 0.0) -> torch.Tensor:
        pu, qi, qj = self.user(u), self.item(i), self.item(j)
        x = (pu * (qi - qj)).sum(dim=1)
        loss = -nn.functional.logsigmoid(x).mean()
        if reg > 0:
            loss = loss + reg * (pu.norm(2).pow(2) + qi.norm(2).pow(2) + qj.norm(2).pow(2)) / u.numel()
        return loss


def build_interactions(processed: Path, vocab: dict, threshold: float, split: str = "train"):
    table = load_table(processed / f"{split}.npz")
    user_ids = vocab["userId"]
    movie_ids = vocab["movieId"]
    u_lookup = np.full(int(user_ids.max()) + 1, -1, dtype=np.int32)
    u_lookup[user_ids] = np.arange(len(user_ids), dtype=np.int32)
    m_lookup = np.full(int(movie_ids.max()) + 1, -1, dtype=np.int32)
    m_lookup[movie_ids] = np.arange(len(movie_ids), dtype=np.int32)
    mask = table["rating"] >= threshold
    u = u_lookup[table["userId"][mask]]
    i = m_lookup[table["movieId"][mask]]
    if (u < 0).any() or (i < 0).any():
        raise ValueError("评分表出现词表之外的 id")
    return u.astype(np.int32), i.astype(np.int32)


def load_pretrained(model: BPRMF, path: Path) -> bool:
    if not path.exists():
        log(f"未找到预训练模型 {path}，改为随机初始化")
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    with torch.no_grad():
        model.user.weight.copy_(state["user_embedding.weight"].float())
        model.item.weight.copy_(state["movie_embedding.weight"].float())
    log(f"已从 {path.name} 热启动（epoch={ckpt.get('epoch')}，valid_RMSE={ckpt.get('valid_rmse', float('nan')):.4f}）")
    return True


@torch.no_grad()
def score_pairs(user_emb: torch.Tensor, item_emb: torch.Tensor,
                ucodes: np.ndarray, icodes: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """分批计算 (u, i) 的点积得分，避免一次性 gather 全部候选项。"""
    out = np.empty(len(ucodes), dtype=np.float32)
    for s in range(0, len(ucodes), chunk):
        e = min(s + chunk, len(ucodes))
        u = torch.from_numpy(ucodes[s:e]).to(DEVICE, non_blocking=True)
        i = torch.from_numpy(icodes[s:e]).to(DEVICE, non_blocking=True)
        out[s:e] = (user_emb[u] * item_emb[i]).sum(dim=1).cpu().numpy()
    return out


def evaluate_ndcg(model: BPRMF, cand: CandidateSet, users: np.ndarray, k: int = 10) -> float:
    """在给定用户子集上计算平均 NDCG@k（用于监控训练）。"""
    user_emb = model.user.weight.detach()
    item_emb = model.item.weight.detach()
    values = []
    for u in users:
        sl = cand.user_slice(int(u))
        if sl.stop == sl.start:
            continue
        ucode = np.full(sl.stop - sl.start, u - 1, dtype=np.int32)
        scores = score_pairs(user_emb, item_emb, ucode, cand.item_codes[sl])
        pos = set(cand.mid[sl][cand.label[sl] == 1].tolist())
        if not pos:
            continue
        top = cand.mid[sl][np.argsort(-scores, kind="stable")[:k]].tolist()
        values.append(ndcg_at_k(top, pos, k))
    return float(np.mean(values)) if values else 0.0


def train(args) -> dict:
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])
    models_dir = resolve(cfg["paths"].get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = load_table(processed / "vocab.npz")
    n_users, n_items = len(vocab["userId"]), len(vocab["movieId"])
    u, i = build_interactions(processed, vocab, cfg["positive_threshold"])
    log(f"训练交互（rating>={cfg['positive_threshold']}）：{len(u):,}，用户 {n_users:,}，电影 {n_items:,}")

    model = BPRMF(n_users, n_items, args.dim).to(DEVICE)
    if args.init_from:
        load_pretrained(model, resolve(args.init_from))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    item_counts = np.bincount(i, minlength=n_items).astype(np.float64)
    if args.neg_power == 0.0:
        neg_prob = None
    else:
        neg_prob = item_counts ** args.neg_power
        neg_prob = neg_prob / neg_prob.sum()

    cand_valid = CandidateSet(processed / "candidates_valid.npz", processed / "vocab.npz")
    valid_users = np.flatnonzero(cand_valid.counts)[: args.eval_users]
    log(f"验证监控用户数：{len(valid_users):,}")

    rng = np.random.default_rng(args.seed)
    n = len(u)
    steps = (n + args.batch - 1) // args.batch
    best_ndcg, best_state, best_epoch = -1.0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        t0 = time.perf_counter()
        total_loss, seen = 0.0, 0
        for step in range(steps):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            ub = torch.from_numpy(u[idx]).to(DEVICE, non_blocking=True)
            ib = torch.from_numpy(i[idx]).to(DEVICE, non_blocking=True)
            jb_np = rng.choice(n_items, size=len(idx), p=neg_prob) if neg_prob is not None \
                else rng.integers(0, n_items, size=len(idx))
            jb = torch.from_numpy(jb_np.astype(np.int64)).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = model.bpr_loss(ub, ib, jb, reg=args.reg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)

        train_seconds = time.perf_counter() - t0
        ndcg_valid = evaluate_ndcg(model, cand_valid, valid_users)
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(seen, 1), 5),
                        "valid_ndcg@10": round(ndcg_valid, 5), "seconds": round(train_seconds, 1)})
        log(f"epoch {epoch}/{args.epochs} loss={total_loss / max(seen, 1):.5f} "
            f"valid_NDCG@10={ndcg_valid:.4f} 用时 {train_seconds:.0f}s")

        if ndcg_valid > best_ndcg:
            best_ndcg, best_epoch = ndcg_valid, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"回载最优 epoch={best_epoch}（valid NDCG@10={best_ndcg:.4f}）")

    model_path = models_dir / f"{args.model_name}.pt"
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "n_users": n_users, "n_items": n_items, "dim": args.dim,
        "epoch": best_epoch, "valid_ndcg@10": best_ndcg, "args": vars(args),
    }, model_path)
    log(f"模型已保存：{model_path}")

    return {"model": "bpr_mf", "valid_ndcg@10": best_ndcg, "best_epoch": best_epoch,
            "history": history, "model_path": str(model_path), "device": DEVICE}


def predict(args) -> dict:
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])
    models_dir = resolve(cfg["paths"].get("models_dir", "models"))
    ckpt = torch.load(models_dir / f"{args.model_name}.pt", map_location="cpu", weights_only=False)
    model = BPRMF(ckpt["n_users"], ckpt["n_items"], ckpt["dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cand = CandidateSet(processed / f"candidates_{args.split}.npz", processed / "vocab.npz")
    ucodes = (cand.uid - 1).astype(np.int32)

    t0 = time.perf_counter()
    scores = score_pairs(model.user.weight, model.item.weight, ucodes, cand.item_codes)
    scoring_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    columns = cand.topk(scores, k=10)
    ranking_seconds = time.perf_counter() - t1

    tag = "bpr" if args.model_name == "bpr_mf" else args.model_name
    out_path = results / (f"top10_{tag}.csv" if args.split == "test" else f"top10_{tag}_{args.split}.csv")
    write_top10(out_path, columns)
    n_users = len(np.unique(columns["userId"]))
    log(f"[bpr] 打分 {scoring_seconds:.1f}s，排序 {ranking_seconds:.1f}s，用户 {n_users:,}")
    log(f"[bpr] 结果写入 {out_path}")
    return {"pred_file": str(out_path), "users": int(n_users),
            "scoring_seconds": round(scoring_seconds, 1), "ranking_seconds": round(ranking_seconds, 1),
            "latency_ms_per_user": round((scoring_seconds + ranking_seconds) / max(n_users, 1) * 1000, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description="BPR 矩阵分解")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", default="all", choices=["train", "predict", "all"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--neg-power", type=float, default=0.0,
                        help="负采样按 count^power 加权，0 表示均匀采样")
    parser.add_argument("--eval-users", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--init-from", default=None,
                        help="热启动权重路径，例如 E:/VS_tools/Project/Big_Data/Big_Data/movielens_mf.pt")
    parser.add_argument("--model-name", default="bpr_mf", help="模型文件名（不含 .pt）")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(load_config(args.config)["seed"])

    summary = {}
    if args.mode in ("train", "all"):
        summary["train"] = train(args)
    if args.mode in ("predict", "all"):
        summary["predict"] = predict(args)

    results = resolve(load_config(args.config)["paths"]["results_dir"])
    if args.mode == "all":
        out = results / "metrics_bpr_train.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(json.dumps(summary.get("train", {}).get("history", []), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())