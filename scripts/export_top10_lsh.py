#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务二补交付：用 item-kNN（与 LSH 同一口径的头部条目集合）生成可评测的 Top10。

背景：
    src/similarity/itemknn.py 只产出相似度/召回率报告与「头部 3,000 部 × 20 邻居」的表，
    没有独立推荐结果文件。本脚本复用同一条相似度路线（列归一化的二值 item 向量 +
    余弦相似度），把头部条目的邻居扩展到全部 84,432 部电影，再在测试候选集内取 Top10。

打分规则：
    score(user, c) = Σ_{h ∈ 训练历史 ∩ 头部条目} cos(h, c)
    同分时按训练集条目热度降序（避免"候选集顺序"这种假信号）。

用法：
    python scripts/export_top10_lsh.py                       # 复用缓存邻居表
    python scripts/export_top10_lsh.py --rebuild             # 重新计算邻居表
    python scripts/evaluate.py --pred results/top10_lsh.csv --name lsh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_set import CandidateSet, write_top10  # noqa: E402
from src.data_io import load_config, load_table, resolve  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_matrix(train: dict, vocab: dict) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """user x item 的二值矩阵（列归一化），以及 userId/movieId -> 0 基编码的查表。"""
    n_users, n_items = len(vocab["userId"]), len(vocab["movieId"])
    u_lookup = np.zeros(int(vocab["userId"].max()) + 1, dtype=np.int32)
    u_lookup[vocab["userId"]] = np.arange(n_users, dtype=np.int32)
    m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(n_items, dtype=np.int32)
    u = u_lookup[train["userId"]]
    m = m_lookup[train["movieId"]]
    matrix = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, m)), shape=(n_users, n_items))
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    matrix = matrix.multiply(sp.csr_matrix(1.0 / norms)).tocsr()
    return matrix, u_lookup, m_lookup


def build_neighbors(matrix: sp.csr_matrix, head: np.ndarray, k: int,
                    chunk: int = 250) -> tuple[np.ndarray, np.ndarray, float]:
    """头部条目 -> 全库条目的余弦 top-k 邻居（分块计算，避免 3000x84432 一次占满内存）。"""
    t0 = time.perf_counter()
    item_matrix_t = matrix.T.tocsr()
    n_items = matrix.shape[1]
    out_codes = np.empty((len(head), k), dtype=np.int32)
    out_sims = np.empty((len(head), k), dtype=np.float32)
    for start in range(0, len(head), chunk):
        part = head[start:start + chunk]
        sim = (item_matrix_t[part] @ matrix).toarray()
        rows = np.arange(len(part))
        sim[rows, part] = -np.inf          # 去掉自身
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        order = np.argsort(-sim[rows[:, None], idx], axis=1)
        idx = idx[rows[:, None], order]
        out_codes[start:start + len(part)] = idx
        out_sims[start:start + len(part)] = sim[rows[:, None], idx]
        log(f"  邻居计算 {min(start + chunk, len(head)):,}/{len(head):,}")
    return out_codes, out_sims, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="item-kNN 头部邻居 -> 测试候选集 Top10")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--head", type=int, default=3000, help="头部条目数（与 itemknn.py 一致）")
    parser.add_argument("--neighbors", type=int, default=200, help="每个头部条目保留的邻居数")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rebuild", action="store_true", help="强制重算邻居表")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    feature_dir = resolve(cfg["paths"].get("features_dir", "data/features"))
    results = resolve(cfg["paths"]["results_dir"])

    cand = CandidateSet(processed / "candidates_test.npz", processed / "vocab.npz")
    train = load_table(processed / "train.npz")
    vocab = load_table(processed / "vocab.npz")
    matrix, _, m_lookup = build_matrix(train, vocab)
    n_items = matrix.shape[1]
    log(f"候选集 {cand.rows:,} 行 / {int((cand.counts > 0).sum()):,} 用户；训练矩阵 {matrix.shape}，非零 {matrix.nnz:,}")

    counts = np.bincount(m_lookup[train["movieId"]], minlength=n_items)
    head = np.argsort(-counts)[:args.head]
    head = head[counts[head] > 0]

    cache = feature_dir / f"item_neighbors_full_k{args.neighbors}.npz"
    if cache.exists() and not args.rebuild:
        table = np.load(cache)
        items, neighbors, sims = table["items"], table["neighbors"], table["sims"]
        log(f"复用邻居表缓存 {cache.name}")
    else:
        log(f"计算邻居表：头部 {len(head):,} 部 × {args.neighbors} 邻居（全库）")
        nb, sv, seconds = build_neighbors(matrix, head, args.neighbors)
        items, neighbors, sims = head.astype(np.int32), nb, sv
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, items=items, neighbors=neighbors, sims=sims)
        log(f"邻居表计算完成 {seconds:.1f}s，缓存到 {cache.name}")

    # 头部条目 code -> 邻居表行号
    head_row = np.full(n_items, -1, dtype=np.int32)
    head_row[items] = np.arange(len(items), dtype=np.int32)

    # 训练历史（userId 1 基，与 CandidateSet 下标一致）
    uid = train["userId"].astype(np.int64)
    codes = cand.movie_to_code(train["movieId"].astype(np.int64))
    order = np.argsort(uid, kind="stable")
    uid_s, codes_s = uid[order], codes[order]
    hist_counts = np.bincount(uid_s, minlength=cand.user_count + 1).astype(np.int64)
    hist_starts = np.concatenate([[0], np.cumsum(hist_counts)[:-1]]).astype(np.int64)

    # 热度先验：仅用于同分排序（量级 1e-4，不会改变主排序）
    pop = counts.astype(np.float32)
    pop = (pop / max(float(pop.max()), 1.0)).astype(np.float32)

    pos_of = np.full(n_items, -1, dtype=np.int32)
    top_uid, top_mid, top_score, top_rank = [], [], [], []
    no_history = 0
    n_cand_max = int(cand.counts.max())
    buf = np.zeros(n_cand_max, dtype=np.float64)
    t_rank = time.perf_counter()

    for u in np.flatnonzero(cand.counts):
        sl = cand.user_slice(int(u))
        cc = cand.item_codes[sl]
        n = len(cc)
        if n == 0:
            continue
        s, c = int(hist_starts[u]), int(hist_counts[u])
        hist = codes_s[s:s + c]
        rows = head_row[hist] if hist.size else np.empty(0, dtype=np.int32)
        rows = rows[rows >= 0]

        score = np.zeros(n, dtype=np.float64)
        if rows.size:
            nb = neighbors[rows].ravel()
            sv = sims[rows].ravel().astype(np.float64)
            pos_of[cc] = np.arange(n, dtype=np.int32)
            pos = pos_of[nb]
            keep = pos >= 0
            if keep.any():
                score = np.bincount(pos[keep], weights=sv[keep], minlength=n)[:n]
            pos_of[cc] = -1
        else:
            no_history += 1

        score = score + 1e-4 * pop[cc]
        take = min(args.k, n)
        idx = np.argsort(-score, kind="stable")[:take]
        top_uid.append(np.full(take, cand.user_ids[u - 1], dtype=np.int32))
        top_mid.append(cand.mid[sl][idx])
        top_score.append(score[idx].astype(np.float32))
        top_rank.append(np.arange(1, take + 1, dtype=np.int8))

    cols = {
        "userId": np.concatenate(top_uid),
        "rank": np.concatenate(top_rank),
        "movieId": np.concatenate(top_mid),
        "score": np.concatenate(top_score),
    }
    out_path = Path(args.out) if args.out else results / "top10_lsh.csv"
    write_top10(out_path, cols)
    elapsed = time.perf_counter() - t_rank
    users = int(len(top_uid))

    summary = {
        "model": "lsh",
        "method": "item-kNN：头部 3,000 部 × 全库余弦 top-200 邻居，在测试候选集内累加相似度；同分按热度",
        "relation_to_lsh": "与 src/similarity/itemknn.py 的 LSH 使用同一头部集合与同一相似度定义，本文件是精确邻居版本",
        "users": users,
        "rows": int(len(cols["userId"])),
        "k": args.k,
        "head_items": int(len(items)),
        "neighbors_per_item": int(args.neighbors),
        "users_without_head_history": int(no_history),
        "ranking_seconds": round(elapsed, 2),
        "latency_ms_per_user": round(elapsed / max(users, 1) * 1000, 4),
        "pred_file": out_path.name,
    }
    out_json = results / "similarity_top10_lsh.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log(f"用户 {users:,}，无头部历史 {no_history:,}，排序 {elapsed:.1f}s（{summary['latency_ms_per_user']:.4f} ms/用户）")
    log(f"结果：{out_path} / {out_json}，总耗时 {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
