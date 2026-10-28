# -*- coding: utf-8 -*-
"""任务二：精确相似度计算 + LSH 近似候选召回。

规模说明（写进报告的关键取舍）：
    全量 84,432 部电影的精确 item-item 余弦矩阵约 7.1e9 个元素，
    单机内存无法承受，因此精确计算在"头部条目集合"（默认按训练交互数取前 3000 部）
    上进行，LSH 也在同一集合上做近似召回，两者口径一致、可直接比较召回率。

用法：
    python -m src.similarity.itemknn --config configs/base.yaml --head 3000 --k 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_item_user_matrix(user_codes: np.ndarray, item_codes: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    """user x item 的 0/1 交互矩阵。"""
    data = np.ones(len(user_codes), dtype=np.float32)
    return sp.csr_matrix((data, (user_codes, item_codes)), shape=(n_users, n_items))


def l2_normalize_columns(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """把每一列（每个条目的用户集合）归一化为单位向量，便于算余弦。"""
    matrix = matrix.tocsr().astype(np.float32)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    return matrix.multiply(sp.csr_matrix(1.0 / norms)).tocsr()


def exact_topk(item_matrix: sp.csr_matrix, head: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """在头部条目集合内做精确余弦 top-k。"""
    t0 = time.perf_counter()
    sub = item_matrix[:, head]
    sim = (sub.T @ sub).toarray()
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    idx = idx[rows, order]
    values = sim[rows, idx]
    seconds = time.perf_counter() - t0
    return head[idx], values, seconds


def lsh_topk(item_matrix: sp.csr_matrix, head: np.ndarray, k: int,
             num_perm: int = 64, threshold: float = 0.3) -> tuple[dict, float, int]:
    """用 MinHash + LSH 做近似近邻召回，返回 item -> 候选集合。"""
    from datasketch import MinHash, MinHashLSH

    t0 = time.perf_counter()
    csc = item_matrix.tocsc()
    signatures = {}
    minhashes = {}
    for j, item in enumerate(head):
        users = csc.indices[csc.indptr[item]:csc.indptr[item + 1]]
        m = MinHash(num_perm=num_perm)
        for u in users.tolist():
            m.update(str(u).encode("utf-8"))
        minhashes[int(item)] = m
        signatures[int(item)] = m.hashvalues
    build_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as session:
        for item, m in minhashes.items():
            session.insert(item, m)
    candidates = {}
    for item, m in minhashes.items():
        result = set(lsh.query(m)) - {item}
        candidates[item] = result
    query_seconds = time.perf_counter() - t0
    return candidates, build_seconds + query_seconds, len(minhashes)


def evaluate_recall(exact: np.ndarray, head: np.ndarray, candidates: dict, k: int) -> dict:
    """LSH 召回集合与精确 top-k 的重合情况。"""
    index = {int(item): i for i, item in enumerate(head)}
    recalls, precisions, sizes = [], [], []
    for item, cand in candidates.items():
        i = index[item]
        truth = set(int(x) for x in exact[i][:k])
        got = set(cand)
        recalls.append(len(truth & got) / max(len(truth), 1))
        precisions.append(len(truth & got) / max(len(got), 1))
        sizes.append(len(got))
    return {
        "recall_at_k": float(np.mean(recalls)),
        "precision_at_k": float(np.mean(precisions)),
        "avg_candidates": float(np.mean(sizes)),
    }


def run(config_path: str, head_size: int, k: int, num_perm: int, threshold: float) -> dict:
    from ..data_io import load_config, load_table, resolve

    cfg = load_config(config_path)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])

    train = load_table(processed / "train.npz")
    vocab = load_table(processed / "vocab.npz")
    n_users, n_items = len(vocab["userId"]), len(vocab["movieId"])
    u_lookup = np.zeros(int(vocab["userId"].max()) + 1, dtype=np.int32)
    u_lookup[vocab["userId"]] = np.arange(n_users, dtype=np.int32)
    m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(n_items, dtype=np.int32)

    u = u_lookup[train["userId"]]
    m = m_lookup[train["movieId"]]
    matrix = build_item_user_matrix(u, m, n_users, n_items)
    matrix = l2_normalize_columns(matrix)
    log(f"用户-条目矩阵：{matrix.shape}，非零 {matrix.nnz:,}")

    counts = np.bincount(m, minlength=n_items)
    head = np.argsort(-counts)[:head_size]
    head = head[counts[head] > 0]
    log(f"头部条目集合：{len(head):,} 部（训练交互数前 {head_size:,}）")

    exact, exact_sim, exact_seconds = exact_topk(matrix, head, k)
    log(f"精确 top-{k} 计算完成：{exact_seconds:.1f}s")

    candidates, lsh_seconds, n_items_lsh = lsh_topk(matrix, head, k, num_perm, threshold)
    quality = evaluate_recall(exact, head, candidates, k)
    log(f"LSH 召回完成：{lsh_seconds:.1f}s，平均候选 {quality['avg_candidates']:.1f}，"
        f"recall@{k}={quality['recall_at_k']:.4f}")

    out = {
        "head_items": int(len(head)),
        "k": k,
        "num_perm": num_perm,
        "lsh_threshold": threshold,
        "total_items": int(n_items),
        "exact_block_seconds": round(exact_seconds, 2),
        "lsh_seconds": round(lsh_seconds, 2),
        "speedup_on_head": round(lsh_seconds / max(exact_seconds, 1e-9), 3),
        "avg_exact_similarity": float(np.mean(exact_sim)),
        **quality,
        "note": "全量 84,432x84,432 精确余弦约 7.1e9 元素，单机不可行，故精确计算在头部条目集合内完成",
    }
    out_path = results / "similarity_lsh.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    log(f"结果写入 {out_path}")

    # 存一份精确邻居表，供图/召回实验复用
    np.savez(results.parent / "data" / "features" / "item_neighbors_exact.npz",
             items=head, neighbors=exact[:len(head), :k], sims=exact_sim[:len(head), :k].astype(np.float32))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="精确相似度与 LSH 召回对比")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--head", type=int, default=3000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    run(args.config, args.head, args.k, args.num_perm, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())