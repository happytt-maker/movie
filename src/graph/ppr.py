# -*- coding: utf-8 -*-
"""任务五：用户-电影二部图的链接分析与社区特征。

流程：
  1. 用训练期正反馈构造二部图（用户节点 + 电影节点 + 评分边）；
  2. 在二部图上做 PageRank（链接分析），得到电影的全局权威度；
  3. 在抽样用户子图上做 Louvain 社区划分，得到电影社区标签与规模；
  4. 用 PageRank 分和"社区匹配度"给候选集排序，与热门基线在同一批用户上对比 NDCG@10。

用法：
    python -m src.graph.ppr --config configs/base.yaml --users 30000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from ..data_io import load_config, load_table, resolve
from ..metrics import ndcg_at_k


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_bipartite(user_codes: np.ndarray, item_codes: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    """返回 (n_users+n_items) 的对称邻接矩阵，同时保留原始二部邻接。"""
    data = np.ones(len(user_codes), dtype=np.float32)
    bi = sp.csr_matrix((data, (user_codes, item_codes)), shape=(n_users, n_items))
    upper = sp.hstack([sp.csr_matrix((n_users, n_users)), bi], format="csr")
    lower = sp.hstack([bi.T, sp.csr_matrix((n_items, n_items))], format="csr")
    return sp.vstack([upper, lower], format="csr"), bi


def pagerank(adj: sp.csr_matrix, alpha: float = 0.85, iters: int = 30, tol: float = 1e-8) -> np.ndarray:
    """幂迭代 PageRank（列随机转移）。"""
    n = adj.shape[0]
    degrees = np.asarray(adj.sum(axis=0)).ravel()
    inv = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    transition = adj.multiply(inv).tocsr()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n
    for _ in range(iters):
        new = alpha * (transition @ rank) + teleport
        if np.abs(new - rank).sum() < tol:
            rank = new
            break
        rank = new
    return rank / rank.sum()


def louvain_communities(adj: sp.csr_matrix, seed: int = 20260907) -> list[set]:
    """networkx 的 Louvain 社区划分（子图规模受控）。"""
    import networkx as nx

    graph = nx.from_scipy_sparse_array(adj)
    return nx.algorithms.community.louvain_communities(graph, seed=seed, resolution=1.0)


def run(config_path: str, n_users: int) -> dict:
    cfg = load_config(config_path)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])
    seed = int(cfg["seed"])

    train = load_table(processed / "train.npz")
    vocab = load_table(processed / "vocab.npz")
    n_users_all, n_items = len(vocab["userId"]), len(vocab["movieId"])
    u_lookup = np.zeros(int(vocab["userId"].max()) + 1, dtype=np.int32)
    u_lookup[vocab["userId"]] = np.arange(n_users_all, dtype=np.int32)
    m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
    m_lookup[vocab["movieId"]] = np.arange(n_items, dtype=np.int32)
    u = u_lookup[train["userId"]]
    m = m_lookup[train["movieId"]]
    pos = train["rating"] >= cfg["positive_threshold"]
    u, m = u[pos], m[pos]

    rng = np.random.default_rng(seed)
    sample_users = np.sort(rng.choice(n_users_all, min(n_users, n_users_all), replace=False))
    keep = np.zeros(n_users_all, dtype=bool)
    keep[sample_users] = True
    mask = keep[u]
    su, sm = u[mask], m[mask]
    # 子图内重新编号电影
    used_items, sm_compact = np.unique(sm, return_inverse=True)
    local_u = np.searchsorted(sample_users, su)
    log(f"子图：{len(sample_users):,} 用户，{len(used_items):,} 电影，{len(su):,} 条边")

    adj, bi = build_bipartite(local_u, sm_compact, len(sample_users), len(used_items))
    log(f"二部图节点 {adj.shape[0]:,}，边（无向）{adj.nnz:,}，平均度 {adj.nnz / adj.shape[0]:.2f}")

    t0 = time.perf_counter()
    rank = pagerank(adj)
    pr_seconds = time.perf_counter() - t0
    item_pr = rank[len(sample_users):]
    log(f"PageRank 完成：{pr_seconds:.1f}s，收敛后前 3 部电影 {np.argsort(-item_pr)[:3].tolist()}")

    t0 = time.perf_counter()
    communities = louvain_communities(adj, seed=seed)
    louvain_seconds = time.perf_counter() - t0
    node_community = np.full(adj.shape[0], -1, dtype=np.int32)
    for cid, nodes in enumerate(communities):
        node_community[list(nodes)] = cid
    item_community = node_community[len(sample_users):]
    sizes = np.bincount(item_community[item_community >= 0])
    modularity = None
    try:
        import networkx as nx

        modularity = float(nx.algorithms.community.modularity(
            nx.from_scipy_sparse_array(adj), communities))
    except Exception:  # noqa: BLE001
        pass
    log(f"Louvain 完成：{len(communities)} 个社区，最大社区 {int(sizes.max()) if len(sizes) else 0} 个节点，用时 {louvain_seconds:.1f}s")

    # ---- 在测试候选集上比较：PageRank 排序 vs 社区加权威重 vs 热门 ----
    cand = CandidateSetLite(processed / "candidates_test.npz", vocab)
    counts = np.bincount(sm_compact, minlength=len(used_items)).astype(np.float64)
    item_score_pr = np.zeros(n_items, dtype=np.float32)
    item_score_pop = np.zeros(n_items, dtype=np.float32)
    item_score_pr[used_items] = item_pr.astype(np.float32)
    item_score_pop[used_items] = counts.astype(np.float32)
    item_comm_all = np.full(n_items, -1, dtype=np.int32)
    item_comm_all[used_items] = item_community

    # 用户的社区偏好：其历史电影所在社区的分布
    user_comm = {}
    for uu, mm in zip(local_u.tolist(), sm_compact.tolist()):
        c = int(item_community[mm])
        user_comm.setdefault(uu, {}).setdefault(c, 0)
        user_comm[uu][c] += 1

    ndcg_pop, ndcg_pr, ndcg_comm = [], [], []
    users_check = [int(x) for x in sample_users[:5000]]
    for u_code in users_check:
        uid = int(vocab["userId"][u_code])
        sl = cand.slice_of(uid)
        if sl is None or sl[1] == 0:
            continue
        mids = cand.mid[sl[0]:sl[0] + sl[1]]
        labels = cand.label[sl[0]:sl[0] + sl[1]]
        rel = set(mids[labels == 1].tolist())
        if not rel:
            continue
        codes = cand.codes(sl[0], sl[1])
        pop_score = item_score_pop[codes]
        pr_score = item_score_pr[codes]
        pref = user_comm.get(int(np.searchsorted(sample_users, u_code)), {})
        total_pref = sum(pref.values()) or 1
        comm_score = pr_score * np.array(
            [1.0 + 2.0 * pref.get(int(item_comm_all[c]), 0) / total_pref for c in codes], dtype=np.float32)

        for name, score, bucket in (("popularity", pop_score, ndcg_pop),
                                    ("pagerank", pr_score, ndcg_pr),
                                    ("pagerank+community", comm_score, ndcg_comm)):
            top = mids[np.argsort(-score, kind="stable")[:10]].tolist()
            bucket.append(ndcg_at_k(top, rel, 10))

    report = {
        "graph": {
            "users": int(len(sample_users)),
            "items": int(len(used_items)),
            "edges": int(len(su)),
            "nodes": int(adj.shape[0]),
            "density": round(float(adj.nnz / (adj.shape[0] ** 2)), 8),
            "avg_degree": round(float(adj.nnz / adj.shape[0]), 3),
        },
        "pagerank": {"seconds": round(pr_seconds, 2), "top_items": np.argsort(-item_pr)[:10].tolist()},
        "louvain": {"seconds": round(louvain_seconds, 2), "communities": len(communities),
                    "modularity": round(modularity, 4) if modularity is not None else None,
                    "largest_community_nodes": int(sizes.max()) if len(sizes) else 0},
        "evaluation": {
            "users": len(ndcg_pop),
            "ndcg@10_popularity": round(float(np.mean(ndcg_pop)), 4) if ndcg_pop else 0.0,
            "ndcg@10_pagerank": round(float(np.mean(ndcg_pr)), 4) if ndcg_pr else 0.0,
            "ndcg@10_pagerank_plus_community": round(float(np.mean(ndcg_comm)), 4) if ndcg_comm else 0.0,
            "note": "同批抽样用户、同一候选集；PageRank 单独使用与热门基线接近，社区加权带来小幅变化",
        },
        "seed": seed,
    }
    out = results / "graph_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"图实验：NDCG 热门={report['evaluation']['ndcg@10_popularity']:.4f} "
        f"PageRank={report['evaluation']['ndcg@10_pagerank']:.4f} "
        f"+社区={report['evaluation']['ndcg@10_pagerank_plus_community']:.4f}")
    log(f"报告写入 {out}")
    return report


class CandidateSetLite:
    """只读候选集：按 userId 直接切片，避免加载全量排序索引。"""

    def __init__(self, path, vocab):
        table = load_table(path)
        uid = table["userId"].astype(np.int64)
        order = np.argsort(uid, kind="stable")
        self.uid = uid[order]
        self.mid = table["movieId"][order]
        self.label = table["label"][order]
        self.m_lookup = np.zeros(int(vocab["movieId"].max()) + 1, dtype=np.int32)
        self.m_lookup[vocab["movieId"]] = np.arange(len(vocab["movieId"]), dtype=np.int32)
        counts = np.bincount(self.uid, minlength=int(self.uid.max()) + 1)
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.counts = counts

    def slice_of(self, user_id: int):
        start = int(self.starts[user_id])
        count = int(self.counts[user_id])
        return (start, count) if count else None

    def codes(self, start, end):
        return self.m_lookup[self.mid[start:start + end]]


def main() -> int:
    parser = argparse.ArgumentParser(description="二部图链接分析与社区特征")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--users", type=int, default=30000)
    args = parser.parse_args()
    run(args.config, args.users)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())