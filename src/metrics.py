# -*- coding: utf-8 -*-
"""评价指标：NDCG@k / Recall@k / MRR@k（二值相关性，正反馈 = rating >= threshold）。

与任务书第五节第 5 条对应：正反馈记 1，其余记 0；IDCG 按 min(|relevant|, k) 计算。
"""

from __future__ import annotations

import math

import numpy as np

_DISCOUNT = 1.0 / np.log2(np.arange(2, 100) + 1)  # 1/log2(2..100)


def dcg_at_k(ranked_items, relevant: set, k: int = 10) -> float:
    hits = np.array([1.0 if item in relevant else 0.0 for item in ranked_items[:k]], dtype=np.float64)
    if hits.size == 0:
        return 0.0
    return float((hits * _DISCOUNT[: hits.size]).sum())


def ndcg_at_k(ranked_items, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = dcg_at_k(ranked_items, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = float(_DISCOUNT[:ideal_hits].sum())
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_items[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_items, relevant: set, k: int = 10) -> float:
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_predictions(pred: dict[int, list[int]], truth: dict[int, set], k: int = 10) -> dict:
    """批量计算并按用户求平均。

    pred  : {userId: [按分数降序的 movieId]}
    truth : {userId: {测试正样本 movieId}}
    """
    common = sorted(set(pred) & set(truth))
    if not common:
        raise ValueError("预测与真值没有交集用户")
    ndcg = [ndcg_at_k(pred[u], truth[u], k) for u in common]
    recall = [recall_at_k(pred[u], truth[u], k) for u in common]
    mrr = [mrr_at_k(pred[u], truth[u], k) for u in common]
    return {
        "users": len(common),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"recall@{k}": float(np.mean(recall)),
        f"mrr@{k}": float(np.mean(mrr)),
    }


def macro_ndcg(ranked_by_user: dict[int, np.ndarray], truth: dict[int, set], k: int = 10) -> float:
    """对 numpy 数组形式的排序结果求平均 NDCG。"""
    values = [ndcg_at_k(list(map(int, ranked_by_user[u][:k])), truth[u], k) for u in truth if u in ranked_by_user]
    return float(np.mean(values)) if values else 0.0