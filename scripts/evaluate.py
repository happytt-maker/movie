#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一评价脚本：NDCG@10 / Recall@10。

对应任务书「五 数据划分与评价协议」第 5 条：
    教师使用统一脚本计算 NDCG@10，并同时记录 Recall@10、吞吐量和单用户推荐延迟。

用法：
    python scripts/evaluate.py --pred results/top10_popularity.csv --config configs/base.yaml

TODO（实现时按需补全）：
    - NDCG@10 计算（注意：相关性用二值正反馈，rating>=4 记 1）
    - Recall@10 计算
    - 单用户推荐延迟统计
"""

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ndcg_at_k(ranked_items, relevant_items, k: int = 10) -> float:
    """计算单个用户的 NDCG@k。

    参数
    ----
    ranked_items : list
        按推荐顺序排列的 itemId（长度应为 k）
    relevant_items : set
        该用户的真实正反馈 itemId 集合
    k : int
        截断位置

    返回
    ----
    float
        NDCG@k 值；若无相关项则返回 0.0
    """
    import math

    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)  # i 从 0 开始，位置从 1 开始

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items, relevant_items, k: int = 10) -> float:
    """计算单个用户的 Recall@k。"""
    if not relevant_items:
        return 0.0
    hits = len(set(ranked_items[:k]) & relevant_items)
    return hits / len(relevant_items)


def main() -> int:
    parser = argparse.ArgumentParser(description="NDCG@10 / Recall@10 评价")
    parser.add_argument("--pred", required=True, help="Top10 推荐结果 CSV")
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[配置] 主指标 = {cfg['metrics']['primary']}, 截断位置 k = 10")
    print(f"[输入] 推荐结果 = {args.pred}")

    # TODO: 读取预测与真值，批量计算并汇总输出到 results/
    print("\n[待实现] 评价主流程尚未编写。")
    print("         ndcg_at_k() 与 recall_at_k() 已可用，可直接复用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
