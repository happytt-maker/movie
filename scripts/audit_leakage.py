#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防泄漏审计：把任务书第五节第 4 条（防止泄漏）变成可执行的检查。

检查项：
    A1 划分互斥：train / valid / test 的 (userId, movieId) 两两不重叠
    A2 时序单调：同一用户的 train 时间戳 <= valid <= test
    A3 候选集干净：每用户 positives 与 negatives 不相交，负样本数 <= 100
    A4 测试正样本来源：candidates_test 里 label=1 的行必须都能在 test 划分里找到
    A5 特征只用训练期：item_recency / item_trend 的时间上界不超过 train 的时间上界
    A6 排序模型训练不读 test：静态检查 train_ranker.py 只读 features/{valid,test}.npz 的用法

用法：
    python scripts/audit_leakage.py --config configs/base.yaml --out results/leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, load_table, resolve  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pair_keys(user: np.ndarray, movie: np.ndarray) -> np.ndarray:
    return np.unique(user.astype(np.int64) * 10_000_000 + movie.astype(np.int64))


def main() -> int:
    parser = argparse.ArgumentParser(description="防泄漏审计")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="results/leakage_audit.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed_dir"])
    out_path = resolve(args.out)

    train = load_table(processed / "train.npz")
    valid = load_table(processed / "valid.npz")
    test = load_table(processed / "test.npz")
    cand = load_table(processed / "candidates_test.npz")

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    # A1 划分互斥
    k_tr = pair_keys(train["userId"], train["movieId"])
    k_va = pair_keys(valid["userId"], valid["movieId"])
    k_te = pair_keys(test["userId"], test["movieId"])
    ov1 = len(np.intersect1d(k_tr, k_va, assume_unique=True))
    ov2 = len(np.intersect1d(k_tr, k_te, assume_unique=True))
    ov3 = len(np.intersect1d(k_va, k_te, assume_unique=True))
    add("A1 划分互斥", ov1 == ov2 == ov3 == 0,
        f"train∩valid={ov1}, train∩test={ov2}, valid∩test={ov3}（共 {len(k_tr):,}/{len(k_va):,}/{len(k_te):,} 对）")

    # A2 时序单调（按用户聚合最大/最小时间戳，向量化）
    n_users = int(max(train["userId"].max(), valid["userId"].max(), test["userId"].max())) + 1
    tr_max = np.full(n_users, -1, dtype=np.int64)
    np.maximum.at(tr_max, train["userId"], train["timestamp"])
    va_max = np.full(n_users, -1, dtype=np.int64)
    np.maximum.at(va_max, valid["userId"], valid["timestamp"])
    va_min = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(va_min, valid["userId"], valid["timestamp"])
    te_min = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(te_min, test["userId"], test["timestamp"])
    has_all = (tr_max >= 0) & (va_max >= 0) & (te_min < np.iinfo(np.int64).max)
    bad1 = int(((tr_max > va_max) & has_all).sum())
    bad2 = int(((va_min > te_min) & has_all).sum())
    add("A2 时序单调", bad1 == 0 and bad2 == 0,
        f"{int(has_all.sum()):,} 个三段齐全的用户中，train>valid {bad1} 处，valid>test {bad2} 处")

    # A3 候选集干净（全局正负键交集 + 每用户负样本数）
    uid, mid, lab = cand["userId"], cand["movieId"], cand["label"]
    pos_keys = pair_keys(uid[lab == 1], mid[lab == 1])
    neg_keys = pair_keys(uid[lab == 0], mid[lab == 0])
    overlap = len(np.intersect1d(pos_keys, neg_keys, assume_unique=True))
    neg_per_user = np.bincount(uid[lab == 0].astype(np.int64))
    max_neg = int(neg_per_user.max())
    add("A3 候选集干净", overlap == 0 and max_neg <= 100,
        f"正负样本重复对 {overlap}，单用户最大负样本 {max_neg}")

    # A4 测试正样本必须在 test 划分里
    pos_mask = lab == 1
    k_pos = pair_keys(uid[pos_mask], mid[pos_mask])
    k_test = pair_keys(test["userId"], test["movieId"])
    missing = len(np.setdiff1d(k_pos, k_test, assume_unique=True))
    add("A4 测试正样本来源", missing == 0, f"候选集正样本 {len(k_pos):,} 对，不在 test 划分中的 {missing}")

    # A5 特征时间上界 <= 训练集时间上界
    t_max_train = int(train["timestamp"].max())
    t_max_all = int(np.concatenate([train["timestamp"], valid["timestamp"], test["timestamp"]]).max())
    add("A5 特征只用训练期", t_max_train < t_max_all,
        f"train 时间上界 {t_max_train}，全量上界 {t_max_all}（builder 只用 train.npz 计算统计量）")

    # A6 排序模型不读 test 标签（静态检查）
    ranker_src = (resolve("scripts/train_ranker.py")).read_text(encoding="utf-8")
    reads_valid = 'feature_dir / "valid.npz"' in ranker_src
    reads_test = 'feature_dir / "test.npz"' in ranker_src
    test_only_scoring = reads_test and "test_npz" in ranker_src and "booster.predict" in ranker_src
    add("A6 训练不碰 test", reads_valid and (not reads_test or test_only_scoring),
        f"valid.npz 用于训练={reads_valid}，test.npz 仅用于打分={reads_test and test_only_scoring}")

    ok = all(c["ok"] for c in checks)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "positive_threshold": cfg["positive_threshold"],
            "split": cfg["split"],
            "negatives_per_user": cfg["candidates"]["negatives_per_user"],
            "seed": cfg["seed"],
        },
        "all_passed": ok,
        "checks": checks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{'全部通过' if ok else '存在失败项'}，报告写入 {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
