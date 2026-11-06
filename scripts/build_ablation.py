#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务六补交付：把散在 metrics_ranker_*.json 里的消融结果汇总成一张表。

用法：
    python scripts/build_ablation.py --config configs/base.yaml
    python scripts/build_ablation.py --config configs/base.yaml \
        --configs ranker_full,ranker_no_bpr,ranker_no_genre,ranker_no_temporal

产物：results/ablation.csv
    列：config, dropped_features, ndcg@10, recall@10, delta_ndcg, delta_pct, best_iteration, users, note
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, resolve  # noqa: E402

DEFAULT_CONFIGS = [
    ("ranker_full", "全量 9 特征（最终模型）"),
    ("ranker_no_bpr", "去掉 BPR 分数"),
    ("ranker_no_genre", "去掉类型亲和度"),
    ("ranker_no_temporal", "去掉时序三项（trend / recency / last_activity）"),
]

FIELDS = ["config", "dropped_features", "ndcg@10", "recall@10",
          "delta_ndcg", "delta_pct", "best_iteration", "users", "note"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="消融结果汇总")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--configs", default=None, help="逗号分隔的模型名，顺序即表格顺序")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = resolve(cfg["paths"]["results_dir"])

    if args.configs:
        names = [(n.strip(), "") for n in args.configs.split(",") if n.strip()]
    else:
        names = DEFAULT_CONFIGS

    rows, missing = [], []
    base_ndcg = None
    for name, note in names:
        path = results / f"metrics_{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        ndcg = float(m.get("ndcg@10") or 0.0)
        recall = float(m.get("recall@10") or 0.0)
        if base_ndcg is None:
            base_ndcg = ndcg
        dropped = ";".join(m.get("excluded") or [])
        rows.append({
            "config": name,
            "dropped_features": dropped,
            "ndcg@10": round(ndcg, 5),
            "recall@10": round(recall, 5),
            "delta_ndcg": round(ndcg - base_ndcg, 5),
            "delta_pct": round((ndcg - base_ndcg) / base_ndcg * 100, 2) if base_ndcg else 0.0,
            "best_iteration": int(m.get("best_iteration") or 0),
            "users": int(m.get("users") or 0),
            "note": note or (f"excluded={dropped}" if dropped else ""),
        })

    out_path = Path(args.out) if args.out else results / "ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        log(f"{r['config']:20s} ndcg={r['ndcg@10']:.4f} recall={r['recall@10']:.4f} "
            f"delta={r['delta_ndcg']:+.4f} ({r['delta_pct']:+.2f}%)")
    log(f"写入 {out_path}（{len(rows)} 行）")
    if missing:
        log(f"缺少指标文件，已跳过：{missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
