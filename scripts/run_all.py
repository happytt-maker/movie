#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键跑通全流程（数据 -> 基线 -> 相似度 -> 图 -> 流式 -> 排序 -> 汇总）。

用法：
    python scripts/run_all.py --config configs/local.yaml
    python scripts/run_all.py --config configs/local.yaml --skip prepare,knn

说明：
    * Spark 任务（任务三）不在本脚本内，需用 spark-submit 单独运行，见 README；
    * 每一步都会打印耗时，失败即停止并返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "prepare": ["scripts/prepare_data.py"],
    "features_valid": ["scripts/build_features.py", "--split", "valid"],
    "features_test": ["scripts/build_features.py", "--split", "test"],
    "popularity": ["-m", "src.baseline.popularity"],
    "knn": ["-m", "src.similarity.itemknn"],
    "graph": ["-m", "src.graph.ppr"],
    "streaming": ["-m", "src.streaming.replay"],
    "bpr": ["-m", "src.ranking.bpr", "--mode", "all", "--epochs", "4"],
    "ranker": ["scripts/train_ranker.py", "--name", "ranker_full"],
}

ORDER = ["prepare", "popularity", "knn", "graph", "streaming", "bpr",
         "features_valid", "features_test", "ranker"]


def run(cmd: list[str], config: str) -> float:
    full = [sys.executable, *cmd, "--config", config]
    t0 = time.time()
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(full, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"步骤失败（exit={result.returncode}）：{' '.join(cmd)}")
    return time.time() - t0


def summarize(config: str) -> None:
    cfg_path = ROOT / config
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "extends" in cfg:
        with open(ROOT / "configs" / cfg["extends"], "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
        cfg = {**base, **cfg}
    results = ROOT / cfg["paths"]["results_dir"]
    rows = []
    for path in sorted(results.glob("metrics_*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        rows.append((m.get("model", path.stem), m.get("ndcg@10"), m.get("recall@10")))
    if not rows:
        return
    print("\n模型                NDCG@10   Recall@10")
    print("-" * 42)
    for name, ndcg, recall in sorted(rows, key=lambda r: -(r[1] or 0)):
        print(f"{name:20s} {ndcg if ndcg is not None else 0:>8.4f} {recall if recall is not None else 0:>10.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="一键运行全流程")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--skip", default="", help="逗号分隔的跳过步骤")
    parser.add_argument("--only", default="", help="逗号分隔的只跑某些步骤")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    timings = {}
    for name in ORDER:
        if name in skip or (only and name not in only):
            continue
        timings[name] = round(run(STEPS[name], args.config), 1)
        print(f"<<< {name} 完成，{timings[name]}s", flush=True)

    print("\n各步骤耗时（秒）：")
    for name, sec in timings.items():
        print(f"  {name:16s} {sec:>8.1f}")
    summarize(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())