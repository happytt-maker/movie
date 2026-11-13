#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一致性校验：对比两次运行产出的 metrics_*.json，逐字段给出偏差。

用途（分工文档 4.3）：换个目录、换台机器复跑同一份数据，结果须完全一致。

用法：
    python scripts/check_consistency.py --a results --b C:/tmp/movie_check_results
    python scripts/check_consistency.py --a results --b ... --keys ndcg@10,recall@10 --tol 1e-9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_KEYS = ["ndcg@10", "recall@10", "mrr@10", "users", "best_iteration"]


def load_metrics(folder: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(folder.glob("metrics_*.json")):
        try:
            out[path.stem.replace("metrics_", "", 1)] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="两次运行结果一致性对比")
    parser.add_argument("--a", required=True, help="基准结果目录")
    parser.add_argument("--b", required=True, help="对照结果目录")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    a = load_metrics(Path(args.a))
    b = load_metrics(Path(args.b))
    common = sorted(set(a) & set(b))
    if not common:
        print(f"两个目录没有同名模型：A={sorted(a)} B={sorted(b)}")
        return 2

    rows, worst, failed = [], 0.0, []
    for name in common:
        for key in keys:
            va, vb = a[name].get(key), b[name].get(key)
            if va is None or vb is None:
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff = abs(float(va) - float(vb))
                worst = max(worst, diff)
                ok = diff <= args.tol
                if not ok:
                    failed.append(f"{name}.{key}")
                rows.append((name, key, va, vb, diff, ok))
            elif va != vb:
                failed.append(f"{name}.{key}")
                rows.append((name, key, va, vb, "n/a", False))

    print(f"{'模型':22s} {'字段':14s} {'基准':>14s} {'对照':>14s} {'偏差':>12s}  结果")
    print("-" * 86)
    for name, key, va, vb, diff, ok in rows:
        d = f"{diff:.2e}" if isinstance(diff, float) else str(diff)
        print(f"{name:22s} {key:14s} {va:>14} {vb:>14} {d:>12s}  {'OK' if ok else 'DIFF'}")

    print("-" * 86)
    if failed:
        print(f"不一致：{len(failed)} 项（最大偏差 {worst:.2e}）-> {failed[:8]}")
        return 1
    print(f"完全一致：{len(rows)} 项，最大偏差 {worst:.2e}（容差 {args.tol:g}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
