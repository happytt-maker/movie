#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据检查、统计与训练/验证/测试划分。

对应任务书「五 数据划分与评价协议」：
  1. 正反馈：rating >= 4
  2. 时间划分：每位用户按 timestamp 升序，前 80% / 中间 10% / 最后 10%
  3. 候选集：每个测试用户 = 测试正样本 + 100 个固定负样本（种子 20260907）
  4. 防泄漏：不得使用推荐时刻之后的数据

用法：
    python scripts/prepare_data.py --config configs/base.yaml

TODO（实现时按需补全）：
    - 数据质量检查：缺失值、重复记录、评分分布、时间范围
    - 描述性统计输出到 results/data_stats.json
    - 划分结果写入 data/processed/
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="MovieLens 数据检查与划分")
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    if not raw_dir.exists():
        print(f"[错误] 未找到数据目录：{raw_dir}", file=sys.stderr)
        print("       请先下载 MovieLens 32M 并解压到该目录，详见 data/README.md", file=sys.stderr)
        return 1

    print("[配置] 已加载：")
    print(f"       随机种子       = {cfg['seed']}")
    print(f"       正反馈阈值     = {cfg['positive_threshold']}")
    print(f"       划分比例       = {cfg['split']}")
    print(f"       负采样数       = {cfg['candidates']['negatives_per_user']}")

    # TODO: 在此实现数据检查、统计与划分逻辑
    print("\n[待实现] 数据检查与划分逻辑尚未编写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
