# -*- coding: utf-8 -*-
"""项目内部的数据读写与配置工具。

processed 目录里的表统一存成 .npz（每列一个数组 + 一个 manifest.json 描述），
避免对 pyarrow 的硬依赖；需要 CSV 的地方由脚本单独导出。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = {
    "ratings": ("userId", "movieId", "rating", "timestamp"),
    "candidates": ("userId", "movieId", "label"),
    "pred": ("userId", "rank", "movieId", "score"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """读取 YAML 配置；相对路径按项目根目录解析。

    支持 ``extends: base.yaml``：先读基础配置，再用当前文件覆盖，避免复制整份协议。
    """
    path = Path(path) if path else PROJECT_ROOT / "configs" / "base.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        cfg = _deep_merge(load_config(parent_path), cfg)
    return cfg


def resolve(path_str: str | Path) -> Path:
    """把配置里的相对路径解析到项目根目录。"""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def save_table(path: str | Path, allow_unequal: bool = False, **columns: np.ndarray) -> Path:
    """把若干数组写成一个 .npz，并更新同目录的 manifest.json。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = {len(v) for v in columns.values()}
    if not allow_unequal and len(lengths) != 1:
        raise ValueError(f"列长度不一致：{ {k: len(v) for k, v in columns.items()} }")
    np.savez(path, **columns)

    manifest_path = path.parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    manifest[path.name] = {
        "rows": sorted(lengths),
        "columns": {k: {"dtype": str(v.dtype), "len": int(len(v))} for k, v in columns.items()},
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_table(path: str | Path) -> dict[str, np.ndarray]:
    """读取 .npz，返回列名字典。"""
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def read_ratings(path: str | Path) -> dict[str, np.ndarray]:
    ratings = load_table(path)
    for col in COLUMNS["ratings"]:
        if col not in ratings:
            raise KeyError(f"{path} 缺少列 {col}")
    return ratings


def write_top10_csv(path: str | Path, user_ids, movie_ids, scores=None, ranks=None) -> Path:
    """按统一格式写 Top-10 结果：userId,rank,movieId[,score]。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(user_ids)
    if ranks is None:
        ranks = np.ones(n, dtype=np.int32)
    if scores is None:
        scores = np.zeros(n, dtype=np.float32)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("userId,rank,movieId,score\n")
        for u, k, m, s in zip(user_ids, ranks, movie_ids, scores):
            fh.write(f"{int(u)},{int(k)},{int(m)},{float(s):.6f}\n")
    return path