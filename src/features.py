# -*- coding: utf-8 -*-
"""任务六的特征工程：为候选集的每一行生成排序特征。

特征分三类：
  召回/协同类  bpr（BPR-MF 打分）、knn（item-kNN 相似度）
  内容类        genre_affinity（用户类型画像与电影类型的匹配度）
  统计/时序类  pop_log、item_mean、item_rate、recency、user_activity

每个特征都是与候选集行数等长的 float32 数组，按 split 存成 npz，
排序模型只需按列拼接，避免每次重复计算。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data_io import load_table

GENRE_SEP = "|"


def movie_genre_matrix(movies_csv: Path, movie_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """构造 (n_items, n_genres) 的 0/1 矩阵，行顺序与 vocab 的 movieId 一致。"""
    movies = pd.read_csv(movies_csv, usecols=["movieId", "genres"], dtype={"movieId": "int64", "genres": "string"})
    movies = movies.set_index("movieId")
    all_genres = sorted({g for gs in movies["genres"].dropna() for g in str(gs).split(GENRE_SEP) if g and g != "(no genres listed)"})
    index = {g: i for i, g in enumerate(all_genres)}
    matrix = np.zeros((len(movie_ids), len(all_genres)), dtype=np.float32)
    for row, movie_id in enumerate(movie_ids):
        gs = movies["genres"].get(int(movie_id))
        if not isinstance(gs, str):
            continue
        for g in gs.split(GENRE_SEP):
            j = index.get(g)
            if j is not None:
                matrix[row, j] = 1.0
    return matrix, all_genres


def user_genre_profile(train: dict[str, np.ndarray], user_codes: np.ndarray, item_codes: np.ndarray,
                       genre: np.ndarray, n_users: int, positive_threshold: float) -> np.ndarray:
    """用户类型画像：训练期正反馈电影的类型的加权和，再做 L2 归一化。"""
    mask = train["rating"] >= positive_threshold
    rows = user_codes[mask]
    cols = item_codes[mask]
    values = np.ones(mask.sum(), dtype=np.float32)
    interaction = sp.csr_matrix((values, (rows, cols)), shape=(n_users, genre.shape[0]))
    profile = interaction @ genre  # (n_users, n_genres)
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (profile / norms).astype(np.float32)


def item_stat_features(train: dict[str, np.ndarray], item_codes: np.ndarray, n_items: int,
                       global_mean: float, prior_count: float = 50.0) -> dict[str, np.ndarray]:
    counts = np.bincount(item_codes, minlength=n_items).astype(np.float64)
    rating_sum = np.bincount(item_codes, weights=train["rating"].astype(np.float64), minlength=n_items)
    mean = np.divide(rating_sum, counts, out=np.zeros_like(rating_sum), where=counts > 0)
    bayes = (rating_sum + prior_count * global_mean) / (counts + prior_count)

    # 条目最近一次被评分的时间（越接近数据末尾越"新"）
    t = train["timestamp"].astype(np.int64)
    last_seen = np.zeros(n_items, dtype=np.int64)
    np.maximum.at(last_seen, item_codes, t)
    t_min, t_max = int(t.min()), int(t.max())
    recency = (last_seen - t_min) / max(t_max - t_min, 1)

    # 条目在最后 10% 时间窗口内的评分占比，衡量热度趋势
    cutoff = t_min + int((t_max - t_min) * 0.9)
    recent = np.bincount(item_codes[t >= cutoff], minlength=n_items).astype(np.float64)
    trend = np.divide(recent, counts, out=np.zeros_like(recent), where=counts > 0)

    return {
        "pop_log": np.log1p(counts),
        "item_mean": mean,
        "item_bayes": bayes,
        "item_recency": recency,
        "item_trend": trend,
    }


def user_stat_features(train: dict[str, np.ndarray], n_users: int) -> dict[str, np.ndarray]:
    counts = np.bincount(train["userId"].astype(np.int64) - 1, minlength=n_users).astype(np.float64)
    t = train["timestamp"].astype(np.int64)
    last = np.zeros(n_users, dtype=np.int64)
    np.maximum.at(last, train["userId"].astype(np.int64) - 1, t)
    t_min, t_max = int(t.min()), int(t.max())
    return {
        "user_activity": np.log1p(counts),
        "user_last_activity": (last - t_min) / max(t_max - t_min, 1),
    }


def bpr_scores(model_path: Path, user_codes: np.ndarray, item_codes: np.ndarray) -> np.ndarray:
    """用 BPR-MF 的因子给候选对打分。"""
    import torch

    from .ranking.bpr import BPRMF, score_pairs

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = BPRMF(ckpt["n_users"], ckpt["n_items"], ckpt["dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        return score_pairs(model.user.weight, model.item.weight, user_codes.astype(np.int32), item_codes.astype(np.int32))