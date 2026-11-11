# -*- coding: utf-8 -*-
"""公共模块：数据加载、时间切分、偏置模型、评估。

数据事实（见 reports/audit.json）：
  - 31,900,204 条评分，396,060 用户 / 142,609 电影
  - 评分 0.5~5.0，均值 3.5404，标准差 1.0590
  - 两个半边（真实 / 合成）ID 区间完全隔离，统计分布几乎一致
    → 可用同一个模型，不必分组建模
  - 测试集 10 万行全部晚于该用户的全部训练评分（时间外推）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "output"
for _d in (MODELS, REPORTS, OUTPUT):
    _d.mkdir(exist_ok=True)

# 两个半边的 ID 边界（来自数据附带的 id_ranges.json）
ML_USER_MAX = 200948
ML_MOVIE_MAX = 87585

# 本地验证的截断点：复刻老师「按全局时间截断、留出之后全部评分」的规则。
# 选 2023-05-01 是因为留出量与老师的测试集同量级（约 10.7 万行 / 7,198 用户 / 每人中位 6 条，
# 老师为 10 万行 / 5,498 用户 / 中位 4 条），形态可比。
VALID_CUTOFF = int(pd.Timestamp("2023-05-01", tz="UTC").timestamp())
MIN_TRAIN_HISTORY = 5

RATING_MIN, RATING_MAX = 0.5, 5.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rmse(pred, truth) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred, truth) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.mean(np.abs(p - t)))


def load_ratings() -> pd.DataFrame:
    dtype = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    df = pd.read_csv(DATA / "ratings.csv", dtype=dtype)
    df["half"] = np.where(df.userId <= ML_USER_MAX, 0, 1).astype(np.int8)  # 0=真实 1=合成
    return df


def load_test_pairs() -> pd.DataFrame:
    tp = pd.read_csv(DATA / "test_pairs.csv")
    tp["half"] = np.where(tp.userId <= ML_USER_MAX, 0, 1).astype(np.int8)
    return tp


def temporal_split(df: pd.DataFrame, cutoff: int = VALID_CUTOFF,
                   min_history: int = MIN_TRAIN_HISTORY) -> tuple[np.ndarray, np.ndarray]:
    """按全局时间截断：cutoff 之前训练，之后留出。

    与老师口径一致的一条约束：留出用户必须在训练期至少有 min_history 条评分，
    否则该用户的评分无法估计，不纳入留出（他们的评分仍留在训练侧）。
    返回 (train_mask, valid_mask)。
    """
    before = df.timestamp.to_numpy() < cutoff
    counts = np.bincount(df.userId.to_numpy()[before], minlength=int(df.userId.max()) + 1)
    eligible_user = counts[df.userId.to_numpy()] >= min_history
    valid = (~before) & eligible_user
    train = before
    return train, valid


TAIL_CUTOFF = int(pd.Timestamp("2023-07-01", tz="UTC").timestamp())


def tail_split(df: pd.DataFrame, cutoff: int = TAIL_CUTOFF,
               min_history: int = MIN_TRAIN_HISTORY) -> tuple[np.ndarray, np.ndarray]:
    """按老师的「时间尾部」口径重建留出集。

    依据：老师的 10 万测试行全部晚于该用户训练期最后一条评分，间隔中位数仅 0.1 天
    （p75 = 18 天），可见它不是「全局截断半年」，而是「紧邻用户历史的尾部一段」。
    原先 VALID_CUTOFF 切出来的间隔中位是 120 天，会把「用户越往后打分越低」的
    时间漂移放大，从而高估误差。

    取 TAIL_CUTOFF 是为了让留出集的用户结构对齐老师测试集：
    本口径 5,480 用户 / 每人中位 4 条  vs  老师 5,498 用户 / 每人中位 4 条。
    """
    return temporal_split(df, cutoff=cutoff, min_history=min_history)


def sparse_sim_split(df: pd.DataFrame, n_users: int = 1043, keep: int = 5,
                     holdout_per_user: int = 62, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """模拟测试集里最重的那一群用户：只留 5 条历史，其余全部待预测。

    动机（见 reports/final_eval.json 与测试集统计）：10 万测试行里有 65,168 行（65.2%）
    来自仅 1,043 位「训练表里恰好 5 条评分」的用户，人均要预测 62 条。
    任何按时间切分的验证集都造不出这种形态（5 条历史是构造出来的，不是自然截断的结果），
    所以这里显式模拟：抽 1,043 位评分足够多的用户，只保留他们最早的 5 条进训练，
    之后的 62 条留出评估——与测试集同构。
    """
    ts = df.timestamp.to_numpy()
    uid = df.userId.to_numpy()
    order = np.argsort(ts, kind="stable")
    cnt = np.bincount(uid)
    pool = np.flatnonzero(cnt >= keep + holdout_per_user)
    rng = np.random.default_rng(seed)
    picked = rng.choice(pool, size=min(n_users, len(pool)), replace=False)
    picked_set = np.zeros(len(cnt), dtype=bool)
    picked_set[picked] = True

    # 每个被抽中的用户：按时间前 keep 条进训练，其后的 holdout_per_user 条留出
    idx = np.flatnonzero(picked_set[uid])
    order2 = idx[np.lexsort((ts[idx], uid[idx]))]          # 组内按时间升序
    su = uid[order2]
    is_new = np.empty(len(order2), dtype=bool)
    is_new[0] = True
    is_new[1:] = su[1:] != su[:-1]
    grp_start = np.flatnonzero(is_new)
    grp_len = np.diff(np.append(grp_start, len(order2)))
    pos = np.arange(len(order2)) - np.repeat(grp_start, grp_len)   # 组内序号

    train = np.ones(len(df), dtype=bool)
    train[idx] = False                                     # 被抽中用户的全部行先全部排除
    valid = np.zeros(len(df), dtype=bool)
    train[order2[pos < keep]] = True
    valid[order2[(pos >= keep) & (pos < keep + holdout_per_user)]] = True
    return train, valid


def describe_split(df: pd.DataFrame, train: np.ndarray, valid: np.ndarray) -> dict:
    v = df[valid]
    t = df[train]
    g = v.groupby("userId").size()
    return {
        "train_rows": int(len(t)),
        "valid_rows": int(len(v)),
        "valid_users": int(v.userId.nunique()),
        "valid_movies": int(v.movieId.nunique()),
        "valid_per_user_median": float(g.median()) if len(g) else 0.0,
        "valid_per_user_max": int(g.max()) if len(g) else 0,
        "valid_by_half": {"ml": int((v.half == 0).sum()), "synthetic": int((v.half == 1).sum())},
        "train_time_max": pd.to_datetime(int(t.timestamp.max()), unit="s").strftime("%Y-%m-%d"),
        "valid_time_min": pd.to_datetime(int(v.timestamp.min()), unit="s").strftime("%Y-%m-%d"),
        "valid_time_max": pd.to_datetime(int(v.timestamp.max()), unit="s").strftime("%Y-%m-%d"),
    }


class BiasModel:
    """μ + b_u + b_i 的交替最小二乘。

    只做偏置，不含隐向量；既是基线，也为 MF 提供偏置初值。
    """

    def __init__(self, n_users: int, n_movies: int, lam_u: float = 12.0, lam_i: float = 8.0):
        self.n_users = n_users
        self.n_movies = n_movies
        self.lam_u = lam_u
        self.lam_i = lam_i
        self.mu = 0.0
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_movies, dtype=np.float32)

    def fit(self, uid: np.ndarray, mid: np.ndarray, y: np.ndarray, rounds: int = 8) -> "BiasModel":
        uid = uid.astype(np.int64)
        mid = mid.astype(np.int64)
        y = y.astype(np.float64)
        self.mu = float(y.mean())
        cnt_u = np.bincount(uid, minlength=self.n_users).astype(np.float64)
        cnt_m = np.bincount(mid, minlength=self.n_movies).astype(np.float64)
        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_movies)
        for _ in range(rounds):
            resid = y - self.mu - bi[mid]
            bu = np.bincount(uid, weights=resid, minlength=self.n_users) / (cnt_u + self.lam_u)
            resid = y - self.mu - bu[uid]
            bi = np.bincount(mid, weights=resid, minlength=self.n_movies) / (cnt_m + self.lam_i)
        self.bu = bu.astype(np.float32)
        self.bi = bi.astype(np.float32)
        return self

    def predict(self, uid: np.ndarray, mid: np.ndarray) -> np.ndarray:
        return (self.mu + self.bu[uid] + self.bi[mid]).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi)

    @classmethod
    def load(cls, path: Path) -> "BiasModel":
        z = np.load(path)
        m = cls(len(z["bu"]), len(z["bi"]))
        m.mu = float(z["mu"])
        m.bu, m.bi = z["bu"], z["bi"]
        return m


def append_experiment(record: dict, path: Path | None = None) -> None:
    path = path or (REPORTS / "experiments.json")
    hist = []
    if path.exists():
        hist = json.loads(path.read_text(encoding="utf-8"))
    hist.append(record)
    path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
