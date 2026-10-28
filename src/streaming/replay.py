# -*- coding: utf-8 -*-
"""任务四：流式增量处理（按时间戳回放评分流）。

实现三件事（对应任务书"完成采样、窗口统计和用户画像增量更新"）：
  1. 采样：按 Bernoulli 采样或蓄水池采样，从流中抽取统计样本；
  2. 窗口统计：按 1 天滚动窗口聚合事件量、均分和类型分布；
  3. 画像增量更新：每个用户维护"类型计数画像"，事件到达即更新，不做全量重算。

用法：
    python -m src.streaming.replay --config configs/base.yaml --limit 500000
    python -m src.streaming.replay --config configs/base.yaml --split test --decay 0.999
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from ..data_io import load_config, load_table, resolve


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GenreIndex:
    """movieId -> 类型下标列表。"""

    def __init__(self, movies_csv: Path):
        import csv

        self.genres = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
            "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western",
        ]
        index = {g: i for i, g in enumerate(self.genres)}
        self.map: dict[int, list[int]] = {}
        with open(movies_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                gs = (row.get("genres") or "").split("|")
                self.map[int(row["movieId"])] = [index[g] for g in gs if g in index]

    def get(self, movie_id: int) -> list[int]:
        return self.map.get(movie_id, [])


class StreamingState:
    """增量状态：窗口聚合 + 用户画像 + 采样样本。"""

    def __init__(self, n_genres: int, window_seconds: int = 86400, decay: float = 1.0,
                 sample_rate: float = 0.001, reservoir_size: int = 5000, seed: int = 20260907):
        self.n_genres = n_genres
        self.window_seconds = window_seconds
        self.decay = decay
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

        self.profiles: dict[int, np.ndarray] = {}
        self.last_ts: dict[int, int] = {}
        self.window_start = None
        self.window_count = 0
        self.window_sum = 0.0
        self.window_genres = np.zeros(n_genres, dtype=np.int64)
        self.windows = []
        self.sampled = 0
        self.seen = 0
        self.reservoir = deque(maxlen=reservoir_size)
        self.update_seconds = 0.0

    def apply(self, user_id: int, movie_id: int, rating: float, timestamp: int, genre_ids: list[int]) -> None:
        self.seen += 1

        # 1) 采样：Bernoulli 采样 + 蓄水池
        if self.rng.random() < self.sample_rate:
            self.sampled += 1
        if len(self.reservoir) < self.reservoir.maxlen:
            self.reservoir.append((user_id, movie_id, rating))
        else:
            j = int(self.rng.integers(0, self.seen))
            if j < self.reservoir.maxlen:
                self.reservoir[j] = (user_id, movie_id, rating)

        # 2) 滚动窗口统计
        if self.window_start is None:
            self.window_start = timestamp
        if timestamp - self.window_start >= self.window_seconds:
            self.windows.append({
                "start": int(self.window_start),
                "events": self.window_count,
                "mean_rating": round(self.window_sum / max(self.window_count, 1), 4),
                "top_genres": self._top_genres(),
            })
            self.window_start = timestamp
            self.window_count = 0
            self.window_sum = 0.0
            self.window_genres = np.zeros(self.n_genres, dtype=np.int64)
        self.window_count += 1
        self.window_sum += rating
        for g in genre_ids:
            self.window_genres[g] += 1

        # 3) 用户画像增量更新（可选时间衰减）
        t0 = time.perf_counter()
        profile = self.profiles.get(user_id)
        if profile is None:
            profile = np.zeros(self.n_genres, dtype=np.float32)
            self.profiles[user_id] = profile
            self.last_ts[user_id] = timestamp
        elif self.decay < 1.0:
            days = max((timestamp - self.last_ts[user_id]) / 86400.0, 0.0)
            profile *= math.pow(self.decay, days)
            self.last_ts[user_id] = timestamp
        for g in genre_ids:
            profile[g] += 1.0
        self.update_seconds += time.perf_counter() - t0

    def _top_genres(self, n: int = 3):
        if self.window_genres.sum() == 0:
            return []
        top = np.argsort(-self.window_genres)[:n]
        return [{"genre_index": int(i), "count": int(self.window_genres[i])} for i in top if self.window_genres[i] > 0]

    def finish(self) -> None:
        if self.window_count:
            self.windows.append({
                "start": int(self.window_start or 0),
                "events": self.window_count,
                "mean_rating": round(self.window_sum / max(self.window_count, 1), 4),
                "top_genres": self._top_genres(),
            })


def run(config_path: str, split: str, limit: int | None, decay: float, sample_rate: float) -> dict:
    cfg = load_config(config_path)
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])
    raw_dir = resolve(cfg["paths"]["raw_dir"])

    genre_index = GenreIndex(raw_dir / "movies.csv")
    table = load_table(processed / f"{split}.npz")
    order = np.argsort(table["timestamp"], kind="stable")
    if limit:
        order = order[:limit]
    log(f"回放 {split} 划分：{len(order):,} 条事件（时间戳升序）")

    state = StreamingState(len(genre_index.genres), decay=decay, sample_rate=sample_rate,
                           seed=int(cfg["seed"]), window_seconds=86400)

    t0 = time.perf_counter()
    users = table["userId"][order]
    movies = table["movieId"][order]
    ratings = table["rating"][order]
    stamps = table["timestamp"][order]
    for u, m, r, ts in zip(users.tolist(), movies.tolist(), ratings.tolist(), stamps.tolist()):
        state.apply(u, m, r, ts, genre_index.get(m))
    state.finish()
    elapsed = time.perf_counter() - t0

    # 与批量计算对齐：画像计数应等于该用户在流中出现次数的类型累计
    check_user = int(users[0])
    batch = np.zeros(len(genre_index.genres), dtype=np.float32)
    for m in movies[users == check_user].tolist():
        for g in genre_index.get(m):
            batch[g] += 1.0
    profile = state.profiles[check_user]
    consistent = bool(np.allclose(profile, batch)) if decay >= 1.0 else None

    report = {
        "split": split,
        "events": int(len(order)),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_events_per_sec": round(len(order) / max(elapsed, 1e-9), 0),
        "avg_update_us": round(state.update_seconds / max(len(order), 1) * 1e6, 3),
        "windows": len(state.windows),
        "window_hours": 24,
        "sampled_events": int(state.sampled),
        "sample_rate": sample_rate,
        "reservoir_size": len(state.reservoir),
        "users_with_profile": len(state.profiles),
        "decay_per_day": decay,
        "profile_consistency_with_batch": consistent,
        "first_windows": state.windows[:5],
        "last_window": state.windows[-1] if state.windows else None,
        "events_per_window": {
            "min": min((w["events"] for w in state.windows), default=0),
            "max": max((w["events"] for w in state.windows), default=0),
            "mean": round(float(np.mean([w["events"] for w in state.windows])), 1) if state.windows else 0,
        },
    }
    out = results / "streaming_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"处理 {len(order):,} 事件，{elapsed:.1f}s，吞吐 {report['throughput_events_per_sec']:,.0f} 事件/秒，"
        f"画像更新 {report['avg_update_us']:.2f} µs/次")
    log(f"窗口 {len(state.windows):,} 个（24h），采样 {state.sampled:,} 条，画像用户 {len(state.profiles):,}")
    log(f"报告写入 {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="评分流回放与增量画像")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"])
    parser.add_argument("--limit", type=int, default=None, help="只回放前 N 条事件")
    parser.add_argument("--decay", type=float, default=1.0, help="画像按天衰减系数，1.0 表示不衰减")
    parser.add_argument("--sample-rate", type=float, default=0.001)
    args = parser.parse_args()
    run(args.config, args.split, args.limit, args.decay, args.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())