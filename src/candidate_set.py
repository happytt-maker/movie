# -*- coding: utf-8 -*-
"""候选集的内存表示：按用户连续存放，便于逐用户排序与打分。

候选集由 prepare_data.py 生成，每行是 (userId, movieId, label)。
userId 在 MovieLens 中是从 1 开始的连续整数，因此内部统一用 ``userId - 1`` 作为下标。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_io import load_table


class CandidateSet:
    """按用户分组的候选集。

    属性
    ----
    user_ids : 下标 -> 原始 userId
    movie_ids: 编码 -> 原始 movieId
    starts / counts: 每个用户在下标中的区间
    item_index: 电影维度上的下标（0 基编码）
    """

    def __init__(self, path: str | Path, vocab_path: str | Path | None = None):
        table = load_table(path)
        uid = table["userId"].astype(np.int64)
        mid = table["movieId"].astype(np.int64)
        label = table["label"].astype(np.int8)

        order = np.argsort(uid, kind="stable")
        uid, mid, label = uid[order], mid[order], label[order]

        self.path = Path(path)
        self.user_ids, self.movie_ids = self._load_vocab(vocab_path)
        # 候选集内部以原始 userId 作为 0 基下标（MovieLens 的 userId 从 1 连续到 N），
        # 因此 code = userId - 1。这里显式校验，避免默默错位。
        if self.user_ids[0] != 1 or self.user_ids[-1] != len(self.user_ids) or \
                not np.array_equal(self.user_ids, np.arange(1, len(self.user_ids) + 1)):
            raise ValueError("userId 不是从 1 开始的连续整数，CandidateSet 的下标假设不成立")
        self.starts, self.counts = self._user_offsets(uid)
        self.uid = uid
        self.mid = mid
        self.label = label
        self.item_index = self._item_index_from_movies()
        self.item_codes = self.item_index.take(mid)
        self.rows = len(uid)

    def _load_vocab(self, vocab_path):
        vocab_file = Path(vocab_path) if vocab_path else self.path.parent / "vocab.npz"
        vocab = load_table(vocab_file)
        user_ids = vocab["userId"].astype(np.int64)
        movie_ids = vocab["movieId"].astype(np.int64)
        self._movie_id_to_code = None
        return user_ids, movie_ids

    def _user_offsets(self, uid: np.ndarray):
        n_users = int(uid.max()) + 1
        counts = np.bincount(uid, minlength=n_users).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        return starts, counts

    def _item_index_from_movies(self):
        """movieId -> 0 基编码（与 vocab 中 movieId 的顺序一致）。"""
        lookup = np.full(int(self.movie_ids.max()) + 1, -1, dtype=np.int32)
        lookup[self.movie_ids] = np.arange(len(self.movie_ids), dtype=np.int32)
        codes = lookup[self.mid]
        if (codes < 0).any():
            missing = int((codes < 0).sum())
            raise ValueError(f"候选集中有 {missing:,} 行 movieId 不在 vocab 中")
        self._movie_id_to_code = lookup
        return lookup

    @property
    def user_count(self) -> int:
        return len(self.user_ids)

    @property
    def item_count(self) -> int:
        return len(self.movie_ids)

    def movie_to_code(self, movie_ids):
        """原始 movieId -> 内部编码。"""
        if self._movie_id_to_code is None:
            self._item_index_from_movies()
        arr = np.asarray(movie_ids, dtype=np.int64)
        codes = self._movie_id_to_code[arr]
        if (codes < 0).any():
            raise KeyError("movieId 不在 vocab 中")
        return codes

    def user_slice(self, user_code: int) -> slice:
        start = int(self.starts[user_code])
        return slice(start, start + int(self.counts[user_code]))

    def positives(self, user_code: int) -> np.ndarray:
        seg = self.mid[self.user_slice(user_code)]
        mask = self.label[self.user_slice(user_code)] == 1
        return seg[mask]

    def truth_map(self) -> dict[int, set[int]]:
        """userId -> 正样本集合（供评价脚本使用）。"""
        truth = {}
        for u in np.flatnonzero(self.counts):
            mask = self.label[self.user_slice(u)] == 1
            truth[int(self.user_ids[u - 1])] = set(self.mid[self.user_slice(u)][mask].tolist())
        return truth

    def topk(self, scores: np.ndarray, k: int = 10) -> dict[str, np.ndarray]:
        """对所有用户按 scores 取前 k 个，返回可直接写 CSV 的列。

        scores: 长度与候选集行数一致，与用户分段对齐。
        """
        if len(scores) != self.rows:
            raise ValueError(f"scores 长度 {len(scores)} 与候选集行数 {self.rows} 不一致")
        scores = np.asarray(scores, dtype=np.float32)

        top_uid, top_mid, top_score, top_rank = [], [], [], []
        for u in np.flatnonzero(self.counts):
            sl = self.user_slice(u)
            n = sl.stop - sl.start
            if n == 0:
                continue
            seg = scores[sl]
            take = min(k, n)
            if take < n:
                part = np.argpartition(-seg, take - 1)[:take]
            else:
                part = np.arange(n)
            order = np.argsort(-seg[part], kind="stable")
            idx = part[order]
            top_uid.append(np.full(take, self.user_ids[u - 1], dtype=np.int32))
            top_mid.append(self.mid[sl][idx])
            top_score.append(seg[idx])
            top_rank.append(np.arange(1, take + 1, dtype=np.int8))

        return {
            "userId": np.concatenate(top_uid),
            "rank": np.concatenate(top_rank),
            "movieId": np.concatenate(top_mid),
            "score": np.concatenate(top_score),
        }


def write_top10(path, columns) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame({
        "userId": columns["userId"],
        "rank": columns["rank"],
        "movieId": columns["movieId"],
        "score": columns["score"],
    }).to_csv(path, index=False, float_format="%.6f")
    return path