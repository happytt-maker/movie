"""模型加载与推理引擎。

启动时一次性加载：
  1. BPR 召回模型（models/bpr_mf.pt）
  2. LightGBM 排序模型（启动时训练 ~8s，缓存到内存）
  3. 候选集、特征、词表等数据

提供 predict_single / predict_batch 接口供 FastAPI 调用。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
import yaml

# ---------- 项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "base.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = cfg_path.parent / parent_path
        with open(parent_path, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        cfg = {**base, **cfg}
    return cfg


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ============================================================
# BPR 模型（PyTorch）
# ============================================================

class BPRMF(torch.nn.Module):
    """与 src/ranking/bpr.py 中完全一致的结构，用于加载权重。"""

    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = torch.nn.Embedding(n_users, dim)
        self.item = torch.nn.Embedding(n_items, dim)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(dim=1)


# ============================================================
# CandidateSet 精简版（只保留推理需要的部分）
# ============================================================

class CandidateSetLite:
    """按用户分组的候选集，只保留推理所需字段。"""

    def __init__(self, cand_path: Path, vocab_path: Path):
        table = _load_npz(cand_path)
        uid = table["userId"].astype(np.int64)
        mid = table["movieId"].astype(np.int64)
        label = table["label"].astype(np.int8)

        order = np.argsort(uid, kind="stable")
        uid, mid, label = uid[order], mid[order], label[order]

        vocab = _load_npz(vocab_path)
        self.user_ids = vocab["userId"].astype(np.int64)
        self.movie_ids = vocab["movieId"].astype(np.int64)

        # userId 从 1 开始连续 → code = userId - 1
        n_users = int(uid.max()) + 1
        counts = np.bincount(uid, minlength=n_users).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

        self.starts = starts
        self.counts = counts
        self.uid = uid
        self.mid = mid
        self.label = label
        self.rows = len(uid)

        # movieId → 0-based code
        lookup = np.full(int(self.movie_ids.max()) + 1, -1, dtype=np.int32)
        lookup[self.movie_ids] = np.arange(len(self.movie_ids), dtype=np.int32)
        self.item_codes = lookup[mid]

    def user_slice(self, user_code: int) -> slice:
        start = int(self.starts[user_code])
        return slice(start, start + int(self.counts[user_code]))

    def get_candidates_for_user(self, original_uid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回该用户的 (item_codes, movieIds, labels)。

        注意：userId 从 1 开始，bincount 索引 = userId（因为 bincount([1,1,2]) = [0,2,1]）。
        """
        code = original_uid  # bincount index == userId for 1-based ids
        if code < 0 or code >= len(self.starts) or self.counts[code] == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int64), np.array([], dtype=np.int8)
        sl = self.user_slice(code)
        return self.item_codes[sl], self.mid[sl], self.label[sl]


# ============================================================
# 特征构建（复用 build_features.py 的逻辑，但只做单用户/小批量）
# ============================================================

def _build_genre_matrix(movies_csv: Path) -> tuple[np.ndarray, list[str]]:
    """从 movies.csv 构建 movieId → genre 向量矩阵（向量化，无 iterrows）。"""
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer

    df = pd.read_csv(movies_csv)
    df["genre_list"] = df["genres"].fillna("").str.split("|")
    mlb = MultiLabelBinarizer()
    mat_sparse = mlb.fit_transform(df["genre_list"])
    # 按 movieId 索引对齐（movieId 从 1 开始连续）
    n_movies = int(df["movieId"].max()) + 1
    mat = np.zeros((n_movies, mat_sparse.shape[1]), dtype=np.float32)
    mat[df["movieId"].values] = mat_sparse.astype(np.float32)
    return mat, list(mlb.classes_)


def _compute_item_stats(train_npz: dict, movie_ids_arr: np.ndarray,
                        cache_path: Path | None = None) -> dict[str, np.ndarray]:
    """计算条目级统计特征。支持磁盘缓存，二次启动秒级加载。"""
    # 尝试从缓存加载
    if cache_path and cache_path.exists():
        cached = _load_npz(cache_path)
        return {k: cached[k] for k in ["item_count", "item_mean", "item_bayes", "item_recency", "item_trend"]}

    uid = train_npz["userId"]
    mid = train_npz["movieId"]
    rating = train_npz["rating"]
    ts = train_npz["timestamp"]

    n_items = int(movie_ids_arr.max()) + 1
    counts = np.bincount(mid, minlength=n_items).astype(np.float64)
    sums = np.zeros(n_items, dtype=np.float64)
    np.add.at(sums, mid, rating)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0, sums / counts, 0.0)

    global_mean = float(rating.mean())
    C = 10.0
    bayes = (counts * means + C * global_mean) / (counts + C)

    max_ts = np.zeros(n_items, dtype=np.float64)
    np.maximum.at(max_ts, mid, ts.astype(np.float64))
    valid_max = max_ts[max_ts > 0]
    ts_range = valid_max.max() - valid_max.min() if len(valid_max) > 0 else 1.0
    recency = np.where(max_ts > 0, (max_ts - valid_max.min()) / ts_range, 0.0)

    # Trend: numpy 分组循环（~22s 首次，之后走缓存）
    sort_idx = np.lexsort((ts, mid))
    mid_s = mid[sort_idx]
    rating_s = rating[sort_idx].astype(np.float64)
    boundaries = np.concatenate([[0], np.where(np.diff(mid_s) != 0)[0] + 1, [len(mid_s)]])
    trend = np.zeros(n_items, dtype=np.float64)
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        n = e - s
        if n < 4:
            continue
        cut = max(1, n // 5)
        movie_id = int(mid_s[s])
        if 0 <= movie_id < n_items:
            trend[movie_id] = rating_s[e - cut:e].mean() - rating_s[s:s + cut].mean()

    result = {
        "item_count": counts, "item_mean": means, "item_bayes": bayes,
        "item_recency": recency, "item_trend": trend,
    }

    # 写入缓存
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **result)

    return result


def _compute_user_stats(train_npz: dict, user_ids_arr: np.ndarray) -> dict[str, np.ndarray]:
    """计算用户级统计特征。"""
    uid = train_npz["userId"]
    ts = train_npz["timestamp"]

    n_users = int(user_ids_arr.max()) + 1
    activity = np.bincount(uid, minlength=n_users).astype(np.float64)
    last_ts = np.zeros(n_users, dtype=np.float64)
    np.maximum.at(last_ts, uid, ts.astype(np.float64))
    ts_range = last_ts.max() - last_ts[last_ts > 0].min() if (last_ts > 0).any() else 1.0
    last_activity = np.where(last_ts > 0, (last_ts - last_ts[last_ts > 0].min()) / ts_range, 0.0)

    return {
        "user_activity": activity,
        "user_last_activity": last_activity,
    }


# ============================================================
# 推理引擎
# ============================================================

class RecommendationEngine:
    """封装全部模型与数据，提供推理接口。"""

    def __init__(self):
        self.cfg = _load_config()
        self.processed = _resolve(self.cfg["paths"]["processed_dir"])
        self.features_dir = _resolve(self.cfg["paths"].get("features_dir", "data/features"))
        self.models_dir = _resolve(self.cfg["paths"].get("models_dir", "models"))
        self.results_dir = _resolve(self.cfg["paths"]["results_dir"])
        self.raw_dir = _resolve(self.cfg["paths"]["raw_dir"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_time = 0.0
        self.loaded_models: list[str] = []

        # 延迟加载的组件
        self._bpr_model: BPRMF | None = None
        self._booster: lgb.Booster | None = None
        self._cand: CandidateSetLite | None = None
        self._vocab: dict[str, np.ndarray] | None = None
        self._genre_mat: np.ndarray | None = None
        self._genre_names: list[str] | None = None
        self._item_stats: dict[str, np.ndarray] | None = None
        self._user_stats: dict[str, np.ndarray] | None = None
        self._train_npz: dict[str, np.ndarray] | None = None
        self._feature_names: list[str] | None = None
        self._movies_df = None  # for movie name lookup

    def load_all(self) -> float:
        """一次性加载所有模型和数据，返回耗时（秒）。"""
        t0 = time.perf_counter()

        # 1. BPR 模型
        bpr_path = self.models_dir / "bpr_mf.pt"
        if bpr_path.exists():
            ckpt = torch.load(bpr_path, map_location="cpu", weights_only=False)
            self._bpr_model = BPRMF(ckpt["n_users"], ckpt["n_items"], ckpt["dim"]).to(self.device)
            self._bpr_model.load_state_dict(
                {k: v.to(self.device) for k, v in ckpt["model_state"].items()}
            )
            self._bpr_model.eval()
            for p in self._bpr_model.parameters():
                p.requires_grad_(False)
            self.loaded_models.append(f"bpr_mf.pt ({ckpt.get('valid_ndcg@10', '?'):.4f})")
        else:
            raise FileNotFoundError(f"BPR 模型不存在: {bpr_path}")

        # 2. 候选集与词表
        self._cand = CandidateSetLite(
            self.processed / "candidates_test.npz",
            self.processed / "vocab.npz",
        )
        self._vocab = _load_npz(self.processed / "vocab.npz")

        # 3. 训练数据（用于特征计算）
        self._train_npz = _load_npz(self.processed / "train.npz")

        # 4. 条目/用户统计特征（带磁盘缓存，首次 ~22s，之后 < 0.1s）
        stats_cache = self.processed / "item_stats_cache.npz"
        self._item_stats = _compute_item_stats(self._train_npz, self._vocab["movieId"], cache_path=stats_cache)
        self._user_stats = _compute_user_stats(self._train_npz, self._vocab["userId"])

        # 5. 类型矩阵
        movies_csv = self.raw_dir / "movies.csv"
        if movies_csv.exists():
            self._genre_mat, self._genre_names = _build_genre_matrix(movies_csv)
            # 加载电影名用于展示
            import pandas as pd
            self._movies_df = pd.read_csv(movies_csv)[["movieId", "title", "genres"]]
        else:
            self._genre_mat = np.zeros((int(self._vocab["movieId"].max()) + 1, 19), dtype=np.float32)
            self._genre_names = [f"genre_{i}" for i in range(19)]
            self._movies_df = None

        # 6. LightGBM 排序模型（启动时训练）
        self._train_ranker()

        # 7. 特征名
        self._feature_names = [
            "pop_log", "item_mean", "item_bayes", "item_recency", "item_trend",
            "user_activity", "user_last_activity", "genre_affinity", "bpr",
        ]

        self.load_time = round(time.perf_counter() - t0, 2)
        return self.load_time

    def _train_ranker(self) -> None:
        """启动时训练或加载 LightGBM LambdaRank（首次 ~12s，之后 < 0.1s）。"""
        import lightgbm as lgb

        # 优先加载离线流水线落盘的同一版权重（scripts/train_ranker.py 产出），
        # 保证线上/线下同一模型、同一打分；没有才回退到"启动时自训 + 缓存"。
        shared_model = self.models_dir / "ranker_full.txt"
        shared_meta = self.models_dir / "ranker_full.meta.json"
        if shared_model.exists() and shared_meta.exists():
            import json as _json
            meta = _json.loads(shared_meta.read_text(encoding="utf-8"))
            self._booster = lgb.Booster(model_file=str(shared_model))
            self.loaded_models.append(
                f"LightGBM (models/ranker_full.txt, best_iter={meta.get('best_iteration')}, "
                f"tiebreak={meta.get('tiebreak', 'none')})"
            )
            return

        # 兼容旧路径：LightGBM 的 C 库不支持非 ASCII 路径，缓存放到纯 ASCII 目录
        cache_dir = Path(r"C:\temp\movie_recsys_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "ranker_cache.txt"
        if cache_path.exists():
            self._booster = lgb.Booster(model_file=str(cache_path))
            self.loaded_models.append(f"LightGBM (cached, best_iter={self._booster.best_iteration})")
            return

        valid_feat = np.load(self.features_dir / "valid.npz")
        names = [k for k in valid_feat.files if k not in {"label", "userId", "movieId"}]
        uid = valid_feat["userId"]
        label = valid_feat["label"]

        uniq_users = np.unique(uid)
        rng = np.random.default_rng(int(self.cfg["seed"]))
        keep = np.sort(rng.choice(uniq_users, min(80000, len(uniq_users)), replace=False))
        lookup = np.zeros(int(uniq_users.max()) + 1, dtype=bool)
        lookup[keep] = True
        mask = lookup[uid]

        X = np.column_stack([valid_feat[n][mask] for n in names]).astype(np.float32)
        y = label[mask].astype(np.int8)
        groups_all = np.bincount(uid[mask])
        user_all = uid[mask]

        users_train = np.flatnonzero(groups_all > 0)
        is_valid_user = (users_train % 10 == 0)
        valid_users_set = set(users_train[is_valid_user].tolist())
        valid_mask = np.isin(user_all, np.fromiter(valid_users_set, dtype=np.int64))
        train_mask = ~valid_mask

        def build(m, la):
            sub_u = user_all[m]
            g = np.bincount(sub_u)
            g = g[g > 0]
            return X[m], la[m], g

        X_tr, y_tr, g_tr = build(train_mask, y)
        X_va, y_va, g_va = build(valid_mask, y)

        dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=names, free_raw_data=True)
        dvalid = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, feature_name=names, free_raw_data=True)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 30,
            "num_threads": 12,
            "seed": int(self.cfg["seed"]),
            "verbose": -1,
        }
        self._booster = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        # 缓存到纯 ASCII 路径（LightGBM C 库不支持非 ASCII 路径）
        cache_dir = Path(r"C:\temp\movie_recsys_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "ranker_cache.txt"
        self._booster.save_model(str(cache_path))
        self.loaded_models.append(f"LightGBM (trained, best_iter={self._booster.best_iteration})")

    def _build_features_for_candidates(
        self, item_codes: np.ndarray, movie_ids: np.ndarray, user_id: int, bpr_scores: np.ndarray
    ) -> np.ndarray:
        """为单个用户的候选集构建 9 维特征矩阵。"""
        n = len(item_codes)
        if n == 0:
            return np.empty((0, 9), dtype=np.float32)

        feats = np.zeros((n, 9), dtype=np.float32)

        # pop_log
        counts = self._item_stats["item_count"]
        feats[:, 0] = np.log1p(counts[movie_ids.clip(0, len(counts) - 1)])

        # item_mean
        feats[:, 1] = self._item_stats["item_mean"][movie_ids.clip(0, len(self._item_stats["item_mean"]) - 1)]

        # item_bayes
        feats[:, 2] = self._item_stats["item_bayes"][movie_ids.clip(0, len(self._item_stats["item_bayes"]) - 1)]

        # item_recency
        feats[:, 3] = self._item_stats["item_recency"][movie_ids.clip(0, len(self._item_stats["item_recency"]) - 1)]

        # item_trend
        feats[:, 4] = self._item_stats["item_trend"][movie_ids.clip(0, len(self._item_stats["item_trend"]) - 1)]

        # user_activity
        ua = self._user_stats["user_activity"]
        feats[:, 5] = ua[user_id] if user_id < len(ua) else 0.0

        # user_last_activity
        ula = self._user_stats["user_last_activity"]
        feats[:, 6] = ula[user_id] if user_id < len(ula) else 0.0

        # genre_affinity: 用户历史类型偏好 vs 候选电影类型的余弦相似度
        if self._genre_mat is not None and self._train_npz is not None:
            train_mid = self._train_npz["movieId"]
            train_uid = self._train_npz["userId"]
            user_mask = train_uid == user_id
            if user_mask.any():
                user_movies = train_mid[user_mask]
                valid_mids = user_movies[user_movies < self._genre_mat.shape[0]]
                if len(valid_mids) > 0:
                    user_genre_vec = self._genre_mat[valid_mids].mean(axis=0)
                    norm = np.linalg.norm(user_genre_vec)
                    if norm > 0:
                        user_genre_vec /= norm
                        cand_mids = movie_ids[movie_ids < self._genre_mat.shape[0]]
                        if len(cand_mids) == n:
                            cand_genre = self._genre_mat[cand_mids]
                            norms = np.linalg.norm(cand_genre, axis=1, keepdims=True)
                            norms = np.where(norms > 0, norms, 1.0)
                            cand_genre_normed = cand_genre / norms
                            feats[:, 7] = cand_genre_normed @ user_genre_vec

        # bpr score
        feats[:, 8] = bpr_scores

        return feats

    def _bpr_score_candidates(self, user_id: int, item_codes: np.ndarray) -> np.ndarray:
        """用 BPR 模型给候选打分。"""
        if len(item_codes) == 0:
            return np.array([], dtype=np.float32)
        u_tensor = torch.full((len(item_codes),), user_id - 1, dtype=torch.long, device=self.device)
        i_tensor = torch.from_numpy(item_codes.astype(np.int64)).to(self.device)
        with torch.no_grad():
            scores = self._bpr_model.score(u_tensor, i_tensor).cpu().numpy()
        return scores.astype(np.float32)

    def predict_single(self, user_id: int, top_k: int = 10) -> dict[str, Any]:
        """单用户推荐。"""
        t0 = time.perf_counter()

        item_codes, movie_ids, _labels = self._cand.get_candidates_for_user(user_id)
        if len(item_codes) == 0:
            return {"userId": user_id, "recommendations": [], "latencyMs": 0.0}

        # BPR 打分
        bpr_scores = self._bpr_score_candidates(user_id, item_codes)

        # 构建特征
        features = self._build_features_for_candidates(item_codes, movie_ids, user_id, bpr_scores)

        # LightGBM 打分
        lgb_scores = self._booster.predict(features, num_iteration=self._booster.best_iteration)

        # Top-K（同分用 BPR 分数打破，与离线 CandidateSet.topk 保持完全一致）
        k = min(top_k, len(lgb_scores))
        top_idx = np.lexsort((-bpr_scores, -lgb_scores))[:k]

        recs = []
        for rank, idx in enumerate(top_idx, 1):
            recs.append({
                "rank": rank,
                "movieId": int(movie_ids[idx]),
                "score": float(lgb_scores[idx]),
            })

        latency = (time.perf_counter() - t0) * 1000
        return {"userId": user_id, "recommendations": recs, "latencyMs": round(latency, 2)}

    def predict_batch(self, user_ids: list[int], top_k: int = 10) -> dict[str, Any]:
        """批量推荐。"""
        t0 = time.perf_counter()
        results = []
        for uid in user_ids:
            r = self.predict_single(uid, top_k)
            results.append(r)
        total_latency = (time.perf_counter() - t0) * 1000
        return {"totalUsers": len(user_ids), "results": results, "totalLatencyMs": round(total_latency, 2)}

    def get_movie_info(self, movie_id: int) -> dict[str, str] | None:
        """查询电影名称和类型。"""
        if self._movies_df is None:
            return None
        row = self._movies_df[self._movies_df["movieId"] == movie_id]
        if row.empty:
            return None
        return {"title": row.iloc[0]["title"], "genres": row.iloc[0]["genres"]}

    def evaluate(self, pred_file: str, name: str = "api_eval") -> dict[str, Any]:
        """对预测结果文件做评估。"""
        import pandas as pd
        from src.metrics import ndcg_at_k, recall_at_k

        pred_path = _resolve(pred_file)
        if not pred_path.exists():
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")

        df = pd.read_csv(pred_path, dtype={"userId": "int64", "rank": "int32", "movieId": "int64"})
        if not {"userId", "rank", "movieId"}.issubset(df.columns):
            raise ValueError(f"缺少必要列，实际列: {list(df.columns)}")

        # 加载真值
        truth = self._cand_truth()
        k = 10
        ndcgs, recalls, mrrs = [], [], []

        for uid, group in df.groupby("userId", sort=False):
            top = group.sort_values("rank")["movieId"].head(k).tolist()
            rel = truth.get(int(uid), set())
            if not rel:
                continue
            ndcgs.append(ndcg_at_k(top, rel, k))
            hits = len(set(top) & rel)
            recalls.append(hits / len(rel))
            # MRR
            for rank, m in enumerate(top, 1):
                if m in rel:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)

        return {
            "model": name,
            "ndcg10": round(float(np.mean(ndcgs)), 5) if ndcgs else 0.0,
            "recall10": round(float(np.mean(recalls)), 5) if recalls else 0.0,
            "mrr10": round(float(np.mean(mrrs)), 5) if mrrs else None,
            "users": len(ndcgs),
            "latencyMs": 0.0,
        }

    def _cand_truth(self) -> dict[int, set[int]]:
        """从候选集提取真值（正样本集合）。"""
        truth = {}
        for u in np.flatnonzero(self._cand.counts):
            sl = self._cand.user_slice(u)
            pos = self._cand.mid[sl][self._cand.label[sl] == 1]
            truth[int(self._cand.user_ids[u - 1])] = set(pos.tolist())
        return truth


# 全局单例
_engine: RecommendationEngine | None = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
        _engine.load_all()
    return _engine
