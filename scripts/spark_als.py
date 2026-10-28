#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务三：Spark MLlib ALS 分布式矩阵分解（隐式反馈）+ 扩展性实验。

运行（Windows 上注意两点，见 docs 说明）：
    1. 脚本路径必须是纯 ASCII，否则 JVM 的路径解析会报 Bad pathname；
    2. 通过 spark-submit 启动，Python 侧直接创建 SparkSession 在部分沙箱环境下会挂起。

    spark-submit --master local[4] --driver-memory 6g scripts/spark_als.py \
        --train data/spark/train_positive.csv \
        --candidates data/spark/candidates_sample.csv \
        --factors models/als_factors.npz --report results/als_report.json

扩展性实验：
    用 --scales 0.25,0.5,1.0 对不同数据规模分别训练，记录耗时与迭代收敛情况。
"""

from __future__ import annotations

import argparse
import json
import math
import time
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def build_spark(app: str, master: str, driver_memory: str, shuffle_partitions: int) -> SparkSession:
    return (SparkSession.builder
            .master(master)
            .appName(app)
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.ui.enabled", "false")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate())


def train_als(spark, train_path: str, rank: int, iters: int, reg: float, alpha: float,
              seed: int, scale: float = 1.0) -> tuple[object, dict]:
    t0 = time.perf_counter()
    df = (spark.read.csv(train_path, header=True, inferSchema=False)
          .select(F.col("userId").cast("int"), F.col("movieId").cast("int"),
                  F.col("rating").cast("float")))
    if scale < 1.0:
        df = df.sample(fraction=scale, seed=seed)
    n_rows = df.count()
    log(f"训练样本 {n_rows:,} 行（scale={scale}），读取耗时 {time.perf_counter()-t0:.1f}s")

    als = ALS(rank=rank, regParam=reg, alpha=alpha, implicitPrefs=True,
              maxIter=iters, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", seed=seed, nonnegative=True)
    t1 = time.perf_counter()
    model = als.fit(df)
    fit_seconds = time.perf_counter() - t1
    log(f"ALS 训练完成：{fit_seconds:.1f}s（rank={rank}, iters={iters}）")
    info = {"rows": int(n_rows), "fit_seconds": round(fit_seconds, 1),
            "read_seconds": round(t1 - t0, 1), "rank": rank, "iters": iters, "scale": scale}
    return model, info


def evaluate(spark, model, cand_path: str, k: int = 10) -> dict:
    t0 = time.perf_counter()
    cand = (spark.read.csv(cand_path, header=True, inferSchema=False)
            .select(F.col("userId").cast("int"), F.col("movieId").cast("int"),
                    F.col("label").cast("int")))
    preds = model.transform(cand).select("userId", "movieId", "label", "prediction")
    rows = preds.collect()
    score_seconds = time.perf_counter() - t0
    log(f"候选打分完成：{len(rows):,} 行，{score_seconds:.1f}s")

    by_user: dict[int, list[tuple[float, int]]] = {}
    truth: dict[int, set[int]] = {}
    for r in rows:
        u = int(r["userId"])
        by_user.setdefault(u, []).append((float(r["prediction"]), int(r["movieId"])))
        if int(r["label"]) == 1:
            truth.setdefault(u, set()).add(int(r["movieId"]))

    ndcgs = []
    for u, items in by_user.items():
        if u not in truth:
            continue
        items.sort(key=lambda x: -x[0])
        top = [m for _, m in items[:k]]
        ndcgs.append(ndcg_at_k(top, truth[u], k))
    metrics = {
        "users": len(ndcgs),
        "ndcg@10": round(float(np.mean(ndcgs)), 5) if ndcgs else 0.0,
        "score_seconds": round(score_seconds, 1),
        "rows": len(rows),
    }
    log(f"ALS 评估：NDCG@10={metrics['ndcg@10']:.4f}（{metrics['users']:,} 用户）")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Spark ALS 分布式推荐")
    parser.add_argument("--train", default="data/spark/train_positive.csv")
    parser.add_argument("--candidates", default="data/spark/candidates_sample.csv")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--driver-memory", default="6g")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--scales", default="", help="扩展性实验的数据比例，如 0.25,0.5,1.0")
    parser.add_argument("--factors", default="models/als_factors.npz")
    parser.add_argument("--report", default="results/als_report.json")
    args = parser.parse_args()

    spark = build_spark("movielens-als", args.master, args.driver_memory, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    log(f"Spark {spark.version} 启动，master={args.master}")

    report = {"engine": "Spark MLlib ALS", "spark_version": spark.version, "runs": []}
    try:
        if args.scales:
            for scale in [float(x) for x in args.scales.split(",")]:
                model, info = train_als(spark, args.train, args.rank, args.iters,
                                        args.reg, args.alpha, args.seed, scale)
                info["ndcg@10"] = evaluate(spark, model, args.candidates)["ndcg@10"]
                report["runs"].append(info)
                log(f"scale={scale} 完成：{info}")
        else:
            model, info = train_als(spark, args.train, args.rank, args.iters,
                                    args.reg, args.alpha, args.seed, 1.0)
            info.update(evaluate(spark, model, args.candidates))
            report["runs"].append(info)
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)

            try:
                user_factors = model.userFactors.select("id", "features").toPandas()
                item_factors = model.itemFactors.select("id", "features").toPandas()
                users = np.stack([np.asarray(x, dtype=np.float32) for x in user_factors["features"]])
                items = np.stack([np.asarray(x, dtype=np.float32) for x in item_factors["features"]])
                np.savez(args.factors, user_ids=user_factors["id"].to_numpy(np.int64),
                         item_ids=item_factors["id"].to_numpy(np.int64), user=users, item=items)
                log(f"因子已保存：{args.factors}（users {users.shape}, items {items.shape}）")
                report["factors"] = {"users": list(users.shape), "items": list(items.shape)}
            except Exception as exc:  # noqa: BLE001
                log(f"因子保存失败（不影响训练结论）：{type(exc).__name__}: {exc}")
                report["factors"] = f"failed: {type(exc).__name__}"
    finally:
        spark.stop()

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"报告写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())