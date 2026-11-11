# -*- coding: utf-8 -*-
"""生成提交文件：把若干模型的测试集预测按验证集上优化的权重融合，写出 CSV。

用法：
    python scripts/make_submission.py --preds mf64_final mf128_final --out output/test_pairs_pred.csv
    python scripts/make_submission.py --preds mf64_final --weights 1.0 --out ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common as C


def load_valid(tag: str):
    z = np.load(C.MODELS / f"valid_pred_{tag}.npz")
    return z["y"], z["pred"], z["half"], z["uid"], z["mid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="参与融合的模型 tag")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="手工权重（留空则自动优化）")
    ap.add_argument("--no-valid", action="store_true",
                    help="跳过验证集权重优化，直接用等权/手工权重（用于 --final 模型，它们没有验证集预测）")
    ap.add_argument("--out", default="output/test_pairs_pred.csv")
    args = ap.parse_args()

    tags = args.preds

    if args.no_valid:
        # ---- 不做验证集优化：等权或手工权重 ----
        if args.weights and len(args.weights) == len(tags):
            w = np.array(args.weights, dtype=np.float64)
        else:
            w = np.ones(len(tags), dtype=np.float64)
        w = w / w.sum()
        names = tags
        blend, y = None, None
        C.log(f"跳过验证集优化，权重 {dict(zip(names, np.round(w, 4)))}")
    else:
        # ---- 在验证集上确定融合权重 ----
        y, _, half, _, _ = load_valid(tags[0])
        mats, names = [], []
        for t in tags:
            yy, p, hh, _, _ = load_valid(t)
            assert len(p) == len(y) and np.allclose(yy, y), "各模型验证集不一致"
            mats.append(p.astype(np.float64))
            names.append(t)
        P = np.vstack(mats)

        if args.weights and len(args.weights) == len(tags):
            w = np.array(args.weights, dtype=np.float64)
            w = w / w.sum()
        else:
            # 非负最小二乘：对截距不敏感的闭式近似，再用网格精修
            best = None
            grid = np.linspace(0, 1, 21) if len(tags) == 2 else None
            if grid is not None:
                for a in grid:
                    cand = a * P[0] + (1 - a) * P[1]
                    r = C.rmse(np.clip(cand, C.RATING_MIN, C.RATING_MAX), y)
                    if best is None or r < best[0]:
                        best = (r, [a, 1 - a])
                w = np.array(best[1])
            else:
                # 多于两个：坐标下降
                w = np.ones(len(tags)) / len(tags)
                cur = C.rmse(np.clip(w @ P, C.RATING_MIN, C.RATING_MAX), y)
                for _ in range(60):
                    improved = False
                    for k in range(len(tags)):
                        for delta in (0.2, 0.1, 0.05, -0.2, -0.1, -0.05):
                            w2 = w.copy()
                            w2[k] = max(0.0, w2[k] + delta)
                            if w2.sum() <= 0:
                                continue
                            w2 /= w2.sum()
                            r = C.rmse(np.clip(w2 @ P, C.RATING_MIN, C.RATING_MAX), y)
                            if r < cur - 1e-7:
                                cur, w, improved = r, w2, True
                    if not improved:
                        break

        blend = np.clip(w @ P, C.RATING_MIN, C.RATING_MAX)
        C.log(f"权重 {dict(zip(names, np.round(w, 4)))}")
        C.log(f"融合后验证集 RMSE {C.rmse(blend, y):.5f} | 真实 {C.rmse(blend[half==0], y[half==0]):.5f} "
              f"| 合成 {C.rmse(blend[half==1], y[half==1]):.5f}")
        for t, p in zip(names, P):
            C.log(f"  单模型 {t:16} RMSE {C.rmse(np.clip(p, C.RATING_MIN, C.RATING_MAX), y):.5f}")

    # ---- 应用同一权重到测试集 ----
    tp = C.load_test_pairs()
    test_mats = []
    for t in tags:
        f = C.MODELS / f"pred_test_{t}.npy"
        if not f.exists():
            raise FileNotFoundError(f"缺少 {f}，请先用 --final 训练该模型")
        test_mats.append(np.load(f).astype(np.float64))
        assert len(test_mats[-1]) == len(tp), "测试集预测长度与 test_pairs 不一致"
    pred = np.clip(w @ np.vstack(test_mats), C.RATING_MIN, C.RATING_MAX)

    out = C.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["userId,movieId,timestamp,pred_rating"]
    for u, m, ts, p in zip(tp.userId, tp.movieId, tp.timestamp, pred):
        lines.append(f"{int(u)},{int(m)},{int(ts)},{p:.4f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "file": str(out.relative_to(C.ROOT)), "rows": int(len(pred)),
        "models": tags, "weights": [float(x) for x in w],
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        "pred_min": float(pred.min()), "pred_max": float(pred.max()),
        "pred_hist": np.histogram(pred, bins=np.arange(0.5, 5.51, 0.5))[0].tolist(),
    }
    if blend is not None:
        meta["valid_rmse"] = C.rmse(blend, y)
        meta["valid_rmse_ml"] = C.rmse(blend[half == 0], y[half == 0])
        meta["valid_rmse_synthetic"] = C.rmse(blend[half == 1], y[half == 1])
    else:
        meta["valid_rmse"] = None      # 用 --final 模型，没有对应验证集预测
        meta["note"] = "权重来自同配置的验证集实验（见 reports/final_eval.json 与报告 §7.3）"
    (C.REPORTS / "submission.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    C.log(f"已写出 {out}  {len(pred):,} 行 | 预测均值 {pred.mean():.4f} 标准差 {pred.std():.4f}")


if __name__ == "__main__":
    main()
