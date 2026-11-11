# -*- coding: utf-8 -*-
"""生成 PPT / 报告用的图表，并把图中用到的数字写进 reports/fig_stats.json。

用法：python scripts/make_figs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#C9D8DE"
plt.rcParams["text.color"] = "#12333B"
plt.rcParams["axes.labelcolor"] = "#12333B"
plt.rcParams["xtick.color"] = "#3D5A62"
plt.rcParams["ytick.color"] = "#3D5A62"

INK = "#12333B"
TEAL = "#0E7C86"
TEAL_L = "#7FC0C6"
ORANGE = "#E08A3C"
RED = "#B5453A"
GRAY = "#9FB4BC"

FIGS = C.ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
STATS: dict = {}

# 误差分析用哪个模型的验证集预测：优先用与老师切法对齐的尾部切分（T1）
VAL_TAG = "T1" if (C.MODELS / "valid_pred_T1.npz").exists() else "f_d64l2e2"


SHOW_TITLE = os.environ.get("FIG_TITLE", "0") == "1"


def ttl(ax, s: str, **kw):
    """图内标题开关：PPT 母版自带顶条标题，默认关掉以免重复。设 FIG_TITLE=1 可打开。"""
    if SHOW_TITLE:
        ax.set_title(s, **kw)


def save(fig, name: str) -> None:
    p = FIGS / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  图 {p.relative_to(C.ROOT)}")


# ---------------------------------------------------------------- 1 基线阶梯
def fig_ladder() -> None:
    ev = json.loads((C.REPORTS / "final_eval.json").read_text(encoding="utf-8"))
    lad = ev["ladder"]
    order = ["全局均值", "用户均值", "电影均值", "偏置 ALS",
             "Bias-MF（单 seed 最优）", "三 seed 等权融合"]
    labels = [k for k in order if k in lad]
    vals = [lad[k] for k in labels]
    colors = [GRAY] * 3 + [TEAL_L, TEAL, ORANGE]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.006, yy, f"{v:.4f}", va="center", ha="left",
                fontsize=12, color=INK, fontweight="bold")
    for i in range(1, len(vals)):
        drop = vals[i - 1] - vals[i]
        ax.annotate(f"−{drop:.3f}", xy=(vals[i] * 0.52, y[i] + 0.37),
                    fontsize=9.5, color="#5C7A82", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("验证集 RMSE（越低越好）", fontsize=11)
    ax.grid(axis="x", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ttl(ax, f"模型能力阶梯：从 {vals[0]:.3f} 到 {vals[-1]:.3f}", fontsize=15,
                 fontweight="bold", color=INK, pad=12, loc="left")
    save(fig, "01_基线阶梯.png")
    STATS["ladder"] = dict(zip(labels, vals))


# ------------------------------------------------- 2 误差来源：按电影热度分桶
def fig_error_source(cnt_m, mid_v, err2) -> None:
    edges = [1, 10, 50, 200, 1000, 10 ** 9]
    names = ["1–9 条", "10–49", "50–199", "200–999", "≥1000 条"]
    n_rows, share_err, rmse_b = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (cnt_m[mid_v] >= a) & (cnt_m[mid_v] < b)
        n_rows.append(int(m.sum()))
        share_err.append(float(err2[m].sum() / err2.sum() * 100))
        rmse_b.append(float(np.sqrt(err2[m].mean())) if m.sum() else 0.0)
    share_row = np.array(n_rows) / len(mid_v) * 100

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, share_row, w, label="样本占比", color=TEAL_L, zorder=3)
    ax.bar(x + w / 2, share_err, w, label="平方误差占比", color=ORANGE, zorder=3)
    for xi, a, b, r in zip(x, share_row, share_err, rmse_b):
        ax.text(xi - w / 2, a + 1.2, f"{a:.1f}%", ha="center", fontsize=9.5, color=INK)
        ax.text(xi + w / 2, b + 1.2, f"{b:.1f}%", ha="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(xi, -4.6, f"RMSE {r:.2f}", ha="center", fontsize=8.8, color="#6B858C")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_xlabel("电影被评分次数（训练集统计）", fontsize=11)
    ax.set_ylabel("百分比（%）", fontsize=11)
    ax.set_ylim(-7, 68)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ttl(ax, "误差集中在哪里：热门片主导损失", fontsize=15, fontweight="bold",
                 color=INK, pad=12, loc="left")
    save(fig, "02_误差来源.png")
    STATS["error_source"] = {"buckets": names, "n_rows": n_rows,
                             "share_row_pct": share_row.tolist(),
                             "share_err_pct": share_err, "rmse": rmse_b}


# ------------------------------------- 3 门控消融：短历史群体上的 κ 对比
def fig_gate_ablation() -> None:
    """稀疏历史模拟集上，三档门控强度的 RMSE。"""
    data = [("κ = 0\n（不门控）", 0.92358, TEAL),
            ("κ = 10\n（原设计）", 0.92801, TEAL_L),
            ("κ = 100\n（强门控）", 0.93949, GRAY)]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.5, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.0012, f"{v:.4f}", ha="center", fontsize=12,
                color=INK, fontweight="bold")
    ax.axhline(0.0, color="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.900, 0.950)
    ax.set_ylabel("稀疏群验证集 RMSE", fontsize=11)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.02, 0.95, "对照：同一模型在长历史群体上为 0.839\n两者是不同验证集，仅作量级参考",
            transform=ax.transAxes, fontsize=9.5, color="#6B858C", va="top")
    ttl(ax, "门控的消融：压小因子反而更差", fontsize=15, fontweight="bold",
                 color=INK, pad=12, loc="left")
    save(fig, "03_门控消融.png")
    STATS["gate_ablation"] = {d[0].replace("\n", " "): d[1] for d in data}


# ------------------------------------------------------------- 4 残差分布
def fig_residual(resid) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.hist(np.clip(resid, -4, 4), bins=160, color=TEAL_L, edgecolor="none", zorder=3)
    ax.axvline(0, color=RED, lw=1.6, ls="--", zorder=4)
    mu, sd = float(resid.mean()), float(resid.std())
    ax.text(0.98, 0.92, f"均值 {mu:+.3f}   标准差 {sd:.3f}", transform=ax.transAxes,
            ha="right", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F2F8FA", ec="#CFE0E5"))
    ax.set_xlabel("残差 = 真实评分 − 预测评分", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ttl(ax, "残差近似对称，无明显系统偏移", fontsize=15, fontweight="bold",
                 color=INK, pad=12, loc="left")
    save(fig, "04_残差分布.png")
    STATS["residual"] = {"mean": mu, "std": sd}


# ------------------------------------------------------- 5 训练曲线（读日志）
def fig_curve() -> None:
    log = C.REPORTS / "curve.log"
    if not log.exists():
        print("  跳过训练曲线（缺 reports/curve.log）")
        return
    ep, tr, va = [], [], []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "epoch" not in line or "train_MSE" not in line:
            continue
        parts = line.replace("(", " ").replace(")", " ").replace("|", " ").split()
        try:
            e = int(parts[parts.index("epoch") + 1].split("/")[0])
            trm = float(parts[parts.index("train_MSE") + 1])
        except (ValueError, IndexError):
            continue
        va_v = np.nan
        if "valid_RMSE" in parts:
            k = parts.index("valid_RMSE")
            try:
                va_v = float(parts[k + 1].rstrip("*"))
            except (ValueError, IndexError):
                va_v = np.nan
        ep.append(e)
        tr.append(np.sqrt(trm))
        va.append(va_v)
    if not ep or all(np.isnan(va)):
        print("  跳过训练曲线（日志里没有 valid_RMSE）")
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(ep, tr, "-o", color=TEAL_L, lw=2.2, ms=5.5, label="训练集 RMSE")
    ax.plot(ep, va, "-o", color=ORANGE, lw=2.4, ms=6, label="验证集 RMSE")
    best_i = int(np.nanargmin(va))
    ax.scatter([ep[best_i]], [va[best_i]], s=140, facecolor="none",
               edgecolor=RED, lw=2, zorder=5)
    ax.annotate(f"验证最优 {va[best_i]:.4f}（第 {ep[best_i]} 轮）\n之后训练误差继续降、验证误差回升",
                xy=(ep[best_i], va[best_i]), xytext=(ep[best_i] + 1.2, va[best_i] - 0.032),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("训练轮次（epoch）", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.grid(color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="center right")
    ttl(ax, "早停的依据：验证误差第 8 轮触底，之后开始过拟合", fontsize=14.5,
                 fontweight="bold", color=INK, pad=12, loc="left")
    save(fig, "05_训练曲线.png")
    STATS["curve"] = {"epochs": ep, "train_rmse": tr, "valid_rmse": va,
                      "best_epoch": ep[best_i], "best_rmse": va[best_i]}


# ------------------------------------------- 6 测试集结构：历史长度决定误差重心
def fig_test_structure() -> None:
    tp = C.load_test_pairs()
    df = C.load_ratings()
    cnt_u = np.bincount(df.userId.to_numpy())
    nu = cnt_u[tp.userId.to_numpy()]

    edges = [5, 6, 10, 30, 100, 300, 10 ** 9]
    names = ["=5", "6–9", "10–29", "30–99", "100–299", "≥300"]
    rows, users = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (nu >= a) & (nu < b)
        rows.append(int(m.sum()))
        users.append(int(tp.userId[m].nunique()))
    share = np.array(rows) / len(tp) * 100

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(names))
    colors = [ORANGE] + [TEAL_L] * (len(names) - 1)
    bars = ax.bar(x, share, 0.62, color=colors, zorder=3)
    for xi, v, u in zip(x, share, users):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=11,
                color=INK, fontweight="bold")
        ax.text(xi, -5.2, f"{u:,} 人", ha="center", fontsize=8.5, color="#6B858C")
    bars[0].set_edgecolor(RED)
    bars[0].set_linewidth(1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_xlabel("测试行所属用户的训练历史条数 n_u（分桶）", fontsize=11)
    ax.set_ylabel("占 10 万测试行的比例（%）", fontsize=11)
    ax.set_ylim(-8, 74)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ttl(ax, "测试集的真实形态：三分之二的行来自「只有 5 条历史」的用户",
                 fontsize=14.5, fontweight="bold", color=INK, pad=12, loc="left")
    ax.annotate("1,043 位用户\n人均 62 条待预测", xy=(0, share[0] * 0.55),
                xytext=(1.15, 46), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    save(fig, "06_测试集结构.png")
    STATS["test_structure"] = {"buckets": names, "rows": rows,
                               "share_pct": share.tolist(), "users": users}


# --------------------------------------------------- 7 评分档位分布（极度集中）
def fig_rating_dist() -> None:
    df = C.load_ratings()
    vals = np.arange(0.5, 5.01, 0.5)
    h = np.histogram(df.rating.to_numpy(), bins=np.append(vals - 0.25, 5.25))[0]
    share = h / h.sum() * 100
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    colors = [TEAL if s >= 10 else TEAL_L for s in share]
    ax.bar(vals, share, width=0.42, color=colors, zorder=3)
    for v, s in zip(vals, share):
        ax.text(v, s + 0.5, f"{s:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=10.5)
    ax.set_xlabel("评分档位（只有 10 个离散值）", fontsize=11)
    ax.set_ylabel("占比（%）", fontsize=11)
    ax.set_ylim(0, 30)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ttl(ax, "评分分布：3.0–4.0 三档占 58.5%，均值 3.5404", fontsize=14,
                 fontweight="bold", color=INK, pad=12, loc="left")
    save(fig, "07_评分分布.png")
    STATS["rating_dist"] = {"values": vals.tolist(), "share_pct": share.tolist()}


# --------------------------------------------------------- 8 模型结构示意图
def fig_model() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    terms = [
        ("μ   全局均值", "3.5404 —— 全表平均口味", 5.35, "#E8F4F6"),
        ("b_u   用户偏置", "这个人比平均挑剔多少", 4.55, "#E8F4F6"),
        ("b_i   电影偏置", "这部片比平均好多少", 3.75, "#E8F4F6"),
        ("p_u · q_i   因子项", "64 维隐向量：口味与风格的匹配度", 2.95, "#FCEFE2"),
    ]
    for label, desc, y, fc in terms:
        ax.add_patch(plt.Rectangle((0.25, y - 0.33), 3.75, 0.72, facecolor=fc,
                                   edgecolor="#9CBEC5", lw=1.2, zorder=3))
        ax.text(0.45, y + 0.16, label, fontsize=12.5, color=INK, fontweight="bold", va="center")
        ax.text(0.45, y - 0.15, desc, fontsize=9.6, color="#53737B", va="center")
        ax.annotate("", xy=(5.05, 4.15), xytext=(4.02, y),
                    arrowprops=dict(arrowstyle="-|>", color="#7FA6AE", lw=1.4,
                                    connectionstyle="arc3,rad=0.05"), zorder=2)

    ax.add_patch(plt.Circle((5.45, 4.15), 0.42, facecolor=TEAL, edgecolor="none", zorder=4))
    ax.text(5.45, 4.15, "Σ", fontsize=17, color="white", ha="center", va="center", zorder=5)
    ax.annotate("", xy=(7.05, 4.15), xytext=(5.9, 4.15),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.8), zorder=4)
    ax.text(7.2, 4.45, "预测评分", fontsize=16, color=INK, fontweight="bold")
    ax.text(7.2, 3.88, "裁剪到 [0.5, 5.0]", fontsize=10, color="#53737B")
    ax.text(0.25, 6.05, "预测：四项相加", fontsize=12.5, color="#53737B", fontweight="bold")

    # 训练流程（下方，与上面的预测结构留出空隙）
    ax.plot([0.25, 9.75], [2.18, 2.18], color="#D9E7EB", lw=1.0)
    ax.text(0.25, 1.72, "训练：偏置先解、因子后学", fontsize=12, color="#53737B",
            fontweight="bold")
    steps = ["① ALS 预拟合\nμ, b_u, b_i", "② 作为初值\n交给网络", "③ SparseAdam\n只更新批内参数",
             "④ 早停在第 6 轮\n输出预测"]
    for i, s in enumerate(steps):
        x = 0.25 + i * 2.45
        ax.add_patch(plt.Rectangle((x, 0.62), 2.12, 0.78, facecolor="#F4FAFB",
                                   edgecolor="#B7D2D8", lw=1.0, zorder=3))
        ax.text(x + 1.06, 1.01, s, fontsize=9.8, color=INK, ha="center", va="center")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.38, 1.01), xytext=(x + 2.16, 1.01),
                        arrowprops=dict(arrowstyle="-|>", color="#9FB4BC", lw=1.2))
    save(fig, "08_模型结构.png")


def main() -> None:
    C.log("载入评分表…")
    df = C.load_ratings()
    uid = df["userId"].to_numpy(np.int32)
    mid = df["movieId"].to_numpy(np.int32)
    cnt_u = np.bincount(uid).astype(np.int64)
    cnt_m = np.bincount(mid).astype(np.int64)
    STATS["counts"] = {
        "n_rows": int(len(df)),
        "user_median": int(np.median(cnt_u[cnt_u > 0])),
        "user_p90": int(np.percentile(cnt_u[cnt_u > 0], 90)),
        "item_median": int(np.median(cnt_m[cnt_m > 0])),
        "item_p90": int(np.percentile(cnt_m[cnt_m > 0], 90)),
        "top1pct_item_rating_share": float(
            np.sort(cnt_m)[::-1][: int((cnt_m > 0).sum() * 0.01)].sum() / cnt_m.sum() * 100),
    }
    # 误差分析里的「电影热度 / 用户历史长度」必须只按训练行统计，否则会把留出行算进去
    trm, _ = C.tail_split(df)
    # minlength 保证长度覆盖全表 id 上界（有些电影只出现在留出集里）
    cnt_u_tr = np.bincount(uid[trm], minlength=cnt_u.size).astype(np.int64)
    cnt_m_tr = np.bincount(mid[trm], minlength=cnt_m.size).astype(np.int64)
    del df, uid, mid

    z = np.load(C.MODELS / f"valid_pred_{VAL_TAG}.npz")
    y, pred, half, uid_v, mid_v = z["y"], z["pred"], z["half"], z["uid"], z["mid"]

    # 误差分析统一用「融合模型」，与报告中的表保持一致
    ev = json.loads((C.REPORTS / "final_eval.json").read_text(encoding="utf-8"))
    wts = ev["models"]["ensemble"].get("weights")
    if wts and all((C.MODELS / f"valid_pred_{t}.npz").exists() for t in wts):
        blend = np.zeros(len(y))
        for t, w in wts.items():
            zt = np.load(C.MODELS / f"valid_pred_{t}.npz")
            blend += w * zt["pred"].astype(np.float64)
        pred = np.clip(blend, 0.5, 5.0)
        C.log(f"误差分析使用融合模型（权重 {wts}），RMSE {C.rmse(pred, y):.5f}")
    else:
        C.log(f"误差分析使用单模型 {VAL_TAG}，RMSE {C.rmse(pred, y):.5f}")
    err2 = (y - pred) ** 2

    C.log("画图…")
    fig_ladder()
    fig_error_source(cnt_m_tr, mid_v, err2)
    fig_gate_ablation()
    fig_residual(y - pred)
    fig_test_structure()
    fig_rating_dist()
    fig_model()
    fig_curve()

    (C.REPORTS / "fig_stats.json").write_text(
        json.dumps(STATS, ensure_ascii=False, indent=2), encoding="utf-8")
    C.log(f"统计量写入 reports/fig_stats.json")


if __name__ == "__main__":
    main()
