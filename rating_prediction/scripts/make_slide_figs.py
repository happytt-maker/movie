# -*- coding: utf-8 -*-
"""为答辩 PPT 单独绘制两张图，字号按幻灯片栏宽放大。

为什么单独出一版：make_figs.py 里的图宽 7.6-8.2 英寸，放到 PPT 右栏
（约 5.3 英寸）时等效字号只有 7pt 左右，投影上读不清。这里把画布收窄到
6-6.2 英寸、字号保持 10-11pt，放进栏位后等效字号约 9pt 以上。

顺带修掉报告版图 3 的一个画法问题：那版是三根从 0.900 起画的柱子，
轴被截断后柱长完全由轴下限决定，视觉上放大了差异。幻灯片版改成
点图（只画点，不画长度），差异读数交给标注。

产出：assets/s05_curve.png、assets/s03_gate.png
用法：python scripts/make_slide_figs.py
"""

from __future__ import annotations

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

INK = "#12333B"
TEAL = "#0E7C86"
TEAL_L = "#7FC0C6"
ORANGE = "#E08A3C"
RED = "#B5453A"
GRAY = "#9FB4BC"

_ASSETS = C.ROOT / "docs" / "答辩PPT" / "assets"
OUT_CURVE = _ASSETS / "s05_curve.png"
OUT_GATE = _ASSETS / "s03_gate.png"
OUT_DIST = _ASSETS / "s07_dist.png"


def parse_log(path: Path):
    ep, tr, va = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "epoch" not in line or "train_MSE" not in line:
            continue
        parts = line.replace("(", " ").replace(")", " ").replace("|", " ").split()
        try:
            e = int(parts[parts.index("epoch") + 1].split("/")[0])
            trm = float(parts[parts.index("train_MSE") + 1])
        except (ValueError, IndexError):
            continue
        v = np.nan
        if "valid_RMSE" in parts:
            k = parts.index("valid_RMSE")
            try:
                v = float(parts[k + 1].rstrip("*"))
            except (ValueError, IndexError):
                v = np.nan
        ep.append(e)
        tr.append(np.sqrt(trm))
        va.append(v)
    return ep, tr, va


def fig_curve_slide() -> None:
    log = C.REPORTS / "curve.log"
    if not log.exists():
        raise SystemExit(f"缺日志：{log}")
    ep, tr, va = parse_log(log)
    if not ep or all(np.isnan(va)):
        raise SystemExit("日志里没有 valid_RMSE，无法成图")

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(ep, tr, "-o", color=TEAL_L, lw=2.0, ms=4.5, label="训练集 RMSE")
    ax.plot(ep, va, "-o", color=ORANGE, lw=2.4, ms=5.6, label="验证集 RMSE")

    best = int(np.nanargmin(va))
    ax.scatter([ep[best]], [va[best]], s=150, facecolor="none",
               edgecolor=RED, lw=2.0, zorder=5)
    ax.annotate(f"第 {ep[best]} 轮触底 {va[best]:.4f}",
                xy=(ep[best], va[best]),
                xytext=(ep[best] + 1.6, va[best] - 0.045),
                fontsize=10.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax.set_xlabel("训练轮次（epoch）", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout(pad=0.9)

    fig.savefig(OUT_CURVE, dpi=200)
    plt.close(fig)
    print(f"已出图 {OUT_CURVE.name}  最优 {va[best]:.5f} @ 第 {ep[best]} 轮")


def fig_gate_slide() -> None:
    """门控消融：点图而非柱图。轴被截断时，点图不会把差异画成夸张的长度比。"""
    data = [("κ = 0（不门控）", 0.92358, TEAL),
            ("κ = 10（原设计）", 0.92801, TEAL_L),
            ("κ = 100（强门控）", 0.93949, GRAY)]
    labels = [f"κ = {lbl.split('κ = ')[1]}" for lbl, _, _ in data]
    vals = [v for _, v, _ in data]
    colors = [c for _, _, c in data]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    y = np.arange(len(vals))[::-1]

    ax.axvline(vals[0], color=TEAL, lw=1.2, ls="--", zorder=1)
    for yi, v, c in zip(y, vals, colors):
        ax.scatter([v], [yi], s=260, color=c, zorder=4, edgecolor="white", lw=1.5)

    for yi, v, lbl in zip(y, vals, labels):
        delta = v - vals[0]
        tag = "最优" if abs(delta) < 1e-9 else f"Δ +{delta:.4f}"
        ax.text(v + 0.0016, yi, f"{v:.5f}   {tag}", va="center", fontsize=11.5,
                color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.918, 0.954)
    ax.set_ylim(-0.7, len(vals) - 0.3)
    ax.set_xlabel("稀疏群验证集 RMSE（越低越好）", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.7)

    fig.savefig(OUT_GATE, dpi=200)
    plt.close(fig)
    print(f"已出图 {OUT_GATE.name}  三档 {vals}")


def fig_rating_slide() -> None:
    """评分档位分布。数据直接读 reports/fig_stats.json，不再扫 3,190 万行原始表。"""
    import json
    stats = json.loads((C.REPORTS / "fig_stats.json").read_text(encoding="utf8"))
    vals = stats["rating_dist"]["values"]
    share = stats["rating_dist"]["share_pct"]

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    colors = [TEAL if v in (3.0, 3.5, 4.0) else TEAL_L for v in vals]
    bars = ax.bar([str(f"{v:g}") for v in vals], share, 0.62, color=colors, zorder=3)
    for b, sv in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, sv + 0.4, f"{sv:.1f}",
                ha="center", fontsize=10.5, color=INK, fontweight="bold")

    ax.set_ylim(0, 31)
    ax.set_xlabel("评分档位（只有 10 个离散值）", fontsize=11)
    ax.set_ylabel("占比（%）", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="#E6EFF2", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.0, 1.04, "深色三档合计 58.5%　均值 3.5404",
            transform=ax.transAxes, fontsize=10.5, color=TEAL, fontweight="bold")
    fig.tight_layout(pad=0.7)

    fig.savefig(OUT_DIST, dpi=200)
    plt.close(fig)
    print(f"已出图 {OUT_DIST.name}")

def main() -> None:
    fig_curve_slide()
    fig_gate_slide()
    fig_rating_slide()


if __name__ == "__main__":
    main()
