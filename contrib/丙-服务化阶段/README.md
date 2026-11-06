# 丙 · 服务化阶段交付包

> 本目录 **只新增文件，不修改仓库里任何已有内容**。
> 目的是让下一位负责人拿到完整上下文，能直接接着做。

## 目录说明

```
contrib/丙-服务化阶段/
├── README.md          ← 本文件
└── patched/           ← 改过的 4 个核心文件（未被覆盖到仓库原位置）
    ├── src/candidate_set.py
    ├── scripts/train_ranker.py
    ├── scripts/spark_als.py
    └── app/models.py
```

## 为什么有 patched/

仓库里原有的这 4 个文件被本阶段改过（同分规则、模型落盘、ALS 因子、接口加载模型），
但为了避免改动已有文件，改动后的版本放在这里，**需要时手动覆盖**：

```bash
cp contrib/丙-服务化阶段/patched/src/candidate_set.py   src/candidate_set.py
cp contrib/丙-服务化阶段/patched/scripts/train_ranker.py scripts/train_ranker.py
cp contrib/丙-服务化阶段/patched/scripts/spark_als.py    scripts/spark_als.py
cp contrib/丙-服务化阶段/patched/app/models.py           app/models.py
```

不覆盖也能跑，但复现不出 0.8604（会得到旧口径的 0.8894）。

## 四处改动的具体内容

| 文件 | 改动 |
|---|---|
| `src/candidate_set.py` | `topk(scores, k, tiebreak=None)`：同分时按 `tiebreak` 降序，而不是按候选行序 |
| `scripts/train_ranker.py` | ① 训练后保存 `models/<name>.txt` + `.meta.json`；② 传入 `test.npz["bpr"]` 做同分排序；③ metrics 记录 `tiebreak` |
| `scripts/spark_als.py` | ① `df.cache()` 后计时，fit 不含 CSV 解析；② 保存 `models/als_factors.npz`；③ 输出 `results/scalability.csv` |
| `app/models.py` | ① 优先加载 `models/ranker_full.txt`（与离线同版权重）；② 排序改为 `np.lexsort((-bpr, -score))` |

## 数字变化（如果统一文档，按这个替换）

| 位置 | 旧值 | 新值 |
|---|---|---|
| README 第 4 行 | 0.8894 / +20.8% | 0.8604 / +16.8% |
| README 结果表末行 | 0.8894 / 0.8301 / 0.9338 / +20.8% | 0.8604 / 0.8237 / 0.8825 / +16.8% |
| README ALS 行 | 0.7896 | 0.7710 |
| README 消融表 | 0.8894 / −5.4% / 0.8727 / −1.9% | 0.8604 / −2.2% / 0.8524 / −0.9% |
| `docs/项目报告.md` | 同上四处 | 同上 |
| `docs/服务化接口说明.md` | 0.8894 | 0.8604 |

## 交付物清单（本次新增）

- `docs/接口契约.md`（v1.1）
- `docs/核对报告.md`（10 章）
- `docs/交接说明-丙.md`
- `docs/个人贡献说明-丙.md`
- `docs/答辩讲解稿-丙.md`、`docs/答辩问答-丙.md`、`docs/演示材料-丙.pptx`
- `scripts/`：`predict.py`、`audit_leakage.py`、`check_consistency.py`、`check_tiebreak.py`、
  `export_top10_lsh.py`、`export_top10_als.py`、`build_ablation.py`
- `results/`：`ablation.csv`、`scalability.csv`、`leakage_audit.json`、`metrics_lsh.json`、
  `metrics_als.json`、`similarity_top10_lsh.json`、`similarity_top10_als.json`

大文件（`top10_lsh.csv` / `top10_als.csv` / `als_factors.npz`）未上传，用上面的脚本重新生成。
