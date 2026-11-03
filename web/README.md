# web/ —— 推荐系统前端（成员乙）

服务化阶段的前端与可视化交付：把 `results/` 里的模型产物变成**能演示、能答疑**的页面。

两个页面，一条命令启动：

```bash
cd web
npm install          # 第一次；国内慢可加 --registry=https://registry.npmmirror.com
npm run dev          # 打开 http://127.0.0.1:5173
```

> Node 版本 ≥ 18（本项目在 Node 22 + npm 10 上验证）。

---

## 1. 两个页面

| 路由 | 页面 | 对应分工要求 |
|---|---|---|
| `#/` | 推荐展示 `Recommend.vue` | 选用户、看 Top10、每张卡片带片名/类型/分数/类型亲和度/推荐理由 |
| `#/dashboard` | 指标看板 `Dashboard.vue` | 六模型 NDCG 对比、消融视图、特征重要性、各任务指标、模型明细表 |

推荐展示页的要点：

- **选用户**：左栏支持按 `userId` 搜索，也可以「随机抽取」；用户来自导出快照（对应 `data/processed/vocab.npz` 的抽样）。
- **推荐卡片**：片名与年份、类型标签、BPR 召回分（带条形长度）、类型亲和度、候选集排名、一句话推荐理由。
- **性能信息**：底部条显示本次延迟、p50、p95、吞吐与最近 N 次延迟的迷你曲线。静态模式下会明确标注「只含本地快照读取」，接上接口后自动变成真实推理延迟。

指标看板页的要点：

- 顶部四个 KPI：最终 NDCG@10、相对热门提升、Recall@10、MRR@10。
- 主图：六模型柱状（NDCG@10）+ 折线（相对热门提升）+ 任务书 10% 目标虚线。
- 消融：完整模型 / 去 BPR / 去类型亲和度，标注相对跌幅。
- 特征重要性：LightGBM 分裂增益，对数轴（BPR 分数比其余特征高两个数量级，线性轴会压平）。
- 任务指标卡：流式吞吐、Spark ALS、LSH、二部图社区。
- 明细表：六个模型的 NDCG / Recall / MRR / 相对提升 / 评估用户数，抽样模型带「抽样」标记。

---

## 2. 数据从哪来

页面默认读 `public/data/` 下的四个 JSON，全部由 `scripts/export_web_data.py` 从仓库真实产物导出，不手写：

| 文件 | 内容 | 来源 |
|---|---|---|
| `meta.json` | 生成时间、Top10 来源、用户数 | `results/*.csv` |
| `metrics.json` | 六模型指标、消融、特征重要性、各任务指标 | `results/metrics_*.json`、`results/*_report.json` |
| `movies.json` | 被引用到的电影片名与类型 | `data/ml-32m/movies.csv` |
| `recommend_static.json` | 按用户分组的 Top10（含 BPR 分与类型亲和度） | `results/sample_top10.csv`，可选 `data/features/test.npz` |

重新导出（在**仓库根目录**执行）：

```bash
python web/scripts/export_web_data.py
# 想要更多用户：python web/scripts/export_web_data.py --max-users 300
# 用全量预测：  python web/scripts/export_web_data.py --pred results/top10_ranker_full.csv --max-users 200
```

脚本行为：

- 找到 `data/features/{split}.npz` 就**自动带上** `bpr` 与 `genre_affinity`（每行特征），卡片上的「类型亲和度」从「待接入」变成真实数值；
- 找不到就退化为 null，页面显示「待接入」，不会报错；
- 找不到 `data/ml-32m/movies.csv` 时，片名显示为 `MovieLens #<id>`，其余功能不受影响。

---

## 3. 切换到甲的接口

甲的 FastAPI 一起来，只改环境变量，页面代码一行不动：

```bash
cp .env.example .env.local
```

`.env.local`：

```ini
VITE_DATA_SOURCE=api
VITE_PROXY_TARGET=http://127.0.0.1:8001
```

然后重启 `npm run dev`。请求走 Vite 的 `/api` 反向代理到甲的服务，**不需要甲改 CORS**。

联调检查顺序：

1. 打开页面右上角的数据源徽标，应为「实时接口」；
2. 浏览器控制台 `fetch('/api/health')` 应返回 `{status, modelVersion, ...}`；
3. 换用户时底部的「本次延迟 / p50 / p95」应变成毫秒级真实值。

如果接口没起来，页面会自己退回静态快照（徽标显示「接口离线」），演示不会开天窗。

### 字段兼容

`src/api/client.js` 里对甲的返回结构做了归一化，下面几种写法前端都能吃：

```json
[{"rank":1,"movieId":307,"score":0.097}]
{"items":[{"rank":1,"movieId":307,"score":0.097}]}
{"recommendations":[{"rank":1,"movieId":307,"bpr_score":0.097}]}
```

- `movieId / movie_id / itemId / mid`、`score / bpr / bpr_score`、`rank / position` 都能识别；
- 只回 `movieId` 时，前端用 `movies.json` 本地补齐片名与类型；
- 返回里带 `genreAffinity`（或 `genre_affinity`）时，卡片直接显示真实值。

接口契约建议（与分工文档一致）：

| 接口 | 方法 | 前端用法 |
|---|---|---|
| `/health` | GET | 右上角数据源徽标、模型版本 |
| `/predict` | POST `{userId, topK}` | 推荐展示页的每次换用户 |
| `/metrics` | GET（可选） | 看板优先读它；没有就用静态 `metrics.json` |
| `/predict/batch`、`/evaluate` | POST | 前端不用，留给丙的核对脚本 |

---

## 4. 目录结构

```
web/
├── index.html
├── vite.config.js          # /api 反向代理（target 走 VITE_PROXY_TARGET）
├── .env.example
├── public/data/            # 导出的静态数据（见第 2 节）
├── scripts/
│   └── export_web_data.py  # 从 results/ 与 data/ 导出前端数据
└── src/
    ├── api/client.js       # 双数据源客户端（static / api）
    ├── store/app.js        # 全局状态、性能采样、p50/p95
    ├── composables/useChart.js
    ├── components/         # AppHeader / UserPicker / MovieCard / KpiCard / PerfStrip
    ├── views/
    │   ├── Recommend.vue
    │   └── Dashboard.vue
    ├── utils/labels.js     # 类型中英对照、数字格式、推荐理由生成
    └── styles/main.css     # 设计令牌与通用样式
```

---

## 5. 常见问题

**Q：`npm install` 卡住或报网络错误？**
换镜像：`npm install --registry=https://registry.npmmirror.com`。仍然失败就删掉 `node_modules` 与 `package-lock.json` 重来。

**Q：端口 5173 被占用？**
`npm run dev -- --port 5174`。

**Q：想在别人的电脑上只演示、不装 Node？**
`npm run build` 生成 `dist/`，然后把 `dist/` 拷过去用任意静态服务器托管：

```bash
npm run build
python -m http.server 8080 -d dist    # 打开 http://127.0.0.1:8080
```

**Q：看板上的数字和 README 对不上？**
看板读的是 `results/metrics_*.json`（脚本实测产物）。若与 README 表格有千分位差异，属于 LightGBM/PyTorch 版本波动，以 `results/` 为准，并在核对报告里说明。

**Q：为什么用 hash 路由（`#/dashboard`）？**
答辩现场可能用最简单的静态服务器托管 `dist/`，hash 路由刷新不会 404。

---

## 6. 与验收标准的对照

| 分工文档要求 | 本工程对应 |
|---|---|
| `web/` 可 `npm run dev` 启动 | ✅ Vite 5 + Vue 3，已验证 `dev` 与 `build` |
| `web/src/views/Recommend.vue` 选定用户后正确显示 Top10 | ✅ 10 张卡片，数据来自真实 Top10 CSV |
| `web/src/views/Dashboard.vue` 六模型对比、数据来自接口 | ✅ 六模型柱状 + 接口/静态双源 |
| 用户选择：搜索或随机抽取 | ✅ 左栏搜索 + 随机抽取 |
| 卡片含片名、类型、BPR 分数、类型亲和度、推荐理由 | ✅ 五项齐备（亲和度需导出脚本读到特征文件） |
| 指标看板：热门 0.7363 → 最终 0.8894 | ✅ 柱状图 + 10% 目标线 + 相对提升折线 |
| 消融视图：0.8417 / 0.8727 | ✅ 消融柱状 + 相对跌幅标注 |
| 性能信息：延迟与吞吐 | ✅ 本次/p50/p95/吞吐 + 最近 N 次曲线 |
| `docs/演示录屏.mp4` | ⬜ 需按 `docs/演示录屏脚本.md` 自行录制（3 分钟分镜已写好） |
