# 数据目录

原始数据**不纳入版本控制**（见 `.gitignore`）。

请从 GroupLens 官方下载 MovieLens 32M 到本目录：

<https://grouplens.org/datasets/movielens/32m/>

解压后结构：

```
data/
└── ml-32m/
    ├── ratings.csv
    ├── movies.csv
    ├── tags.csv
    └── links.csv
```

然后运行：

```bash
python scripts/prepare_data.py --config configs/base.yaml
```
