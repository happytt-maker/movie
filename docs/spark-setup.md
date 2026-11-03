# Spark 环境安装与 ALS 运行

本文记录 Windows 本地的 Spark 运行配置，以及如何用项目自己的输入数据运行 ALS。Spark 脚本通过 spark-submit 启动；在 Windows 上，Java 路径与项目路径应使用 ASCII 字符，避免 JVM 报 Bad pathname。

## 验证环境

| 项目 | 版本或路径 |
|---|---|
| Windows | Windows 11 x64 |
| Python | 3.12.4，项目虚拟环境 .venv |
| PySpark / Spark | 3.5.3 |
| Java | Temurin Runtime 17.0.20.1 |
| Spark master | local[4] |

requirements.txt 固定了本机验证过的 Python 包版本。Spark 3.5.3 支持 Java 17；正式使用时建议安装 JDK 17。运行本项目 spark-submit 只需要可用的 Java 17 runtime。

## 安装 Python 依赖

从仓库根目录执行：

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

仓库路径和数据路径尽量不要包含中文或空格。若当前路径不满足该要求，可把仓库放到 C:\movie 一类的纯 ASCII 路径。

## 设置 Java 与 Spark 环境变量

先安装 Java 17，并把 JAVA_HOME 改成 Java 安装目录。以下是本机便携式 Java runtime 的路径示例：

    $env:JAVA_HOME = "D:\env\jdk-17.0.20.1+1-jre"
    $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
    $env:SPARK_HOME = (Resolve-Path ".venv\Lib\site-packages\pyspark").Path
    $env:PYSPARK_PYTHON = (Resolve-Path ".venv\Scripts\python.exe").Path
    $env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON

这些变量只对当前 PowerShell 窗口生效。检查版本：

    & "$env:JAVA_HOME\bin\java.exe" -version
    .\.venv\Scripts\python.exe -c "import pyspark; print(pyspark.__version__)"
    .\.venv\Scripts\spark-submit.cmd --version

本机检查结果为 Java 17.0.20.1 和 Spark 3.5.3。通过 pip 安装 PySpark 后，Windows 下还需设置 SPARK_HOME 指向虚拟环境中的 pyspark 包目录；否则 spark-submit.cmd 可能报找不到 Spark jars。

## 准备 MovieLens 输入

从 [GroupLens MovieLens 32M 官方页面](https://grouplens.org/datasets/movielens/32m/) 下载数据并解压到 data/ml-32m/。至少需要 ratings.csv 和 movies.csv。

在仓库根目录、已安装依赖的虚拟环境中运行：

    .\.venv\Scripts\python.exe scripts\prepare_data.py
    .\.venv\Scripts\python.exe scripts\export_for_spark.py --sample-users 20000

第二条命令生成 ALS 所需的训练正反馈和抽样候选集：

    data/spark/train_positive.csv
    data/spark/candidates_sample.csv

## 运行 ALS

在同一 PowerShell 窗口中执行：

    .\.venv\Scripts\spark-submit.cmd --master "local[4]" --driver-memory 6g scripts\spark_als.py --train data\spark\train_positive.csv --candidates data\spark\candidates_sample.csv --factors models\als_factors.npz --report results\als_report.json

成功时终端会打印 Spark 版本、训练耗时、候选打分耗时和 NDCG@10。结果报告写入 results/als_report.json，矩阵因子写入 models/als_factors.npz。

若需要扩展性曲线，可运行：

    .\.venv\Scripts\spark-submit.cmd --master "local[4]" --driver-memory 6g scripts\spark_als.py --scales 0.25,0.5,1.0 --iters 3

## 常见问题

- Bad pathname：检查仓库和数据路径，移到不含中文、空格的 ASCII 路径。
- 找不到 Spark jars：在当前终端重新设置 SPARK_HOME，确认其目录下有 jars 文件夹。
- 找不到 spark-submit：使用虚拟环境脚本的完整路径 .venv\Scripts\spark-submit.cmd。
- Spark 启动后挂起：不要在 Python 脚本里直接创建 SparkSession；按本文通过 spark-submit 启动。
- Java 版本不符：检查 JAVA_HOME 是否指向 Java 17，而不是其他已安装版本。
- 内存不足：降低 --driver-memory 和本地并行度，例如使用 local[2]。

## 本机执行记录

本机确认 `spark-submit --version` 可识别 Spark 3.5.3 和 Java 17.0.20.1，并用临时合成小样本完整运行了 `scripts/spark_als.py`：12 条训练记录、8 条候选记录、rank=2、迭代 1 次；ALS 训练约 2.5 秒，候选评估得到 NDCG@10=0.81546，成功导出用户/电影因子。该结果只用于确认 Java → PySpark → ALS → 因子导出的链路，不代表 MovieLens 指标。记录见 `results/als_smoke_report.json`。

当前仓库没有 `data/ml-32m/` 原始数据。本机尝试从 GroupLens 下载时速度过低，故未运行或改写已有的 `results/als_report.json`（该文件保留原有 20k 用户抽样实验记录）。获取数据并生成 Spark 输入后，再按上文命令复现完整数据实验。

Windows 运行时可能提示未配置 `winutils.exe`、改用 Hadoop 内置 Java 实现；本机 local 模式 ALS 已正常完成。若出现退出码 9009，通常是 Spark 子进程找不到 Python，按上文同时设置 `PYSPARK_PYTHON` 与 `PYSPARK_DRIVER_PYTHON` 指向项目虚拟环境。
