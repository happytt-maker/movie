# -*- coding: utf-8 -*-
"""MovieLens 矩阵分解（PyTorch）：训练与推荐。

相对原版的修改点（逐条对应之前指出的缺陷）：
1. 偏置初始化：user_bias / movie_bias 之前沿用 nn.Embedding 默认的 N(0, 1)，而因子被显式初始化成
   std=0.05，量级不匹配。后果有两个：初始预测偏离评分均值约 1.4 分；更重要的是"训练集中零交互"
   的条目永远拿不到梯度，偏置会永久停留在 N(0, 1) 的随机值上，在推荐时噪声打分挤占 top-K。
   现改为零初始化，并用训练集均分初始化 global_bias。
2. 模型一致性：原版保存的是 valid 最优 checkpoint，但测试集评估用的是内存里最后一轮的模型，
   打印出来的测试指标和 .pt 文件不是同一个模型。现在在内存里保留 best_state，测试前回载。
3. 推荐冷启动：recommend 增加长尾/冷启动过滤（--min-item-ratings），剔除训练样本过少的条目。
4. 指标口径：train 侧同时打印 MSE 与 RMSE，避免和 valid_RMSE 混在一起看导致误判。
5. 正则化：weight_decay 只作用于因子矩阵，bias 与 global_bias 不参与衰减；优化器改用 AdamW。
6. --max-rows：改为"全量读取后随机抽样"。ratings.csv 按 userId 排序，取前 N 行只会得到最前面
   几个用户，指标与全量不可比。
7. 数据校验：增加缺失值检查。pd.factorize 对 NaN 会给出 -1 编码，直接送进 nn.Embedding 会
   IndexError，报错点离根因很远。
8. 边角处理：模型文件不存在、训练未产生有效 checkpoint、可用候选不足等情况给出明确提示/报错。
9. 可选：--plateau-patience / --early-stop-patience（默认关闭，不改变原有训练行为）。
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


# 本项目改动：仅把两处默认路径改成相对项目根目录的写法，其余逻辑与原始脚本一致。
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # legacy/big_data/ -> 项目根目录
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "ml-32m")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "movielens_mf.pt")

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


class MatrixFactorization(nn.Module):
    """带用户/物品偏置的矩阵分解：r_hat = <p_u, q_i> + b_u + b_i + mu"""

    def __init__(self, user_count, movie_count, embedding_dim, global_bias_init=0.0, sparse=False):
        super().__init__()
        # sparse=True 时 embedding 反向只产生"被 batch 用到的那几行"的稀疏梯度，
        # 配合 SparseAdam 可以把每步开销从"整张表"降到"一个 batch"。
        self.user_embedding = nn.Embedding(user_count, embedding_dim, sparse=sparse)
        self.movie_embedding = nn.Embedding(movie_count, embedding_dim, sparse=sparse)
        self.user_bias = nn.Embedding(user_count, 1)
        self.movie_bias = nn.Embedding(movie_count, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        # 关键修复：nn.Embedding 的默认初始化是 N(0, 1)，对 0.5~5 的评分尺度来说太大，
        # 而且没被训练到的行会一直保留这个随机值。偏置必须显式零初始化。
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        with torch.no_grad():
            self.global_bias.fill_(float(global_bias_init))

    def forward(self, user_ids, movie_ids):
        interaction = (self.user_embedding(user_ids) * self.movie_embedding(movie_ids)).sum(dim=1)
        return (
            interaction
            + self.user_bias(user_ids).squeeze(1)
            + self.movie_bias(movie_ids).squeeze(1)
            + self.global_bias
        )


def load_data(data_dir, max_rows=None, seed=42):
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"找不到评分文件：{ratings_path}")

    dtype = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"], dtype=dtype)
    if ratings.empty:
        raise ValueError("ratings.csv 没有可用数据")
    if ratings[["userId", "movieId", "rating"]].isna().any().any():
        raise ValueError("ratings.csv 存在缺失值：评分列的 NaN 会污染 loss，id 列的 NaN 会产生非法索引")

    rng = np.random.default_rng(seed)

    # --max-rows：在"全量读取"之后做随机抽样。注意 ratings.csv 是按 userId 排序的，
    # 如果按前 N 行截断，只会拿到最前面若干个用户，得到的指标与全量结果没有可比性。
    if max_rows is not None and 0 < max_rows < len(ratings):
        keep = np.sort(rng.choice(len(ratings), size=max_rows, replace=False))
        ratings = ratings.iloc[keep].reset_index(drop=True)
        print(f"已从全量数据中随机抽样 {len(ratings):,} 条评分")

    user_codes, user_ids = pd.factorize(ratings["userId"], sort=True)
    movie_codes, movie_ids = pd.factorize(ratings["movieId"], sort=True)
    # np.asarray 对 ndarray / Index 都成立（factorize 对 Series 返回的是 Index）
    # 索引用 int32 存：3200 万条 record 下比 int64 省一半内存（约 400MB），
    # 训练时再按 batch 转成 Embedding 需要的 int64，代价可以忽略。
    users = np.asarray(user_codes, dtype=np.int32)
    movies = np.asarray(movie_codes, dtype=np.int32)
    values = ratings["rating"].to_numpy(dtype=np.float32, copy=False)

    order = rng.permutation(np.arange(len(values), dtype=np.int32))
    test_size = max(1, int(len(values) * TEST_RATIO))
    valid_size = max(1, int(len(values) * VALID_RATIO))
    test_idx = order[:test_size]
    valid_idx = order[test_size:test_size + valid_size]
    train_idx = order[test_size + valid_size:]
    if len(train_idx) == 0:
        raise ValueError("数据太少，无法划分训练集")

    movies_info = {}
    if movies_path.exists():
        movie_table = pd.read_csv(
            movies_path, usecols=["movieId", "title"], dtype={"movieId": "int32", "title": "string"}
        )
        movies_info = dict(zip(movie_table["movieId"].tolist(), movie_table["title"].tolist()))

    user_count = len(user_ids)
    movie_count = len(movie_ids)

    return {
        "train": (users[train_idx], movies[train_idx], values[train_idx]),
        "valid": (users[valid_idx], movies[valid_idx], values[valid_idx]),
        "test": (users[test_idx], movies[test_idx], values[test_idx]),
        "user_ids": np.asarray(user_ids, dtype=np.int64),
        "movie_ids": np.asarray(movie_ids, dtype=np.int64),
        "movies_info": movies_info,
        "train_mean": float(values[train_idx].mean()),
        # 记录每个条目/用户在训练集里的交互数，供推荐阶段做长尾过滤
        "item_train_counts": np.bincount(movies[train_idx], minlength=movie_count).astype(np.int64),
        "user_train_counts": np.bincount(users[train_idx], minlength=user_count).astype(np.int64),
    }


class TensorBatches:
    """按 batch 直接从整块张量里 gather，替代 DataLoader。

    原来的 DataLoader + TensorDataset + 默认 collate 会为每个 batch 逐样本调用 2048×3 次
    __getitem__ 再把结果 stack 起来，全部发生在 Python 主线程里。实测每批要 67 ms，
    而真正的 GPU 训练步只要 5-10 ms——即 90% 以上的时间花在"取数"上，GPU 全程空转
    （任务管理器里表现为 GPU 0% + CPU 占用很低，但每轮耗时巨大）。
    这里改成一次 C++ 索引操作取一整批，实测 0.55 ms/批，快约 120 倍。

    device 不为 None 时，整块数据会一次性搬进显存，每批的 gather 完全在 GPU 上完成：
    主机侧不再参与数据供给，也就不会因为 Windows 压缩内存 / 缺页而随轮次变慢
    （实测每批 0.045 ms，且每轮耗时稳定）。
    """

    def __init__(self, part, batch_size, shuffle, seed=0, device=None):
        users, movies, ratings = part
        self.device = device
        # from_numpy 与 numpy 数组共享内存，不产生额外拷贝
        self.users = torch.from_numpy(users)
        self.movies = torch.from_numpy(movies)
        self.ratings = torch.from_numpy(ratings)
        if device is not None:
            # 直接以 int64 落到显存，这样训练循环里连类型转换都不需要
            self.users = self.users.to(device=device, dtype=torch.long)
            self.movies = self.movies.to(device=device, dtype=torch.long)
            self.ratings = self.ratings.to(device=device, dtype=torch.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(ratings)
        self._rng = np.random.default_rng(seed)
        self._base = np.arange(self.n, dtype=np.int32)
        self._device_base = torch.arange(self.n, device=device) if device is not None else None

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.device is None:
            order = self._rng.permutation(self._base) if self.shuffle else self._base
            for start in range(0, self.n, self.batch_size):
                index = torch.from_numpy(order[start:start + self.batch_size])
                yield self.users[index], self.movies[index], self.ratings[index]
        else:
            order = torch.randperm(self.n, device=self.device) if self.shuffle else self._device_base
            for start in range(0, self.n, self.batch_size):
                index = order[start:start + self.batch_size]
                yield self.users[index], self.movies[index], self.ratings[index]

    def vram_bytes(self):
        if self.device is None:
            return 0
        return (self.users.numel() + self.movies.numel()) * 8 + self.ratings.numel() * 4


def as_index(tensor, device):
    """索引张量：numpy 侧用 int32 省内存，送进 nn.Embedding 前按 batch 转成 int64。"""
    return tensor.to(device=device, dtype=torch.long, non_blocking=True)


def move_batch(users, movies, ratings, device, on_device=False):
    """数据已在显存时直接返回；否则做 gather 后的 H2D 搬运与类型转换。"""
    if on_device:
        return users, movies, ratings
    return as_index(users, device), as_index(movies, device), ratings.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    on_device = batches.device is not None
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for users, movies, ratings in batches:
        users, movies, ratings = move_batch(users, movies, ratings, device, on_device)
        predictions = model(users, movies)
        error = predictions - ratings
        squared_error += (error * error).sum().item()
        absolute_error += error.abs().sum().item()
        count += ratings.numel()
    if count == 0:
        raise ValueError("评估集为空")
    return (squared_error / count) ** 0.5, absolute_error / count


def build_optimizer(model, learning_rate, weight_decay):
    """weight_decay 只作用于因子矩阵；偏置参与 L2 会让 bias 被无意义地压向 0。"""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if name == "global_bias" or name.endswith("_bias.weight"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def build_optimizers(model, learning_rate, weight_decay, sparse):
    """返回优化器列表。

    sparse=False：单个 AdamW，对所有参数做稠密更新。
    sparse=True ：因子矩阵交给 SparseAdam（只更新 batch 里出现过的行），其余参数交给 AdamW。
                  SparseAdam 不支持 weight_decay，所以因子矩阵上的 L2 会被跳过
                  （原默认值 1e-6 本来就可以忽略）。
    """
    if not sparse:
        return [build_optimizer(model, learning_rate, weight_decay)]

    sparse_names = ("user_embedding.weight", "movie_embedding.weight")
    sparse_params = [param for name, param in model.named_parameters() if name in sparse_names]
    dense_params = [param for name, param in model.named_parameters() if name not in sparse_names]
    return [
        torch.optim.SparseAdam(sparse_params, lr=learning_rate),
        torch.optim.AdamW(dense_params, lr=learning_rate, weight_decay=weight_decay),
    ]


def train(args):
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_start = time.perf_counter()
    data = load_data(args.data_dir, args.max_rows, seed=args.seed)
    data_seconds = time.perf_counter() - data_start
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available and not args.cpu else "cpu")

    # 环境诊断：这三行用来确认"到底跑在哪个设备上"
    cuda_note = ""
    if cuda_available:
        cuda_note = f"，CUDA {torch.version.cuda}，显卡 {torch.cuda.get_device_name(0)}"
    print(f"PyTorch：{torch.__version__}，CUDA 可用：{cuda_available}{cuda_note}")
    if args.cpu and cuda_available:
        print("注意：已通过 --cpu 强制使用 CPU（本机其实有可用的 CUDA）")
    elif device.type == "cpu":
        print("提示：没有检测到可用的 CUDA，正在用 CPU 训练。"
              "若本机有 NVIDIA 显卡，说明装的是 CPU 版 PyTorch，需要改装 CUDA 版。")
    else:
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    if device.type == "cpu":
        print(f"CPU 线程数：{torch.get_num_threads()}")

    n_train = len(data["train"][2])
    n_valid = len(data["valid"][2])
    n_test = len(data["test"][2])
    print(f"设备：{device}")
    print(f"数据加载与预处理耗时：{data_seconds:.1f} s")
    print(f"记录数：train={n_train:,} valid={n_valid:,} test={n_test:,} 合计={n_train + n_valid + n_test:,}")
    print(f"用户数：{len(data['user_ids']):,}，电影数：{len(data['movie_ids']):,}")
    print(f"训练集平均分：{data['train_mean']:.4f}")
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    print(f"每轮步数：{steps_per_epoch:,}（batch_size={args.batch_size}）")

    model = MatrixFactorization(
        len(data["user_ids"]),
        len(data["movie_ids"]),
        args.embedding_dim,
        global_bias_init=data["train_mean"],
        sparse=args.sparse,
    ).to(device)
    optimizers = build_optimizers(model, args.learning_rate, args.weight_decay, args.sparse)
    main_optimizer = optimizers[0]
    if args.sparse:
        print("已启用稀疏梯度（SparseAdam）：因子矩阵仅更新 batch 中出现过的行，但不使用 weight_decay")
    scheduler = None
    if args.plateau_patience >= 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            main_optimizer, mode="min", factor=0.5, patience=args.plateau_patience
        )
    loss_function = nn.MSELoss()

    # --data-on-gpu：把 train/valid/test 整块放进显存。这样每批的随机 gather 完全在 GPU 上，
    # 主机侧不参与取数，也就不会因为 Windows 压缩内存/缺页而出现"第 3 轮起变慢"。
    data_device = None
    if args.data_on_gpu:
        if device.type != "cuda":
            print("提示：--data-on-gpu 需要 CUDA，已忽略")
        else:
            need_bytes = (n_train + n_valid + n_test) * (8 + 8 + 4)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes < need_bytes * 1.3:
                print(f"提示：显存不足（需约 {need_bytes / 1024 ** 2:.0f} MB，"
                      f"可用 {free_bytes / 1024 ** 2:.0f} MB），--data-on-gpu 已忽略")
            else:
                data_device = device
                print(f"数据将常驻显存：约占 {need_bytes / 1024 ** 2:.0f} MB "
                      f"（可用 {free_bytes / 1024 ** 2:.0f} MB / 共 {total_bytes / 1024 ** 2:.0f} MB）")

    train_batches = TensorBatches(data["train"], args.batch_size, shuffle=True, seed=args.seed,
                                  device=data_device)
    valid_batches = TensorBatches(data["valid"], args.batch_size, shuffle=False, device=data_device)
    test_batches = TensorBatches(data["test"], args.batch_size, shuffle=False, device=data_device)
    on_device = data_device is not None
    if on_device:
        # 数据已经在显存里，主机侧这份 int32/float32 数组可以释放，缓解内存压力
        used = train_batches.vram_bytes() + valid_batches.vram_bytes() + test_batches.vram_bytes()
        for key in ("train", "valid", "test"):
            data[key] = None
        print(f"已释放主机侧数据数组，显存中数据占用约 {used / 1024 ** 2:.0f} MB")

    best_valid_rmse = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_count = 0
        for users, movies, ratings in train_batches:
            users, movies, ratings = move_batch(users, movies, ratings, device, on_device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(users, movies), ratings)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total_loss += loss.item() * ratings.numel()
            total_count += ratings.numel()

        train_mse = total_loss / total_count
        train_seconds = time.perf_counter() - epoch_start
        valid_start = time.perf_counter()
        valid_rmse, valid_mae = evaluate(model, valid_batches, device)
        valid_seconds = time.perf_counter() - valid_start
        epoch_seconds = train_seconds + valid_seconds
        eta_minutes = epoch_seconds * (args.epochs - epoch) / 60.0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_MSE={train_mse:.4f} | train_RMSE={train_mse ** 0.5:.4f} "
            f"| valid_RMSE={valid_rmse:.4f} | valid_MAE={valid_mae:.4f} "
            f"| lr={main_optimizer.param_groups[0]['lr']:.2e} "
            f"| 本轮={epoch_seconds:.1f}s(train={train_seconds:.1f}s valid={valid_seconds:.1f}s) "
            f"预计剩余={eta_minutes:.1f}分钟"
        )

        if scheduler is not None:
            scheduler.step(valid_rmse)
            # 稀疏路径下有两个优化器，把调度后的学习率同步过去
            current_lr = main_optimizer.param_groups[0]["lr"]
            for other in optimizers[1:]:
                for group in other.param_groups:
                    group["lr"] = current_lr

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            # 内存里留一份最优权重：否则测试集评估用的是"最后一轮"模型，
            # 和落盘的 checkpoint 不是同一个模型，打印出来的指标对不上。
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            checkpoint = {
                "model_state": best_state,
                "user_count": len(data["user_ids"]),
                "movie_count": len(data["movie_ids"]),
                "embedding_dim": args.embedding_dim,
                "user_ids": data["user_ids"],
                "movie_ids": data["movie_ids"],
                "movies_info": data["movies_info"],
                "item_train_counts": data["item_train_counts"],
                "user_train_counts": data["user_train_counts"],
                "train_mean": data["train_mean"],
                "valid_rmse": valid_rmse,
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(checkpoint, args.model_path)
        else:
            epochs_without_improvement += 1

        if (
            args.early_stop_patience >= 0
            and epochs_without_improvement > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(f"valid_RMSE 连续 {epochs_without_improvement} 轮未提升，提前停止训练。")
            break

    if best_state is None:
        print("警告：训练过程中没有产生有效的 valid_RMSE，未保存任何模型，下面用当前模型评估测试集。")
    else:
        model.load_state_dict(best_state)
        print(f"回载最优模型：epoch={best_epoch}，valid_RMSE={best_valid_rmse:.4f}")
        print(f"训练总耗时（不含数据加载）：{time.perf_counter() - train_start:.1f} s")

    test_rmse, test_mae = evaluate(model, test_batches, device)
    print(f"测试集：RMSE={test_rmse:.4f}，MAE={test_mae:.4f}")
    if best_state is not None:
        print(f"模型已保存：{Path(args.model_path).resolve()}")


def load_seen_movie_ids(data_dir, user_id, chunk_size=500_000):
    """流式扫描 ratings.csv，取出该用户已经评过分的 movieId。"""
    ratings_path = Path(data_dir) / "ratings.csv"
    if not ratings_path.exists():
        print(f"提示：找不到 {ratings_path}，无法排除用户已经看过的电影。")
        return set()
    seen = set()
    dtype = {"userId": "int32", "movieId": "int32"}
    for chunk in pd.read_csv(ratings_path, usecols=["userId", "movieId"], dtype=dtype, chunksize=chunk_size):
        seen.update(chunk.loc[chunk["userId"] == user_id, "movieId"].tolist())
    return seen


@torch.no_grad()
def recommend(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path.resolve()}，请先执行 --mode train")

    # weights_only=False：checkpoint 里存了 numpy 数组与 movies_info 字典
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MatrixFactorization(checkpoint["user_count"], checkpoint["movie_count"], checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    user_id = int(args.user_id)
    matching = np.where(checkpoint["user_ids"] == user_id)[0]
    if len(matching) == 0:
        raise ValueError(f"模型中不存在 userId={user_id}")
    user_index = int(matching[0])

    user_train_counts = checkpoint.get("user_train_counts")
    if user_train_counts is not None and int(user_train_counts[user_index]) == 0:
        print(f"警告：userId={user_id} 在训练集中没有任何交互，推荐结果不可靠。")

    movie_count = int(checkpoint["movie_count"])
    movie_indices = torch.arange(movie_count, dtype=torch.long)
    user_indices = torch.full_like(movie_indices, user_index)
    scores = model(user_indices, movie_indices)

    masked = torch.zeros(movie_count, dtype=torch.bool)

    # 1) 长尾/冷启动过滤：训练样本极少的条目，其 embedding 与偏置基本没被训练，
    #    打分会接近随机初值，很容易噪声性地挤进 top-K。
    item_train_counts = checkpoint.get("item_train_counts")
    if item_train_counts is not None:
        counts = torch.as_tensor(np.asarray(item_train_counts, dtype=np.int64))
        if counts.numel() != movie_count:
            raise ValueError("checkpoint 中的 item_train_counts 与 movie_count 不一致")
        masked |= counts < args.min_item_ratings
    else:
        print("提示：该 checkpoint 未记录条目训练样本数，跳过长尾过滤，建议重新训练。")

    # 2) 过滤用户已经看过的电影
    if not args.include_seen:
        seen_movie_ids = load_seen_movie_ids(args.data_dir, user_id)
        movie_id_to_index = {int(movie_id): index for index, movie_id in enumerate(checkpoint["movie_ids"])}
        seen_indices = [movie_id_to_index[movie_id] for movie_id in seen_movie_ids if movie_id in movie_id_to_index]
        if seen_indices:
            masked[torch.tensor(seen_indices, dtype=torch.long)] = True

    scores[masked] = float("-inf")

    available = int((~masked).sum())
    if available == 0:
        raise ValueError("所有候选电影都被过滤掉了，请降低 --min-item-ratings 或加 --include-seen")
    top_k = min(args.top_k, available)
    if top_k < args.top_k:
        print(f"提示：可用候选只有 {available} 部，top_k 自动调整为 {top_k}")

    top_indices = torch.topk(scores, k=top_k).indices.tolist()
    for rank, index in enumerate(top_indices, 1):
        movie_id = int(checkpoint["movie_ids"][index])
        title = checkpoint["movies_info"].get(movie_id, "未知标题")
        sample_info = f" | 训练样本数={int(item_train_counts[index])}" if item_train_counts is not None else ""
        print(f"{rank:02d}. movieId={movie_id} | 预测评分={scores[index].item():.3f}{sample_info} | {title}")


def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens-32M PyTorch 矩阵分解推荐模型")
    parser.add_argument("--mode", choices=["train", "recommend"], default="train")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="默认 8192。本模型每步的固定开销来自整张因子表的更新，几乎不随 batch 变化，"
             "所以加大 batch 能近似线性缩短每轮时间（2048 -> 8192 约快 3 倍），"
             "代价是每轮参数更新次数变少，可能需要更多轮",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.02,
        help="默认 0.02（原来的 1e-6 等于没有正则：实测 valid_RMSE 在第 3 轮触底后一路恶化到第 10 轮，"
             "而 train_RMSE 还在下降，是典型过拟合）。注意 --sparse 下 SparseAdam 不支持 weight_decay",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅随机抽样 N 条评分用于快速测试（会先读取整个 ratings.csv，抽样保证代表性）",
    )
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="因子矩阵使用稀疏梯度 + SparseAdam：每步只更新 batch 中出现过的行，"
             "在大词表上通常能大幅缩短每轮时间（代价：因子矩阵不使用 weight_decay）",
    )
    parser.add_argument(
        "--data-on-gpu",
        action="store_true",
        help="把 train/valid/test 整块放进显存（3200 万条约 640MB）。每批的随机 gather 全在 GPU 上完成，"
             "主机侧不参与取数：实测能消除第 3 轮起每轮变慢约 40%% 的现象，同时释放主机内存",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=-1,
        help="valid_RMSE 停滞多少轮后把学习率减半，负数表示关闭（默认关闭）",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="valid_RMSE 连续多少轮未提升就提前停止（默认 3；负数表示关闭）",
    )
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--min-item-ratings",
        type=int,
        default=5,
        help="推荐时只保留训练集中交互数不少于该值的电影（默认 5，设为 0 关闭过滤）",
    )
    parser.add_argument("--include-seen", action="store_true", help="允许推荐用户已经看过的电影")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "train":
        train(arguments)
    else:
        recommend(arguments)