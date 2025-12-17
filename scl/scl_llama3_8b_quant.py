import math
import random
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 通用工具
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_modules(model: nn.Module):
    """
    收集所有需要量化的 Linear 层。
    这里简单地把所有 nn.Linear 都量化，必要时你可以筛掉 lm_head 等。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


# ============================================================
# 2. 8-bit 均匀标量量化（SCL-Scalar Quant 8-bit）
# ============================================================

def uniform_scalar_quantize_8bit(
    weight: torch.Tensor,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """
    均匀标量量化 + 反量化，返回重建后的权重张量（float32）。
    weight: torch.Tensor on CPU (float32)
    """
    w = weight.view(-1)

    if clip_ratio < 1.0:
        lo = w.quantile((1 - clip_ratio) / 2)
        hi = w.quantile(1 - (1 - clip_ratio) / 2)
    else:
        lo = w.min()
        hi = w.max()

    # 避免 hi == lo
    if (hi - lo) < 1e-8:
        return weight.clone()

    L = 256  # 8-bit
    delta = (hi - lo) / (L - 1)

    # 映射到 [0, L-1]
    q = torch.clamp(torch.round((w - lo) / delta), 0, L - 1)

    # 反量化
    w_rec = lo + q * delta
    return w_rec.view_as(weight)


# ============================================================
# 3. Lloyd-Max 标量量化（用 1D k-means 实现）
# ============================================================

def kmeans_1d(
    x: torch.Tensor,
    num_clusters: int = 256,
    num_iters: int = 20,
    device: str = "cuda",
) -> torch.Tensor:
    """
    简单 1D k-means，用来近似 Lloyd-Max。
    x: 1D tensor
    返回: cluster centers [num_clusters]
    """
    x = x.to(device)
    N = x.numel()

    # 初始化中心：等间隔
    xmin, xmax = x.min(), x.max()
    centers = torch.linspace(xmin, xmax, num_clusters, device=device)

    for _ in range(num_iters):
        # [N, K] 距离矩阵
        # 注意：如果 N 很大，这里是 O(NK)；但我们后面会对 x 做采样。
        dists = (x[:, None] - centers[None, :]) ** 2
        labels = torch.argmin(dists, dim=1)

        # 重新计算每个簇的均值
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                centers[k] = x[mask].mean()

    return centers.detach().cpu()


def lloyd_max_quantize_8bit(
    weight: torch.Tensor,
    sample_size: int = 200_000,
    num_iters: int = 25,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Lloyd-Max 标量量化（近似）：先在采样子集上跑 1D k-means，得到 256 个 level，
    再对整个 tensor 做最近中心分配 + 反量化。
    weight: CPU float32 tensor
    """
    w = weight.view(-1)
    N = w.numel()

    # 采样子集来训练 codebook（避免对全部 7B 权重做 k-means）
    if N > sample_size:
        idx = torch.randperm(N)[:sample_size]
        sample = w[idx]
    else:
        sample = w

    centers = kmeans_1d(
        sample,
        num_clusters=256,
        num_iters=num_iters,
        device=device,
    )  # [256] on CPU

    # 用训练好的 centers 对整个 tensor 量化 + 反量化
    centers_gpu = centers.to(device)

    w_rec = torch.empty_like(w)
    chunk_size = 1_000_000  # 按块处理，避免距离矩阵过大

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = w[start:end].to(device)  # [M]

        dists = (chunk[:, None] - centers_gpu[None, :]) ** 2  # [M, 256]
        labels = torch.argmin(dists, dim=1)                  # [M]
        rec = centers_gpu[labels]                            # [M]

        w_rec[start:end] = rec.cpu()

    return w_rec.view_as(weight)


# ============================================================
# 4. 向量量化（8-bit VQ：256 codewords）
# ============================================================

def kmeans_nd(
    x: torch.Tensor,
    num_clusters: int = 256,
    num_iters: int = 20,
    device: str = "cuda",
) -> torch.Tensor:
    """
    简单 ND k-means，用于 VQ。
    x: [N, D]
    返回: centers [K, D]
    """
    x = x.to(device)
    N, D = x.shape

    # 随机选 K 个点做初始化
    perm = torch.randperm(N, device=device)
    init_idx = perm[:num_clusters]
    centers = x[init_idx].clone()  # [K, D]

    for _ in range(num_iters):
        # [N, K] 距离
        dists = torch.cdist(x, centers, p=2)  # 比手写 (x-c)^2 好点

        labels = torch.argmin(dists, dim=1)  # [N]

        # 更新中心
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                centers[k] = x[mask].mean(dim=0)

    return centers.detach().cpu()


def vector_quantize_8bit(
    weight: torch.Tensor,
    block_size: int = 4,
    sample_size: int = 200_000,
    num_iters: int = 25,
    device: str = "cuda",
) -> torch.Tensor:
    """
    8-bit VQ：将权重看成 dim=block_size 的向量，使用 256 个 codeword。
    weight: CPU float32 tensor
    返回：反量化后的重建权重
    """
    w = weight.view(-1)
    N = w.numel()

    # 填充到 block_size 的整数倍
    pad = (-N) % block_size
    if pad > 0:
        w_padded = torch.cat([w, torch.zeros(pad, dtype=w.dtype)])
    else:
        w_padded = w
    N_pad = w_padded.numel()
    num_vecs = N_pad // block_size

    vectors = w_padded.view(num_vecs, block_size)  # [num_vecs, D]

    # 采样部分向量来训练 codebook
    if num_vecs > sample_size:
        idx = torch.randperm(num_vecs)[:sample_size]
        sample = vectors[idx]
    else:
        sample = vectors

    centers = kmeans_nd(
        sample,
        num_clusters=256,
        num_iters=num_iters,
        device=device,
    )  # [256, D] on CPU

    centers_gpu = centers.to(device)

    # 对所有向量做 VQ 编码 + 重建
    rec_vectors = torch.empty_like(vectors)
    chunk_size = 250_000  # 控制显存

    for start in range(0, num_vecs, chunk_size):
        end = min(start + chunk_size, num_vecs)
        chunk = vectors[start:end].to(device)  # [M, D]

        dists = torch.cdist(chunk, centers_gpu, p=2)  # [M, 256]
        labels = torch.argmin(dists, dim=1)           # [M]
        rec = centers_gpu[labels]                     # [M, D]

        rec_vectors[start:end] = rec.cpu()

    w_rec_padded = rec_vectors.view(-1)

    # 去掉 padding，reshape 回原形状
    w_rec = w_rec_padded[:N].view_as(weight)
    return w_rec


# ============================================================
# 5. 对 LLaMA-2-7B 应用三种量化
# ============================================================

QuantMethod = Literal["scalar_uniform", "lloyd_max", "vector_vq"]


@dataclass
class QuantConfig:
    method: QuantMethod
    clip_ratio: float = 1.0          # 仅用于 uniform scalar
    sample_size: int = 200_000       # 用于 Lloyd-Max 和 VQ
    num_iters: int = 25
    block_size: int = 4              # 用于 VQ
    device_for_kmeans: str = "cuda"  # "cuda" or "cpu"


def quantize_linear_weight(
    w: torch.Tensor,
    cfg: QuantConfig,
) -> torch.Tensor:
    """
    输入: Linear.weight 的 tensor（在 GPU 或 CPU 都可）
    输出: 量化+反量化后的 tensor（同 dtype）
    """
    dtype = w.dtype
    device = w.device

    # 为了省显存，先搬到 CPU 做 quant
    w_cpu = w.detach().float().cpu()

    if cfg.method == "scalar_uniform":
        w_rec_cpu = uniform_scalar_quantize_8bit(
            w_cpu,
            clip_ratio=cfg.clip_ratio,
        )
    elif cfg.method == "lloyd_max":
        w_rec_cpu = lloyd_max_quantize_8bit(
            w_cpu,
            sample_size=cfg.sample_size,
            num_iters=cfg.num_iters,
            device=cfg.device_for_kmeans,
        )
    elif cfg.method == "vector_vq":
        w_rec_cpu = vector_quantize_8bit(
            w_cpu,
            block_size=cfg.block_size,
            sample_size=cfg.sample_size,
            num_iters=cfg.num_iters,
            device=cfg.device_for_kmeans,
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # 回到原来的 device + dtype
    return w_rec_cpu.to(device=device, dtype=dtype)


def apply_quant_to_model(
    model: nn.Module,
    cfg: QuantConfig,
    skip_lm_head: bool = True,
):
    """
    遍历模型里的 Linear 层做量化（+反量化）。
    """
    total_params = 0
    with torch.no_grad():
        for name, module in get_linear_modules(model):
            if skip_lm_head and "lm_head" in name:
                print(f"[skip] {name}")
                continue

            w = module.weight
            numel = w.numel()
            total_params += numel

            print(f"[quant] {name} | shape={tuple(w.shape)} | params={numel/1e6:.2f}M")

            w_rec = quantize_linear_weight(w, cfg)
            module.weight.data.copy_(w_rec)

    print(f"Total quantized params in Linear layers: {total_params/1e9:.3f}B")


# ============================================================
# 6. 简单测试：load LLaMA-2-7B，做一次前向
# ============================================================

def load_llama3_8b(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str = "cuda",
):
    """
    加载原始 LLaMA-2-7B FP16 模型。
    注意：需要你已经在 HF 上申请过访问权限。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},  # 整个模型放一张卡
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def quick_inference(tokenizer, model, prompt: str, max_new_tokens: int = 32):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="scalar_uniform",
        choices=["scalar_uniform", "lloyd_max", "vector_vq"],
    )
    parser.add_argument("--sample_size", type=int, default=200_000)
    parser.add_argument("--num_iters", type=int, default=25)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--clip_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    # 可选：自定义输出目录前缀
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="保存量化模型的目录（默认根据 method 自动命名）",
    )

    args = parser.parse_args()

    set_seed(42)

    print(f"Loading model {args.model_name} ...")
    tokenizer, model = load_llama3_8b(args.model_name, device=args.device)

    # 原始模型做一次 forward
    print("\n=== Original model sample ===")
    quick_inference(tokenizer, model, "The capital of France is")

    cfg = QuantConfig(
        method=args.method,          # 三种方法在这里选
        clip_ratio=args.clip_ratio,
        sample_size=args.sample_size,
        num_iters=args.num_iters,
        block_size=args.block_size,
        device_for_kmeans=args.device,
    )
    print(f"\nQuant config: {cfg}")

    # 对模型做量化
    apply_quant_to_model(model, cfg)

    # 量化后再测一次
    print("\n=== Quantized model sample ===")
    quick_inference(tokenizer, model, "The capital of France is")

    # ========= 在这里保存模型 =========
    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        # 根据 method 自动起名
        if args.method == "scalar_uniform":
            save_dir = "./llama3_8b_scl_scalar8"
        elif args.method == "lloyd_max":
            save_dir = "./llama3_8b_scl_lloyd8"
        else:  # vector_vq
            save_dir = "./llama3_8b_scl_vq8"

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving quantized model to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done saving.")



if __name__ == "__main__":
    main()
