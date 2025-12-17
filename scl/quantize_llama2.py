import argparse
from typing import Literal

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------
#  K-means / Lloyd-Max helper
# ---------------------------

def kmeans(
    x: torch.Tensor,
    k: int = 256,
    iters: int = 20,
    verbose: bool = True,
) -> torch.Tensor:
    """
    简单 k-means，用于 1D 或低维向量。
    x: [N, D]
    return: centroids [k, D]
    """
    device = x.device
    N, D = x.shape

    # 随机初始化中心
    idx = torch.randperm(N, device=device)[:k]
    centroids = x[idx].clone()

    for it in range(iters):
        # 分配
        x_exp = x.unsqueeze(1)           # [N,1,D]
        c_exp = centroids.unsqueeze(0)   # [1,k,D]
        dist = (x_exp - c_exp).pow(2).sum(-1)  # [N,k]

        labels = dist.argmin(dim=1)      # [N]

        # 更新中心
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=k).float().to(device)

        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids[i] = x[mask].mean(dim=0)
            else:
                # 如果某个中心没人选，随机重置
                ridx = torch.randint(0, N, (1,), device=device)
                new_centroids[i] = x[ridx]

        shift = (centroids - new_centroids).pow(2).sum().sqrt()
        centroids = new_centroids

        if verbose:
            print(f"[kmeans] iter {it+1}/{iters}, shift={shift.item():.4f}")

        if shift < 1e-4:
            break

    return centroids


# ---------------------------
#  三种量化方法
# ---------------------------

def quantize_linear_uniform(m: nn.Linear, n_bits: int = 8):
    """均匀标量量化：对 weight 做对称 per-tensor 8bit。"""
    w = m.weight.data
    w_f = w.float()
    max_val = w_f.abs().max()
    qmax = 2 ** (n_bits - 1) - 1  # 127

    scale = max_val / qmax if max_val > 0 else 1.0
    q = torch.round(w_f / scale).clamp_(-qmax - 1, qmax)
    w_q = (q * scale).to(w.dtype)
    m.weight.data.copy_(w_q)


def quantize_linear_lloyd(
    m: nn.Linear,
    n_bits: int = 8,
    sample_size: int = 1_000_000,
    batch_size: int = 500_000,
):
    """
    稳定版 Lloyd-Max 标量量化：
    - 先在子样本上训练 k-means codebook
    - 再用 mini-batch 给整层权重分配最近中心，避免显存爆炸
    """
    w = m.weight.data
    w_f = w.float()
    flat = w_f.view(-1)
    N = flat.numel()
    k = 2 ** n_bits
    device = flat.device

    # 1) 训练 codebook（在 sample 上）
    if N > sample_size:
        perm = torch.randperm(N, device=device)[:sample_size]
        sample = flat[perm]
    else:
        sample = flat

    print(f"[Lloyd-Max] training codebook on {sample.numel()} samples, k={k}")
    sample = sample.unsqueeze(1)  # [Ns,1]
    centroids = kmeans(sample, k=k, iters=20, verbose=False).squeeze(1)  # [k]

    # 防御性检查
    if not torch.isfinite(centroids).all():
        print("Warning: non-finite centroids detected, applying nan_to_num")
        centroids = torch.nan_to_num(centroids)

    # 2) 分 batch 量化整层
    print("[Lloyd-Max] quantizing full tensor in batches...")
    labels = torch.empty(N, dtype=torch.long, device=device)

    c_exp = centroids.unsqueeze(0)  # [1,k]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = flat[start:end].unsqueeze(1)  # [B,1]
        dist = (chunk - c_exp).pow(2)        # [B,k]
        labels[start:end] = dist.argmin(dim=1)

    w_q = centroids[labels].view_as(w_f)

    if not torch.isfinite(w_q).all():
        print("Warning: non-finite quantized weights, applying nan_to_num")
        w_q = torch.nan_to_num(w_q)

    m.weight.data.copy_(w_q.to(w.dtype))




def quantize_linear_vq(
    m: nn.Linear,
    n_bits: int = 8,
    group_size: int = 4,
    sample_size: int = 500_000,
    batch_size: int = 200_000,   # 👈 新增：量化阶段每次处理多少个向量
):
    """
    稳定版 VQ：
    - 把 weight 沿输入维度分成 group_size 维的向量
    - 在子样本上训练 k-means codebook
    - 再用 mini-batch 给所有向量分配最近中心，避免显存爆炸
    """
    w = m.weight.data
    w_f = w.float()  # [out, in]
    out_f, in_f = w_f.shape
    device = w_f.device

    k = 2 ** n_bits

    # 1) 沿 last dim 分组
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        w_padded = torch.nn.functional.pad(w_f, (0, pad))
    else:
        w_padded = w_f

    new_in = w_padded.shape[1]
    vecs = w_padded.view(out_f, new_in // group_size, group_size)
    vecs = vecs.reshape(-1, group_size)        # [N, group_size]
    N = vecs.size(0)

    # 2) 在子样本上训练 codebook
    if N > sample_size:
        perm = torch.randperm(N, device=device)[:sample_size]
        sample = vecs[perm]
    else:
        sample = vecs

    print(f"[VQ] training codebook on {sample.shape[0]} vectors, "
          f"dim={group_size}, k={k}")
    centroids = kmeans(sample, k=k, iters=20, verbose=False)  # [k, group_size]

    if not torch.isfinite(centroids).all():
        print("Warning: non-finite VQ centroids, applying nan_to_num")
        centroids = torch.nan_to_num(centroids)

    # 3) 分 batch 量化所有向量
    print("[VQ] quantizing full tensor in batches...")
    labels = torch.empty(N, dtype=torch.long, device=device)
    c_exp = centroids.unsqueeze(0)   # [1, K, D]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = vecs[start:end]                  # [B, D]
        x = chunk.unsqueeze(1)                   # [B, 1, D]
        dist = (x - c_exp).pow(2).sum(-1)        # [B, K]
        labels[start:end] = dist.argmin(dim=1)

    q_vecs = centroids[labels]                  # [N, D]

    if not torch.isfinite(q_vecs).all():
        print("Warning: non-finite quantized VQ vectors, applying nan_to_num")
        q_vecs = torch.nan_to_num(q_vecs)

    # 4) reshape 回原来的 weight 形状
    q_vecs = q_vecs.view(out_f, new_in // group_size, group_size)
    q_padded = q_vecs.reshape(out_f, new_in)

    if pad > 0:
        q = q_padded[:, :in_f]
    else:
        q = q_padded

    m.weight.data.copy_(q.to(w.dtype))


def quantize_model(model: nn.Module, method: Literal["scalar", "lloyd", "vq"]):
    model.eval()
    total = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    print(f"Found {total} Linear layers.")

    i = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            i += 1
            # ✅ 先别对 lm_head 做 lloyd
            if method == "lloyd" and name == "lm_head":
                print(f"[{i}/{total}] skip lloyd for {name}")
                continue

            print(f"[{i}/{total}] quantizing {name} with {method}...")
            if method == "scalar":
                quantize_linear_uniform(module)
            elif method == "lloyd":
                quantize_linear_lloyd(module)
            elif method == "vq":
                quantize_linear_vq(module)
            else:
                raise ValueError(method)
    print("Quantization done.")


def check_model_finite(model):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print("Found non-finite values in", name)
            return False
    return True


# ---------------------------
#  PPL on WikiText-2 helper
# ---------------------------

@torch.no_grad()
def evaluate_wikitext_ppl(
    model,
    tokenizer,
    device="cuda",
    stride: int = 512,
    max_length: int = 2048,
):
    """
    HF 官方教程的 PPL 模板。
    """
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids.to(device)

    nlls = []
    for i in range(0, input_ids.size(1) - 1, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1) - 1)
        trg_len = end_loc - i  # may be shorter at end

        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_slice, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


# ---------------------------
#  main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["scalar", "lloyd", "vq"],
        required=True,
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="只量化，不在 WikiText-2 上评估",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    device = args.device

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map={"": device},   # 单卡
    )

    # 先把模型搬到 device 上，防止后面量化时有 CPU/CUDA 混用问题
    model.to(device)

    # 量化
    quantize_model(model, args.method)
    if not check_model_finite(model):
        print("Quantized model has NaN/Inf weights, aborting eval.")
        return

    if args.no_eval:
        print("Skip evaluation as --no-eval is set.")
        return

    print("Evaluating WikiText-2 perplexity...")
    ppl = evaluate_wikitext_ppl(model, tokenizer, device=device)
    print(f"Method = {args.method}, WikiText-2 PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
