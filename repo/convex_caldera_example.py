"""
Example usage of Convex-CALDERA algorithm with evaluation metrics.

This script demonstrates how to use the new Convex-CALDERA implementation
with both penalty and constrained forms, and how to evaluate the results.
"""

import torch
import sys
import os
import numpy as np
sys.path.append('rank-constrained-regression-main')

from src.caldera.decomposition.convex_caldera import (
    convex_caldera_decompose as convex_caldera,
    ConvexCalderaParams
)
from src.caldera.utils.metrics import (
    evaluate_compression,
    plot_bit_allocation_heatmap,
    plot_accuracy_vs_bits,
    plot_loss_vs_rank,
    plot_singular_value_spectra,
    compute_singular_values
)
# ---- replace sample W/H with actual LLaMA-2-7B layer ----
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example 1: Single weight matrix compression with penalty form
print("\n" + "="*60)
print("Example 1: Convex-CALDERA with Penalty Form")
print("="*60)





model_name = "meta-llama/Llama-2-13b-hf"
# load model (注意：需HF访问权限)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"   # 或 device_map={"": "cuda:0"} 视显存而定
)

# Example: pick one layer to compress, e.g. first MLP gate_proj
layer_name = "model.layers.0.mlp.gate_proj"  # 根据实际模型结构确认 name
# locate module
target_module = None
for name, mod in model.named_modules():
    if name.endswith(layer_name) or name == layer_name:
        target_module = mod
        break
if target_module is None:
    raise RuntimeError(f"Can't find layer {layer_name} in model")

# Get weight and move to device used by convex_caldera (float32)
W = target_module.weight.detach().to(torch.float32).to(device)

# Hessian H: either from precomputed diag_Hessians.pt or identity if not available
h_path = "diag_Hessians.pt"
if os.path.exists(h_path):
    Hall = torch.load(h_path)
    # Hall should map layer names to diagonal arrays: Hall[name]
    h_diag = Hall.get(layer_name, None)
    if h_diag is None:
        H = torch.eye(W.shape[1], device=device)
    else:
        H = torch.diag_embed(h_diag.to(device).to(torch.float32))
else:
    H = torch.eye(W.shape[1], device=device)
# ---- end replacement ----



# Configure parameters for penalty form
params_penalty = ConvexCalderaParams(
    B_tot=2.0,  # Target 2 bits per parameter
    b_min=1.0,
    b_max=8.0,
    mu=0.1,  # Nuclear norm penalty weight
    tau_star=None,  # Not using constrained form
    lambda_reg=0.01,
    k=1.0,
    discrete_bits=[2, 3, 4, 8],
    solver="SCS",
    solver_verbose=True
)

# Run Convex-CALDERA
print("Running Convex-CALDERA (penalty form)...")
decomp_penalty = convex_caldera(
    W=W,
    H=H,
    params=params_penalty,
    device=device
)

print(f"\nResults (Penalty Form):")
print(f"  Solver status: {decomp_penalty.solver_status}")
print(f"  Solve time: {decomp_penalty.solve_time:.2f}s")
print(f"  Average bit-width: {decomp_penalty.avg_bit_width:.2f}")
print(f"  Effective rank: {decomp_penalty.effective_rank:.2f}")
print(f"  Duality gap: {decomp_penalty.duality_gap:.4f}")
print(f"  Residual norm: {decomp_penalty.residual_norm:.4f}")

# Example 2: Constrained form
print("\n" + "="*60)
print("Example 2: Convex-CALDERA with Constrained Form")
print("="*60)

# Configure parameters for constrained form
params_constrained = ConvexCalderaParams(
    B_tot=2.0,
    b_min=2.0,
    b_max=8.0,
    tau_star=100.0,  # Nuclear norm bound
    mu=None,  # Not using penalty form
    lambda_reg=0.01,
    k=1.0,
    discrete_bits=[2, 3, 4, 8],
    solver="SCS",
    solver_verbose=True
)

print("Running Convex-CALDERA (constrained form)...")
decomp_constrained = convex_caldera(
    W=W,
    H=H,
    params=params_constrained,
    device=device
)

print(f"\nResults (Constrained Form):")
print(f"  Solver status: {decomp_constrained.solver_status}")
print(f"  Solve time: {decomp_constrained.solve_time:.2f}s")
print(f"  Average bit-width: {decomp_constrained.avg_bit_width:.2f}")
print(f"  Effective rank: {decomp_constrained.effective_rank:.2f}")
print(f"  Duality gap: {decomp_constrained.duality_gap:.4f}")
print(f"  Residual norm: {decomp_constrained.residual_norm:.4f}")

# Example 3: Evaluation metrics
print("\n" + "="*60)
print("Example 3: Evaluation Metrics")
print("="*60)

# Compute comprehensive metrics
metrics = evaluate_compression(
    W_original=W,
    W_compressed=decomp_penalty.W_compressed,
    avg_bit_width=decomp_penalty.avg_bit_width,
    effective_rank=decomp_penalty.effective_rank,
    duality_gap=decomp_penalty.duality_gap
)

print(f"\nCompression Metrics:")
print(f"  Bits per parameter: {metrics.bits_per_parameter:.3f}")
print(f"  Relative error: {metrics.relative_error:.4f}")
print(f"  Compression ratio: {metrics.compression_ratio:.2f}x")
print(f"  Model size: {metrics.model_size_mb:.2f} MB")
print(f"  Effective rank: {metrics.effective_rank:.2f}")
print(f"  Duality gap: {metrics.duality_gap:.4f}")

# Example 4: Qualitative plots
print("\n" + "="*60)
print("Example 4: Generating Qualitative Plots")
print("="*60)

# Create output directory
os.makedirs("plots", exist_ok=True)

# Plot singular value spectra
print("Plotting singular value spectra...")
sv_original = compute_singular_values(W)
sv_compressed = compute_singular_values(decomp_penalty.W_compressed)
plot_singular_value_spectra(
    sv_original,
    sv_compressed,
    save_path="plots/singular_values.png"
)

# Plot bit allocation (for multiple layers, this would be more interesting)
print("Plotting bit allocation heatmap...")
bit_allocations = np.array([[decomp_penalty.b_discrete]])
plot_bit_allocation_heatmap(
    bit_allocations,
    layer_names=["Layer 0"],
    save_path="plots/bit_allocation.png"
)

# 在 Example 5 前面加
print("\n=== DEBUG INFO ===")
decomp_test = convex_caldera(W=W, H=H, params=ConvexCalderaParams(B_tot=2.0), device=device)
print(f"avg_bit_width: {decomp_test.avg_bit_width}")
print(f"b_discrete: {decomp_test.b_discrete}")
print(f"effective_rank: {decomp_test.effective_rank}")
print(f"W_compressed dtype: {decomp_test.W_compressed.dtype}")
print(f"W original norm: {torch.norm(W):.4f}")
print(f"W_comp norm: {torch.norm(decomp_test.W_compressed):.4f}")
print(f"Error: {torch.norm(W - decomp_test.W_compressed):.4f}")

# Example 5: Compare different bit budgets
print("\n" + "="*60)
print("Example 5: Accuracy vs Bits Trade-off")
print("="*60)

bits_list = []
accuracy_list = []  # Using relative error as proxy for accuracy

for B_tot in [1.5, 2.0, 2.5, 3.0, 4.0]:
    params = ConvexCalderaParams(
        B_tot=B_tot,
        b_min=1.0,
        b_max=8.0,
        mu=0.1,
        solver="SCS",
        solver_verbose=False
    )
    
    decomp = convex_caldera(W=W, H=H, params=params, device=device)
    metrics = evaluate_compression(
        W_original=W,
        W_compressed=decomp.W_compressed,
        avg_bit_width=decomp.avg_bit_width,
        effective_rank=decomp.effective_rank,
        duality_gap=decomp.duality_gap
    )
    
    bits_list.append(metrics.bits_per_parameter)
    accuracy_list.append(1.0 - metrics.relative_error)  # Higher is better
    
    print(f"  B_tot={B_tot:.1f}: bits={metrics.bits_per_parameter:.3f}, "
          f"error={metrics.relative_error:.4f}")

# Plot accuracy vs bits
plot_accuracy_vs_bits(
    bits_list,
    accuracy_list,
    method_name="Convex-CALDERA",
    save_path="plots/accuracy_vs_bits.png"
)

print("\n" + "="*60)
print("All examples completed!")
print("Plots saved in 'plots/' directory")
print("="*60)

# after obtaining W_compressed
target_module.weight.data.copy_(decomp_penalty.W_compressed.to(target_module.weight.dtype).to(target_module.weight.device))
# 保存 HuggingFace 模型
save_dir = "./llama2_13b_convex_caldera_quant"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)





import os
import numpy as np
import pandas as pd
import torch


# -----------------
# Config
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)

# Choose ONE layer/weight for the sweep
# Example: random matrix (replace with real layer weight)
m, n = 4096, 4096
W = torch.randn(m, n, device=device, dtype=torch.float32)
H = torch.eye(n, device=device)

# Sweeps
rank_list = [32, 64, 128]
bits_list = [2, 3, 4, 8]   # change as needed

# Baseline: you can decide fp16 or fp32 baseline in evaluate_compression (your current is fp16-style)
# This script will just call evaluate_compression as you implemented.

# -----------------
# Precompute singular values on CPU for tau_star mapping
# -----------------
S = compute_singular_values(W)  # numpy array descending

def tau_for_rank(r: int) -> float:
    r = min(r, len(S))
    return float(np.sum(S[:r]))

# -----------------
# Run sweep: Constrained rank-controlled
# -----------------
rows = []
for r in rank_list:
    tau = tau_for_rank(r)
    for b in bits_list:
        params = ConvexCalderaParams(
            B_tot=float(b),
            b_min=2.0,
            b_max=8.0,
            tau_star=tau,   # rank-controlled via tau
            mu=None,
            discrete_bits=[2,3,4,8],
            solver_verbose=False,
        )
        decomp = convex_caldera(W=W, H=H, params=params, device=device)

        metrics = evaluate_compression(
            W_original=W,
            W_compressed=decomp.W_compressed,
            avg_bit_width=decomp.avg_bit_width,
            effective_rank=decomp.effective_rank,
            duality_gap=decomp.duality_gap,
        )

        rows.append({
            "mode": "constrained",
            "target_rank": r,
            "tau_star": tau,
            "target_bits": b,
            "avg_bit_width": float(decomp.avg_bit_width),
            "effective_rank": float(decomp.effective_rank),
            "residual_norm": float(decomp.residual_norm),
            "rel_error": float(metrics.relative_error),
            "accuracy_proxy": float(1.0 - metrics.relative_error),
            "bits_per_param": float(metrics.bits_per_parameter),
            "compression_ratio": float(metrics.compression_ratio),
            "model_size_mb": float(metrics.model_size_mb),
            "solve_time_s": float(decomp.solve_time),
        })

df = pd.DataFrame(rows)
df.to_csv("results_rank_bits.csv", index=False)
print("Saved: results_rank_bits.csv")
print(df.head())

# -----------------
# Plotting (matplotlib only)
# -----------------
import matplotlib.pyplot as plt

# Heatmap: rel_error with rows=rank, cols=bits
pivot_err = df.pivot(index="target_rank", columns="target_bits", values="rel_error")
plt.figure()
plt.imshow(pivot_err.values, aspect="auto")
plt.xticks(range(len(pivot_err.columns)), pivot_err.columns)
plt.yticks(range(len(pivot_err.index)), pivot_err.index)
plt.colorbar(label="Relative error")
plt.xlabel("Bits (target)")
plt.ylabel("Rank (target)")
plt.title("Constrained Convex-CALDERA: Relative Error Heatmap")
plt.tight_layout()
plt.savefig("plots/heatmap_error_rank_bits.png")
plt.close()

# Heatmap: bits_per_param (what you're actually paying under your metric)
pivot_bpp = df.pivot(index="target_rank", columns="target_bits", values="bits_per_param")
plt.figure()
plt.imshow(pivot_bpp.values, aspect="auto")
plt.xticks(range(len(pivot_bpp.columns)), pivot_bpp.columns)
plt.yticks(range(len(pivot_bpp.index)), pivot_bpp.index)
plt.colorbar(label="Bits per parameter")
plt.xlabel("Bits (target)")
plt.ylabel("Rank (target)")
plt.title("Constrained Convex-CALDERA: Bits/Param Heatmap")
plt.tight_layout()
plt.savefig("plots/heatmap_bits_rank_bits.png")
plt.close()

# Curves: error vs bits for each rank
plt.figure()
for r in rank_list:
    sub = df[df["target_rank"] == r].sort_values("target_bits")
    plt.plot(sub["bits_per_param"], sub["rel_error"], marker="o", label=f"rank={r}")
plt.xlabel("Bits per parameter")
plt.ylabel("Relative error")
plt.title("Constrained Convex-CALDERA: Error vs Bits (by Rank)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/curve_error_vs_bits_by_rank.png")
plt.close()

print("Saved plots under plots/")




