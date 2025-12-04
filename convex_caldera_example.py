"""
Example usage of Convex-CALDERA algorithm with evaluation metrics.

This script demonstrates how to use the new Convex-CALDERA implementation
with both penalty and constrained forms, and how to evaluate the results.
"""

import torch
import sys
import os
sys.path.append('rank-constrained-regression-main')

from src.caldera.decomposition.convex_caldera import (
    convex_caldera,
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example 1: Single weight matrix compression with penalty form
print("\n" + "="*60)
print("Example 1: Convex-CALDERA with Penalty Form")
print("="*60)

# Create a sample weight matrix
m, n = 1024, 1024
W = torch.randn(m, n, device=device)

# Create Hessian (identity for simplicity, or use precomputed)
H = torch.eye(n, device=device)

# Configure parameters for penalty form
params_penalty = ConvexCalderaParams(
    B_tot=2.0,  # Target 2 bits per parameter
    b_min=2.0,
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
bit_allocations = decomp_penalty.b_discrete.reshape(1, -1)
plot_bit_allocation_heatmap(
    bit_allocations,
    layer_names=["Layer 0"],
    save_path="plots/bit_allocation.png"
)

# Example 5: Compare different bit budgets
print("\n" + "="*60)
print("Example 5: Accuracy vs Bits Trade-off")
print("="*60)

bits_list = []
accuracy_list = []  # Using relative error as proxy for accuracy

for B_tot in [1.5, 2.0, 2.5, 3.0, 4.0]:
    params = ConvexCalderaParams(
        B_tot=B_tot,
        b_min=2.0,
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

