"""
Example usage of SCL (Stanford Compression Library) baselines for quantization.

This script demonstrates scalar quantization, vector quantization, and Lloyd-Max
quantization as baselines for comparison with CALDERA and Convex-CALDERA.

Based on: https://stanforddatacompressionclass.github.io/notes/lossy/quant.html
"""

import torch
import sys
import os
sys.path.append('rank-constrained-regression-main')

from src.caldera.utils.scl_baselines import (
    scl_quantize,
    SCLQuantizationParams,
    apply_scl_baseline_to_model
)
from src.caldera.utils.metrics import evaluate_compression, plot_singular_value_spectra

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example 1: Scalar Uniform Quantization
print("\n" + "="*60)
print("Example 1: Scalar Uniform Quantization")
print("="*60)

# Create a sample weight matrix
W = torch.randn(512, 512, device=device)

params_scalar = SCLQuantizationParams(
    num_bits=2,
    method="scalar",
    distortion_metric="mse"
)

result_scalar = scl_quantize(W, params_scalar, device)

print(f"Method: {result_scalar.method}")
print(f"Rate: {result_scalar.rate:.3f} bits per sample")
print(f"Distortion (MSE): {result_scalar.distortion:.6f}")
print(f"Compression ratio: {result_scalar.compression_ratio:.2f}x")
print(f"Codebook size: {result_scalar.num_codebook_entries}")

# Example 2: Lloyd-Max Quantization (Optimal for MSE)
print("\n" + "="*60)
print("Example 2: Lloyd-Max Quantization (Optimal for MSE)")
print("="*60)

params_lloyd = SCLQuantizationParams(
    num_bits=2,
    method="lloyd_max",
    max_iterations=100,
    tolerance=1e-6,
    distortion_metric="mse"
)

result_lloyd = scl_quantize(W, params_lloyd, device)

print(f"Method: {result_lloyd.method}")
print(f"Rate: {result_lloyd.rate:.3f} bits per sample")
print(f"Distortion (MSE): {result_lloyd.distortion:.6f}")
print(f"Compression ratio: {result_lloyd.compression_ratio:.2f}x")
print(f"Codebook size: {result_lloyd.num_codebook_entries}")

# Compare with uniform quantization
print(f"\nImprovement over uniform: "
      f"{(result_scalar.distortion - result_lloyd.distortion) / result_scalar.distortion * 100:.2f}% reduction in MSE")

# Example 3: Vector Quantization (K-means)
print("\n" + "="*60)
print("Example 3: Vector Quantization (K-means / Generalized Lloyd)")
print("="*60)

params_vector = SCLQuantizationParams(
    num_bits=2,
    method="vector",
    vector_dim=4,  # 4-dimensional vectors
    max_iterations=100,
    tolerance=1e-6,
    distortion_metric="mse"
)

result_vector = scl_quantize(W, params_vector, device)

print(f"Method: {result_vector.method}")
print(f"Rate: {result_vector.rate:.3f} bits per sample")
print(f"Distortion (MSE): {result_vector.distortion:.6f}")
print(f"Compression ratio: {result_vector.compression_ratio:.2f}x")
print(f"Codebook size: {result_vector.num_codebook_entries}")
print(f"Vector dimension: {params_vector.vector_dim}")

# Example 4: Compare all methods
print("\n" + "="*60)
print("Example 4: Comparison of All SCL Methods")
print("="*60)

methods = ["scalar", "lloyd_max", "vector"]
results = {}

for method in methods:
    if method == "vector":
        params = SCLQuantizationParams(
            num_bits=2,
            method=method,
            vector_dim=4,
            distortion_metric="mse"
        )
    else:
        params = SCLQuantizationParams(
            num_bits=2,
            method=method,
            distortion_metric="mse"
        )
    
    result = scl_quantize(W, params, device)
    results[method] = result
    
    print(f"\n{method.upper()}:")
    print(f"  Rate: {result.rate:.3f} bits/sample")
    print(f"  Distortion: {result.distortion:.6f}")
    print(f"  Compression: {result.compression_ratio:.2f}x")

# Example 5: Rate-Distortion Trade-off
print("\n" + "="*60)
print("Example 5: Rate-Distortion Trade-off")
print("="*60)

bits_list = [1, 2, 3, 4, 6, 8]
distortion_list = []

for num_bits in bits_list:
    params = SCLQuantizationParams(
        num_bits=num_bits,
        method="lloyd_max",
        distortion_metric="mse"
    )
    result = scl_quantize(W, params, device)
    distortion_list.append(result.distortion)
    print(f"Bits: {num_bits}, Rate: {result.rate:.3f}, Distortion: {result.distortion:.6f}")

# Example 6: Evaluation Metrics
print("\n" + "="*60)
print("Example 6: Comprehensive Evaluation Metrics")
print("="*60)

from src.caldera.utils.metrics import evaluate_compression

# Evaluate Lloyd-Max quantization
metrics = evaluate_compression(
    W_original=W,
    W_compressed=result_lloyd.quantized,
    avg_bit_width=result_lloyd.rate,
    effective_rank=W.shape[0],  # Full rank for scalar quantization
    duality_gap=0.0,  # No optimization, so no duality gap
)

print(f"Bits per parameter: {metrics.bits_per_parameter:.3f}")
print(f"Relative error: {metrics.relative_error:.4f}")
print(f"Compression ratio: {metrics.compression_ratio:.2f}x")
print(f"Model size: {metrics.model_size_mb:.2f} MB")

# Example 7: Singular Value Spectra Comparison
print("\n" + "="*60)
print("Example 7: Singular Value Spectra Comparison")
print("="*60)

os.makedirs("plots", exist_ok=True)

from src.caldera.utils.metrics import compute_singular_values

sv_original = compute_singular_values(W)
sv_scalar = compute_singular_values(result_scalar.quantized)
sv_lloyd = compute_singular_values(result_lloyd.quantized)
sv_vector = compute_singular_values(result_vector.quantized)

# Plot comparison
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original
axes[0, 0].plot(sv_original, 'o-', label='Original', markersize=3)
axes[0, 0].set_title('Original')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Scalar
axes[0, 1].plot(sv_original, 'o-', label='Original', markersize=3, alpha=0.5)
axes[0, 1].plot(sv_scalar, 's-', label='Scalar Uniform', markersize=3)
axes[0, 1].set_title('Scalar Uniform Quantization')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Lloyd-Max
axes[1, 0].plot(sv_original, 'o-', label='Original', markersize=3, alpha=0.5)
axes[1, 0].plot(sv_lloyd, '^-', label='Lloyd-Max', markersize=3)
axes[1, 0].set_title('Lloyd-Max Quantization')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Vector
axes[1, 1].plot(sv_original, 'o-', label='Original', markersize=3, alpha=0.5)
axes[1, 1].plot(sv_vector, 'd-', label='Vector Quantization', markersize=3)
axes[1, 1].set_title('Vector Quantization (K-means)')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/scl_singular_values_comparison.png", dpi=300, bbox_inches='tight')
print("Saved singular value spectra comparison to plots/scl_singular_values_comparison.png")
plt.close()

print("\n" + "="*60)
print("All SCL baseline examples completed!")
print("="*60)

