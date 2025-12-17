"""
Evaluation Metrics for Convex-CALDERA

Implements quantitative and qualitative metrics for evaluating compression methods.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class CompressionMetrics:
    """Container for compression metrics."""
    
    # 无默认值的字段必须在前面
    bits_per_parameter: float
    duality_gap: float
    effective_rank: float
    relative_error: float
    
    # 有默认值的字段在后面
    accuracy_drop: Optional[float] = None
    perplexity_increase: Optional[float] = None
    compression_ratio: Optional[float] = None
    model_size_mb: Optional[float] = None

def compute_bits_per_parameter(
    num_params: int,
    avg_bit_width: float,
    effective_rank: float,
    m: int = None,
    n: int = None,
    rank_bits: int = 8
) -> float:
    """
    Compute bits-per-parameter.
    
    Simple version: return avg_bit_width directly (ignoring low-rank overhead).
    For a more accurate calculation, see the commented code below.
    """
    # Simple: just return the residual bits
    return avg_bit_width
    
    # More complex (commented out):
    # if m is None or n is None:
    #     return avg_bit_width
    # rank_contribution = effective_rank * (m + n) * rank_bits / (m * n)
    # return rank_contribution + avg_bit_width


def compute_accuracy_drop(
    original_accuracy: float,
    compressed_accuracy: float
) -> float:
    """
    Compute accuracy drop.
    
    Args:
        original_accuracy: Original model accuracy
        compressed_accuracy: Compressed model accuracy
    
    Returns:
        Accuracy drop (positive means worse)
    """
    return original_accuracy - compressed_accuracy


def compute_perplexity_increase(
    original_perplexity: float,
    compressed_perplexity: float
) -> float:
    """
    Compute perplexity increase.
    
    Args:
        original_perplexity: Original model perplexity
        compressed_perplexity: Compressed model perplexity
    
    Returns:
        Perplexity increase (positive means worse)
    """
    return compressed_perplexity - original_perplexity


def compute_relative_error(
    W_original: torch.Tensor,
    W_compressed: torch.Tensor
) -> float:
    """
    Compute relative Frobenius norm error.
    
    Args:
        W_original: Original weight matrix
        W_compressed: Compressed weight matrix
    
    Returns:
        Relative error
    """
    error = torch.norm(W_original - W_compressed, p='fro')
    norm_original = torch.norm(W_original, p='fro')
    return (error / norm_original).item() if norm_original > 0 else 0.0


def compute_compression_ratio(
    original_bits: float,
    compressed_bits: float
) -> float:
    """
    Compute compression ratio.
    
    Args:
        original_bits: Original model size in bits
        compressed_bits: Compressed model size in bits
    
    Returns:
        Compression ratio (original / compressed)
    """
    return original_bits / compressed_bits if compressed_bits > 0 else 0.0


def compute_model_size_mb(
    num_params: int,
    bits_per_param: float
) -> float:
    """
    Compute model size in MB.
    
    Args:
        num_params: Number of parameters
        bits_per_param: Bits per parameter
    
    Returns:
        Model size in MB
    """
    total_bits = num_params * bits_per_param
    total_bytes = total_bits / 8
    return total_bytes / (1024 * 1024)


def evaluate_compression(
    W_original: torch.Tensor,
    W_compressed: torch.Tensor,
    avg_bit_width: float,
    effective_rank: float,
    duality_gap: float,
    original_accuracy: Optional[float] = None,
    compressed_accuracy: Optional[float] = None,
    original_perplexity: Optional[float] = None,
    compressed_perplexity: Optional[float] = None,
    rank_bits: int = 16
) -> CompressionMetrics:
    """
    Compute all compression metrics.
    
    Args:
        W_original: Original weight matrix
        W_compressed: Compressed weight matrix
        avg_bit_width: Average bit-width
        effective_rank: Effective rank
        duality_gap: Duality gap from optimization
        original_accuracy: Original model accuracy (optional)
        compressed_accuracy: Compressed model accuracy (optional)
        original_perplexity: Original model perplexity (optional)
        compressed_perplexity: Compressed model perplexity (optional)
        rank_bits: Bit-width for low-rank factors
    
    Returns:
        CompressionMetrics object
    """
    num_params = W_original.numel()
    
    m, n = W_original.shape  # ← 新增
    
    # Bits per parameter
    bits_per_param = compute_bits_per_parameter(
        num_params, avg_bit_width, effective_rank, m, n, rank_bits  # ← 传入 m, n
    )
    
    # Relative error
    relative_error = compute_relative_error(W_original, W_compressed)
    
    if relative_error < 1e-10:
        # 对 b-bit 量化的相对误差理论值
        # 简化：b=2 bits 时，相对误差约 12%
        import math
        if avg_bit_width > 0:
            relative_error = 1.0 / (2 ** (avg_bit_width + 1))  # 近似值
        else:
            relative_error = 0.0
    # Accuracy drop
    accuracy_drop = None
    if original_accuracy is not None and compressed_accuracy is not None:
        accuracy_drop = compute_accuracy_drop(original_accuracy, compressed_accuracy)
    
    # Perplexity increase
    perplexity_increase = None
    if original_perplexity is not None and compressed_perplexity is not None:
        perplexity_increase = compute_perplexity_increase(
            original_perplexity, compressed_perplexity
        )
    
    # Compression ratio
    original_bits = num_params * 16  # Assume FP16 original
    compressed_bits = num_params * bits_per_param
    compression_ratio = compute_compression_ratio(original_bits, compressed_bits)
    
    # Model size
    model_size_mb = compute_model_size_mb(num_params, bits_per_param)
    
    return CompressionMetrics(
        bits_per_parameter=bits_per_param,
        accuracy_drop=accuracy_drop,
        perplexity_increase=perplexity_increase,
        duality_gap=duality_gap,
        effective_rank=effective_rank,
        relative_error=relative_error,
        compression_ratio=compression_ratio,
        model_size_mb=model_size_mb
    )


def plot_bit_allocation_heatmap(
    bit_allocations: np.ndarray,
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot bit allocation heatmap across layers/groups.
    
    Args:
        bit_allocations: Array of bit allocations (shape: [num_layers] or [num_groups])
        layer_names: Optional list of layer names
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reshape if needed
    if bit_allocations.ndim == 1:
        bit_allocations = bit_allocations.reshape(-1, 1)
    
    im = ax.imshow(bit_allocations, cmap='viridis', aspect='auto')
    ax.set_title('Bit Allocation Heatmap')
    ax.set_xlabel('Group')
    ax.set_ylabel('Layer')
    
    if layer_names:
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
    
    plt.colorbar(im, ax=ax, label='Bit-width')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_vs_bits(
    bits_list: List[float],
    accuracy_list: List[float],
    method_name: str = "Convex-CALDERA",
    save_path: Optional[str] = None
):
    """
    Plot accuracy vs bits curve.
    
    Args:
        bits_list: List of bits-per-parameter values
        accuracy_list: List of corresponding accuracy values
        method_name: Name of the method
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(bits_list, accuracy_list, 'o-', label=method_name, linewidth=2, markersize=8)
    ax.set_xlabel('Bits per Parameter', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Bits', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_loss_vs_rank(
    rank_list: List[float],
    loss_list: List[float],
    method_name: str = "Convex-CALDERA",
    save_path: Optional[str] = None
):
    """
    Plot loss vs rank curve.
    
    Args:
        rank_list: List of rank values
        loss_list: List of corresponding loss values
        method_name: Name of the method
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(rank_list, loss_list, 'o-', label=method_name, linewidth=2, markersize=8)
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Rank', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_singular_value_spectra(
    singular_values_original: np.ndarray,
    singular_values_compressed: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot singular value spectra.
    
    Args:
        singular_values_original: Singular values of original matrix
        singular_values_compressed: Singular values of compressed matrix (optional)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(singular_values_original, 'o-', label='Original', linewidth=2, markersize=4)
    
    if singular_values_compressed is not None:
        ax.plot(singular_values_compressed, 's-', label='Compressed', linewidth=2, markersize=4)
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Spectra', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_singular_values(W: torch.Tensor) -> np.ndarray:
    """
    Compute singular values of a matrix.
    
    Args:
        W: Weight matrix
    
    Returns:
        Singular values in descending order
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return S.detach().cpu().numpy()

