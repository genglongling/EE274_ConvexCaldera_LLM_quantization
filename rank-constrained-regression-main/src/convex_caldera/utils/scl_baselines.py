"""
SCL (Stanford Compression Library) Baselines for Lossy Compression and Quantization

Implements scalar quantization, vector quantization, and rate-distortion optimized
quantization as baselines for comparison with CALDERA and Convex-CALDERA.

Based on: https://stanforddatacompressionclass.github.io/notes/lossy/quant.html
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import warnings


@dataclass
class SCLQuantizationParams:
    """Parameters for SCL quantization baselines."""
    
    # Quantization parameters
    num_bits: int = field(default=2, metadata={"help": "Number of bits for quantization"})
    method: str = field(
        default="scalar",
        metadata={"help": "Quantization method: 'scalar', 'vector', 'lloyd_max'"}
    )
    
    # Vector quantization parameters
    vector_dim: int = field(default=1, metadata={"help": "Dimension for vector quantization"})
    num_codebook_entries: Optional[int] = field(
        default=None,
        metadata={"help": "Number of codebook entries (auto if None based on bits)"}
    )
    
    # Lloyd-Max / K-means parameters
    max_iterations: int = field(default=100, metadata={"help": "Max iterations for Lloyd-Max"})
    tolerance: float = field(default=1e-6, metadata={"help": "Convergence tolerance"})
    random_seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    
    # Distortion metric
    distortion_metric: str = field(
        default="mse",
        metadata={"help": "Distortion metric: 'mse' or 'mae'"}
    )


@dataclass
class SCLQuantizationResult:
    """Results from SCL quantization."""
    
    # Quantized data
    quantized: torch.Tensor
    codebook: torch.Tensor
    indices: torch.Tensor
    
    # Metrics
    rate: float  # Bits per sample
    distortion: float  # MSE or MAE
    compression_ratio: float
    
    # Additional info
    num_codebook_entries: int
    method: str


def scalar_quantize_uniform(
    data: torch.Tensor,
    num_bits: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Scalar uniform quantization.
    
    Quantizes each element independently using uniform quantization.
    
    Args:
        data: Input tensor to quantize
        num_bits: Number of bits per element
        min_val: Minimum value (auto if None)
        max_val: Maximum value (auto if None)
    
    Returns:
        quantized: Quantized tensor
        codebook: Codebook values
        step_size: Quantization step size
    """
    if min_val is None:
        min_val = data.min().item()
    if max_val is None:
        max_val = data.max().item()
    
    # Number of quantization levels
    num_levels = 2 ** num_bits
    
    # Step size
    step_size = (max_val - min_val) / (num_levels - 1)
    
    # Quantize: round to nearest level
    quantized_indices = torch.clamp(
        torch.round((data - min_val) / step_size),
        0,
        num_levels - 1
    ).long()
    
    # Create codebook
    codebook = torch.linspace(min_val, max_val, num_levels)
    
    # Dequantize
    quantized = codebook[quantized_indices]
    
    return quantized, codebook, step_size


def scalar_quantize_lloyd_max(
    data: torch.Tensor,
    num_bits: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Scalar quantization using Lloyd-Max algorithm (optimal for MSE).
    
    Implements the Lloyd-Max algorithm to find optimal quantization levels
    that minimize MSE distortion.
    
    Args:
        data: Input tensor (flattened)
        num_bits: Number of bits
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        random_seed: Random seed
    
    Returns:
        quantized: Quantized tensor
        codebook: Optimal codebook
        distortion: Final MSE distortion
    """
    np.random.seed(random_seed)
    
    # Flatten data
    data_flat = data.flatten().cpu().numpy()
    num_levels = 2 ** num_bits
    
    # Initialize codebook (uniform)
    min_val, max_val = data_flat.min(), data_flat.max()
    codebook = np.linspace(min_val, max_val, num_levels)
    
    distortion_prev = float('inf')
    
    for iteration in range(max_iterations):
        # Step 1: Assign data to nearest codebook entry (partition)
        distances = np.abs(data_flat[:, np.newaxis] - codebook)
        assignments = np.argmin(distances, axis=1)
        
        # Step 2: Update codebook (centroids = conditional expectation)
        codebook_new = np.zeros_like(codebook)
        for i in range(num_levels):
            mask = (assignments == i)
            if np.sum(mask) > 0:
                # Optimal reconstruction level is the mean (for MSE)
                codebook_new[i] = data_flat[mask].mean()
            else:
                # Keep old value if no assignments
                codebook_new[i] = codebook[i]
        
        # Compute distortion
        quantized_flat = codebook_new[assignments]
        distortion = np.mean((data_flat - quantized_flat) ** 2)
        
        # Check convergence
        if abs(distortion_prev - distortion) < tolerance:
            break
        
        codebook = codebook_new
        distortion_prev = distortion
    
    # Quantize
    distances = np.abs(data_flat[:, np.newaxis] - codebook)
    assignments = np.argmin(distances, axis=1)
    quantized_flat = codebook[assignments]
    
    # Reshape to original shape
    quantized = torch.from_numpy(quantized_flat.reshape(data.shape)).float()
    codebook_torch = torch.from_numpy(codebook).float()
    
    return quantized, codebook_torch, distortion


def vector_quantize_kmeans(
    data: torch.Tensor,
    num_bits: int,
    vector_dim: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Vector quantization using K-means (Generalized Lloyd algorithm).
    
    Implements vector quantization by grouping data into vectors and
    applying K-means clustering.
    
    Args:
        data: Input tensor
        num_bits: Number of bits per vector
        vector_dim: Dimension of each vector
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        random_seed: Random seed
    
    Returns:
        quantized: Quantized tensor
        codebook: Codebook (centroids)
        indices: Assignment indices
        distortion: Final MSE distortion
    """
    np.random.seed(random_seed)
    
    # Reshape data into vectors
    data_flat = data.flatten().cpu().numpy()
    num_samples = len(data_flat)
    
    # Pad if necessary
    if num_samples % vector_dim != 0:
        padding = vector_dim - (num_samples % vector_dim)
        data_flat = np.pad(data_flat, (0, padding), mode='constant', constant_values=0)
        num_samples = len(data_flat)
    
    # Reshape to vectors
    vectors = data_flat.reshape(-1, vector_dim)
    num_vectors = vectors.shape[0]
    
    # Number of codebook entries
    num_codebook = 2 ** num_bits
    
    if num_codebook > num_vectors:
        warnings.warn(f"Codebook size {num_codebook} > number of vectors {num_vectors}. Using {num_vectors}.")
        num_codebook = num_vectors
    
    # Initialize centroids (random selection)
    indices = np.random.choice(num_vectors, num_codebook, replace=False)
    centroids = vectors[indices].copy()
    
    distortion_prev = float('inf')
    
    for iteration in range(max_iterations):
        # Step 1: Assign vectors to nearest centroid
        distances = np.sum((vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        
        # Step 2: Update centroids
        centroids_new = np.zeros_like(centroids)
        for i in range(num_codebook):
            mask = (assignments == i)
            if np.sum(mask) > 0:
                centroids_new[i] = vectors[mask].mean(axis=0)
            else:
                centroids_new[i] = centroids[i]
        
        # Compute distortion
        quantized_vectors = centroids_new[assignments]
        distortion = np.mean((vectors - quantized_vectors) ** 2)
        
        # Check convergence
        if abs(distortion_prev - distortion) < tolerance:
            break
        
        centroids = centroids_new
        distortion_prev = distortion
    
    # Final assignment
    distances = np.sum((vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    quantized_vectors = centroids[assignments]
    
    # Reshape back
    quantized_flat = quantized_vectors.flatten()[:data.numel()]
    quantized = torch.from_numpy(quantized_flat.reshape(data.shape)).float()
    codebook = torch.from_numpy(centroids).float()
    indices_torch = torch.from_numpy(assignments).long()
    
    return quantized, codebook, indices_torch, distortion


def compute_distortion(
    original: torch.Tensor,
    quantized: torch.Tensor,
    metric: str = "mse"
) -> float:
    """
    Compute distortion between original and quantized tensors.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
        metric: Distortion metric ('mse' or 'mae')
    
    Returns:
        Distortion value
    """
    if metric == "mse":
        return torch.mean((original - quantized) ** 2).item()
    elif metric == "mae":
        return torch.mean(torch.abs(original - quantized)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def scl_quantize(
    data: torch.Tensor,
    params: Optional[SCLQuantizationParams] = None,
    device: str = "cpu"
) -> SCLQuantizationResult:
    """
    Main SCL quantization function.
    
    Applies scalar or vector quantization based on parameters.
    
    Args:
        data: Input tensor to quantize
        params: SCL quantization parameters
        device: Device to run on
    
    Returns:
        SCLQuantizationResult with quantized data and metrics
    """
    if params is None:
        params = SCLQuantizationParams()
    
    data = data.to(device)
    
    if params.method == "scalar":
        # Uniform scalar quantization
        quantized, codebook, step_size = scalar_quantize_uniform(
            data, params.num_bits
        )
        indices = None
        distortion = compute_distortion(data, quantized, params.distortion_metric)
        
    elif params.method == "lloyd_max":
        # Lloyd-Max scalar quantization (optimal for MSE)
        quantized, codebook, distortion = scalar_quantize_lloyd_max(
            data,
            params.num_bits,
            params.max_iterations,
            params.tolerance,
            params.random_seed
        )
        indices = None
        
    elif params.method == "vector":
        # Vector quantization (K-means)
        quantized, codebook, indices, distortion = vector_quantize_kmeans(
            data,
            params.num_bits,
            params.vector_dim,
            params.max_iterations,
            params.tolerance,
            params.random_seed
        )
    else:
        raise ValueError(f"Unknown method: {params.method}")
    
    # Compute rate (bits per sample)
    num_codebook_entries = len(codebook)
    if params.method == "vector":
        # For vector quantization: rate = log2(N) / vector_dim bits per sample
        # where N is codebook size, vector_dim is dimension of each vector
        rate = np.log2(num_codebook_entries) / params.vector_dim
    else:
        # For scalar quantization: rate = log2(N) bits per sample
        rate = np.log2(num_codebook_entries)
    
    # Compute compression ratio
    original_bits = data.numel() * 16  # Assume FP16
    compressed_bits = data.numel() * rate
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0.0
    
    return SCLQuantizationResult(
        quantized=quantized,
        codebook=codebook,
        indices=indices,
        rate=rate,
        distortion=distortion,
        compression_ratio=compression_ratio,
        num_codebook_entries=num_codebook_entries,
        method=params.method
    )


def apply_scl_baseline_to_model(
    model: torch.nn.Module,
    layer_names: Optional[List[str]] = None,
    params: Optional[SCLQuantizationParams] = None,
    device: str = "cpu"
) -> Dict[str, SCLQuantizationResult]:
    """
    Apply SCL quantization baseline to model layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to quantize (all if None)
        params: SCL quantization parameters
        device: Device to run on
    
    Returns:
        Dictionary mapping layer names to quantization results
    """
    if params is None:
        params = SCLQuantizationParams()
    
    results = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if layer_names is None or name in layer_names:
                weight = module.weight.data
                
                # Apply quantization
                result = scl_quantize(weight, params, device)
                
                # Replace weight with quantized version
                module.weight.data = result.quantized.to(device)
                
                results[name] = result
                
                print(f"Applied {params.method} quantization to {name}: "
                      f"rate={result.rate:.3f} bits, distortion={result.distortion:.6f}")
    
    return results

