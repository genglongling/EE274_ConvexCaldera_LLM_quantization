# SCL Baselines Implementation Summary

## Overview

This document describes the implementation of SCL (Stanford Compression Library) baselines for lossy compression and quantization. These baselines provide classical quantization methods for comparison with CALDERA and Convex-CALDERA.

Based on: [EE274 Course Notes on Lossy Compression and Quantization](https://stanforddatacompressionclass.github.io/notes/lossy/quant.html)

## Implemented Methods

### 1. Scalar Uniform Quantization

**Method**: `scalar`

Independent quantization of each element using uniform quantization levels.

- **Algorithm**: Simple uniform binning
- **Rate**: `R = log2(N)` bits per sample, where `N = 2^num_bits`
- **Distortion**: MSE or MAE
- **Complexity**: O(n) - very fast
- **Use case**: Fast baseline, simple quantization

**Implementation**:
```python
quantized, codebook, step_size = scalar_quantize_uniform(data, num_bits=2)
```

### 2. Lloyd-Max Quantization

**Method**: `lloyd_max`

Optimal scalar quantization for MSE distortion using the Lloyd-Max algorithm (also known as the Generalized Lloyd algorithm or K-means for 1D).

- **Algorithm**: Iterative optimization
  1. Initialize codebook (uniform)
  2. Assign data to nearest codebook entry (partition)
  3. Update codebook as conditional expectation (centroids)
  4. Repeat until convergence
- **Rate**: `R = log2(N)` bits per sample
- **Distortion**: Minimizes MSE
- **Complexity**: O(n * k * iterations) where k is number of levels
- **Use case**: Optimal scalar quantization baseline

**Historical Note**: 
- First proposed by Stuart Lloyd in 1957 (Bell Labs)
- Published in 1982
- Independently developed by Joel Max in 1960

**Implementation**:
```python
quantized, codebook, distortion = scalar_quantize_lloyd_max(
    data, num_bits=2, max_iterations=100, tolerance=1e-6
)
```

### 3. Vector Quantization (K-means)

**Method**: `vector`

Vector quantization using the Generalized Lloyd algorithm (K-means clustering) to exploit correlations between dimensions.

- **Algorithm**: K-means clustering on vectorized data
  1. Reshape data into k-dimensional vectors
  2. Initialize codebook (random selection)
  3. Assign vectors to nearest centroid
  4. Update centroids as mean of assigned vectors
  5. Repeat until convergence
- **Rate**: `R = log2(N) / k` bits per sample, where k is vector dimension
- **Distortion**: MSE
- **Complexity**: O(n * k * N * iterations)
- **Use case**: Exploits correlations, better rate-distortion trade-off

**Advantages over Scalar Quantization**:
- Can exploit dependence between vector components
- More general decision regions
- Better rate-distortion trade-off for correlated data

**Implementation**:
```python
quantized, codebook, indices, distortion = vector_quantize_kmeans(
    data, num_bits=2, vector_dim=4, max_iterations=100
)
```

## API Usage

### Basic Usage

```python
from src.caldera.utils.scl_baselines import (
    scl_quantize,
    SCLQuantizationParams
)

# Configure parameters
params = SCLQuantizationParams(
    num_bits=2,              # 2-bit quantization
    method="lloyd_max",      # or "scalar", "vector"
    distortion_metric="mse"  # or "mae"
)

# Apply quantization
result = scl_quantize(weight_matrix, params, device="cuda")

# Access results
W_compressed = result.quantized
rate = result.rate
distortion = result.distortion
compression_ratio = result.compression_ratio
```

### Apply to Model Layers

```python
from src.caldera.utils.scl_baselines import apply_scl_baseline_to_model

# Apply to specific layers
results = apply_scl_baseline_to_model(
    model,
    layer_names=["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
    params=params,
    device="cuda"
)
```

## Rate-Distortion Theory

The SCL baselines implement fundamental quantization methods that demonstrate the rate-distortion trade-off:

- **Rate (R)**: Bits per sample used to represent the quantized data
- **Distortion (D)**: Measure of information loss (MSE or MAE)
- **Trade-off**: Higher rate → Lower distortion, Lower rate → Higher distortion

For scalar quantization:
- Uniform: `R = log2(N)`, simple but suboptimal
- Lloyd-Max: `R = log2(N)`, optimal for MSE (minimizes distortion for given rate)

For vector quantization:
- `R = log2(N) / k` where k is vector dimension
- Can achieve same distortion with lower rate by exploiting correlations

## Comparison with CALDERA Methods

| Method | Rate | Distortion | Low-Rank | Notes |
|--------|------|------------|----------|-------|
| Scalar Uniform | log2(N) | High | No | Fast, simple |
| Lloyd-Max | log2(N) | Lower | No | Optimal scalar |
| Vector Quantization | log2(N)/k | Lower | No | Exploits correlations |
| CALDERA | Variable | Lower | Yes | Low-rank + quantization |
| Convex-CALDERA | Variable | Lowest | Yes | Optimal with certificates |

## Theoretical Background

### Scalar Quantization

For a continuous source X, scalar quantization maps:
- `Q: R → C` where `C = {y_i}_{i=1}^N` is the codebook
- Partition: `{S_i}_{i=1}^N` such that `Q(x) = y_i` if `x ∈ S_i`
- Rate: `R = log2(N)` bits per sample

### Vector Quantization

For k-dimensional vectors:
- `Q: R^k → C` where `C = {y_i}_{i=1}^N` is the codebook
- Partition: `{S_i}_{i=1}^N` covering R^k
- Rate: `R = (log2 N) / k` bits per sample

**Key Insight**: Vector quantization can achieve the same distortion with lower rate by exploiting correlations between dimensions.

### Optimal Quantization

The Lloyd-Max algorithm finds optimal quantization levels that minimize MSE:
- **Optimal reconstruction level**: Conditional expectation `E[X | X ∈ S_i]`
- **Optimal decision regions**: Nearest neighbor assignment
- **Convergence**: Guaranteed to converge to local optimum

## References

1. [EE274 Course Notes: Lossy Compression and Quantization](https://stanforddatacompressionclass.github.io/notes/lossy/quant.html)
2. Lloyd, S. "Least squares quantization in PCM." IEEE Transactions on Information Theory, 1982.
3. Max, J. "Quantizing for minimum distortion." IRE Transactions on Information Theory, 1960.
4. Gersho, A., & Gray, R. M. "Vector Quantization and Signal Compression." Kluwer Academic Publishers, 1992.

