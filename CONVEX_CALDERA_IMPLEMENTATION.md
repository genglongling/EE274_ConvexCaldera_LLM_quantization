# Convex-CALDERA Implementation Summary

## Overview

This document describes the implementation of the Convex-CALDERA algorithm as described in Algorithm 1 of the paper. The implementation supports both **penalty form** and **constrained form** of the optimization problem.

## Files Created

1. **`rank-constrained-regression-main/src/caldera/decomposition/convex_caldera.py`**
   - Main implementation of Convex-CALDERA algorithm
   - Implements all 7 steps from Algorithm 1
   - Supports both penalty and constrained forms

2. **`rank-constrained-regression-main/src/caldera/utils/metrics.py`**
   - Comprehensive evaluation metrics
   - Quantitative metrics: bits-per-parameter, accuracy drop, perplexity, duality gap, effective rank
   - Qualitative plotting functions: heatmaps, curves, singular value spectra

3. **`convex_caldera_example.py`**
   - Example usage script demonstrating both forms
   - Shows how to use evaluation metrics
   - Generates plots

## Algorithm Implementation

### Step 1: Calibration
- Computes Hessian square root `H_sqrt`
- Computes sensitivity parameter `κ`
- Computes rate-distortion constant `c`

### Step 2: Convex Solve
Solves the optimization problem:

**Penalty Form:**
```
min Σ_g [1/2 ||(W_g - L_g - R_g)H_g^(1/2)||_F^2 + μ||L_g||_* + λq_g]
```

**Constrained Form:**
```
min Σ_g [1/2 ||(W_g - L_g - R_g)H_g^(1/2)||_F^2 + λq_g]
s.t. ||L_g||_* ≤ τ*
```

Both forms include:
- Residual energy constraint: `||R_g||_F^2 ≤ ξ_g ≤ κ_g q_g`
- Exponential cone constraint: `q_g ≥ c_g exp(-k b_g)`
- Bit range: `b_min ≤ b_g ≤ b_max`
- Global budget: `Σ_g p_g b_g ≤ B_tot`

### Step 3: Rounding/Repair
- Discretizes continuous bit allocations `b*` to discrete values `{2, 3, 4, 8, ...}`
- Adjusts if budget is exceeded

### Step 4: Low-Rank Factorization
- Computes SVD of `L*`
- Truncates based on nuclear norm constraint
- Optionally quantizes factors

### Step 5: Quantization
- Computes step size: `Δ = 2t / (2^b - 1)`
- Integerizes residual: `R_int = round(R* / Δ)`
- Reconstructs: `W = L* + Δ R_int`

### Step 6: Verification
- Computes certificates: average bit-width, effective rank, duality gap
- Computes residual norms

### Step 7: Return
- Returns compressed weights and certificates

## API Usage

### Basic Usage

```python
from src.caldera.decomposition.convex_caldera import (
    convex_caldera,
    ConvexCalderaParams
)

# Configure parameters
params = ConvexCalderaParams(
    B_tot=2.0,      # Target bits per parameter
    mu=0.1,         # Penalty weight (penalty form)
    # OR
    tau_star=100.0, # Nuclear norm bound (constrained form)
    discrete_bits=[2, 3, 4, 8],
    solver="SCS"
)

# Run algorithm
decomp = convex_caldera(
    W=weight_matrix,
    H=hessian_matrix,
    params=params,
    device="cuda"
)

# Access results
W_compressed = decomp.W_compressed
avg_bit_width = decomp.avg_bit_width
effective_rank = decomp.effective_rank
duality_gap = decomp.duality_gap
```

### Evaluation Metrics

```python
from src.caldera.utils.metrics import evaluate_compression

metrics = evaluate_compression(
    W_original=W,
    W_compressed=decomp.W_compressed,
    avg_bit_width=decomp.avg_bit_width,
    effective_rank=decomp.effective_rank,
    duality_gap=decomp.duality_gap
)

print(f"Bits per parameter: {metrics.bits_per_parameter}")
print(f"Compression ratio: {metrics.compression_ratio}x")
print(f"Relative error: {metrics.relative_error}")
```

### Plotting

```python
from src.caldera.utils.metrics import (
    plot_singular_value_spectra,
    plot_accuracy_vs_bits,
    plot_bit_allocation_heatmap
)

# Plot singular values
plot_singular_value_spectra(sv_original, sv_compressed, save_path="sv.png")

# Plot accuracy vs bits
plot_accuracy_vs_bits(bits_list, accuracy_list, save_path="acc_vs_bits.png")

# Plot bit allocation
plot_bit_allocation_heatmap(bit_allocations, save_path="bit_alloc.png")
```

## Dependencies

- `torch`: PyTorch for tensor operations
- `cvxpy`: Convex optimization solver
- `numpy`: Numerical operations
- `matplotlib`: Plotting (for metrics)

Install with:
```bash
pip install torch cvxpy numpy matplotlib
```

## Solver Options

The implementation supports multiple CVXPY solvers:
- **SCS** (default): Splitting Conic Solver - fast, open-source
- **MOSEK**: Commercial solver (requires license)
- **ECOS**: Embedded Conic Solver - open-source

## Notes

1. The exponential cone constraint is implemented using CVXPY's `ExpCone` constraint.
2. For large matrices, the convex optimization step may be slow. Consider using smaller matrices or adjusting solver tolerance.
3. The implementation currently supports single-group compression. Multi-group support can be added by extending the optimization problem.
4. The duality gap is approximated using relative error. For exact duality gap, the dual solution would need to be extracted from the solver.

## Future Improvements

- [ ] Multi-group support for layer-wise compression
- [ ] Exact duality gap computation
- [ ] Integration with quantization-aware training (QAT)
- [ ] Support for different Hessian computation methods
- [ ] Batch processing for multiple layers

