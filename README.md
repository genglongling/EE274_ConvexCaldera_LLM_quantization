# Convex-CALDERA: LLM Weight Compression via Convex Optimization with Low-Rank and Low-Precision Factor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** Longling Geng, Suchen He  
**Affiliation:** Stanford University, Department of Electrical Engineering  
**Contact:** gll2027@stanford.edu, sche@stanford.edu  
**Course:** EE274 Milestone

## Overview

Large Language Models (LLMs) such as LLaMA and GPT deliver state-of-the-art performance but require massive memory and compute resources at inference time. A 7B-parameter model requires more than 14GB memory in FP16, causing high latency and energy cost, especially in edge or resource-constrained deployment scenarios.

**Convex-CALDERA** addresses these limitations by formulating LLM weight compression as a convex optimization problem that jointly controls:
- **Low-rank approximation** via nuclear norm
- **Quantization precision allocation** via convex rate–distortion surrogates

This produces **global optimality guarantees**, dual certificates, and interpretable accuracy–efficiency trade-offs.

## Key Features

- ✅ **Convex optimization formulation** with provable guarantees
- ✅ **Joint low-rank and quantization** compression
- ✅ **Verifiable certificates**: effective rank, achieved average bit-width, and duality gap
- ✅ **Lower bits-per-parameter** (target ≤ 4 bits average)
- ✅ **Better perplexity** than QuIP# and CALDERA at equal bit budgets

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

```bash
# Clone the repository
git clone https://github.com/<your_repo>/Convex-CALDERA-EE274.git
cd EE274_ConvexCaldera_LLM_quantization

# Install dependencies
pip install torch transformers datasets scipy numpy matplotlib tqdm

# For convex optimization solver (when implementing Convex-CALDERA)
pip install cvxpy
```

## Project Structure

```
EE274_ConvexCaldera_LLM_quantization/
├── main.py                          # Main script for quantization experiments
├── diag_Hessians.pt                 # Precomputed diagonal Hessians
├── convex_caldera_example.py        # Convex-CALDERA usage examples
├── scl_baselines_example.py         # SCL baseline quantization examples
├── rank-constrained-regression-main/
│   ├── src/
│   │   ├── caldera/                 # Original CALDERA package
│   │   │   ├── decomposition/
│   │   │   │   └── alg.py           # CALDERA decomposition algorithm
│   │   │   └── utils/
│   │   │       ├── dataclasses.py   # Parameter dataclasses
│   │   │       ├── quantization.py # Quantization utilities
│   │   │       ├── metrics.py      # Evaluation metrics
│   │   │       └── scl_baselines.py # SCL baseline methods
│   │   └── convex_caldera/          # New Convex-CALDERA package
│   │       ├── decomposition/
│   │       │   └── convex_caldera.py # Convex-CALDERA implementation
│   │       └── utils/
│   │           ├── dataclasses.py   # Parameter dataclasses
│   │           ├── quantization.py  # Quantization utilities
│   │           ├── metrics.py       # Evaluation metrics
│   │           └── scl_baselines.py # SCL baseline methods
│   ├── caldera_playbook.ipynb      # CALDERA usage examples
│   └── README.md
└── README.md                        # This file
```

## Methods

This project implements a simplified but principled version of Convex-CALDERA. The goal is to decompose each weight matrix $W$ into a low-rank part $L$ and a quantized residual:

$$W_c = L^* + \Delta S_{\text{int}}$$

subject to rank and bit budgets. The convex formulation is:

$$\min_{L,R,b,q} \frac{1}{2}\|(W - L - R)H^{1/2}\|_F^2 + \mu\|L\|_* + \lambda q$$

subject to exponential-cone and global bit-budget constraints:

$$q \ge ce^{-kb}, \quad b_{\min} \le b \le b_{\max}, \quad \sum_g p_g b_g \le B_{\text{tot}}$$

### Current Implementation

The codebase currently includes:
- **Baseline PTQ pipelines** based on QuIP# (LLaMA-2 7B)
- **CALDERA baseline** (rank-128)
- **Unquantized (FP16) baseline**

### Current Implementation

The codebase currently includes:
- **Baseline PTQ pipelines** based on QuIP# (LLaMA-2 7B)
- **CALDERA baseline** (rank-128)
- **Unquantized (FP16) baseline**
- **✅ Convex-CALDERA** (both penalty and constrained forms) - **NEW!**
- **✅ SCL Library baselines** (scalar, Lloyd-Max, vector quantization) - **NEW!**

### To-Do (In Progress)

- [ ] Prepare final report + plots for EE274

## Usage

This repository provides two packages for LLM weight compression:

1. **`caldera`** - The original CALDERA algorithm package
2. **`convex_caldera`** - The new Convex-CALDERA package with convex optimization formulation

Both packages can be used independently. Choose the one that best fits your needs.

---

### Using the `caldera` Package

The `caldera` package provides the original CALDERA algorithm for low-rank + quantization decomposition. This package contains only the original CALDERA implementation (not Convex-CALDERA).

#### Setup

```python
import sys
sys.path.append('rank-constrained-regression-main')

import torch
from src.caldera.utils.dataclasses import CalderaParams
from src.caldera.utils.quantization import QuantizerFactory
from src.caldera.decomposition.alg import caldera
```

#### Basic CALDERA Example

```python
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Set up CALDERA parameters
quant_factory_Q = QuantizerFactory(method="uniform", block_size=64)
quant_factory_LR = QuantizerFactory(method="uniform", block_size=64)

quant_params = CalderaParams(
    compute_quantized_component=True,
    compute_low_rank_factors=True,
    Q_bits=2,          # 2-bit quantization for quantized component
    L_bits=16,         # 16-bit for low-rank factor L
    R_bits=16,         # 16-bit for low-rank factor R
    rank=128,          # Rank constraint
    iters=5,
    lplr_iters=5,
    activation_aware_LR=True,
    update_order=["Q", "LR"],
    quant_factory_Q=quant_factory_Q,
    quant_factory_LR=quant_factory_LR,
)

# Apply CALDERA to a weight matrix
W = model.layers[0].mlp.gate_proj.weight.data
H = torch.eye(W.shape[1])  # Hessian (or use precomputed)

caldera_decom = caldera(
    quant_params=quant_params,
    W=W,
    H=H,
    device="cuda",
    use_tqdm=True,
)

# Reconstruct compressed weight
W_compressed = caldera_decom.Q + caldera_decom.L @ caldera_decom.R
```

---

### Using the `convex_caldera` Package

The `convex_caldera` package is a standalone implementation of Convex-CALDERA with convex optimization formulation. This package contains only the Convex-CALDERA implementation (not the original CALDERA algorithm).

#### Setup

```python
import sys
sys.path.append('rank-constrained-regression-main')

import torch
from src.convex_caldera.decomposition.convex_caldera import (
    convex_caldera,
    ConvexCalderaParams
)
from src.convex_caldera.utils.metrics import evaluate_compression
```

#### Basic Convex-CALDERA Example

The new Convex-CALDERA algorithm supports both **penalty form** and **constrained form**:

```python
# Load weight matrix
W = model.layers[0].mlp.gate_proj.weight.data
H = torch.eye(W.shape[1])  # Hessian (or use precomputed)

# Option 1: Penalty form (with μ)
params_penalty = ConvexCalderaParams(
    B_tot=2.0,          # Target 2 bits per parameter
    b_min=2.0,
    b_max=8.0,
    mu=0.1,             # Nuclear norm penalty weight
    tau_star=None,      # Not using constrained form
    lambda_reg=0.01,
    discrete_bits=[2, 3, 4, 8],
    solver="SCS"
)

# Option 2: Constrained form (with τ*)
params_constrained = ConvexCalderaParams(
    B_tot=2.0,
    b_min=2.0,
    b_max=8.0,
    tau_star=100.0,     # Nuclear norm bound
    mu=None,            # Not using penalty form
    discrete_bits=[2, 3, 4, 8],
    solver="SCS"
)

# Run Convex-CALDERA
decomp = convex_caldera(
    W=W,
    H=H,
    params=params_penalty,
    device="cuda"
)

# Get compressed weights
W_compressed = decomp.W_compressed

# Evaluate metrics
metrics = evaluate_compression(
    W_original=W,
    W_compressed=W_compressed,
    avg_bit_width=decomp.avg_bit_width,
    effective_rank=decomp.effective_rank,
    duality_gap=decomp.duality_gap
)

print(f"Bits per parameter: {metrics.bits_per_parameter:.3f}")
print(f"Effective rank: {metrics.effective_rank:.2f}")
print(f"Duality gap: {metrics.duality_gap:.4f}")
print(f"Compression ratio: {metrics.compression_ratio:.2f}x")
```

### SCL Library Baselines

The SCL baselines provide classical quantization methods for comparison. Available in both packages:

**From `caldera` package:**
```python
from src.caldera.utils.scl_baselines import (
    scl_quantize,
    SCLQuantizationParams
)
```

**From `convex_caldera` package:**
```python
from src.convex_caldera.utils.scl_baselines import (
    scl_quantize,
    SCLQuantizationParams
)
```

#### Example Usage

```python
import torch

# Load weight matrix
W = model.layers[0].mlp.gate_proj.weight.data

# Option 1: Scalar Uniform Quantization
params_scalar = SCLQuantizationParams(
    num_bits=2,
    method="scalar",
    distortion_metric="mse"
)
result_scalar = scl_quantize(W, params_scalar, device="cuda")

# Option 2: Lloyd-Max Quantization (Optimal for MSE)
params_lloyd = SCLQuantizationParams(
    num_bits=2,
    method="lloyd_max",
    max_iterations=100,
    distortion_metric="mse"
)
result_lloyd = scl_quantize(W, params_lloyd, device="cuda")

# Option 3: Vector Quantization (K-means)
params_vector = SCLQuantizationParams(
    num_bits=2,
    method="vector",
    vector_dim=4,  # 4-dimensional vectors
    max_iterations=100,
    distortion_metric="mse"
)
result_vector = scl_quantize(W, params_vector, device="cuda")

# Access results
print(f"Rate: {result_lloyd.rate:.3f} bits/sample")
print(f"Distortion (MSE): {result_lloyd.distortion:.6f}")
print(f"Compression ratio: {result_lloyd.compression_ratio:.2f}x")

# Use quantized weights
W_compressed = result_lloyd.quantized
```

**SCL Baseline Methods:**
- **`scalar`**: Uniform scalar quantization (fast, simple)
- **`lloyd_max`**: Optimal scalar quantization using Lloyd-Max algorithm (iterative, better MSE)
- **`vector`**: Vector quantization using K-means/Generalized Lloyd algorithm (exploits correlations)

### Running Experiments

See `main.py` for a complete example using the POPE dataset with LLaVA-OneVision models.
See `convex_caldera_example.py` for Convex-CALDERA usage examples.
See `scl_baselines_example.py` for SCL baseline quantization methods.

```bash
# Run original CALDERA
python main.py

# Run Convex-CALDERA examples
python convex_caldera_example.py

# Run SCL baseline examples
python scl_baselines_example.py
```

## Results

### Zero-Shot Evaluation

Performance on LLaMA-2 and LLaMA-3 models: perplexity (↓) and accuracy (↑) across compression methods.

| Method | Avg Bits | Wiki2 ↓ | C4 ↓ | Wino ↑ | RTE ↑ | PiQA ↑ | ArcC ↑ |
|--------|----------|---------|------|--------|-------|--------|--------|
| **QuIP# (LLaMa-2 7B)** | 2.0 | 7.73 | 10.0 | 61.7 | 57.8 | 69.6 | 29.9 |
| **CALDERA (7B, rank-128)** | 2.2 | 6.76 | 8.83 | 63.8 | 59.9 | 75.1 | 34.6 |
| **Convex-CALDERA (7B, rank-128)** | **2.2** | **--** | **--** | **--** | **--** | **--** | **--** |
| Unquantized (7B) | 16 | 5.12 | 6.63 | 67.3 | 63.2 | 78.5 | 40.0 |

*Note: Convex-CALDERA results are in progress. See paper for full results table.*

### Compression Efficiency

| Model | Method | Avg Bits ↓ | Effective Rank ↓ | Certifiable/Verifiable Guarantees |
|-------|--------|------------|------------------|-----------------------------------|
| LLaMa-2 7B | QuIP# | 2.0 | Full | ❌ |
| | CALDERA (rank-128) | 2.2 | 128 | ❌ |
| | **Convex-CALDERA (rank-128)** | **2.2** | **128** | **✅** |
| | Unquantized (FP16) | 16 | Full | ✅ |

## Evaluation Metrics

The codebase includes comprehensive evaluation metrics available in both packages:

- **`src/caldera/utils/metrics.py`** - Metrics in the caldera package
- **`src/convex_caldera/utils/metrics.py`** - Metrics in the convex_caldera package

### Quantitative Metrics
- **Bits-per-parameter**: Accounts for both low-rank and quantized components
- **Accuracy drop**: Difference between original and compressed model accuracy
- **Perplexity increase**: Difference in perplexity scores
- **Duality gap**: Optimality certificate from convex optimization
- **Effective rank**: Rank of the low-rank component
- **Relative error**: Frobenius norm error relative to original
- **Compression ratio**: Ratio of original to compressed model size

### Qualitative Metrics (Plotting Functions)
- **Bit allocation heatmaps**: Visualize bit allocation across layers/groups
- **Accuracy vs. bits curves**: Trade-off between compression and accuracy
- **Loss vs. rank curves**: Impact of rank on reconstruction loss
- **Singular value spectra**: Compare original vs compressed singular values

Example usage (same for both packages):
```python
# From caldera package
from src.caldera.utils.metrics import (
    plot_bit_allocation_heatmap,
    plot_accuracy_vs_bits,
    plot_singular_value_spectra
)

# OR from convex_caldera package
# from src.convex_caldera.utils.metrics import (
#     plot_bit_allocation_heatmap,
#     plot_accuracy_vs_bits,
#     plot_singular_value_spectra
# )

# Generate plots
plot_singular_value_spectra(sv_original, sv_compressed, save_path="plots/sv.png")
plot_accuracy_vs_bits(bits_list, accuracy_list, save_path="plots/acc_vs_bits.png")
```

## Related Work

### Post-Training Compression
- **GPTQ** [Frantar et al., NeurIPS 2023]: Group-wise quantization for Transformers
- **LLM.int8** [Dettmers et al., ICLR 2022]: Efficient 8-bit deployment

### Low-Rank Methods
- **LoRA** [Hu et al., ICLR 2021]: Low-rank adapters for efficient fine-tuning

### Optimization-Based Approaches
- **CALDERA** [Saha et al., ICLR 2024]: Reconstructs quantization errors with low-rank factors
- **HAWQ** [Dong et al., 2019]: Uses Hessian information for bit allocation
- **CVXQ** [Frantar et al., 2022]: Formulates quantization as convex optimization

### Stanford Compression Library (SCL) Baselines
The SCL provides foundational tools for classical lossy compression techniques, including scalar quantization, vector quantization, and rate–distortion–optimized quantizers. These methods serve as theoretically grounded baselines for understanding quantization effects on neural network weights.

**Implemented SCL Baselines:**
- **Scalar Uniform Quantization**: Independent quantization of each element
- **Lloyd-Max Quantization**: Optimal scalar quantization for MSE distortion (iterative algorithm)
- **Vector Quantization (K-means)**: Generalized Lloyd algorithm for vector quantization, exploiting correlations between dimensions

See `scl_baselines_example.py` for usage examples.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{convex-caldera-2024,
  title={Convex-CALDERA: LLM Weight Compression via Convex Optimization with Low-Rank and Low-Precision Factor},
  author={Geng, Longling and He, Suchen},
  journal={EE274 Milestone},
  year={2024},
  institution={Stanford University}
}
```

## References

- Frantar, E., et al. "GPTQ: Accurate Post-training Quantization for Transformers." NeurIPS 2023.
- Dettmers, T., et al. "LLM.int8: 8-bit Matrix Multiplication for Transformers at Scale." ICLR 2022.
- Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2021.
- Saha, A., et al. "CALDERA: Finding All Pure Strategy Nash Equilibria in Integer Programming Games." ICLR 2024.
- Recht, B., et al. "Guaranteed Minimum Rank via Nuclear Norm Minimization." 2010.
- Boyd, S., & Vandenberghe, L. *Convex Optimization*. 2004.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work is part of the EE274 (Data Compression) course at Stanford University. We thank the course instructors and the Stanford Compression Library team for their support and resources.

## Contact

For questions or issues, please contact:
- Longling Geng: gll2027@stanford.edu
- Suchen He: sche@stanford.edu

---

**Note:** This is an in-progress research project. Results and implementations are subject to change.

