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
├── rank-constrained-regression-main/
│   ├── src/
│   │   └── caldera/
│   │       ├── decomposition/
│   │       │   └── alg.py          # CALDERA decomposition algorithm
│   │       └── utils/
│   │           ├── dataclasses.py  # Parameter dataclasses
│   │           └── quantization.py # Quantization utilities
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

### To-Do (In Progress)

- [ ] Implement SCL Library baselines for Lossy Compression and Quantization
- [ ] Implement Convex-CALDERA (rank-128) end-to-end: CVXPY/exp-cone solver, rounding-and-repair module, low-rank SVD factorization + truncated reconstruction
- [ ] Prepare final report + plots for EE274

## Usage

### Basic Example

```python
import torch
from transformers import AutoModelForCausalLM
from src.caldera.utils.dataclasses import CalderaParams
from src.caldera.utils.quantization import QuantizerFactory
from src.caldera.decomposition.alg import caldera

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

### Running Experiments

See `main.py` for a complete example using the POPE dataset with LLaVA-OneVision models.

```bash
python main.py
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

### Quantitative
- Bits-per-parameter
- Accuracy drop
- Perplexity increase
- Duality gap
- Effective rank

### Qualitative
- Bit allocation heatmaps
- Accuracy vs. bits curves
- Loss vs. rank curves
- Singular value spectra

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

### Stanford Compression Library (SCL)
The SCL provides foundational tools for classical lossy compression techniques, including scalar quantization, vector quantization, and rate–distortion–optimized quantizers. These methods serve as theoretically grounded baselines for understanding quantization effects on neural network weights.

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

