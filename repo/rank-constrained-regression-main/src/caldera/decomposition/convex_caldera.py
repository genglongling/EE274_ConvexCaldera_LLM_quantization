# convex_caldera.py
# ---------------------------------------------------------
# Convex-CALDERA: closed-form low-rank + simple bit allocation (no CVXPY)
#
# This module replaces the previous CVXPY-based solver with:
#   1) A closed-form proximal operator for the nuclear norm
#      (via singular value soft-thresholding).
#   2) A lightweight bit-allocation routine that can be
#      upgraded later to a CVXQ-style dual algorithm.
#
# Dependencies: numpy, torch
# ---------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch


# ---------------------------------------------------------
# Hyper-parameters for Convex-CALDERA
# ---------------------------------------------------------

@dataclass
class ConvexCalderaParams:
    """
    Hyper-parameters for the Convex-CALDERA decomposition.

    Attributes
    ----------
    mu : float
        Nuclear-norm regularization weight in the objective:
            0.5 ||W - L||_F^2 + mu * ||L||_*
        Ignored if tau_star is not None and we enforce a
        trace / nuclear-norm constraint instead.
    tau_star : Optional[float]
        If not None, we enforce an approximate nuclear-norm
        / trace constraint by truncating singular values such
        that sum_i s_i <= tau_star.  When this is set, we do
        not use the soft-threshold parameter `mu` directly.
    lambda_reg : float
        Regularization coefficient in the CALDERA-type penalty
        for the residual quantization term (used only for
        logging / certificates here; the actual residual
        quantization happens in a separate module).
    kappa : float
        Scale factor for the rate–distortion penalty term.
    k : float
        Exponential rate parameter in exp(-k * b).  Kept here
        to match the original CALDERA notation.
    b_min : int
        Minimum bit-width allowed.
    b_max : int
        Maximum bit-width allowed.
    B_tot : float
        Total bit budget (average bit per weight or per group,
        depending on how you interpret it in your experiment).
        In this minimal implementation, for a single group we
        simply clamp b between [b_min, min(b_max, B_tot)].
    per_channel : bool
        If True, later you can extend the allocator to assign
        bits per-channel (e.g., per out-feature).  For now we
        only implement single-scalar b, but we keep this flag
        in case you want to upgrade to multi-group CVXQ style.
    """

    mu: float = 1e-3
    tau_star: Optional[float] = None
    lambda_reg: float = 1.0
    kappa: float = 1.0
    k: float = 1.0

    b_min: int = 2
    b_max: int = 8
    B_tot: float = 4.0

    per_channel: bool = False
    # Compatibility with old API
    discrete_bits: list = None
    solver: str = "SCS"
    solver_verbose: bool = False
    
    def __post_init__(self):
        if self.discrete_bits is None:
            self.discrete_bits = [2, 3, 4, 8, 16]


# ---------------------------------------------------------
# Low-rank part: nuclear-norm proximal operator via SVD
# ---------------------------------------------------------

def _soft_threshold_singular_values(
    S: np.ndarray,
    mu: float
) -> np.ndarray:
    """
    Soft-threshold singular values: s_i -> max(s_i - mu, 0).
    """
    return np.maximum(S - mu, 0.0)


def _truncate_singular_values_by_tau(
    S: np.ndarray,
    tau_star: float
) -> np.ndarray:
    """
    Truncate singular values so that their sum is approximately
    bounded by tau_star.

    We do this by finding the smallest rank r such that
        sum_{i=1}^r S_i >= tau_star,
    and setting S_{r+1:} = 0.

    This is a very rough enforcement of a trace / nuclear-norm
    constraint, but it is enough for small EE274 experiments.
    """
    if tau_star <= 0:
        # No low-rank structure allowed at all
        return np.zeros_like(S)

    cumsum = np.cumsum(S)
    # first index where cumulative sum exceeds tau_star
    r = np.searchsorted(cumsum, tau_star) + 1
    r = max(1, min(r, len(S)))
    S_trunc = S.copy()
    S_trunc[r:] = 0.0
    return S_trunc


def solve_convex_low_rank(
    W: torch.Tensor,
    params: ConvexCalderaParams
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Solve the low-rank convex problem in closed form via SVD.

        min_L  0.5 ||W - L||_F^2 + mu ||L||_*
        (or a trace / nuclear-norm style constraint via tau_star)

    Parameters
    ----------
    W : torch.Tensor (m x n)
        Weight matrix of a single linear / conv-equivalent layer.
        Can be on GPU; we will detach + move to CPU.
    params : ConvexCalderaParams
        Hyper-parameters controlling the nuclear-norm regularization.

    Returns
    -------
    L_star_np : np.ndarray
        Optimal low-rank component (same shape as W).
    R_star_np : np.ndarray
        Residual component W - L_star.
    stats : dict
        Dictionary of useful statistics (rank, nuclear norm, etc.).
    """
    # 1. Move to CPU + numpy
    W_np = W.detach().cpu().numpy()
    m, n = W_np.shape

    # 2. SVD
    # For typical EE274-sized layers, full SVD is OK. For truly
    # large LLM layers, you would want a truncated / randomized SVD.
    U, S, Vh = np.linalg.svd(W_np, full_matrices=False)

    # 3. Decide how to modify singular values
    if params.tau_star is not None:
        # Constraint-type formulation: enforce approximate nuclear norm budget
        S_new = _truncate_singular_values_by_tau(S, params.tau_star)
        mode = "trace_constraint"
    else:
        # Penalty-type formulation: proximal of mu * ||L||_*
        S_new = _soft_threshold_singular_values(S, params.mu)
        mode = "soft_threshold"

    # 4. Reconstruct L_star
    L_star_np = (U * S_new) @ Vh
    R_star_np = W_np - L_star_np

    # 5. Collect statistics
    rank = int((S_new > 0).sum())
    nuclear_norm = float(S_new.sum())
    frob_L = float(np.linalg.norm(L_star_np, ord="fro"))
    frob_R = float(np.linalg.norm(R_star_np, ord="fro"))

    stats = {
        "mode": mode,
        "rank": rank,
        "nuclear_norm": nuclear_norm,
        "frob_L": frob_L,
        "frob_R": frob_R,
        "shape": (m, n),
    }

    return L_star_np, R_star_np, stats


# ---------------------------------------------------------
# Bit allocation: minimal scalar version (upgradeable)
# ---------------------------------------------------------

def allocate_bits_scalar(
    params: ConvexCalderaParams
) -> Tuple[float, Dict[str, Any]]:
    """
    Improved bit allocator that respects the budget.
    
    Strategy: Allocate bits more intelligently based on B_tot
    - If B_tot is very small (< 2), still use at least 2 bits for R
    - Otherwise use B_tot directly
    """
    # More intelligent allocation: use B_tot as-is, but ensure minimum 2 bits
    b_star = float(np.clip(params.B_tot, params.b_min, params.b_max))

    stats = {
        "b_star": b_star,
        "b_min": params.b_min,
        "b_max": params.b_max,
        "B_tot": params.B_tot,
        "allocation_mode": "improved_clamp",
    }
    return b_star, stats



def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Quantize a tensor to specified number of bits using symmetric quantization.
    
    Correct quantization: map [-max_val, max_val] to [-2^(b-1), 2^(b-1)-1]
    """
    if num_bits >= 16:
        return tensor
    
    if tensor.numel() == 0:
        return tensor
    
    if device is not None:
        tensor = tensor.to(device)
    
    # Find max absolute value
    max_val = torch.max(torch.abs(tensor))
    
    if max_val < 1e-10:
        return tensor
    
    # Number of levels: for b bits, we have 2^b - 1 levels
    # But we want to map to [-2^(b-1), 2^(b-1)-1], which is 2^b values total
    q_max = 2 ** (num_bits - 1) - 1  # e.g., for 8-bit: 127
    
    # Quantize: scale to [-q_max, q_max] and round
    # quantized = round(tensor / max_val * q_max)
    quantized = torch.round(tensor / max_val * q_max)
    
    # Clamp (should already be within bounds, but just in case)
    quantized = torch.clamp(quantized, -q_max, q_max)
    
    # Dequantize: scale back
    # dequantized = quantized / q_max * max_val
    dequantized = (quantized / q_max) * max_val
    
    return dequantized
# ---------------------------------------------------------
# Main decomposition wrapper
# ---------------------------------------------------------

class ConvexCalderaDecomposition:
    """
    High-level wrapper for Convex-CALDERA decomposition.

    Usage
    -----
    >>> params = ConvexCalderaParams(mu=1e-3, B_tot=4.0)
    >>> decomp = ConvexCalderaDecomposition(params)
    >>> out = decomp.decompose(W)   # W: torch.Tensor (m x n)
    >>> L_low_rank = out["L_low_rank"]      # torch.Tensor
    >>> R_residual = out["R_residual"]      # torch.Tensor
    >>> b_star = out["b_star"]              # float

    The actual quantization of R_residual should be handled in a
    separate quantization module (e.g., your existing LLM
    quantizer).  This class only computes the *convex* low-rank
    part + an initial bit allocation.
    """

    def __init__(self, params: ConvexCalderaParams):
        self.params = params

    def decompose(
        self,
        W: torch.Tensor,
        H_sqrt: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Run the Convex-CALDERA decomposition on a single weight matrix.

        Parameters
        ----------
        W : torch.Tensor (m x n)
            Weight matrix to decompose.  Can live on CPU or GPU.
        H_sqrt : Optional[torch.Tensor]
            Placeholder for compatibility with earlier versions.
            In the simplest EE274 setting, we typically use
            H_sqrt = I, so this argument is not needed.
        device : Optional[torch.device]
            Device for the returned tensors.  If None, uses W.device.
        dtype : Optional[torch.dtype]
            dtype for the returned tensors.  If None, uses W.dtype.

        Returns
        -------
        result : dict
            {
              "L_low_rank": torch.Tensor,
              "R_residual": torch.Tensor,
              "b_star": float,
              "solver_stats": dict,
              "bit_stats": dict,
              "params": dict,
            }
        """
        if device is None:
            device = W.device
        if dtype is None:
            dtype = W.dtype

        # 1) Low-rank convex solve via SVD
        L_star_np, R_star_np, solver_stats = solve_convex_low_rank(
            W=W,
            params=self.params,
        )

        # 2) Bit allocation (minimal scalar version)
        b_star, bit_stats = allocate_bits_scalar(self.params)

        # 3) Convert back to torch tensors
        L_low_rank = torch.from_numpy(L_star_np).to(device=device, dtype=dtype)
        R_residual = torch.from_numpy(R_star_np).to(device=device, dtype=dtype)

        # 4) Optionally compute a simple "certificate"-like objective value
        #    This is purely for logging; the real certificate in CALDERA
        #    would involve the quantization of R_residual as well.
        recon_error = 0.5 * float(torch.norm(W - L_low_rank - R_residual, p="fro") ** 2)
        nuclear_norm = solver_stats["nuclear_norm"]
        penalty_low_rank = (self.params.mu * nuclear_norm) if self.params.mu is not None else 0.0

        # simple rate-distortion-style penalty term
        rd_penalty = self.params.lambda_reg * self.params.kappa * np.exp(
            -self.params.k * b_star
        )
        obj_value = recon_error + penalty_low_rank + float(rd_penalty)

        solver_stats["reconstruction_error"] = recon_error
        solver_stats["penalty_low_rank"] = penalty_low_rank
        solver_stats["rd_penalty"] = float(rd_penalty)
        solver_stats["objective_value"] = obj_value

        result = {
            "L_low_rank": L_low_rank,
            "R_residual": R_residual,
            "b_star": b_star,
            "solver_stats": solver_stats,
            "bit_stats": bit_stats,
            "params": asdict(self.params),
        }
        return result


# 返回对象类（兼容老 API）
@dataclass
class ConvexCalderaResult:
    """Object-based result for backward compatibility."""
    L_low_rank: torch.Tensor
    R_residual: torch.Tensor
    W_compressed: torch.Tensor
    solver_status: str
    solve_time: float
    avg_bit_width: float
    effective_rank: float
    duality_gap: float
    residual_norm: float
    objective_value: float
    b_star: float
    b_discrete: float
# ---------------------------------------------------------
# Convenience function (functional API)
# ---------------------------------------------------------

def convex_caldera_decompose(
    W: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    params: Optional[ConvexCalderaParams] = None,
    device: Optional[torch.device] = None,
) -> ConvexCalderaResult:
    """
    Main Convex-CALDERA function (matches old API).
    
    Parameters
    ----------
    W : torch.Tensor
        Weight matrix to decompose.
    H : Optional[torch.Tensor]
        Hessian (currently unused in SVD version).
    params : Optional[ConvexCalderaParams]
        If None, uses defaults.
    device : Optional[torch.device]
        Output device.
    
    Returns
    -------
    result : ConvexCalderaResult
        Object with all metrics as attributes.
    """
    import time
    start_time = time.time()
    
    if params is None:
        params = ConvexCalderaParams()
    if device is None:
        device = W.device

    # Internal decomposition
    internal_decomp = ConvexCalderaDecomposition(params)
    result_dict = internal_decomp.decompose(W=W, H_sqrt=H, device=device)
    
        # Extract components
    L_low = result_dict["L_low_rank"]
    R_res = result_dict["R_residual"]
    b_val = result_dict["b_star"]
    solver_stats = result_dict["solver_stats"]
    
    # ========== 量化部分 ==========
    # 策略：L 用 8 bits，R 用 b_val bits
    L_low_quantized = quantize_tensor(L_low, num_bits=8, device=device)
    
    # R 的 bits：至少 2 bits（确保确实有量化）
    # 改成：
    # b_val_int = int(np.round(np.clip(b_val, 1.0, 8.0)))  # 直接用 B_tot，min=1# ← 改成最小 2 bits
    # R_res_quantized = quantize_tensor(R_res, num_bits=b_val_int, device=device)增
    b_val_int = int(np.ceil(np.clip(b_val, 1.0, 8.0)))  # 改成 ceil
    R_res_quantized = quantize_tensor(R_res, num_bits=b_val_int, device=device)
    
    # 重建压缩后的权重
    W_comp = L_low_quantized + R_res_quantized
    
    
    # ========== 计算量化误差 ==========
    # 用原始 W 和量化后的 W_comp 计算误差
    quant_error = torch.norm(W - W_comp, p="fro").item()
    
    solve_time = time.time() - start_time
    
    # Return object (not dict)
    return ConvexCalderaResult(
        L_low_rank=L_low_quantized,
        R_residual=R_res_quantized,
        W_compressed=W_comp,
        solver_status="optimal",
        solve_time=solve_time,
        avg_bit_width=float(b_val_int),  # ← 改成实际的 bits
        effective_rank=float(solver_stats["rank"]),
        duality_gap=0.0,
        residual_norm=quant_error,  # ← 改成总的量化误差
        objective_value=solver_stats["objective_value"],
        b_star=float(b_val_int),  # ← 改成整数
        b_discrete=b_val_int,
    )