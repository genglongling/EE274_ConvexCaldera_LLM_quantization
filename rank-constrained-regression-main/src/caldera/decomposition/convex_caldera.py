"""
Convex-CALDERA: Convex Low-Rank + Low-Precision Compression

Implements Algorithm 1 from the paper with both penalty and constrained forms.
"""

import torch
import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
import warnings

from src.caldera.utils.dataclasses import CalderaDecomposition


@dataclass
class ConvexCalderaParams:
    """Parameters for Convex-CALDERA algorithm."""
    
    # Budget constraints
    B_tot: float = field(default=2.0, metadata={"help": "Total bit budget per parameter"})
    b_min: float = field(default=2.0, metadata={"help": "Minimum bit-width"})
    b_max: float = field(default=16.0, metadata={"help": "Maximum bit-width"})
    
    # Rank control (choose one)
    tau_star: Optional[float] = field(default=None, metadata={"help": "Nuclear norm bound (constrained form)"})
    mu: Optional[float] = field(default=0.1, metadata={"help": "Nuclear norm penalty weight (penalty form)"})
    
    # Regularization
    lambda_reg: float = field(default=0.01, metadata={"help": "Regularization weight for qg"})
    
    # Rate-distortion constants
    k: float = field(default=1.0, metadata={"help": "Rate-distortion constant k"})
    
    # Discrete bit-widths
    discrete_bits: List[int] = field(
        default_factory=lambda: [2, 3, 4, 8, 16],
        metadata={"help": "Discrete bit-widths for rounding"}
    )
    
    # Solver settings
    solver: str = field(default="SCS", metadata={"help": "CVXPY solver (SCS, MOSEK, ECOS)"})
    solver_verbose: bool = field(default=False)
    solver_tol: float = field(default=1e-4)
    
    # Verification
    tolerance: float = field(default=0.05, metadata={"help": "Tolerance for accuracy degradation"})
    apply_qat: bool = field(default=False, metadata={"help": "Apply quantization-aware training if needed"})
    
    # Quantization
    quantize_factors: bool = field(default=False, metadata={"help": "Quantize low-rank factors"})
    factor_bits: int = field(default=16, metadata={"help": "Bit-width for low-rank factors if quantized"})


@dataclass
class ConvexCalderaDecomposition:
    """Results from Convex-CALDERA decomposition."""
    
    # Decomposed weights
    L_star: torch.Tensor  # Low-rank component
    R_star: torch.Tensor  # Residual component
    W_compressed: torch.Tensor  # Final compressed weights
    
    # Bit allocations
    b_star: np.ndarray  # Continuous bit allocations
    b_discrete: np.ndarray  # Discrete bit allocations
    
    # Certificates
    avg_bit_width: float  # Average bit-width achieved
    effective_rank: float  # Effective rank
    duality_gap: float  # Duality gap
    residual_norm: float  # Residual norm
    
    # Optimization info
    solve_time: float
    solver_status: str
    objective_value: float
    
    # Group-wise information
    group_info: Dict = field(default_factory=dict)


def compute_hessian_and_sensitivities(
    W: torch.Tensor,
    H: torch.Tensor,
    calibration_data: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Step 1: Calibration - Compute Hessians, sensitivities, and rate-distortion constants.
    
    Args:
        W: Weight matrix
        H: Hessian matrix (or precomputed)
        calibration_data: Optional calibration data for computing H
    
    Returns:
        H_sqrt: Square root of Hessian
        kappa: Sensitivity parameter
        c: Rate-distortion constant
    """
    if H is None:
        if calibration_data is None:
            H = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        else:
            # Compute Gram matrix from calibration data
            H = torch.matmul(calibration_data.T, calibration_data)
    
    # Ensure H is positive definite
    H = (H + H.T) / 2  # Symmetrize
    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = torch.clamp(eigvals, min=1e-8)
    H = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    
    # Compute square root
    H_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    
    # Sensitivity: based on Frobenius norm of weight matrix
    kappa = torch.norm(W, p='fro').item()
    
    # Rate-distortion constant: based on weight variance
    c = torch.var(W).item() * 0.1  # Scaling factor
    
    return H_sqrt, kappa, c


def solve_convex_optimization(
    W: torch.Tensor,
    H_sqrt: torch.Tensor,
    kappa: float,
    c: float,
    params: ConvexCalderaParams,
    p: float = 1.0  # Group size (normalized)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Step 2: Convex Solve - Solve the convex optimization problem.
    
    Supports both penalty form (with mu) and constrained form (with tau_star).
    
    Args:
        W: Weight matrix
        H_sqrt: Square root of Hessian
        kappa: Sensitivity parameter
        c: Rate-distortion constant
        params: Convex-CALDERA parameters
        p: Group size (for multi-group case, default 1.0 for single group)
    
    Returns:
        L_star: Optimal low-rank component
        R_star: Optimal residual component
        b_star: Optimal bit allocation
        objective_value: Objective value at optimum
        status: Solver status
    """
    m, n = W.shape
    W_np = W.detach().cpu().numpy()
    H_sqrt_np = H_sqrt.detach().cpu().numpy()
    
    # Decision variables
    L = cp.Variable((m, n))
    R = cp.Variable((m, n))
    b = cp.Variable(1)
    q = cp.Variable(1)
    xi = cp.Variable(1)
    
    # Objective function
    residual_term = 0.5 * cp.sum_squares((W_np - L - R) @ H_sqrt_np.T)
    
    if params.tau_star is not None:
        # Constrained form: use nuclear norm constraint
        nuclear_norm_term = 0
        constraints = [cp.norm(L, "nuc") <= params.tau_star]
    else:
        # Penalty form: use nuclear norm penalty
        nuclear_norm_term = params.mu * cp.norm(L, "nuc")
        constraints = []
    
    q_term = params.lambda_reg * q
    
    objective = cp.Minimize(residual_term + nuclear_norm_term + q_term)
    
    # Constraints
    # Residual energy constraint: ||R||_F^2 <= xi <= kappa * q
    constraints.extend([
        cp.sum_squares(R) <= xi,
        xi <= kappa * q,
    ])
    
    # Exponential cone constraint: q >= c * exp(-k * b)
    # CVXPY exponential cone: (u, v, w) where u >= v * exp(w/v) for v > 0
    # We want: q >= c * exp(-k * b)
    # Reformulate: q >= c * exp(-k * b) = c * exp(-k * b / c * c / c)
    # Using ExpCone(q, c, -k * b) gives: q >= c * exp(-k * b / c)
    # For small k, this approximates q >= c * exp(-k * b)
    # To get exact: use auxiliary variable or approximate
    c_safe = cp.maximum(c, 1e-8)
    constraints.append(cp.constraints.ExpCone(q, c_safe, -params.k * b * c_safe))
    
    # Bit range constraint
    constraints.append(b >= params.b_min)
    constraints.append(b <= params.b_max)
    
    # Global budget constraint (for single group, p=1.0)
    constraints.append(p * b <= params.B_tot)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        if params.solver == "SCS":
            problem.solve(solver=cp.SCS, verbose=params.solver_verbose, eps=params.solver_tol)
        elif params.solver == "MOSEK":
            problem.solve(solver=cp.MOSEK, verbose=params.solver_verbose)
        elif params.solver == "ECOS":
            problem.solve(solver=cp.ECOS, verbose=params.solver_verbose)
        else:
            problem.solve(solver=cp.SCS, verbose=params.solver_verbose)
        
        status = problem.status
        obj_value = problem.value
        
        if status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(f"Solver status: {status}")
        
        L_star = L.value
        R_star = R.value
        b_star = b.value[0] if b.value is not None else params.b_min
        q_star = q.value[0] if q.value is not None else 0.0
        
        return L_star, R_star, b_star, obj_value, status
        
    except Exception as e:
        warnings.warn(f"Convex optimization failed: {e}. Using fallback.")
        # Fallback: use SVD-based initialization
        U, S, Vh = np.linalg.svd(W_np, full_matrices=False)
        rank = min(128, len(S))
        L_star = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]
        R_star = W_np - L_star
        b_star = params.b_min
        return L_star, R_star, b_star, float('inf'), "failed"


def round_bit_allocations(
    b_star: float,
    discrete_bits: List[int],
    B_tot: float,
    p: float = 1.0
) -> int:
    """
    Step 3: Rounding/Repair - Discretize continuous bit allocations.
    
    Args:
        b_star: Continuous bit allocation
        discrete_bits: Available discrete bit-widths
        B_tot: Total bit budget
        p: Group size
    
    Returns:
        b_discrete: Discrete bit allocation
    """
    # Find closest discrete bit-width
    b_discrete = min(discrete_bits, key=lambda x: abs(x - b_star))
    
    # If exceeds budget, reduce to next lower bit-width
    if p * b_discrete > B_tot:
        valid_bits = [b for b in discrete_bits if p * b <= B_tot]
        if valid_bits:
            b_discrete = max(valid_bits)
        else:
            b_discrete = min(discrete_bits)
    
    return b_discrete


def low_rank_factorization(
    L_star: np.ndarray,
    tau_star: Optional[float] = None,
    mu: Optional[float] = None,
    quantize: bool = False,
    factor_bits: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Step 4: Low-Rank Factorization - Factorize L_star using SVD and truncate.
    
    Args:
        L_star: Low-rank component from optimization
        tau_star: Nuclear norm bound (for constrained form)
        mu: Nuclear norm penalty (for penalty form)
        quantize: Whether to quantize factors
        factor_bits: Bit-width for factors if quantized
    
    Returns:
        L: Low-rank factor L
        R: Low-rank factor R (transpose of V)
        effective_rank: Effective rank
    """
    U, S, Vh = np.linalg.svd(L_star, full_matrices=False)
    
    # Truncate based on nuclear norm constraint
    if tau_star is not None:
        # Find rank such that nuclear norm <= tau_star
        cumsum_norm = np.cumsum(S)
        rank = np.searchsorted(cumsum_norm, tau_star) + 1
        rank = min(rank, len(S))
    else:
        # Use all singular values (or truncate based on threshold)
        threshold = S[0] * 1e-6
        rank = np.sum(S > threshold)
    
    # Truncate
    U_trunc = U[:, :rank]
    S_trunc = S[:rank]
    Vh_trunc = Vh[:rank, :]
    
    # Factorize: L_star = L @ R where L = U @ sqrt(S), R = sqrt(S) @ Vh
    sqrt_S = np.sqrt(S_trunc)
    L = U_trunc @ np.diag(sqrt_S)
    R = np.diag(sqrt_S) @ Vh_trunc
    
    # Convert to torch
    L = torch.from_numpy(L).float()
    R = torch.from_numpy(R).float()
    
    # Quantize if requested
    if quantize:
        # Simple uniform quantization
        L_scale = torch.max(torch.abs(L))
        R_scale = torch.max(torch.abs(R))
        
        L_quant = torch.round(L / L_scale * (2**(factor_bits-1) - 1))
        R_quant = torch.round(R / R_scale * (2**(factor_bits-1) - 1))
        
        L = L_quant / (2**(factor_bits-1) - 1) * L_scale
        R = R_quant / (2**(factor_bits-1) - 1) * R_scale
    
    effective_rank = rank
    
    return L, R, effective_rank


def quantize_residual(
    R_star: np.ndarray,
    b_discrete: int
) -> Tuple[torch.Tensor, float]:
    """
    Step 5: Quantization - Quantize residual R_star.
    
    Args:
        R_star: Residual component
        b_discrete: Discrete bit-width
    
    Returns:
        R_quantized: Quantized residual
        delta: Step size
    """
    R_star_torch = torch.from_numpy(R_star).float()
    
    # Compute step size: delta = 2*t / (2^b - 1)
    t = torch.max(torch.abs(R_star_torch))
    delta = 2 * t / (2**b_discrete - 1) if b_discrete < 16 else t / (2**15)
    
    # Integerize: R_int = round(R_star / delta)
    R_int = torch.round(R_star_torch / delta)
    
    # Clamp to valid range
    max_val = 2**(b_discrete - 1) - 1
    R_int = torch.clamp(R_int, -max_val, max_val)
    
    # Dequantize: R_quantized = delta * R_int
    R_quantized = delta * R_int
    
    return R_quantized, delta.item()


def compute_certificates(
    W: torch.Tensor,
    W_compressed: torch.Tensor,
    b_discrete: int,
    effective_rank: float,
    objective_value: float,
    p: float = 1.0
) -> Dict[str, float]:
    """
    Step 6: Verification - Compute certificates.
    
    Args:
        W: Original weight matrix
        W_compressed: Compressed weight matrix
        b_discrete: Discrete bit-width
        effective_rank: Effective rank
        objective_value: Objective value from optimization
        p: Group size
    
    Returns:
        Dictionary of certificates
    """
    # Average bit-width
    avg_bit_width = b_discrete  # For single group
    
    # Residual norm
    residual = W - W_compressed
    residual_norm = torch.norm(residual, p='fro').item()
    relative_error = residual_norm / torch.norm(W, p='fro').item()
    
    # Duality gap (approximate - would need dual solution for exact)
    # For now, use relative error as proxy
    duality_gap = relative_error
    
    certificates = {
        'avg_bit_width': avg_bit_width,
        'effective_rank': effective_rank,
        'residual_norm': residual_norm,
        'relative_error': relative_error,
        'duality_gap': duality_gap,
        'objective_value': objective_value,
    }
    
    return certificates


def convex_caldera(
    W: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    calibration_data: Optional[torch.Tensor] = None,
    params: Optional[ConvexCalderaParams] = None,
    device: str = "cuda",
    use_tqdm: bool = False
) -> ConvexCalderaDecomposition:
    """
    Main Convex-CALDERA algorithm (Algorithm 1).
    
    Args:
        W: Weight matrix to compress
        H: Optional precomputed Hessian
        calibration_data: Optional calibration data for computing H
        params: Convex-CALDERA parameters
        device: Device to run on
        use_tqdm: Whether to show progress bar
    
    Returns:
        ConvexCalderaDecomposition with compressed weights and certificates
    """
    import time
    start_time = time.time()
    
    if params is None:
        params = ConvexCalderaParams()
    
    # Move to device
    W = W.to(device).float()
    if H is not None:
        H = H.to(device).float()
    
    # Step 1: Calibration
    H_sqrt, kappa, c = compute_hessian_and_sensitivities(W, H, calibration_data)
    
    # Step 2: Convex Solve
    L_star_np, R_star_np, b_star, obj_value, status = solve_convex_optimization(
        W, H_sqrt, kappa, c, params
    )
    
    # Step 3: Rounding/Repair
    b_discrete = round_bit_allocations(b_star, params.discrete_bits, params.B_tot)
    
    # Step 4: Low-Rank Factorization
    L, R_lr, effective_rank = low_rank_factorization(
        L_star_np,
        params.tau_star,
        params.mu,
        params.quantize_factors,
        params.factor_bits
    )
    L = L.to(device)
    R_lr = R_lr.to(device)
    
    # Step 5: Quantization
    R_quantized, delta = quantize_residual(R_star_np, b_discrete)
    R_quantized = R_quantized.to(device)
    
    # Reconstruct compressed weights: W_compressed = L @ R_lr + delta * R_int
    # Note: In the algorithm, L_star is already the low-rank component,
    # so we use it directly and add quantized residual
    L_star = torch.from_numpy(L_star_np).float().to(device)
    W_compressed = L_star + R_quantized
    
    # Step 6: Verification
    certificates = compute_certificates(
        W, W_compressed, b_discrete, effective_rank, obj_value
    )
    
    solve_time = time.time() - start_time
    
    # Create decomposition object
    decomposition = ConvexCalderaDecomposition(
        L_star=L_star,
        R_star=R_quantized,
        W_compressed=W_compressed,
        b_star=np.array([b_star]),
        b_discrete=np.array([b_discrete]),
        avg_bit_width=certificates['avg_bit_width'],
        effective_rank=certificates['effective_rank'],
        duality_gap=certificates['duality_gap'],
        residual_norm=certificates['residual_norm'],
        solve_time=solve_time,
        solver_status=status,
        objective_value=certificates['objective_value'],
        group_info={
            'L': L,
            'R_lr': R_lr,
            'delta': delta,
            'certificates': certificates
        }
    )
    
    return decomposition

