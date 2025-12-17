import torch

from dataclasses import field, dataclass
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm

from src.caldera.utils.dataclasses import *
from src.caldera.utils.quantization import *

def optimized_eigh(H):
    # Check if it's (approximately) an identity matrix
    if torch.allclose(H, torch.eye(H.shape[0], device=H.device, dtype=H.dtype), rtol=1e-5, atol=1e-8):
        # Create a named tuple or class to match torch.linalg.eigh return type
        from collections import namedtuple
        EighResult = namedtuple('EighResult', ['eigenvalues', 'eigenvectors'])
        
        return EighResult(
            eigenvalues=torch.ones(H.shape[0], device=H.device, dtype=H.dtype),
            eigenvectors=torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        )
    else:
        return torch.linalg.eigh(H)
def caldera(
    quant_params: CalderaParams,
    W: torch.Tensor,
    H: torch.Tensor = None,
    device: str = "cuda",
    use_tqdm: bool = True,
    scale_W: bool = True,
):
    """
    Runs the CALDERA algorithm, to decompose a weight matrix into Q + LR, where
    Q is full-rank, L and R are low-rank factors, and all matrices are in a low-
    precision format.
    """
    # scaling
    if scale_W:
        global_scale = W.square().mean().sqrt().item()
    else:
        global_scale = 1
    W = W / global_scale

    if H is None:
        H = torch.eye(W.shape[1]).to(device)

    # Compute the symmetric square root of H, because the data-aware objective can be formulated as min_{L, R} ||(W - LR - Q)H^{1/2}||_F^2.
    EigTuple = namedtuple("EigTuple", ["eigenvalues", "eigenvectors"])
    if not quant_params.activation_aware_LR:
        H_sqrt = H
        eigH = EigTuple(torch.ones(W.shape[1]).to(device), H)
    else:
        #symmetrize the matrix
        H = (H + H.T) / 2
        #import pdb; pdb.set_trace()
        #eigH = torch.linalg.eigh(H)
        eigH = optimized_eigh(H)
        eigvals = eigH.eigenvalues
        if eigvals.min() < quant_params.sigma_reg:
            H = H + (quant_params.sigma_reg - eigvals.min()) * torch.eye(
                H.shape[0], device=H.device, dtype=H.dtype
            )
            eigvals += quant_params.sigma_reg - eigvals.min()
            eigH = EigTuple(eigvals, eigH.eigenvectors)

        H_sqrt = (
            eigH.eigenvectors @ torch.diag(torch.sqrt(eigvals)) @ eigH.eigenvectors.T
        )

    # Initialization
    best_decomp = CalderaDecomposition(
        Q=torch.zeros_like(W).float(),
        L=torch.zeros(W.shape[0], quant_params.rank).to(device),
        R=torch.zeros(quant_params.rank, W.shape[1]).to(device),
    )

    best_decomp.scaleWH = None
    best_decomp.SU = torch.ones(W.shape[1]).to(W.dtype).to(W.device)
    best_decomp.SV = torch.ones(W.shape[0]).to(W.dtype).to(W.device)

    best_decomp.W = W.cpu()
    errors = {}

    # Track the quantization error in both Q and LR components
    for mtx in quant_params.update_order:
        errors[mtx] = []

    min_error = float("inf")
    curr_decomp = deepcopy(best_decomp)

    updated = {mtx: False for mtx in quant_params.update_order}

    to_iter = range(quant_params.iters)
    if use_tqdm:
        to_iter = tqdm(to_iter)
    for _ in to_iter:
        for mtx in quant_params.update_order:
            if mtx == "LR":
                maybe_update_LR(curr_decomp, quant_params, W, H_sqrt, eigH, device)
            elif mtx == "Q":
                maybe_update_Q(curr_decomp, quant_params, W, H, device)
            updated[mtx] = True

            errors[mtx].append(activation_aware_error(W, H, curr_decomp, device))
            if errors[mtx][-1] < min_error and all(updated.values()):
                min_error = errors[mtx][-1]
                best_decomp = deepcopy(curr_decomp)
    best_decomp.errors = errors

    # Update scales
    best_decomp.global_scale = global_scale
    return best_decomp


def maybe_update_LR(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    W: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device,
):
    if quant_params.compute_low_rank_factors:
        residual = W - caldera_info.Q
        update_LR(caldera_info, quant_params, residual, H_sqrt, eigH, device)


def update_LR(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    residual: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device,
):
    """
    Run LPLR on the residual (W - Q)
    """
    data_aware = quant_params.activation_aware_LR

    # Initialization of L, R
    L, R = LR_init(caldera_info, quant_params, H_sqrt, eigH, residual)

    if quant_params.L_bits < 16 or quant_params.R_bits < 16:
        quant_info_L = get_quant_info(
            quant_factory=quant_params.quant_factory_LR,
            bits=quant_params.L_bits,
            device=device,
        )
        quant_info_R = get_quant_info(
            quant_factory=quant_params.quant_factory_LR,
            bits=quant_params.R_bits,
            device=device,
        )

        best_L, best_R = L, R
        best_L_quant_out, best_R_quant_out = None, None
        best_error = float("inf")

        for _ in range(quant_params.lplr_iters):
            # Update L
            if data_aware:
                L = torch.linalg.lstsq((R @ H_sqrt).T, (residual @ H_sqrt).T)[0].T
                if torch.isnan(L).any():
                    L = (residual @ H_sqrt) @ torch.linalg.pinv(R @ H_sqrt)
            else:
                L = torch.linalg.lstsq(R.T, residual.T)[0].T
                if torch.isnan(R).any():
                    L = residual @ torch.linalg.pinv(R)

            quant_out_L = quantize_matrix(L.T, quant_params, quant_info_L)
            L = quant_out_L.A_hat.T

            # Update R
            R = torch.linalg.lstsq(L, residual)[0]
            if torch.isnan(R).any():
                R = torch.linalg.pinv(L) @ residual

            quant_out_R = quantize_matrix(R, quant_params, quant_info_R)
            R = quant_out_R.A_hat

            error = torch.linalg.matrix_norm((residual - L @ R) @ H_sqrt)  # / \
            #  torch.linalg.matrix_norm((residual + caldera_info.Q) @ H_sqrt)
            if error < best_error:
                best_L, best_R = L, R
                best_L_quant_out = quant_out_L
                best_R_quant_out = quant_out_R
                best_error = error

        caldera_info.L_idxs = best_L_quant_out.A_idxs
        caldera_info.R_idxs = best_R_quant_out.A_idxs
        caldera_info.L_scale = best_L_quant_out.scale
        caldera_info.R_scale = best_R_quant_out.scale

        L, R = best_L, best_R

    caldera_info.L = L
    caldera_info.R = R


def LR_init(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    H_sqrt: torch.Tensor,
    eigH: torch.Tensor,
    residual: torch.Tensor,
):
    """Does rank-constrained regression to minimize ||(residual - LR) eigH||_F^2 over L, R in closed-form."""

    if quant_params.activation_aware_LR:
        Y = residual @ H_sqrt @ eigH.eigenvectors
        if quant_params.rand_svd:
            q = min(quant_params.rank * 2, min(*caldera_info.W.shape))
            U, Sigma, V = torch.svd_lowrank(Y, q)
            Vh = V.T
        else:
            U, Sigma, Vh = torch.linalg.svd(Y, full_matrices=False)

        L = U[:, : quant_params.rank]
        R = (
            torch.diag(Sigma[: quant_params.rank])
            @ Vh[: quant_params.rank, :]
            @ torch.diag(1 / eigH.eigenvalues.sqrt())
            @ eigH.eigenvectors.T
        )
    else:
        if quant_params.rand_svd:
            q = min(quant_params.rank * 2, min(*caldera_info.W.shape))
            U, Sigma, V = torch.svd_lowrank(residual, q)
            Vh = V.T
        else:
            U, Sigma, Vh = torch.linalg.svd(residual, full_matrices=False)
        L = U[:, : quant_params.rank] @ torch.diag(Sigma[: quant_params.rank].sqrt())
        R = torch.diag(Sigma[: quant_params.rank].sqrt()) @ Vh[: quant_params.rank, :]
    return L, R


def get_quant_info(quant_factory: QuantizerFactory, bits: int, device: str):
    """Creates and returns quantization information using the specified parameters."""

    quantizer = quant_factory.get_quantizer(bits, device)
    return QuantInfo(quant=quantizer)


def quantize_matrix(A, quant_params, quant_info: QuantInfo = None):
    QuantReturn = namedtuple("QuantReturn", ["A_hat", "A_idxs", "scale"])
    quant_info.quant.block_size = A.shape[0] * A.shape[1]
    A_idxs, scales, shape = quant_info.quant.quantize_block(A)
    A_hat = quant_info.quant.dequantize_block(A_idxs, scales, shape)
    return QuantReturn(A_hat, A_idxs, scales)


def maybe_update_Q(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    W: torch.Tensor,
    H: torch.Tensor,
    device: str,
):

    if quant_params.compute_quantized_component:
        residual = W - caldera_info.L @ caldera_info.R
        if not quant_params.compute_low_rank_factors:
            residual = W
        update_Q_non_data_aware(caldera_info, quant_params, residual, device)


def update_Q_non_data_aware(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    residual: torch.Tensor,
    device: str,
):
    quant_info = get_quant_info(
        quant_factory=quant_params.quant_factory_Q,
        bits=quant_params.Q_bits,
        device=device,
    )

    quant_return = quantize_matrix(residual, quant_params, quant_info)
    caldera_info.Q = quant_return.A_hat
    caldera_info.Q_idxs = quant_return.A_idxs
    caldera_info.Q_scale = quant_return.scale


def activation_aware_error(
    W: torch.Tensor, H: torch.Tensor, caldera_info: CalderaDecomposition, device: str
):
    """Computes the activation-aware loss for a sublayer as tr((W - W_hat) H (W - W_hat).T) / tr(W H^1/2), 
        where H^1/2 is the symmetric square root.
    """

    W = W.to(device).float()
    W_hat = caldera_info.Q + caldera_info.L @ caldera_info.R
    W_hat *= caldera_info.global_scale

    error = (
        (torch.trace((W_hat - W) @ H @ (W_hat - W).T) / torch.trace(W @ H @ W.T))
        .sqrt()
        .item()
    )
    return error
