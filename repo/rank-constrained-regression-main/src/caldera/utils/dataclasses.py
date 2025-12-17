import torch

from src.caldera.utils.quantization import (
    QuantizerFactory,
    AbstractQuantizer,
    LowMemoryQuantizer,
)
from dataclasses import field, dataclass


@dataclass
class CalderaParams:
    """Parameters for the CALDERA decomposition."""

    compute_quantized_component: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether the decomposition should include a quantized full-size"
                "component (denoted Q)."
            )
        },
    )
    compute_low_rank_factors: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether the decomposition should include low-rank factors (L, R)."
            )
        },
    )
    Q_bits: int = field(
        default=2, metadata={"help": "Either 2, 3, or 4 bit lattice quantization"}
    )
    L_bits: int = field(
        default=2, metadata={"help": "Either 2, 3, or 4 bit lattice quantization"}
    )
    R_bits: int = field(
        default=2, metadata={"help": "Either 2, 3, or 4 bit lattice quantization"}
    )
    rank: int = field(default=64, metadata={"help": "Rank of L and R factors"})
    iters: int = field(default=20)
    lplr_iters: int = field(default=5)
    activation_aware_LR: bool = field(
        default=True,
        metadata={"help": "Use activation-aware LPLR for computing the factors."},
    )
    update_order: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                'List specifying whether to update the "LR" factors before '
                '"q" or vice versa. The default is ["LR", "Q"]; pass '
                'in ["Q", "LR"] to swap the update order.'
            )
        },
    )
    quant_factory_Q: QuantizerFactory = field(
        default_factory=QuantizerFactory,
        metadata={
            "help": (
                "(Non-data-aware only) QuantizerFactory (from caldera.utils.quantizers)"
                "  object used to instantiate quantizer for Q. Only used if "
                "activation_aware_Q is False."
            )
        },
    )
    quant_factory_LR: QuantizerFactory = field(
        default_factory=QuantizerFactory,
        metadata={
            "help": (
                "(Non-lattice quant only) QuantizerFactory (from "
                "caldera.utils.quantizers) object used to instantiate quantizer for L "
                "and R. Only used if lattice_quant_LR is False."
            )
        },
    )
    rand_svd: bool = field(
        default=False,
        metadata={"help": "Whether to use randomized SVD for LPLR initialization"},
    )
    sigma_reg: float = field(
        default=0, metadata={"help": "Regularization to make Hessian positive definite"}
    )


@dataclass
class CalderaDecomposition:
    """A dataclass representing the components and parameters in the Caldera decomposition
    """

    Q: torch.Tensor = field(default=None)
    L: torch.Tensor = field(default=None)
    R: torch.Tensor = field(default=None)
    W: torch.Tensor = field(default=None)
    Q_idxs: torch.Tensor = field(default=None)
    L_idxs: torch.Tensor = field(default=None)
    R_idxs: torch.Tensor = field(default=None)
    Q_scale: float = field(default=1)
    L_scale: float = field(default=1)
    R_scale: float = field(default=1)
    global_scale: float = field(default=1)
    SU: torch.Tensor = field(default=None)
    SV: torch.Tensor = field(default=None)
    scaleWH: torch.Tensor = field(default=None)
    errors: dict[str,list[float]] = field(default_factory=dict)


@dataclass
class QuantInfo:
    """ Stores quantization-specific information
    """
    quant: AbstractQuantizer = field(default_factory=LowMemoryQuantizer)

