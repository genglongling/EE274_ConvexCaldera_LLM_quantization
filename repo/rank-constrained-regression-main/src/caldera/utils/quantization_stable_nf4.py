import torch
from abc import ABC, abstractmethod
from typing import List

_BITWIDTHS = [2, 4, 8, 16]
_QUANTIZER_METHODS = ["uniform", "nf4", "nf2"]  # Included 'nf2' and 'nf4'


class AbstractQuantizer(ABC):
    @abstractmethod
    def quantize_block(self, weight): ...

    @abstractmethod
    def dequantize_block(self, weight_quant, weight_params, weight_shape): ...


class LowMemoryQuantizer(AbstractQuantizer):
    """A memory-efficient quantizer for reducing the precision of numerical values.

        This class implements different approaches to quantization while processing data in blocks to minimize memory usage.

        Attributes
        ----------
        num_bits : int
            Number of bits used for quantization.
        method : str
            The quantization method being used.
        block_size : int
            Size of the blocks with common scale.

        Raises
        ------
        AssertionError
            If the specified number of bits is not supported.
        NotImplementedError
            If the specified quantization method is not supported.
    """

    def __init__(
        self, num_bits: int = 2, method: str = "uniform", block_size: int = 64
    ):
        self.num_bits = num_bits
        assert self.num_bits in _BITWIDTHS, "Bit-width not supported!"

        self.method = method.lower()
        if self.method not in _QUANTIZER_METHODS:
            raise NotImplementedError(
                f"Quantization method '{self.method}' not supported yet."
            )

        self.block_size = block_size

        # Prepare quantization levels and codebook
        if self.method in ["nf4", "nf2"]:
            self._prepare_nf_quantization()

    def _prepare_nf_quantization(self):
        """Prepares the quantization levels and codebook for NF quantization."""
        if self.method == "nf4":
            if self.num_bits != 4:
                raise ValueError("NF4 quantization supports only 4 bits.")
            num_levels = 2 ** self.num_bits  # 16 levels
            # Levels are symmetric around zero
            self.levels = torch.tensor(
                [
                    -1.334, -1.0, -0.784, -0.617, -0.476, -0.347, -0.226, -0.112,
                    0.0, 0.112, 0.226, 0.347, 0.476, 0.617, 0.784, 1.0
                ],
                dtype=torch.float32,
            )
        elif self.method == "nf2":
            if self.num_bits != 2:
                raise ValueError("NF2 quantization supports only 2 bits.")
            num_levels = 2 ** self.num_bits  # 4 levels
            self.levels = torch.tensor(
                [-0.8165, -0.3333, 0.3333, 0.8165],
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented.")

        # Compute thresholds between levels
        self.thresholds = (self.levels[:-1] + self.levels[1:]) / 2

    def _quantize_nf(self, weight_standardized):
        """Performs NF quantization (NF4 or NF2) on the input tensor.

            Args:
                weight_standardized (torch.Tensor): Standardized input tensor.

            Returns:
                torch.Tensor: Quantized tensor with indices corresponding to NF levels.
        """
        # Initialize quantized indices tensor
        quantized_indices = torch.zeros_like(weight_standardized, dtype=torch.uint8)

        # Map weights to quantized indices
        for i, threshold in enumerate(self.thresholds):
            quantized_indices += (weight_standardized > threshold).to(torch.uint8)

        return quantized_indices

    def _dequantize_nf(self, weight_quant):
        """Converts NF quantized indices back to floating point numbers.

            Args:
                weight_quant (torch.Tensor): Quantized tensor containing indices from the quantization process.

            Returns:
                torch.Tensor: Dequantized tensor in floating point format.
        """
        # Map indices back to levels using the codebook
        weight_dequant = self.levels.to(weight_quant.device)[weight_quant.long()]
        return weight_dequant

    def _quantize_uniform(self, weight_divabs):
        """Performs uniform quantization on the input tensor.

            Args:
                weight_divabs (torch.Tensor): Input tensor to be quantized, assumed to be
                    normalized by its absolute maximum value.

            Returns:
                torch.Tensor: Quantized tensor.
        """
        weight_scaled = weight_divabs * (2 ** (self.num_bits - 1) - 1)
        weight_scaled = weight_scaled.round()

        # Store the quantized weight as int types
        if self.num_bits <= 8:
            return weight_scaled.to(torch.int8)
        else:
            return weight_scaled.to(torch.int16)

    def _dequantize_uniform(self, weight_quant: torch.Tensor):
        """Converts quantized integer values back to floating point numbers.

            Args:
                weight_quant (torch.Tensor): Quantized tensor containing integer values.

            Returns:
                torch.Tensor: Dequantized tensor in floating point format.
        """
        return weight_quant.float() / (2 ** (self.num_bits - 1) - 1)

    def quantize_block(self, weight: torch.Tensor, epsilon: float = 1e-8):
        """Quantizes a 2D weight matrix by processing it in blocks.

            Args:
                weight (torch.Tensor): Input 2D weight matrix to be quantized
                epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1e-8

            Returns:
                tuple: Contains:
                    - weight_quant (torch.Tensor): Quantized weights
                    - weight_params (tuple): Parameters needed for dequantization
                    - shape (tuple): Original shape of the weight matrix
        """
        if len(weight.shape) != 2:
            raise ValueError(
                f"Support only for 2D matrix, but your input has {len(weight.shape)} dimensions."
            )
        total_elements = weight.shape[0] * weight.shape[1]
        if total_elements % self.block_size != 0:
            raise ValueError(
                f"Weight with shape {weight.shape[0]} x {weight.shape[1]} "
                f"is not divisible by block size {self.block_size}"
            )

        weight_flat = weight.flatten()
        weight_blocks = weight_flat.view(-1, self.block_size)

        if self.method == "uniform":
            # Normalize weights to [-1, 1]
            weight_max = weight_blocks.abs().max(dim=1, keepdim=True)[0]
            weight_max = torch.maximum(
                weight_max, torch.tensor([epsilon], device=weight.device)
            )
            weight_norm = weight_blocks / weight_max

            # Quantize
            weight_quant = self._quantize_uniform(weight_norm)

            # Store quantization parameters
            weight_params = weight_max

        elif self.method in ["nf4", "nf2"]:
            # Standardize weights (zero mean, unit variance)
            weight_mean = weight_blocks.mean(dim=1, keepdim=True)
            weight_std = weight_blocks.std(dim=1, keepdim=True)
            weight_std = torch.maximum(
                weight_std, torch.tensor([epsilon], device=weight.device)
            )
            weight_standardized = (weight_blocks - weight_mean) / weight_std

            # Quantize
            weight_quant = self._quantize_nf(weight_standardized)

            # Store quantization parameters
            weight_params = (weight_mean, weight_std)
        else:
            raise NotImplementedError(f"Quantization method '{self.method}' not implemented.")

        return weight_quant, weight_params, weight.shape

    def dequantize_block(self, weight_quant: torch.Tensor, weight_params, weight_shape: List[int]):
        """Dequantizes a block of weights using the specified quantization method.

            Args:
                weight_quant (torch.Tensor): The quantized weight tensor to be dequantized.
                weight_params (tuple or torch.Tensor): Parameters used during quantization.
                weight_shape (List[int]): The target shape for reshaping the final dequantized weights.

            Returns:
                torch.Tensor: The dequantized and reshaped weight tensor.
        """
        if self.method == "uniform":
            weight_max = weight_params
            weight_norm = self._dequantize_uniform(weight_quant)
            weight_blocks = weight_norm * weight_max
        elif self.method in ["nf4", "nf2"]:
            weight_mean, weight_std = weight_params
            weight_standardized = self._dequantize_nf(weight_quant)
            weight_blocks = weight_standardized * weight_std + weight_mean
        else:
            raise NotImplementedError(f"Dequantization method '{self.method}' not implemented.")

        # Reshape back to original shape
        weight_flat = weight_blocks.view(-1)
        return weight_flat.reshape(weight_shape)


class QuantizerFactory:
    """A factory class for creating quantizer objects.

        This class provides a factory pattern implementation for creating quantizer instances,
        specifically LowMemoryQuantizer objects, with configurable parameters for the 
        quantization method and block size.

        Args:
            method (str, optional): The quantization method to be used. Defaults to "uniform".
            block_size (int, optional): The size of blocks for quantization. Defaults to 64.

        Attributes:
            method (str): The quantization method being used.
            block_size (int): The block size for quantization operations.

        Example:
            >>> factory = QuantizerFactory(method="uniform", block_size=128)
            >>> quantizer = factory.get_quantizer(num_bits=8)
    """

    def __init__(self, method="uniform", block_size=64):
        self.method = method
        self.block_size = block_size

    def get_quantizer(self, num_bits, device="cpu"):
        return LowMemoryQuantizer(num_bits=num_bits, method=self.method, block_size=self.block_size)

    def __str__(self):
        return f"QuantizerFactory(method={self.method}, block_size={self.block_size})"
