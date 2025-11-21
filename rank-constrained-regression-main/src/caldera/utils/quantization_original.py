import torch

from abc import ABC, abstractmethod
from typing import List

_BITWIDTHS = [2, 4, 8, 16]
_QUANTIZER_METHODS = ["uniform"]


class AbstractQuantizer(ABC):
    @abstractmethod
    def quantize_block(self, weight): ...

    @abstractmethod
    def dequantize_block(self, weight_quant, weight_max, weight_shape): ...


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

        self.method = method
        if self.method not in _QUANTIZER_METHODS:
            raise NotImplementedError("Other quantization methods not supported yet.")

        self.block_size = block_size

    def _quantize_uniform(self, weight_divabs):
        """Performs uniform quantization on the input tensor to either int8 or int16.

            Args:
                weight_divabs (torch.Tensor): Input tensor to be quantized, assumed to be
                    normalized by its absolute maximum value.

            Returns:
                torch.Tensor: Quantized tensor in either int8 (if num_bits <= 8) or
                    int16 format. The values are scaled and rounded to utilize the full
                    range of the target data type.
        """

        weight_scaled = weight_divabs * (2 ** (self.num_bits - 1) - 1)
        weight_scaled = weight_scaled.round()

        # Store the quantized weight as either int8 or
        if self.num_bits <= 8:
            return weight_scaled.to(torch.int8)
        else:
            return weight_scaled.to(torch.int16)

    def _dequantize_uniform(self, weight_quant: torch.Tensor):
        """Converts quantized integer values back to floating point numbers using the same scaling factor used in quantization

            Args:
                weight_quant (torch.Tensor): Quantized tensor containing integer values
                    from the quantization process.

            Returns:
                torch.Tensor: Dequantized tensor in floating point format, scaled back
                    to the original value range by dividing by (2^(num_bits-1) - 1).
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
                    - weight_max (torch.Tensor): Maximum absolute values per block
                    - shape (tuple): Original shape of the weight matrix

            Raises:
                ValueError: If input is not 2D or if matrix dimensions are not divisible by block_size
        """

        if len(weight.shape) != 2:
            raise ValueError(
                f"Support only for 2D matrix, but your input has {len(weight.shape)} dimensions."
            )
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Weight with shape {weight.shape[0]} x {weight.shape[1]} is not divisible by block siez {self.block_size}"
            )
        weight_reshape = weight.flatten().reshape(-1, self.block_size)
        weight_max = weight_reshape.abs().max(dim=-1)[0].unsqueeze(-1)
        weight_max = torch.maximum(
            weight_max, torch.Tensor([epsilon]).to(weight.device)
        )
        weight_divabs = weight_reshape / weight_max
        weight_quant = self._quantize_uniform(weight_divabs)

        return weight_quant, weight_max, weight.shape


    def dequantize_block(self, weight_quant: torch.Tensor, weight_max: torch.Tensor, weight_shape: List[int]):
        """Dequantizes a block of weights using uniform dequantization with scaling.

            Args:
                weight_quant (torch.Tensor): The quantized weight tensor to be dequantized.
                weight_max (torch.Tensor): The maximum values used for scaling the dequantized weights.
                weight_shape (List[int]): The target shape for reshaping the final dequantized weights.

            Returns:
                torch.Tensor: The dequantized and reshaped weight tensor, scaled by weight_max.
        """

        weight = self._dequantize_uniform(weight_quant)
        return (weight * weight_max).reshape(weight_shape)
    

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