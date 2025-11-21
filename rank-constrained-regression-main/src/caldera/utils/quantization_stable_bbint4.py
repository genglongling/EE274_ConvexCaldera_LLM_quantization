import torch
from abc import ABC, abstractmethod
from typing import List
import csv
import os

_BITWIDTHS = [2, 4, 8, 16]
_QUANTIZER_METHODS = ["uniform", "nf4", "nf2", "bbint4"]

class AbstractQuantizer(ABC):
    @abstractmethod
    def quantize_block(self, weight): ...

    @abstractmethod
    def dequantize_block(self, weight_quant, weight_params, weight_shape): ...


class LowMemoryQuantizer(AbstractQuantizer):
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
        elif self.method == "bbint4" and self.num_bits != 4:
            raise ValueError("bbint4 quantization supports only 4 bits.")

    def _prepare_nf_quantization(self):
        """Prepares the quantization levels and codebook for NF quantization."""
        if self.method == "nf4":
            if self.num_bits != 4:
                raise ValueError("NF4 quantization supports only 4 bits.")
            # Levels are symmetric around zero and optimized for weight distribution
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
            self.levels = torch.tensor(
                [-0.8165, -0.3333, 0.3333, 0.8165],
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented.")

        # Compute thresholds between levels
        self.thresholds = (self.levels[:-1] + self.levels[1:]) / 2

    def _quantize_nf(self, weight_blocks, scale_factor):
        """Performs NF quantization (NF4 or NF2) on the input tensor.
        
        Args:
            weight_blocks: Input tensor blocks
            scale_factor: Scaling factor to match the levels range
            
        Returns:
            Quantized tensor with indices corresponding to NF levels
        """
        # Scale the weights to match the levels range
        weight_scaled = weight_blocks / scale_factor
        
        # Initialize quantized indices tensor
        quantized_indices = torch.zeros_like(weight_scaled, dtype=torch.uint8)

        # Map weights to quantized indices using thresholds
        for threshold in self.thresholds:
            quantized_indices += (weight_scaled > threshold).to(torch.uint8)

        return quantized_indices

    def _dequantize_nf(self, weight_quant, scale_factor):
        """Converts NF quantized indices back to floating point numbers."""
        # Map indices back to levels using the codebook and restore scale
        weight_dequant = self.levels.to(weight_quant.device)[weight_quant.long()]
        return weight_dequant * scale_factor

    def _quantize_uniform(self, weight_divabs):
        """Performs uniform quantization on the input tensor."""
        weight_scaled = weight_divabs * (2 ** (self.num_bits - 1) - 1)
        weight_scaled = weight_scaled.round()

        if self.num_bits <= 8:
            return weight_scaled.to(torch.int8)
        else:
            return weight_scaled.to(torch.int16)

    def _dequantize_uniform(self, weight_quant: torch.Tensor):
        """Converts quantized integer values back to floating point numbers."""
        return weight_quant.float() / (2 ** (self.num_bits - 1) - 1)

    def _quantize_bbint4(self, weight_blocks: torch.Tensor, epsilon: float = 1e-8, log_file: str = "outlier_log.csv"):

        """Quantizes using bitsandbytes-style INT4 quantization."""
        device = weight_blocks.device
        
        # Detect outliers
        weight_mean = weight_blocks.mean(dim=1, keepdim=True)
        weight_std = weight_blocks.std(dim=1, keepdim=True)
        weight_std = torch.maximum(weight_std, torch.tensor([epsilon], device=device))
        outlier_mask = (weight_blocks - weight_mean).abs() > (6.0 * weight_std)
        
        # Store outliers
        outlier_values = weight_blocks[outlier_mask]
        outlier_indices = torch.nonzero(outlier_mask)

        # Count the number of outliers
        num_outliers = outlier_values.numel()
        print(f"Number of outliers stored: {num_outliers}")

        # Log the number of outliers to a file
        if not os.path.exists(log_file):
            # Create file and write the header if it doesn't exist
            with open(log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Call_ID", "Num_Outliers"])  # Write header

        # Append the data to the log file
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id(self), num_outliers])  # Use `id(self)` as a unique identifier for this call

        # Replace outliers with mean
        weight_blocks = torch.where(outlier_mask, weight_mean, weight_blocks)
        
        # Compute block-wise scaling
        block_min = weight_blocks.min(dim=1, keepdim=True)[0]
        block_max = weight_blocks.max(dim=1, keepdim=True)[0]
        scales = (block_max - block_min) / 15
        scales = torch.maximum(scales, torch.tensor([epsilon], device=device))
        
        # Quantize to 4-bit range
        weight_scaled = (weight_blocks - block_min) / scales
        weight_quant = torch.clamp(torch.round(weight_scaled), 0, 15)
        
        # Pack two 4-bit values into one byte
        weight_packed = (weight_quant[:, ::2] * 16 + weight_quant[:, 1::2]).to(torch.uint8)
        
        return weight_packed, (block_min, scales, outlier_values, outlier_indices)

    def _dequantize_bbint4(self, weight_packed: torch.Tensor, params):
        """Dequantizes bitsandbytes-style INT4 quantized values."""
        block_min, scales, outlier_values, outlier_indices = params
        device = weight_packed.device
        
        # Unpack bytes into 4-bit values
        weight_unpacked = torch.zeros(weight_packed.shape[0], weight_packed.shape[1]*2, 
                                    dtype=torch.float32, device=device)
        weight_unpacked[:, ::2] = weight_packed // 16
        weight_unpacked[:, 1::2] = weight_packed % 16
        
        # Dequantize
        weight_dequant = weight_unpacked * scales + block_min
        
        # Restore outliers
        if outlier_values.numel() > 0:
            weight_dequant[outlier_indices[:, 0], outlier_indices[:, 1]] = outlier_values
        
        return weight_dequant

    def quantize_block(self, weight: torch.Tensor, epsilon: float = 1e-8):
        """Quantizes a 2D weight matrix by processing it in blocks."""
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
            weight_quant = self._quantize_uniform(weight_norm)
            weight_params = weight_max

        elif self.method in ["nf4", "nf2"]:
            # Calculate scale factor based on absolute maximum to match level range
            abs_max = weight_blocks.abs().max(dim=1, keepdim=True)[0]
            scale_factor = torch.maximum(
                abs_max, torch.tensor([epsilon], device=weight.device)
            )
            
            # Quantize using scale factor
            weight_quant = self._quantize_nf(weight_blocks, scale_factor)
            weight_params = scale_factor

        elif self.method == "bbint4":
            weight_quant, weight_params = self._quantize_bbint4(weight_blocks, epsilon)
        else:
            raise NotImplementedError(f"Quantization method '{self.method}' not implemented.")

        return weight_quant, weight_params, weight.shape

    def dequantize_block(self, weight_quant: torch.Tensor, weight_params, weight_shape: List[int]):
        """Dequantizes a block of weights using the specified quantization method."""
        if self.method == "uniform":
            weight_max = weight_params
            weight_norm = self._dequantize_uniform(weight_quant)
            weight_blocks = weight_norm * weight_max
        elif self.method in ["nf4", "nf2"]:
            scale_factor = weight_params
            weight_blocks = self._dequantize_nf(weight_quant, scale_factor)
        elif self.method == "bbint4":
            weight_blocks = self._dequantize_bbint4(weight_quant, weight_params)
        else:
            raise NotImplementedError(f"Dequantization method '{self.method}' not implemented.")

        weight_flat = weight_blocks.view(-1)
        return weight_flat.reshape(weight_shape)


class QuantizerFactory:
    def __init__(self, method="uniform", block_size=64):
        self.method = method
        self.block_size = block_size

    def get_quantizer(self, num_bits, device="cpu"):
        return LowMemoryQuantizer(num_bits=num_bits, method=self.method, block_size=self.block_size)

    def __str__(self):
        return f"QuantizerFactory(method={self.method}, block_size={self.block_size})"