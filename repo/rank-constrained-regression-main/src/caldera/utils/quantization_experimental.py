import torch
from abc import ABC, abstractmethod
from typing import List, Tuple

_BITWIDTHS = [2, 4, 8, 16]
_QUANTIZER_METHODS = ["uniform", "nf4", "nf2"]

def _get_nf4_map():
    return torch.tensor([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0
    ])

def _get_nf2_map():
    return torch.tensor([
        -0.702338,
        -0.200889,
        0.200889,
        0.702338
    ])

class AbstractQuantizer(ABC):
    @abstractmethod
    def quantize_block(self, weight): ...

    @abstractmethod
    def dequantize_block(self, weight_quant, weight_max, weight_shape): ...

class LowMemoryQuantizer(AbstractQuantizer):
    def __init__(
        self, num_bits: int = 2, method: str = "uniform", block_size: int = 64
    ):
        self.num_bits = num_bits
        assert self.num_bits in _BITWIDTHS, "Bit-width not supported!"
        
        self.method = method
        if self.method not in _QUANTIZER_METHODS:
            raise NotImplementedError("Quantization method not supported yet.")
            
        self.block_size = block_size
        
        if self.method == "nf4":
            assert self.num_bits == 4, "NF4 only supports 4-bit quantization!"
        elif self.method == "nf2":
            assert self.num_bits == 2, "NF2 only supports 2-bit quantization!"

    def _quantize_uniform(self, weight_divabs):
        weight_scaled = weight_divabs * (2 ** (self.num_bits - 1) - 1)
        weight_scaled = weight_scaled.round()
        if self.num_bits <= 8:
            return weight_scaled.to(torch.int8)
        else:
            return weight_scaled.to(torch.int16)

    def _dequantize_uniform(self, weight_quant: torch.Tensor):
        return weight_quant.float() / (2 ** (self.num_bits - 1) - 1)

    def _quantize_nf4(self, weight_divabs):
        nf4_map = _get_nf4_map().to(weight_divabs.device)
        x_reshape = weight_divabs.unsqueeze(-1)
        distances = (x_reshape - nf4_map).pow(2)
        indices = distances.argmin(dim=-1)
        return indices.to(torch.int8)

    def _quantize_nf2(self, weight_divabs):
        nf2_map = _get_nf2_map().to(weight_divabs.device)
        x_reshape = weight_divabs.unsqueeze(-1)
        distances = (x_reshape - nf2_map).pow(2)
        indices = distances.argmin(dim=-1)
        return indices.to(torch.int8)

    def _dequantize_nf4(self, weight_quant: torch.Tensor):
        nf4_map = _get_nf4_map().to(weight_quant.device)
        return nf4_map[weight_quant]

    def _dequantize_nf2(self, weight_quant: torch.Tensor):
        nf2_map = _get_nf2_map().to(weight_quant.device)
        return nf2_map[weight_quant]

    def _reshape_to_blocks(self, weight: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        shape = list(weight.shape)
        blocks_row = (shape[0] + self.block_size - 1) // self.block_size
        blocks_col = (shape[1] + self.block_size - 1) // self.block_size
        
        weight_blocks = []
        for i in range(blocks_row):
            for j in range(blocks_col):
                row_start = i * self.block_size
                row_end = min((i + 1) * self.block_size, shape[0])
                col_start = j * self.block_size
                col_end = min((j + 1) * self.block_size, shape[1])
                
                block = weight[row_start:row_end, col_start:col_end]
                
                if block.shape[0] != self.block_size or block.shape[1] != self.block_size:
                    pad_rows = self.block_size - block.shape[0]
                    pad_cols = self.block_size - block.shape[1]
                    block = torch.nn.functional.pad(block, (0, pad_cols, 0, pad_rows))
                
                weight_blocks.append(block.reshape(-1))
        
        return torch.stack(weight_blocks), shape

    def quantize_block(self, weight: torch.Tensor, epsilon: float = 1e-8):
        if len(weight.shape) != 2:
            raise ValueError(f"Support only for 2D matrix, but your input has {len(weight.shape)} dimensions.")
            
        weight_blocks, original_shape = self._reshape_to_blocks(weight)
        weight_max = weight_blocks.abs().max(dim=-1)[0].unsqueeze(-1)
        weight_max = torch.maximum(weight_max, torch.Tensor([epsilon]).to(weight.device))
        weight_divabs = weight_blocks / weight_max
        
        if self.method == "nf4":
            weight_quant = self._quantize_nf4(weight_divabs)
        elif self.method == "nf2":
            weight_quant = self._quantize_nf2(weight_divabs)
        else:
            weight_quant = self._quantize_uniform(weight_divabs)
            
        return weight_quant, weight_max, original_shape

    def dequantize_block(self, weight_quant: torch.Tensor, weight_max: torch.Tensor, weight_shape: List[int]):
        if self.method == "nf4":
            weight = self._dequantize_nf4(weight_quant)
        elif self.method == "nf2":
            weight = self._dequantize_nf2(weight_quant)
        else:
            weight = self._dequantize_uniform(weight_quant)
        
        weight = weight * weight_max
        blocks_row = (weight_shape[0] + self.block_size - 1) // self.block_size
        blocks_col = (weight_shape[1] + self.block_size - 1) // self.block_size
        
        result = torch.zeros(weight_shape, device=weight.device)
        block_idx = 0
        
        for i in range(blocks_row):
            row_start = i * self.block_size
            row_end = min((i + 1) * self.block_size, weight_shape[0])
            
            for j in range(blocks_col):
                col_start = j * self.block_size
                col_end = min((j + 1) * self.block_size, weight_shape[1])
                
                block = weight[block_idx].reshape(self.block_size, self.block_size)
                block = block[:row_end-row_start, :col_end-col_start]
                
                result[row_start:row_end, col_start:col_end] = block
                block_idx += 1
        
        return result

class QuantizerFactory:
    def __init__(self, method="uniform", block_size=64):
        self.method = method
        self.block_size = block_size

    def get_quantizer(self, num_bits, device="cpu"):
        return LowMemoryQuantizer(num_bits=num_bits, method=self.method, block_size=self.block_size)
    
    def __str__(self):
        return f"QuantizerFactory(method={self.method}, block_size={self.block_size})"