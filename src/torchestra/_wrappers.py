from typing import List

import torch


class Stack(torch.nn.Module):
    """
    Stacks a list of tensors along a dimension.

    This is a simple wrapper around `torch.stack` to expose it as a composable module.

    Args:
        dim: The dimension to stack the tensors along.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(x, dim=self.dim)


class Cat(torch.nn.Module):
    """
    Concatenates a list of tensors along a dimension.

    This is a simple wrapper around `torch.cat` to expose it as a composable module.

    Args:
        dim: The dimension to concatenate the tensors along.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, dim=self.dim)


class NanToNum(torch.nn.Module):
    """
    Replaces NaN values in a tensor with zeros.

    This is a simple wrapper around `torch.nan_to_num` to expose it as a composable module.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.nan_to_num()


class Clamp(torch.nn.Module):
    """
    Clamps the values of a tensor to a range.

    This is a simple wrapper around `torch.clamp` to expose it as a composable module.

    Args:
        min: The minimum value to clamp the tensor to.
        max: The maximum value to clamp the tensor to.
    """

    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.min, max=self.max)


class Unsqueeze(torch.nn.Module):
    """
    Unsqueezes a tensor along a dimension.

    This is a simple wrapper around `torch.unsqueeze` to expose it as a composable module.

    Args:
        dim: The dimension to unsqueeze the tensor along.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=self.dim)


class ToStr(torch.nn.Module):
    """
    Converts a tensor to a list of strings.
    """

    def forward(self, x: torch.Tensor) -> List[str]:
        return [str(_x.item()) for _x in x]
