from typing import Tuple

import torch

# The sparseness strategy here is to represent the sparse tensor as a tuple of values and indices,
# where values is a flat tensor of the non-empty values, while indices is a tensor of shape (N, 2)
# representing the begin and end indices into the values tensor for each slice.
#
# For example:
# original = [[1, 2, 3], [], [4, 5]]
# values = [1, 2, 3, 4, 5]
# indices = [[0, 3], [3, 3], [3, 5]]
#
# This has the advantage that numerical preprocessing can be applied to the values tensor
# without caring about its sparseness, and the indices tensor can then be used to reconstruct
# the procecessed values into the original sparse shape. Additionally, it means that
# operations like truncating can also be applied only to the indices tensor, without
# needing to modify the values tensor.
#
# Note that the considerations here don't necessarily apply outside of preprocessing,
# in modeling you're quite likely to want to use built-in PyTorch sparse tensor support,
# as this representation of sparseness is a poor fit for things like matrix
# multiplications.


class SparseTruncIndices(torch.nn.Module):
    """
    Truncates the indices to a fixed length.

    Args:
        length: The fixed length to truncate the indices to.
    """

    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        begin = indices[:, 0]
        end = indices[:, 1]
        end = torch.min(begin + self.length, end)
        return torch.stack([begin, end], dim=1)


class SparseValues(torch.nn.Module):
    """
    Returns the values from a tuple of values and indices representing a sparse tensor.
    """

    def forward(self, values_and_indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return values_and_indices[0]


class SparseIndices(torch.nn.Module):
    """
    Returns the indices from a tuple of values and indices representing a sparse tensor.
    """

    def forward(self, values_and_indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return values_and_indices[1]


class SparseLen(torch.nn.Module):
    """
    Returns the length of each slice in the indices of a sparse tensor.
    """

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return indices[:, 1] - indices[:, 0]


class SparseTrunc(torch.nn.Module):
    """
    Truncates the values of a sparse tensor to a fixed length.

    Args:
        length: The fixed length to truncate the values to.
    """

    def __init__(self, length: int):
        super().__init__()
        self.indices_trunc = SparseTruncIndices(length)

    def forward(self, values_and_indices: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        values, indices = values_and_indices
        indices = self.indices_trunc(indices)
        return values, indices


class SparseTruncPad(torch.nn.Module):
    """
    Truncates and pads the values of a sparse tensor to a fixed length.

    Args:
        length: The fixed length to truncate and pad the values to.
    """

    def __init__(self, length: int, value: float = 0.0):
        super().__init__()
        self.sparse_trunc = SparseTrunc(length)
        self.value = value

    def forward(self, values_and_indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        values, indices = self.sparse_trunc(values_and_indices)
        return torch.stack(
            [
                torch.nn.functional.pad(
                    values[i[0] : i[1]],
                    pad=(0, self.sparse_trunc.indices_trunc.length - int(i[1] - i[0])),
                    value=self.value,
                )
                for i in indices
            ]
        )


class SparseMapSequences(torch.nn.Module):
    """
    Maps a module over the sequences of a sparse tensor.

    The sparse tensor is represented as a tuple of values and indices.

    Args:
        transform: The module to map over the sequences.
    """

    def __init__(self, transform: torch.nn.Module):
        super().__init__()
        self.transform = transform

    def forward(self, values_and_indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        values, indices = values_and_indices
        return torch.stack([self.transform(values[i[0] : i[1]]) for i in indices])
