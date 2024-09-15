from typing import Dict, List

import torch


class SplitToDict(torch.nn.Module):
    """
    Splits a tensor into a dictionary of named tensors.

    Intended mainly for splitting the outputs of parallel/stacked modules for
    debugging purposes.

    Args:
        names: The names of the tensors to split into.
    """

    def __init__(self, names: List[str]):
        super().__init__()
        self.names = names

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert x.shape[1] == len(self.names), "Number of names must match the number of tensors"
        return {name: x[:, i] for i, name in enumerate(self.names)}


class CombineDicts(torch.nn.Module):
    """
    Combines a list of dicts of tensors into a single dict.

    Intended mainly for combining the outputs of parallel modules that are
    collected into dictionaries to maintain the names for debugging purposes.

    If the dictionaries contain items with the same key, an error is raised.
    """

    def forward(self, x: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for d in x:
            for k, v in d.items():
                assert k not in result, f"Key {k} already exists in the result"
                result[k] = v
        return result
