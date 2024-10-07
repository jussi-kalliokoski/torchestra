from typing import Any, Dict, List, Optional

import torch


class CountLookup(torch.nn.Module):
    """
    Maps a list of strings to a tensor of counts.

    The counts are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
    """

    total: torch.Tensor

    def __init__(self, eliminator: Optional[torch.nn.Module] = None):
        super().__init__()
        self.register_buffer("total", torch.tensor(0, dtype=torch.int64))
        self.counts = {"": 0}
        self.eliminator = eliminator or NoThreshold()

    def get_extra_state(self) -> Dict[str, Any]:
        return {"counts": self.counts}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.counts = state["counts"]

    def calculate_stats(self, x: List[str]) -> Dict[str, int]:
        """
        Calculates the key counts from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        counts: Dict[str, int] = {}
        for _x in x:
            counts[_x] = counts.get(_x, 0) + 1
        return counts

    def combine_stats(self, stats: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        counts: Dict[str, int] = {}
        for stat in stats:
            for k, v in stat.items():
                counts[k] = counts.get(k, 0) + v
        return counts

    def apply_stats(self, counts: Dict[str, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            counts: The statistics to apply.
        """
        counts = self.eliminator(counts)
        self.total = torch.tensor(sum(counts.values()), dtype=self.total.dtype)
        self.counts = counts if self.total > 0 else {"": 0}

    def forward(self, x: List[str]) -> torch.Tensor:
        return torch.tensor([self.counts.get(_x, 0) for _x in x], dtype=torch.int64)


class RatioLookup(torch.nn.Module):
    """
    Maps a list of strings to a tensor of the ratios of their occurence.

    The ratios are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
    """

    def __init__(self, eliminator: Optional[torch.nn.Module] = None):
        super().__init__()
        self._count_lookup = CountLookup(eliminator)

    def calculate_stats(self, x: List[str]) -> Dict[str, int]:
        """
        Calculates the key ratios from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        return self._count_lookup.calculate_stats(x)

    def combine_stats(self, stats: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        return self._count_lookup.combine_stats(stats)

    def apply_stats(self, ratios: Dict[str, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            ratios: The statistics to apply.
        """
        self._count_lookup.apply_stats(ratios)

    def forward(self, x: List[str]) -> torch.Tensor:
        if self._count_lookup.total == 0:
            return torch.zeros(len(x)).to(torch.float32)
        return self._count_lookup(x) / self._count_lookup.total


class IndexLookup(torch.nn.Module):
    """
    Maps a list of strings to a tensor of indices.

    The lookups are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
    """

    def __init__(self, eliminator: Optional[torch.nn.Module] = None, padding_idx: int = 0, unknown_idx: int = 1):
        super().__init__()
        self.padding_idx = padding_idx
        self.unknown_idx = unknown_idx
        self.lookup = {"": self.unknown_idx}
        self._count_lookup = CountLookup(eliminator)

    def get_extra_state(self) -> Dict[str, Any]:
        return {"lookup": self.lookup}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.lookup = state["lookup"]

    def dictionary_size(self) -> int:
        """
        Returns the size of the lookup dictionary to use when building embeddings or one-hot encodings.

        Returns:
            The number of unique keys in the lookup.
        """
        if len(self.lookup) == 1 and self.lookup.get("", self.padding_idx) == self.unknown_idx:
            return 2
        return len(self.lookup) + 2  # include unknown and padding

    def calculate_stats(self, x: List[str]) -> Dict[str, int]:
        """
        Calculates the key counts from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        return self._count_lookup.calculate_stats(x)

    def combine_stats(self, stats: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        return self._count_lookup.combine_stats(stats)

    def apply_stats(self, counts: Dict[str, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            counts: The statistics to apply.
        """
        counts = self._count_lookup.eliminator(counts)
        if sum(counts.values()) < 1:
            self.lookup = {"": self.unknown_idx}
            return
        self.lookup = {}
        taken = {self.padding_idx, self.unknown_idx}
        for k in _sort_counts(counts):
            i = len(self.lookup)
            while i in taken:
                i += 1
            self.lookup[k] = i
            taken.add(i)

    def forward(self, x: List[str]) -> torch.Tensor:
        return torch.tensor([self.lookup.get(_x, self.unknown_idx) for _x in x], dtype=torch.int64)


class IntCountLookup(torch.nn.Module):
    """
    Maps a tensor of integers to a tensor of counts.

    The counts are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
        key_dtype: The data type of the keys.
        val_dtype: The data type of the counts.
    """

    total: torch.Tensor
    key_min: torch.Tensor
    key_max: torch.Tensor

    def __init__(self, eliminator: Optional[torch.nn.Module] = None, key_dtype=torch.int64, val_dtype=torch.int64):
        super().__init__()
        self.key_dtype = key_dtype
        self.val_dtype = val_dtype
        # sparse tensors don't work as buffers
        self.lookup = torch.tensor([], dtype=val_dtype).to_sparse()
        self.register_buffer("total", torch.tensor(0, dtype=val_dtype))
        self.register_buffer("key_min", torch.tensor(torch.iinfo(key_dtype).max, dtype=key_dtype))
        self.register_buffer("key_max", torch.tensor(torch.iinfo(key_dtype).min, dtype=key_dtype))
        self.eliminator = eliminator or NoThreshold()

    def get_extra_state(self) -> Dict[str, Any]:
        return {"lookup": self.lookup}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.lookup = state["lookup"]

    def calculate_stats(self, x: torch.Tensor) -> Dict[int, int]:
        """
        Calculates the key counts from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        counts: Dict[int, int] = {}
        for _x in x:
            counts[_x.item()] = counts.get(_x.item(), 0) + 1
        return counts

    def combine_stats(self, stats: List[Dict[int, int]]) -> Dict[int, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        counts: Dict[int, int] = {}
        for stat in stats:
            for k, v in stat.items():
                counts[k] = counts.get(k, 0) + v
        return counts

    def apply_stats(self, counts: Dict[int, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            counts: The statistics to apply.
        """
        str_counts = {str(k): v for k, v in counts.items()}
        counts = {int(k): v for k, v in self.eliminator(str_counts).items()}
        if len(counts) < 1:
            return
        self.key_min = torch.tensor(min(counts), dtype=self.key_dtype)
        self.key_max = torch.tensor(max(counts), dtype=self.key_dtype)
        dense = torch.zeros(int(self.key_max - self.key_min + 1), dtype=self.val_dtype)
        for k, v in counts.items():
            dense[k - self.key_min] = v
        self.lookup = dense.to_sparse()
        self.total = self.lookup.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=self.val_dtype)
        mask = (x >= self.key_min) & (x <= self.key_max)
        out[mask] = self.lookup.index_select(0, x[mask] - self.key_min).to_dense()
        return out


class IntRatioLookup(torch.nn.Module):
    """
    Maps a tensor of integers to a tensor of the ratios of their occurence.

    The ratios are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
        key_dtype: The data type of the keys.
        val_dtype: The data type of the counts.
    """

    def __init__(self, eliminator: Optional[torch.nn.Module] = None, key_dtype=torch.int64, count_dtype=torch.int64):
        super().__init__()
        self._count_lookup = IntCountLookup(eliminator, key_dtype, count_dtype)

    def calculate_stats(self, x: torch.Tensor) -> Dict[int, int]:
        """
        Calculates the key ratios from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        return self._count_lookup.calculate_stats(x)

    def combine_stats(self, stats: List[Dict[int, int]]) -> Dict[int, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        return self._count_lookup.combine_stats(stats)

    def apply_stats(self, ratios: Dict[int, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            ratios: The statistics to apply.
        """
        self._count_lookup.apply_stats(ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._count_lookup.total == 0:
            return torch.zeros(len(x)).to(torch.float32)
        return self._count_lookup(x) / self._count_lookup.total


class IntIndexLookup(torch.nn.Module):
    """
    Maps a tensor of integers to a tensor of indices.

    The lookups are constructed in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        eliminator: A module that eliminates keys from the lookup. Defaults to none being eliminated.
        key_dtype: The data type of the keys.
        padding_idx: The index to use for padding.
        unknown_idx: The index to use for unknown keys.
    """

    key_min: torch.Tensor
    key_max: torch.Tensor

    def __init__(
        self, eliminator: Optional[torch.nn.Module] = None, key_dtype=torch.int64, padding_idx=0, unknown_idx=1
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.unknown_idx = unknown_idx
        self._count_lookup = IntCountLookup(eliminator, key_dtype, torch.int64)
        # sparse tensors don't work as buffers
        self.lookup = torch.tensor([], dtype=torch.int64).to_sparse()
        self.register_buffer("key_min", torch.tensor(torch.iinfo(key_dtype).max, dtype=key_dtype))
        self.register_buffer("key_max", torch.tensor(torch.iinfo(key_dtype).min, dtype=key_dtype))

    def get_extra_state(self) -> Dict[str, Any]:
        return {"lookup": self.lookup}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.lookup = state["lookup"]

    def dictionary_size(self) -> int:
        """
        Returns the size of the lookup dictionary to use when building embeddings or one-hot encodings.

        Returns:
            The number of unique keys in the lookup.
        """
        return int(torch.sum(self.lookup.to_dense() != 0)) + 2  # include unknown and padding

    def calculate_stats(self, x: torch.Tensor) -> Dict[int, int]:
        """
        Calculates the key counts from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The list of keys to calculate statistics from.

        Returns:
            The statistics for the keys.
        """
        return self._count_lookup.calculate_stats(x)

    def combine_stats(self, stats: List[Dict[int, int]]) -> Dict[int, int]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        return self._count_lookup.combine_stats(stats)

    def apply_stats(self, counts: Dict[int, int]) -> None:
        """
        Applies the calculated statistics to the lookup.

        Used by the preprocessing pipeline.

        Args:
            counts: The statistics to apply.
        """
        str_counts = self._count_lookup.eliminator({str(k): v for k, v in counts.items()})
        int_keys = [int(k) for k in _sort_counts(str_counts)]
        if len(int_keys) < 1:
            return
        self.key_min = torch.tensor(min(int_keys), dtype=self.key_min.dtype)
        self.key_max = torch.tensor(max(int_keys), dtype=self.key_max.dtype)
        dense = torch.zeros(int(self.key_max - self.key_min + 1), dtype=torch.int64)
        taken = {self.padding_idx, self.unknown_idx}
        i = 0
        for k in int_keys:
            while i in taken:
                i += 1
            dense[int(k) - self.key_min] = i - self.unknown_idx
            i += 1
        self.lookup = dense.to_sparse()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tensor(self.unknown_idx, dtype=torch.int64).repeat(x.shape)
        mask = (x >= self.key_min) & (x <= self.key_max)
        out[mask] += self.lookup.index_select(0, x[mask] - self.key_min).to_dense()
        return out


class NoThreshold(torch.nn.Module):
    """
    Eliminates no keys.
    """

    def forward(self, counts: Dict[str, int]) -> Dict[str, int]:
        return counts


class MinThreshold(torch.nn.Module):
    """
    Eliminates all keys with counts below the threshold.

    Args:
        threshold: The minimum count to keep a key.
    """

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def forward(self, counts: Dict[str, int]) -> Dict[str, int]:
        new_counts: Dict[str, int] = {}
        for k, v in counts.items():
            if v >= self.threshold:
                new_counts[k] = v
        return new_counts


class RatioThreshold(torch.nn.Module):
    """
    Eliminates all keys with counts below the proportion threshold of the total count.

    Args:
        threshold: The minimum proportion to keep a key.
    """

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, counts: Dict[str, int]) -> Dict[str, int]:
        total = sum(counts.values())
        new_counts: Dict[str, int] = {}
        for k, v in counts.items():
            if v / total >= self.threshold:
                new_counts[k] = v
        return new_counts


class TopK(torch.nn.Module):
    """
    Eliminates all but the top K keys with the highest counts.

    If there are fewer keys than K, all keys are kept.

    Keys with matching counts are picked in alphabetical order.

    Args:
        k: The number of keys to keep.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, counts: Dict[str, int]) -> Dict[str, int]:
        return {k: counts[k] for k in _sort_counts(counts)[: self.k]}


def _sort_counts(counts: Dict[str, int]) -> List[str]:
    keys = sorted(counts.keys())
    key_counts = torch.tensor([counts[k] for i, k in enumerate(keys)])
    indices = key_counts.argsort(descending=True, stable=True)
    return [keys[i.item()] for i in indices]
