from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


class MeanScale(torch.nn.Module):
    """
    Applies mean scaling (mean normalization) to the input data.

    The mean, min and max values are calculated in the stats calculation phase of the
    preprocessing pipeline.
    """

    mean: torch.Tensor
    delta: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("delta", torch.tensor(0.0, dtype=torch.float64))

    def calculate_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean, min and max values from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The data of which to calculate the statistics.
        """
        dim = len(self.mean.shape) - 1
        x = x.transpose(0, dim)
        mean = x.mean(dim=dim)
        vmin = x.min(dim=dim).values
        vmax = x.max(dim=dim).values
        return mean, vmin, vmax

    def combine_stats(
        self, stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.

        Returns:
            The combined statistics.
        """
        dim = len(self.mean.shape) - 1
        mean = torch.stack([m for m, _, _ in stats], dim=dim).mean(dim=dim)
        vmin = torch.stack([v for _, v, _ in stats], dim=dim).min(dim=dim).values
        vmax = torch.stack([v for _, _, v in stats], dim=dim).max(dim=dim).values
        return mean, vmin, vmax

    def apply_stats(self, stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """
        Applies the calculated statistics to the module.

        Used by the preprocessing pipeline.

        Args:
            stats: The statistics to apply.
        """
        self.mean, vmin, vmax = stats
        self.delta = vmax - vmin

    @staticmethod
    def stack(modules: List["MeanScale"]) -> "MeanScale":
        """
        Stacks the provided modules into a single MeanScale operating on a stack of the inputs.

        The stacked MeanScale allows running the calculation of multiple
        features in the same batch, resulting in a more efficient module graph.

        Args:
            modules: The list of modules to stack.

        Returns:
            A module that operates on a stack of the original inputs.
        """
        stacked = MeanScale()
        stacked.mean = torch.stack([m.mean for m in modules])
        stacked.delta = torch.stack([m.delta for m in modules])
        return stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.delta


class MinMaxScale(torch.nn.Module):
    """
    Applies min-max scaling to the input data.

    The minimum and maximum values are calculated in the stats calculation
    phase of the preprocessing pipeline.
    """

    vmin: torch.Tensor
    vdelta: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("vmin", torch.tensor(torch.inf, dtype=torch.float64))
        self.register_buffer("vdelta", torch.tensor(torch.inf, dtype=torch.float64))

    def calculate_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the minimum and maximum values from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The data of which to calculate the statistics.
        """
        dim = len(self.vmin.shape) - 1
        x = x.transpose(0, dim)
        vmin = x.min(dim=dim).values
        vmax = x.max(dim=dim).values
        return vmin, vmax

    def combine_stats(self, stats: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.

        Returns:
            The combined statistics.
        """
        dim = len(self.vmin.shape) - 1
        vmin = torch.stack([v for v, _ in stats], dim=dim).min(dim=dim).values
        vmax = torch.stack([v for _, v in stats], dim=dim).max(dim=dim).values
        return vmin, vmax

    def apply_stats(self, stats: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Applies the calculated statistics to the module.

        Used by the preprocessing pipeline.

        Args:
            stats: The statistics to apply.
        """
        self.vmin, vmax = stats
        self.vdelta = vmax - self.vmin

    @staticmethod
    def stack(modules: List["MinMaxScale"]) -> "MinMaxScale":
        """
        Stacks the provided modules into a single MinMaxScale operating on a stack of the inputs.

        The stacked MinMaxScale allows running the calculation of multiple
        features in the same batch, resulting in a more efficient module graph.

        Args:
            modules: The list of modules to stack.

        Returns:
            A module that operates on a stack of the original inputs.
        """
        stacked = MinMaxScale()
        stacked.vmin = torch.stack([m.vmin for m in modules])
        stacked.vdelta = torch.stack([m.vdelta for m in modules])
        return stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.vmin) / self.vdelta


class StandardScore(torch.nn.Module):
    """
    Applies standard score normalization to the input data.

    The mean and standard deviation are calculated in the stats calculation
    phase of the preprocessing pipeline.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, ddof: int = 1):
        super().__init__()

        self.ddof = ddof
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("std", torch.tensor(1.0, dtype=torch.float64))

    def calculate_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the N, mean and standard deviation from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The data of which to calculate the statistics.
        """
        n = torch.ones_like(self.mean, dtype=torch.int64) * len(x)
        dim = len(n.shape) - 1
        x = x.transpose(0, dim)
        mean = x.mean(dim=dim)
        std = (((x - mean).abs() ** 2).sum(dim=dim) / (n - self.ddof)) ** 0.5
        return n, mean, std

    def combine_stats(
        self, stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.

        Returns:
            The combined statistics.
        """
        # Algorithm based on this explanation:
        # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups#answer-2971563
        n1 = torch.zeros_like(self.mean, dtype=torch.int64)
        m1 = torch.zeros_like(self.mean, dtype=torch.float64)
        s1 = torch.ones_like(self.std, dtype=torch.float64)

        for n2, m2, s2 in stats:
            if torch.all(n1 == 0):
                n1, m1, s1 = n2, m2, s2
                continue

            n = n1 + n2

            # scale the mean weights separately to avoid sum overflow
            w1 = n1 / n
            w2 = n2 / n
            m = m1 * w1 + m2 * w2

            t1 = (n1 - self.ddof) * (s1**2) + n1 * ((m1 - m) ** 2)
            t2 = (n2 - self.ddof) * (s2**2) + n2 * ((m2 - m) ** 2)
            s = ((t1 + t2) / (n - self.ddof)) ** 0.5

            n1, m1, s1 = n, m, s

        return n1, m1, s1

    def apply_stats(self, stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """
        Applies the calculated statistics to the module.

        Used by the preprocessing pipeline.

        Args:
            stats: The statistics to apply.
        """
        _, self.mean, self.std = stats

    @staticmethod
    def stack(modules: List["StandardScore"]) -> "StandardScore":
        """
        Stacks the provided modules into a single StandardScore operating on a stack of the inputs.

        The stacked StandardScore allows running the calculation of multiple
        features in the same batch, resulting in a more efficient module graph.

        Args:
            modules: The list of modules to stack.

        Returns:
            A module that operates on a stack of the original inputs.
        """
        stacked = StandardScore(modules[0].ddof)
        stacked.mean = torch.stack([m.mean for m in modules])
        stacked.std = torch.stack([m.std for m in modules])
        return stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@torch.jit.script
@dataclass
class TDigestStorage:
    max_processed: torch.Tensor
    max_unprocessed: torch.Tensor
    n_processed: torch.Tensor
    n_unprocessed: torch.Tensor
    processed_means: torch.Tensor
    processed_weights: torch.Tensor
    unprocessed_means: torch.Tensor
    unprocessed_weights: torch.Tensor
    processed_weight: torch.Tensor
    unprocessed_weight: torch.Tensor
    mean_min: torch.Tensor
    mean_max: torch.Tensor
    cumulative_weights: torch.Tensor


class TDigest(torch.nn.Module):
    """
    Estimates the quantiles of the input data using the t-digest algorithm.

    This module is intended to be used as a building block for higher-level
    stats-based normalization modules, exposing helpers to work with t-digests
    in a torch-compatible way.

    Args:
        compression: The compression factor of the t-digest.
    """

    compression: torch.Tensor
    pi: torch.Tensor

    def __init__(self, compression: float = 1000.0):
        super().__init__()
        self.register_buffer("compression", torch.tensor(compression, dtype=torch.float64))
        self.register_buffer("pi", torch.acos(torch.tensor(0.0, dtype=torch.float64)) * 2)

    def quantile(self, digest: TDigestStorage, q: torch.Tensor) -> torch.Tensor:
        """
        Returns the esimated values at the given quantile(s) between 0.0 and 1.0.

        Args:
            digest: The digest to use for the estimation.
            q: The quantile(s) to estimate the values for.

        Returns:
            The estimated values at the given quantile(s).
        """
        dtype = q.dtype
        if digest.n_processed == 1:
            return torch.ones_like(q) * digest.processed_means[0].to(dtype)
        q = q.to(digest.processed_means.dtype)
        out = torch.zeros_like(q)
        index = q * digest.processed_weight
        mask = index <= digest.processed_weights[0] / 2.0
        out[mask] = digest.mean_min + 2.0 * index[mask] / digest.processed_weights[0] * (
            digest.processed_means[0] - digest.mean_min
        )
        mask = ~mask
        lower = torch.searchsorted(digest.cumulative_weights, index[mask], side="right")
        lower_proj = torch.zeros_like(index, dtype=torch.int64)
        lower_proj[mask] = lower
        end_mask = mask & (lower_proj >= len(digest.cumulative_weights) - 1)
        z1 = index[end_mask] - digest.processed_weight - digest.processed_weights[-1] / 2.0
        z2 = digest.processed_weights[-1] / 2.0 - z1
        out[end_mask] = self._weighted_average(digest.processed_means[-1], z1, digest.mean_max, z2)
        mask &= ~end_mask
        lower = lower_proj[mask]
        z1 = index[mask] - digest.cumulative_weights[lower - 1]
        z2 = digest.cumulative_weights[lower] - index[mask]
        out[mask] = self._weighted_average(digest.processed_means[lower - 1], z1, digest.processed_means[lower], z2)
        return out.to(dtype)

    def cdf(self, digest: TDigestStorage, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the cumulative distribution function of the input data.

        Args:
            digest: The digest to use for the estimation.
            x: The values to estimate the cumulative distribution for.

        Returns:
            The estimated cumulative distribution of the input data.
        """
        dtype = x.dtype
        if digest.n_processed == 0:
            return torch.zeros_like(x)
        if digest.n_processed == 1:
            width = digest.mean_max - digest.mean_min
            out = (x - digest.mean_min) / width
            lo_mask = x <= digest.mean_min
            out[lo_mask] = 0.0
            mask = ~lo_mask
            hi_mask = mask & (x >= digest.mean_max)
            out[hi_mask] = 1.0
            mask &= ~hi_mask
            mid_mask = mask & ((x - digest.mean_min) <= width) | ((width == 0.0) & (x == digest.mean_min))
            out[mid_mask] = 0.5
            return out.to(dtype)
        x = x.to(digest.processed_means.dtype)
        out = torch.zeros_like(x)
        at_min = x <= digest.mean_min
        at_max = x >= digest.mean_max
        out[at_max] = 1.0
        mask = ~(at_min | at_max)
        m0 = digest.processed_means[0]
        tail_l = mask & (x <= m0)
        if m0 - digest.mean_min > 0.0:
            out[tail_l] = 0.0
        else:
            out[tail_l] = (
                (x[tail_l] - digest.mean_min)
                / (m0 - digest.mean_min)
                * digest.processed_weights[0]
                / digest.processed_weight
                / 2.0
            )
        mn = digest.processed_means[-1]
        tail_r = mask & (x >= mn)
        if digest.mean_max - mn > 0.0:
            out[tail_r] = 1.0
        else:
            out[tail_r] = (
                1.0
                - (digest.mean_max - x[tail_r])
                / (digest.mean_max - mn)
                * digest.processed_weights[-1]
                / digest.processed_weight
                / 2.0
            )
        mask &= ~(tail_l | tail_r)
        upper = torch.searchsorted(digest.processed_means, x[mask])
        z1 = x[mask] - digest.processed_means[upper - 1]
        z2 = digest.processed_means[upper] - x[mask]
        out[mask] = (
            self._weighted_average(digest.cumulative_weights[upper - 1], z1, digest.cumulative_weights[upper], z2)
            / digest.processed_weight
        )
        return out.to(dtype)

    def new_digest(self) -> TDigestStorage:
        """
        Returns a new digest storage instance.

        Returns:
            A new digest storage instance.
        """
        max_processed = self.compression.ceil().to(torch.int64) * 2
        max_unprocessed = self.compression.ceil().to(torch.int64) * 8
        return TDigestStorage(
            max_processed=max_processed,
            max_unprocessed=max_unprocessed,
            n_processed=torch.tensor(0, dtype=torch.int64),
            n_unprocessed=torch.tensor(0, dtype=torch.int64),
            processed_means=torch.zeros(int(max_processed), dtype=torch.float64),
            processed_weights=torch.zeros(int(max_processed), dtype=torch.float64),
            unprocessed_means=torch.zeros(int(max_unprocessed), dtype=torch.float64),
            unprocessed_weights=torch.zeros(int(max_unprocessed), dtype=torch.float64),
            processed_weight=torch.tensor(0.0, dtype=torch.float64),
            unprocessed_weight=torch.tensor(0.0, dtype=torch.float64),
            mean_min=torch.tensor(float("inf"), dtype=torch.float64),
            mean_max=torch.tensor(float("-inf"), dtype=torch.float64),
            cumulative_weights=torch.zeros(0, dtype=torch.float64),
        )

    def merge_digests(self, dst: TDigestStorage, src: TDigestStorage) -> None:
        """
        Merges the source digest into the destination digest.

        WARNING: This is an in-place operation, both digests will be modified.

        Args:
            dst: The destination digest into which to merge.
            src: The source digest from which to merge.
        """
        self._process(src)
        self.add_centroids(dst, src.processed_means[: src.n_processed], src.processed_weights[: src.n_processed])

    def add_centroids(self, digest: TDigestStorage, mean: torch.Tensor, weight: torch.Tensor) -> None:
        """
        Adds the given centroids to the digest.

        WARNING: This is an in-place operation, the digest will be modified.

        Args:
            digest: The digest into which to add the centroids.
            mean: The means of the centroids to add.
            weight: The weights of the centroids to add.
        """
        offset = torch.tensor(0, dtype=torch.int64)
        while offset < len(mean):
            n = torch.min(len(mean) - offset, digest.max_unprocessed - digest.n_unprocessed)
            digest.unprocessed_means[digest.n_unprocessed : digest.n_unprocessed + n] = mean[offset : offset + n]
            digest.unprocessed_weights[digest.n_unprocessed : digest.n_unprocessed + n] = weight[offset : offset + n]
            digest.n_unprocessed += n
            digest.unprocessed_weight += weight[offset : offset + n].sum()
            offset += n
            if digest.n_unprocessed == digest.max_unprocessed:
                self._process(digest)

    def finalize(self, digest: TDigestStorage) -> None:
        """
        Finalizes the digest, preparing it for quantile estimation.

        WARNING: This is an in-place operation, the digest will be modified
        and can no longer be modified afterwards.

        Args:
            digest: The digest to finalize.
        """
        self._process(digest)
        digest.processed_means = digest.processed_means[: digest.n_processed]
        digest.processed_weights = digest.processed_weights[: digest.n_processed]
        digest.cumulative_weights = digest.processed_weights.cumsum(0)
        # clear unused memory
        digest.unprocessed_weights = torch.zeros(0, dtype=digest.unprocessed_weights.dtype)
        digest.unprocessed_means = torch.zeros(0, dtype=digest.unprocessed_means.dtype)

    def _process(self, digest: TDigestStorage) -> None:
        if digest.n_unprocessed == 0 and digest.n_processed <= digest.max_processed:
            return
        n1, n2 = digest.n_processed, digest.n_unprocessed
        means = torch.cat([digest.processed_means[:n1], digest.unprocessed_means[:n2]])
        weights = torch.cat([digest.processed_weights[:n1], digest.unprocessed_weights[:n2]])
        order = torch.argsort(means)

        digest.processed_means[0] = means[order[0]]
        digest.processed_weights[0] = weights[order[0]]
        digest.n_processed = torch.ones_like(digest.n_processed)
        digest.n_unprocessed = torch.zeros_like(digest.n_unprocessed)
        digest.processed_weight += digest.unprocessed_weight
        digest.unprocessed_weight = torch.zeros_like(digest.unprocessed_weight)

        weight_acc = torch.tensor(0.0, dtype=torch.float64)
        limit = digest.processed_weight * self._integrated_q(torch.tensor(1.0, dtype=torch.float64))
        for i in order[1:]:
            mean = means[i]
            weight = weights[i]
            weight_proj = weight_acc + weights[i]
            if weight_proj <= limit:
                weight_acc = weight_proj
                self._combine_centroid(digest, digest.n_processed - 1, mean, weight)
            else:
                k1 = self._integrated_location(weight_acc / digest.processed_weight)
                limit = digest.processed_weight * self._integrated_q(k1 + 1.0)
                weight_acc += weight
                digest.processed_means[digest.n_processed] = mean
                digest.processed_weights[digest.n_processed] = weight
                digest.n_processed += 1
        digest.mean_min = torch.min(digest.mean_min, digest.processed_means[0])
        digest.mean_max = torch.max(digest.mean_max, digest.processed_means[digest.n_processed - 1])

    def _combine_centroid(
        self, digest: TDigestStorage, i: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor
    ) -> None:
        digest.processed_weights[i] += weight
        digest.processed_means[i] += (
            weight * (mean - digest.processed_means[i]) / digest.processed_weights[i]
        ).nan_to_num()

    def _integrated_q(self, k: torch.Tensor) -> torch.Tensor:
        return (1.0 + torch.sin(torch.min(k, self.compression) * self.pi / self.compression - self.pi / 2.0)) / 2.0

    def _integrated_location(self, q: torch.Tensor) -> torch.Tensor:
        return (torch.asin(2.0 * q - 1.0) + self.pi / 2.0) * self.compression / self.pi

    @staticmethod
    def _weighted_average(x1: torch.Tensor, w1: torch.Tensor, x2: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        sorted_x1 = torch.ones_like(x1) * x1
        sorted_w1 = torch.ones_like(x1) * w1
        sorted_x2 = torch.ones_like(x2) * x2
        sorted_w2 = torch.ones_like(x2) * w2
        mask = x1 > x2
        sorted_x1[mask], sorted_x2[mask] = x2[mask], x1[mask]
        sorted_w1[mask], sorted_w2[mask] = w2[mask], w1[mask]
        x = (sorted_x1 * sorted_w1 + sorted_x2 * sorted_w2) / (sorted_w1 + sorted_w2)
        return torch.max(sorted_x1, torch.min(x, sorted_x2))


class TDigestDistribution(torch.nn.Module):
    """
    Estimates the distribution of the input data using the t-digest algorithm.

    The distribution is estimated using the cumulative distribution function
    using statistics calculated in the stats calculation phase of the
    preprocessing pipeline.

    Args:
        compression: The compression factor of the t-digest.
    """

    def __init__(self, compression: float = 1000.0):
        super().__init__()
        self._digests = TDigest(compression)
        self.t = self._digests.new_digest()

    def get_extra_state(self) -> Dict[str, Any]:
        return {"t": self.t}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.t = state["t"]

    def calculate_stats(self, x: torch.Tensor) -> TDigestStorage:
        """
        Calculates the distributin statistics from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The data from which to calculate the statistics.
        """
        t = self._digests.new_digest()
        self._digests.add_centroids(t, x, torch.ones_like(x))
        return t

    def combine_stats(self, stats: List[TDigestStorage]) -> TDigestStorage:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        t = self._digests.new_digest()
        for s in stats:
            self._digests.merge_digests(t, s)
        return t

    def apply_stats(self, t: TDigestStorage) -> None:
        """
        Applies the calculated statistics to the distribution.

        Used by the preprocessing pipeline.

        Args:
            t: The statistics to apply.
        """
        self._digests.finalize(t)
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._digests.cdf(self.t, x)


class RobustScale(torch.nn.Module):
    """
    Applies robust scaling to the input data.

    The median and interquartile range are calculated in the stats calculation
    phase of the preprocessing pipeline using the T-Digest algorithm.
    """

    median: torch.Tensor
    iqr: torch.Tensor

    def __init__(self, compression: float = 1000.0):
        super().__init__()
        self._digests = TDigest(compression)
        self.register_buffer("median", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("iqr", torch.tensor(1.0, dtype=torch.float64))

    def calculate_stats(self, x: torch.Tensor) -> TDigestStorage:
        """
        Calculates the median and interquartile range from the input data.

        Used by the preprocessing pipeline.

        Args:
            x: The data from which to calculate the statistics.
        """
        t = self._digests.new_digest()
        self._digests.add_centroids(t, x, torch.ones_like(x))
        return t

    def combine_stats(self, stats: List[TDigestStorage]) -> TDigestStorage:
        """
        Combines the statistics of multiple datasets.

        Used by the preprocessing pipeline.

        Args:
            stats: The list of statistics to combine.
        """
        t = self._digests.new_digest()
        for s in stats:
            self._digests.merge_digests(t, s)
        return t

    def apply_stats(self, t: TDigestStorage) -> None:
        """
        Applies the calculated statistics to the module.

        Used by the preprocessing pipeline.

        Args:
            t: The statistics to apply.
        """
        self._digests.finalize(t)
        self.median = self._digests.quantile(t, torch.tensor(0.5, dtype=torch.float64))
        q1 = self._digests.quantile(t, torch.tensor(0.25, dtype=torch.float64))
        q3 = self._digests.quantile(t, torch.tensor(0.75, dtype=torch.float64))
        self.iqr = q3 - q1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.median) / self.iqr
