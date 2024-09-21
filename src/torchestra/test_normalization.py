from typing import List, Tuple

import numpy as np
import torch

from torchestra import MinMaxScale, StandardScore, TDigest, TDigestDistribution, TDigestStorage


def test_min_max_scale():
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_combined = torch.cat([train1, train2])
    train_min = train_combined.min()
    train_max = train_combined.max()
    expected = (valid - train_min) / (train_max - train_min)
    module = MinMaxScale()
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(valid)

    assert torch.allclose(module.vmin, train_min)
    assert torch.allclose(module.vdelta, train_max - train_min)
    assert torch.allclose(received, expected)


def test_min_max_scale_state_dict():
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_combined = torch.cat([train1, train2])
    train_min = train_combined.min()
    train_max = train_combined.max()
    expected = (valid - train_min) / (train_max - train_min)
    module = MinMaxScale()
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)
    state_dict = module.state_dict()
    module = MinMaxScale()
    module.load_state_dict(state_dict)

    received = module(valid)

    assert torch.allclose(module.vmin, train_min)
    assert torch.allclose(module.vdelta, train_max - train_min)
    assert torch.allclose(received, expected)


def test_min_max_scale_script_stats():
    class CalculateStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.mod.calculate_stats(x)

    class CombineStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.mod.combine_stats(x)

    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_combined = torch.cat([train1, train2])
    train_min = train_combined.min()
    train_max = train_combined.max()
    expected = (valid - train_min) / (train_max - train_min)
    module = MinMaxScale()
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))

    stats1 = calculate_stats(train1)
    stats2 = calculate_stats(train2)
    stats = combine_stats([stats1, stats2])
    module.apply_stats(stats)
    compiled = torch.jit.script(module)

    received = compiled(valid)

    assert torch.allclose(module.vmin, train_min)
    assert torch.allclose(module.vdelta, train_max - train_min)
    assert torch.allclose(received, expected)


def test_min_max_scale_stacked():
    train1 = torch.rand(50, dtype=torch.float64)
    train2 = torch.rand(50, dtype=torch.float64)
    train3 = torch.rand(50, dtype=torch.float64)
    train4 = torch.rand(50, dtype=torch.float64)
    valid1 = torch.rand(50, dtype=torch.float64)
    valid2 = torch.rand(50, dtype=torch.float64)
    train1_combined = torch.cat([train1, train2])
    train2_combined = torch.cat([train3, train4])
    train_min1 = train1_combined.min()
    train_max1 = train1_combined.max()
    train_min2 = train2_combined.min()
    train_max2 = train2_combined.max()
    expected1 = (valid1 - train_min1) / (train_max1 - train_min1)
    expected2 = (valid2 - train_min2) / (train_max2 - train_min2)
    train1_stacked = torch.cat([train1.view(-1, 1), train3.view(-1, 1)], dim=1)
    train2_stacked = torch.cat([train2.view(-1, 1), train4.view(-1, 1)], dim=1)
    valid_stacked = torch.cat([valid1.view(-1, 1), valid2.view(-1, 1)], dim=1)
    module = MinMaxScale.stack([MinMaxScale(), MinMaxScale()])
    stats1 = module.calculate_stats(train1_stacked)
    stats2 = module.calculate_stats(train2_stacked)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(valid_stacked)
    received1 = received[:, 0]
    received2 = received[:, 1]

    assert torch.allclose(module.vmin[0], train_min1)
    assert torch.allclose(module.vdelta[0], train_max1 - train_min1)
    assert torch.allclose(module.vmin[1], train_min2)
    assert torch.allclose(module.vdelta[1], train_max2 - train_min2)
    assert torch.allclose(received1, expected1)
    assert torch.allclose(received2, expected2)


def test_min_max_scale_pre_calculated_stack():
    train1 = torch.rand(50, dtype=torch.float64)
    train2 = torch.rand(50, dtype=torch.float64)
    train3 = torch.rand(50, dtype=torch.float64)
    train4 = torch.rand(50, dtype=torch.float64)
    valid1 = torch.rand(50, dtype=torch.float64)
    valid2 = torch.rand(50, dtype=torch.float64)
    train1_combined = torch.cat([train1, train2])
    train2_combined = torch.cat([train3, train4])
    train_min1 = train1_combined.min()
    train_max1 = train1_combined.max()
    train_min2 = train2_combined.min()
    train_max2 = train2_combined.max()
    expected1 = (valid1 - train_min1) / (train_max1 - train_min1)
    expected2 = (valid2 - train_min2) / (train_max2 - train_min2)
    valid_stacked = torch.cat([valid1.view(-1, 1), valid2.view(-1, 1)], dim=1)
    module1 = MinMaxScale()
    module2 = MinMaxScale()
    stats1 = module1.calculate_stats(train1)
    stats2 = module1.calculate_stats(train2)
    stats12 = module1.combine_stats([stats1, stats2])
    module1.apply_stats(stats12)
    stats3 = module2.calculate_stats(train3)
    stats4 = module2.calculate_stats(train4)
    stats34 = module2.combine_stats([stats3, stats4])
    module2.apply_stats(stats34)
    module = MinMaxScale.stack([module1, module2])

    received = module(valid_stacked)
    received1 = received[:, 0]
    received2 = received[:, 1]

    assert torch.allclose(module.vmin[0], train_min1)
    assert torch.allclose(module.vdelta[0], train_max1 - train_min1)
    assert torch.allclose(module.vmin[1], train_min2)
    assert torch.allclose(module.vdelta[1], train_max2 - train_min2)
    assert torch.allclose(received1, expected1)
    assert torch.allclose(received2, expected2)


def test_standard_score_biased():
    ddof = 0
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_np = torch.cat([train1, train2]).numpy()
    train_mean = train_np.mean()
    train_std = train_np.std(ddof=ddof)
    expected = (valid.numpy() - train_mean) / train_std
    module = StandardScore(ddof=ddof)
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(valid).numpy()

    assert np.allclose(module.mean.numpy(), train_mean)
    assert np.allclose(module.std.numpy(), train_std)
    assert np.allclose(received, expected)


def test_standard_score_unbiased():
    ddof = 1
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_np = torch.cat([train1, train2]).numpy()
    train_mean = train_np.mean()
    train_std = train_np.std(ddof=ddof)
    expected = (valid.numpy() - train_mean) / train_std
    module = StandardScore(ddof=ddof)
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(valid).numpy()

    assert np.allclose(module.mean.numpy(), train_mean)
    assert np.allclose(module.std.numpy(), train_std)
    assert np.allclose(received, expected)


def test_standard_score_state_dict():
    ddof = 1
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_np = torch.cat([train1, train2]).numpy()
    train_mean = train_np.mean()
    train_std = train_np.std(ddof=ddof)
    expected = (valid.numpy() - train_mean) / train_std
    module = StandardScore(ddof=ddof)
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)
    state_dict = module.state_dict()
    module = StandardScore(ddof=ddof)
    module.load_state_dict(state_dict)

    received = module(valid).numpy()

    assert np.allclose(module.mean.numpy(), train_mean)
    assert np.allclose(module.std.numpy(), train_std)
    assert np.allclose(received, expected)


def test_standard_score_script_stats():
    class CalculateStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.mod.calculate_stats(x)

    class CombineStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(
            self, x: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.mod.combine_stats(x)

    ddof = 1
    train1 = torch.rand(100, dtype=torch.float64)
    train2 = torch.rand(100, dtype=torch.float64)
    valid = torch.rand(100, dtype=torch.float64)
    train_np = torch.cat([train1, train2]).numpy()
    train_mean = train_np.mean()
    train_std = train_np.std(ddof=ddof)
    expected = (valid.numpy() - train_mean) / train_std
    module = StandardScore(ddof=ddof)
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))

    stats1 = calculate_stats(train1)
    stats2 = calculate_stats(train2)
    stats = combine_stats([stats1, stats2])
    module.apply_stats(stats)
    compiled = torch.jit.script(module)
    received = compiled(valid).numpy()

    assert np.allclose(module.mean.numpy(), train_mean)
    assert np.allclose(module.std.numpy(), train_std)
    assert np.allclose(received, expected)


def test_standard_score_stacked():
    ddof = 1
    train1 = torch.rand(50, dtype=torch.float64)
    train2 = torch.rand(50, dtype=torch.float64)
    train3 = torch.rand(50, dtype=torch.float64)
    train4 = torch.rand(50, dtype=torch.float64)
    valid1 = torch.rand(50, dtype=torch.float64)
    valid2 = torch.rand(50, dtype=torch.float64)
    train1_np = torch.cat([train1, train2]).numpy()
    train2_np = torch.cat([train3, train4]).numpy()
    train1_mean = train1_np.mean()
    train1_std = train1_np.std(ddof=ddof)
    train2_mean = train2_np.mean()
    train2_std = train2_np.std(ddof=ddof)
    expected1 = (valid1.numpy() - train1_mean) / train1_std
    expected2 = (valid2.numpy() - train2_mean) / train2_std
    train1_stacked = torch.cat([train1.view(-1, 1), train3.view(-1, 1)], dim=1)
    train2_stacked = torch.cat([train2.view(-1, 1), train4.view(-1, 1)], dim=1)
    valid_stacked = torch.cat([valid1.view(-1, 1), valid2.view(-1, 1)], dim=1)
    module = StandardScore.stack([StandardScore(ddof=ddof), StandardScore(ddof=ddof)])
    stats1 = module.calculate_stats(train1_stacked)
    stats2 = module.calculate_stats(train2_stacked)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(valid_stacked)
    received1 = received[:, 0].numpy()
    received2 = received[:, 1].numpy()

    assert np.allclose(module.mean[0].numpy(), train1_mean)
    assert np.allclose(module.std[0].numpy(), train1_std)
    assert np.allclose(module.mean[1].numpy(), train2_mean)
    assert np.allclose(module.std[1].numpy(), train2_std)
    assert np.allclose(received1, expected1)
    assert np.allclose(received2, expected2)


def test_standard_score_pre_calculated_stack():
    ddof = 1
    train1 = torch.rand(50, dtype=torch.float64)
    train2 = torch.rand(50, dtype=torch.float64)
    train3 = torch.rand(50, dtype=torch.float64)
    train4 = torch.rand(50, dtype=torch.float64)
    valid1 = torch.rand(50, dtype=torch.float64)
    valid2 = torch.rand(50, dtype=torch.float64)
    train1_np = torch.cat([train1, train2]).numpy()
    train2_np = torch.cat([train3, train4]).numpy()
    train1_mean = train1_np.mean()
    train1_std = train1_np.std(ddof=ddof)
    train2_mean = train2_np.mean()
    train2_std = train2_np.std(ddof=ddof)
    expected1 = (valid1.numpy() - train1_mean) / train1_std
    expected2 = (valid2.numpy() - train2_mean) / train2_std
    valid_stacked = torch.cat([valid1.view(-1, 1), valid2.view(-1, 1)], dim=1)
    module1 = StandardScore(ddof=ddof)
    module2 = StandardScore(ddof=ddof)
    stats1 = module1.calculate_stats(train1)
    stats2 = module1.calculate_stats(train2)
    stats12 = module1.combine_stats([stats1, stats2])
    module1.apply_stats(stats12)
    stats3 = module2.calculate_stats(train3)
    stats4 = module2.calculate_stats(train4)
    stats34 = module2.combine_stats([stats3, stats4])
    module2.apply_stats(stats34)
    module = StandardScore.stack([module1, module2])

    received = module(valid_stacked)
    received1 = received[:, 0].numpy()
    received2 = received[:, 1].numpy()

    assert np.allclose(module.mean[0].numpy(), train1_mean)
    assert np.allclose(module.std[0].numpy(), train1_std)
    assert np.allclose(module.mean[1].numpy(), train2_mean)
    assert np.allclose(module.std[1].numpy(), train2_std)
    assert np.allclose(received1, expected1)
    assert np.allclose(received2, expected2)


def test_t_digest_quantile():
    t = TDigest()
    digest = t.new_digest()
    data = torch.tensor([1.0, 2.0, 4.0, 2.0])
    t.add_centroids(digest, data, torch.ones_like(data))
    t.finalize(digest)
    quantile = torch.tensor([0.0, 0.5, 1.0])
    expected = torch.tensor([1.0, 2.0, 4.0])

    received = t.quantile(digest, quantile)

    assert torch.allclose(received, expected)


def test_t_digest_quantile_single_data_point():
    t = TDigest()
    digest = t.new_digest()
    data = torch.tensor([2.0])
    t.add_centroids(digest, data, torch.ones_like(data))
    t.finalize(digest)
    quantile = torch.tensor([0.0, 0.5, 1.0])
    expected = torch.tensor([2.0, 2.0, 2.0])

    received = t.quantile(digest, quantile)

    assert torch.allclose(received, expected)


def test_t_digest_compression():
    t = TDigest()
    digest = t.new_digest()
    data = torch.cat([torch.tensor([1.0, 2.0, 4.0, 2.0])] * 10000)
    t.add_centroids(digest, data, torch.ones_like(data))
    t.finalize(digest)
    quantile = torch.tensor([0.0, 0.5, 1.0])
    expected = torch.tensor([1.0, 2.0, 4.0])

    received = t.quantile(digest, quantile)

    assert torch.allclose(received, expected)


def test_t_digest_cdf_unprocessed():
    t = TDigest()
    digest = t.new_digest()
    quantile = torch.tensor([0.0, 0.5, 1.0])
    expected = torch.tensor([0.0, 0.0, 0.0])

    received = t.cdf(digest, quantile)

    assert torch.allclose(received, expected)


def test_t_digest_cdf_single_data_point():
    t = TDigest()
    digest = t.new_digest()
    data = torch.tensor([2.0])
    t.add_centroids(digest, data, torch.ones_like(data))
    t.finalize(digest)
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([0.0, 0.5, 1.0])

    received = t.cdf(digest, x)

    assert torch.allclose(received, expected)


def test_t_digest_compressed_tails():
    t = TDigest(compression=4)
    digest = t.new_digest()
    data = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(1000) + torch.linspace(0, 0.1, 4000)
    data[0] = 0.5
    data[1] = 4.5
    t.add_centroids(digest, data, torch.ones_like(data))
    t.finalize(digest)
    x = torch.tensor([0.75, 4.25])
    expected = torch.tensor([0.0, 1.0])

    received = t.cdf(digest, x)

    assert torch.allclose(received, expected)


def test_t_digest_distribution():
    train1 = torch.rand(1000, dtype=torch.float64)
    train2 = torch.rand(1000, dtype=torch.float64)
    train_all = torch.cat([train1, train2])
    expected = torch.linspace(0, 1, 101, dtype=torch.float64)
    quantiles = torch.quantile(train_all, expected)
    module = TDigestDistribution()
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)

    received = module(quantiles)

    assert torch.allclose(received, expected, atol=1e-2)


def test_t_digest_distribution_state_dict():
    train1 = torch.rand(1000, dtype=torch.float64)
    train2 = torch.rand(1000, dtype=torch.float64)
    train_all = torch.cat([train1, train2])
    expected = torch.linspace(0, 1, 101, dtype=torch.float64)
    quantiles = torch.quantile(train_all, expected)
    module = TDigestDistribution()
    stats1 = module.calculate_stats(train1)
    stats2 = module.calculate_stats(train2)
    stats = module.combine_stats([stats1, stats2])
    module.apply_stats(stats)
    state_dict = module.state_dict()
    module = TDigestDistribution()
    module.load_state_dict(state_dict)

    received = module(quantiles)

    assert torch.allclose(received, expected, atol=1e-2)


def test_t_digest_distribution_script_stats():
    class CalculateStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x: torch.Tensor) -> TDigestStorage:
            return self.mod.calculate_stats(x)

    class CombineStatsModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x: List[TDigestStorage]) -> TDigestStorage:
            return self.mod.combine_stats(x)

    train1 = torch.rand(1000, dtype=torch.float64)
    train2 = torch.rand(1000, dtype=torch.float64)
    train_all = torch.cat([train1, train2])
    expected = torch.linspace(0, 1, 101, dtype=torch.float64)
    quantiles = torch.quantile(train_all, expected)
    module = TDigestDistribution()
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))

    stats1 = calculate_stats(train1)
    stats2 = calculate_stats(train2)
    stats = combine_stats([stats1, stats2])
    module.apply_stats(stats)
    compiled = torch.jit.script(module)

    received = compiled(quantiles)

    assert torch.allclose(received, expected, atol=1e-2)
