from typing import List, OrderedDict

import torch

from torchestra import STATS_PLAN_INPUT_MAPPING_ALL, STATS_PLAN_INPUT_MAPPING_BYPASS, Parallel, Sequential, StatsPlan


class Add(torch.nn.Module):
    def __init__(self, y: torch.Tensor):
        super().__init__()
        self.y = y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.y


class Mul(torch.nn.Module):
    def __init__(self, y: torch.Tensor):
        super().__init__()
        self.y = y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.y


class StackSum(torch.nn.Module):
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(x).sum()


class TotalAdder(torch.nn.Module):
    total: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.tensor(0.0))

    def calculate_stats(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()

    def combine_stats(self, stats: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(stats).sum()

    def apply_stats(self, stats: torch.Tensor) -> None:
        self.total = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.total


def test_stats_plan_independent():
    graph = Parallel(
        [
            Sequential(
                Add(torch.tensor(3.0)),
                Mul(torch.tensor(2.0)),
                TotalAdder(),
            ),
            Sequential(
                Add(torch.tensor(2.0)),
                Mul(torch.tensor(3.0)),
                TotalAdder(),
            ),
        ]
    )

    stats_plan = StatsPlan(graph)

    assert stats_plan.stats_module_paths == ["0.2", "1.2"]
    assert stats_plan.dependencies == {
        "0.2": set(),
        "1.2": set(),
    }
    assert len(stats_plan.stages) == 1
    assert stats_plan.stages[0].module_paths == ["0.2", "1.2"]
    assert stats_plan.stages[0].input_mapping == [0, 1]
    assert stats_plan.stages[0].graph(torch.tensor(5.0)) == (torch.tensor(16.0), torch.tensor(21.0))


def test_stats_plan_shared_dependencies():
    graph = Sequential(
        TotalAdder(),
        Parallel(
            [
                Sequential(Add(torch.tensor(1.0)), TotalAdder(), Mul(2.0)),
                Sequential(Mul(torch.tensor(2.0)), TotalAdder(), Add(1.0)),
            ]
        ),
        StackSum(),
        TotalAdder(),
    )

    stats_plan = StatsPlan(graph)
    stats_plan.root.get_submodule("0").total += 1
    stats_plan.root.get_submodule("1.0.1").total += 2
    stats_plan.root.get_submodule("1.1.1").total += 3
    stats_plan.root.get_submodule("3").total += 4

    assert stats_plan.stats_module_paths == ["0", "1.0.1", "1.1.1", "3"]
    assert stats_plan.dependencies == {
        "0": set(),
        "1.0.1": {"0"},
        "1.1.1": {"0"},
        "3": {"1.0.1", "1.1.1", "0"},
    }

    assert len(stats_plan.stages) == 3
    assert stats_plan.stages[0].module_paths == ["0"]
    assert stats_plan.stages[0].input_mapping == [STATS_PLAN_INPUT_MAPPING_BYPASS]
    assert isinstance(stats_plan.stages[0].graph, torch.nn.Identity)
    assert stats_plan.stages[1].module_paths == ["1.0.1", "1.1.1"]
    assert stats_plan.stages[1].input_mapping == [0, 1]
    assert stats_plan.stages[1].graph(torch.tensor(5.0)) == (torch.tensor(7.0), torch.tensor(12.0))
    assert stats_plan.stages[2].module_paths == ["3"]
    assert stats_plan.stages[2].input_mapping == [STATS_PLAN_INPUT_MAPPING_ALL]
    assert stats_plan.stages[2].graph(torch.tensor(5.0)) == torch.tensor(34.0)


def test_stats_plan_complex():
    graph = Sequential(
        TotalAdder(),
        Parallel(
            [
                Sequential(Add(torch.tensor(1.0)), TotalAdder(), Mul(2.0), TotalAdder()),
                Sequential(Mul(torch.tensor(2.0)), TotalAdder(), Add(1.0), TotalAdder()),
                Sequential(Add(torch.tensor(2.0)), TotalAdder(), Mul(2.0), TotalAdder()),
            ]
        ),
        StackSum(),
        TotalAdder(),
    )

    stats_plan = StatsPlan(graph)
    stats_plan.root.get_submodule("0").total += 1
    stats_plan.root.get_submodule("1.0.1").total += 2
    stats_plan.root.get_submodule("1.0.3").total += 3
    stats_plan.root.get_submodule("1.1.1").total += 4
    stats_plan.root.get_submodule("1.1.3").total += 5
    stats_plan.root.get_submodule("1.2.1").total += 6
    stats_plan.root.get_submodule("1.2.3").total += 7
    stats_plan.root.get_submodule("3").total += 8

    assert stats_plan.stats_module_paths == [
        "0",
        "1.0.1",
        "1.0.3",
        "1.1.1",
        "1.1.3",
        "1.2.1",
        "1.2.3",
        "3",
    ]
    assert stats_plan.dependencies == {
        "0": set(),
        "1.0.1": {"0"},
        "1.0.3": {"1.0.1", "0"},
        "1.1.1": {"0"},
        "1.1.3": {"1.1.1", "0"},
        "1.2.1": {"0"},
        "1.2.3": {"1.2.1", "0"},
        "3": {
            "0",
            "1.0.1",
            "1.0.3",
            "1.1.1",
            "1.1.3",
            "1.2.1",
            "1.2.3",
        },
    }

    assert len(stats_plan.stages) == 4
    assert stats_plan.stages[0].module_paths == ["0"]
    assert stats_plan.stages[0].input_mapping == [STATS_PLAN_INPUT_MAPPING_BYPASS]
    assert isinstance(stats_plan.stages[0].graph, torch.nn.Identity)
    assert stats_plan.stages[1].module_paths == ["1.0.1", "1.1.1", "1.2.1"]
    assert stats_plan.stages[1].input_mapping == [0, 1, 2]
    assert stats_plan.stages[1].graph(torch.tensor(5.0)) == (torch.tensor(7.0), torch.tensor(12.0), torch.tensor(8.0))
    assert stats_plan.stages[2].module_paths == ["1.0.3", "1.1.3", "1.2.3"]
    assert stats_plan.stages[2].input_mapping == [0, 1, 2]
    assert stats_plan.stages[2].graph(torch.tensor(5.0)) == (torch.tensor(18.0), torch.tensor(17.0), torch.tensor(28.0))
    assert stats_plan.stages[3].module_paths == ["3"]
    assert stats_plan.stages[3].input_mapping == [STATS_PLAN_INPUT_MAPPING_ALL]
    assert stats_plan.stages[3].graph(torch.tensor(5.0)) == torch.tensor(78.0)


def test_sidechain():
    graph = Parallel(
        [
            Sequential(Add(1.0), TotalAdder(), TotalAdder()),
            TotalAdder(),
        ]
    )

    stats_plan = StatsPlan(graph)
    stats_plan.root.get_submodule("0.1").total += 1
    stats_plan.root.get_submodule("0.2").total += 2
    stats_plan.root.get_submodule("1").total += 3

    assert stats_plan.stats_module_paths == ["0.1", "0.2", "1"]
    assert stats_plan.dependencies == {
        "0.1": set(),
        "0.2": {"0.1"},
        "1": set(),
    }

    assert len(stats_plan.stages) == 2
    assert stats_plan.stages[0].module_paths == ["0.1", "1"]
    assert stats_plan.stages[0].input_mapping == [STATS_PLAN_INPUT_MAPPING_ALL, STATS_PLAN_INPUT_MAPPING_BYPASS]
    assert stats_plan.stages[0].graph(torch.tensor(5.0)) == torch.tensor(6.0)
    assert stats_plan.stages[1].module_paths == ["0.2"]
    assert stats_plan.stages[1].input_mapping == [STATS_PLAN_INPUT_MAPPING_ALL]
    assert stats_plan.stages[1].graph(torch.tensor(5.0)) == torch.tensor(7.0)


def test_stats_plan_complete():
    graph = Sequential(
        TotalAdder(),
        Parallel(
            [
                Sequential(Add(torch.tensor(1.0)), TotalAdder(), Mul(2.0), TotalAdder()),
                Sequential(Mul(torch.tensor(2.0)), TotalAdder(), Add(1.0), TotalAdder()),
                Sequential(Add(torch.tensor(2.0)), TotalAdder(), Mul(2.0), TotalAdder()),
            ]
        ),
        StackSum(),
        TotalAdder(),
    )
    ds_shards = [
        [(torch.cos(torch.sin(torch.arange(5.0).to(torch.float64) + i) + j),) for i in range(3)] for j in range(2)
    ]
    ds_combined = torch.cat([torch.cat([batch[0] for batch in ds]) for ds in ds_shards])
    stats_0 = ds_combined.sum()
    stats_1_0_1 = torch.sum(ds_combined + stats_0 + 1.0)
    stats_1_0_3 = torch.sum((ds_combined + stats_0 + 1.0 + stats_1_0_1) * 2.0)
    stats_1_1_1 = torch.sum((ds_combined + stats_0) * 2.0)
    stats_1_1_3 = torch.sum((ds_combined + stats_0) * 2.0 + stats_1_1_1 + 1.0)
    stats_1_2_1 = torch.sum(ds_combined + stats_0 + 2.0)
    stats_1_2_3 = torch.sum((ds_combined + stats_0 + 2.0 + stats_1_2_1) * 2.0)
    stats_3 = torch.stack(
        [
            (ds_combined + stats_0 + 1.0 + stats_1_0_1) * 2.0 + stats_1_0_3,
            (ds_combined + stats_0) * 2.0 + stats_1_1_1 + 1.0 + stats_1_1_3,
            (ds_combined + stats_0 + 2.0 + stats_1_2_1) * 2.0 + stats_1_2_3,
        ]
    ).sum()
    valid = torch.rand(5)
    valid_0 = valid + stats_0
    valid_1_0_0 = valid_0 + 1.0
    valid_1_0_1 = valid_1_0_0 + stats_1_0_1
    valid_1_0_2 = valid_1_0_1 * 2.0
    valid_1_0_3 = valid_1_0_2 + stats_1_0_3
    valid_1_1_0 = valid_0 * 2.0
    valid_1_1_1 = valid_1_1_0 + stats_1_1_1
    valid_1_1_2 = valid_1_1_1 + 1.0
    valid_1_1_3 = valid_1_1_2 + stats_1_1_3
    valid_1_2_0 = valid_0 + 2.0
    valid_1_2_1 = valid_1_2_0 + stats_1_2_1
    valid_1_2_2 = valid_1_2_1 * 2.0
    valid_1_2_3 = valid_1_2_2 + stats_1_2_3
    valid_2 = torch.stack([valid_1_0_3, valid_1_1_3, valid_1_2_3]).sum()
    valid_3 = valid_2 + stats_3
    expected_stats = OrderedDict(
        [
            ("0", stats_0),
            ("1.0.1", stats_1_0_1),
            ("1.0.3", stats_1_0_3),
            ("1.1.1", stats_1_1_1),
            ("1.1.3", stats_1_1_3),
            ("1.2.1", stats_1_2_1),
            ("1.2.3", stats_1_2_3),
            ("3", stats_3),
        ]
    )
    expected = valid_3

    stats_plan = StatsPlan(graph)
    for stage in stats_plan.stages:
        stats_collected = {module_path: [] for module_path in stage.module_paths}
        has_graph = not isinstance(stage.graph, torch.nn.Identity)
        for ds in ds_shards:
            batch_stats = {module_path: [] for module_path in stage.module_paths}
            for batch in ds:
                inputs = stage.graph(*batch) if has_graph else batch
                for module_path, input_idx in zip(stage.module_paths, stage.input_mapping):
                    if input_idx == STATS_PLAN_INPUT_MAPPING_ALL:
                        stat = stats_plan.root.get_submodule(module_path).calculate_stats(inputs)
                    elif input_idx == STATS_PLAN_INPUT_MAPPING_BYPASS:
                        stat = stats_plan.root.get_submodule(module_path).calculate_stats(*batch)
                    else:
                        stat = stats_plan.root.get_submodule(module_path).calculate_stats(inputs[input_idx])
                    batch_stats[module_path].append(stat)
            for module_path, stats in batch_stats.items():
                stats_collected[module_path].append(
                    stats_plan.root.get_submodule(module_path).combine_stats(batch_stats[module_path])
                )
        for module_path in stage.module_paths:
            mod = stats_plan.root.get_submodule(module_path)
            mod.apply_stats(mod.combine_stats(stats_collected[module_path]))
    received = stats_plan.root(valid)

    for k, v in expected_stats.items():
        assert torch.allclose(stats_plan.root.get_submodule(k).total, v), f"stats for `{k}` should match"
    assert torch.allclose(received, expected)
