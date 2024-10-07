from typing import Dict, List

import torch

from torchestra import (
    CountLookup,
    IndexLookup,
    IntCountLookup,
    IntIndexLookup,
    IntRatioLookup,
    MinThreshold,
    RatioLookup,
    RatioThreshold,
    TopK,
)


class CalculateStatsModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x: List[str]) -> Dict[str, int]:
        return self.mod.calculate_stats(x)


class CombineStatsModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x: List[Dict[str, int]]) -> Dict[str, int]:
        return self.mod.combine_stats(x)


class IntCalculateStatsModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x: torch.Tensor) -> Dict[int, int]:
        return self.mod.calculate_stats(x)


class IntCombineStatsModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x: List[Dict[int, int]]) -> Dict[int, int]:
        return self.mod.combine_stats(x)


def test_count_lookup():
    module = CountLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert stats1 == {"a": 5, "b": 4, "c": 2, "d": 2}
    assert stats2 == {"a": 3, "b": 3, "c": 1, "d": 2, "e": 4}
    assert combined_stats == {"a": 8, "b": 7, "c": 3, "d": 4, "e": 4}
    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_count_lookup_eliminator():
    module = CountLookup(MinThreshold(5))
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8, 7, 0, 0, 0, 0]))


def test_count_lookup_state_dict():
    module = CountLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = CountLookup()
    module.load_state_dict(state_dict)

    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_count_lookup_empty():
    module = CountLookup()
    initial = torch.jit.script(module)(["a", "b", "c"])
    stats = module.calculate_stats([])
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(["a", "b", "c"])
    received_compiled = compiled(["a", "b", "c"])

    assert all(initial == torch.tensor([0, 0, 0]))
    assert all(received == torch.tensor([0, 0, 0]))
    assert all(received_compiled == torch.tensor([0, 0, 0]))


def test_count_lookup_script_stats():
    module = CountLookup()
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))
    stats1 = calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert all(compiled(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_ratio_lookup():
    module = RatioLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0]))


def test_ratio_lookup_eliminator():
    module = RatioLookup(MinThreshold(5))
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8 / 15, 7 / 15, 0.0, 0.0, 0.0, 0.0]))


def test_ratio_lookup_state_dict():
    module = RatioLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = RatioLookup()
    module.load_state_dict(state_dict)

    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0]))


def test_ratio_lookup_empty():
    module = RatioLookup()
    initial = torch.jit.script(module)(["a", "b", "c"])
    stats = module.calculate_stats([])
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(["a", "b", "c"])
    received_compiled = compiled(["a", "b", "c"])

    assert all(initial == torch.tensor([0.0, 0.0, 0.0]))
    assert all(received == torch.tensor([0.0, 0.0, 0.0]))
    assert all(received_compiled == torch.tensor([0.0, 0.0, 0.0]))


def test_ratio_lookup_script_stats():
    module = RatioLookup()
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))
    stats1 = calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert all(compiled(["a", "b", "c", "d", "e", "f"]) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0]))


def test_index_lookup():
    module = IndexLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 7
    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_index_lookup_custom_indices():
    module = IndexLookup(padding_idx=1, unknown_idx=2)
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 7
    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([0, 3, 6, 4, 5, 2]))


def test_index_lookup_state_dict():
    module = IndexLookup()
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = IndexLookup()
    module.load_state_dict(state_dict)

    assert module.dictionary_size() == 7
    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_index_lookup_empty():
    module = IndexLookup()
    initial = torch.jit.script(module)(["a", "b", "c"])
    stats = module.calculate_stats([])
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(["a", "b", "c"])
    received_compiled = compiled(["a", "b", "c"])

    assert module.dictionary_size() == 2
    assert all(initial == torch.tensor([1, 1, 1]))
    assert all(received == torch.tensor([1, 1, 1]))
    assert all(received_compiled == torch.tensor([1, 1, 1]))


def test_index_lookup_script_stats():
    module = IndexLookup()
    calculate_stats = torch.jit.script(CalculateStatsModule(module))
    combine_stats = torch.jit.script(CombineStatsModule(module))
    stats1 = calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert module.dictionary_size() == 7
    assert all(compiled(["a", "b", "c", "d", "e", "f"]) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_index_lookup_eliminator():
    module = IndexLookup(eliminator=MinThreshold(5))
    stats1 = module.calculate_stats(["a", "b", "a", "c", "a", "b", "d", "a", "b", "c", "a", "b", "d"])
    stats2 = module.calculate_stats(["a", "b", "a", "e", "e", "e", "d", "e", "b", "c", "a", "b", "d"])
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 4
    assert all(module(["a", "b", "c", "d", "e", "f"]) == torch.tensor([2, 3, 1, 1, 1, 1]))


def test_index_lookup_empty_input():
    module = IndexLookup()

    received = module([])

    assert received.dtype == torch.int64
    assert all(received == torch.tensor([], dtype=torch.int64))


def test_int_count_lookup():
    module = IntCountLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert stats1 == {1: 5, 2: 4, -3: 2, 4: 2}
    assert stats2 == {1: 3, 2: 3, -3: 1, 4: 2, 5: 4}
    assert combined_stats == {1: 8, 2: 7, -3: 3, 4: 4, 5: 4}
    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_int_count_lookup_eliminator():
    module = IntCountLookup(MinThreshold(5))
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8, 7, 0, 0, 0, 0]))


def test_int_count_lookup_state_dict():
    module = IntCountLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = IntCountLookup()
    module.load_state_dict(state_dict)

    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_int_count_lookup_empty():
    module = IntCountLookup()
    initial = torch.jit.script(module)(torch.tensor([1, 2, 3]))
    stats = module.calculate_stats(torch.tensor([]))
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(torch.tensor([1, 2, 3]))
    received_compiled = compiled(torch.tensor([1, 2, 3]))

    assert all(initial == torch.tensor([0, 0, 0]))
    assert all(received == torch.tensor([0, 0, 0]))
    assert all(received_compiled == torch.tensor([0, 0, 0]))


def test_int_count_lookup_script_stats():
    module = IntCountLookup()
    calculate_stats = torch.jit.script(IntCalculateStatsModule(module))
    combine_stats = torch.jit.script(IntCombineStatsModule(module))
    stats1 = calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert all(compiled(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8, 7, 3, 4, 4, 0]))


def test_int_ratio_lookup():
    module = IntRatioLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0]))


def test_int_ratio_lookup_eliminator():
    module = IntRatioLookup(MinThreshold(5))
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8 / 15, 7 / 15, 0.0, 0.0, 0.0, 0.0]))


def test_int_ratio_lookup_state_dict():
    module = IntRatioLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = IntRatioLookup()
    module.load_state_dict(state_dict)

    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0]))


def test_int_ratio_lookup_empty():
    module = IntRatioLookup()
    initial = torch.jit.script(module)(torch.tensor([1, 2, 3]))
    stats = module.calculate_stats(torch.tensor([]))
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(torch.tensor([1, 2, 3]))
    received_compiled = compiled(torch.tensor([1, 2, 3]))

    assert all(initial == torch.tensor([0.0, 0.0, 0.0]))
    assert all(received == torch.tensor([0.0, 0.0, 0.0]))
    assert all(received_compiled == torch.tensor([0.0, 0.0, 0.0]))


def test_int_ratio_lookup_script_stats():
    module = IntRatioLookup()
    calculate_stats = torch.jit.script(IntCalculateStatsModule(module))
    combine_stats = torch.jit.script(IntCombineStatsModule(module))
    stats1 = calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert all(
        compiled(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([8 / 26, 7 / 26, 3 / 26, 4 / 26, 4 / 26, 0.0])
    )


def test_int_index_lookup():
    module = IntIndexLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 7
    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_int_index_lookup_custom_indices():
    module = IntIndexLookup(padding_idx=1, unknown_idx=2)
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 7
    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([0, 3, 6, 4, 5, 2]))


def test_int_index_lookup_state_dict():
    module = IntIndexLookup()
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    state_dict = module.state_dict()
    module = IntIndexLookup()
    module.load_state_dict(state_dict)

    assert module.dictionary_size() == 7
    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_int_index_lookup_empty():
    module = IntIndexLookup()
    initial = torch.jit.script(module)(torch.tensor([1, 2, 3]))
    stats = module.calculate_stats(torch.tensor([]))
    combined_stats = module.combine_stats([stats])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    received = module(torch.tensor([1, 2, 3]))
    received_compiled = compiled(torch.tensor([1, 2, 3]))

    assert module.dictionary_size() == 2
    assert all(initial == torch.tensor([1, 1, 1]))
    assert all(received == torch.tensor([1, 1, 1]))
    assert all(received_compiled == torch.tensor([1, 1, 1]))


def test_int_index_lookup_script_stats():
    module = IntIndexLookup()
    calculate_stats = torch.jit.script(IntCalculateStatsModule(module))
    combine_stats = torch.jit.script(IntCombineStatsModule(module))
    stats1 = calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)
    compiled = torch.jit.script(module)

    assert module.dictionary_size() == 7
    assert all(compiled(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([2, 3, 6, 4, 5, 1]))


def test_int_index_lookup_eliminator():
    module = IntIndexLookup(eliminator=MinThreshold(5))
    stats1 = module.calculate_stats(torch.tensor([1, 2, 1, -3, 1, 2, 4, 1, 2, -3, 1, 2, 4]))
    stats2 = module.calculate_stats(torch.tensor([1, 2, 1, 5, 5, 5, 4, 5, 2, -3, 1, 2, 4]))
    combined_stats = module.combine_stats([stats1, stats2])
    module.apply_stats(combined_stats)

    assert module.dictionary_size() == 4
    assert all(module(torch.tensor([1, 2, -3, 4, 5, 6])) == torch.tensor([2, 3, 1, 1, 1, 1]))


def test_min_threshold():
    module = MinThreshold(20)
    compiled = torch.jit.script(module)

    assert module({"a": 19, "b": 20, "c": 21}) == {"b": 20, "c": 21}
    assert compiled({"a": 19, "b": 20, "c": 21}) == {"b": 20, "c": 21}


def test_ratio_threshold():
    module = RatioThreshold(0.25)
    compiled = torch.jit.script(module)

    assert module({"a": 24, "b": 1, "c": 25, "d": 50}) == {"c": 25, "d": 50}
    assert compiled({"a": 24, "b": 1, "c": 25, "d": 50}) == {"c": 25, "d": 50}


def test_top_k():
    module = TopK(2)
    compiled = torch.jit.script(module)

    assert module({"a": 1, "b": 2, "c": 3, "d": 4}) == {"c": 3, "d": 4}
    assert compiled({"a": 1, "b": 2, "c": 3, "d": 4}) == {"c": 3, "d": 4}


def test_top_k_stable():
    module = TopK(3)
    compiled = torch.jit.script(module)

    assert module({"a": 4, "b": 4, "c": 4, "d": 4}) == {"a": 4, "b": 4, "c": 4}
    assert compiled({"a": 4, "b": 4, "c": 4, "d": 4}) == {"a": 4, "b": 4, "c": 4}
