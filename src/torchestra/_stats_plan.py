from dataclasses import dataclass
from typing import Dict, Iterable, List, OrderedDict, Set, Tuple

import torch

from ._pipes import Parallel, Sequential, TupleAsArgs

STATS_PLAN_INPUT_MAPPING_ALL = -1
STATS_PLAN_INPUT_MAPPING_BYPASS = -2


@dataclass
class StatsPlanStage:
    module_paths: List[str]
    input_mapping: List[int]
    graph: torch.nn.Module


class StatsPlan:
    def __init__(self, root: torch.nn.Module):
        self.root = root
        self.dependencies = self._find_stats_modules(root, found={}, path="")
        self.stats_module_paths = list(self.dependencies.keys())
        self.stages = [*self._build_stages()]

    def _find_stats_modules(
        self, module: torch.nn.Module, found: Dict[str, Set[str]], path: str
    ) -> Dict[str, Set[str]]:
        if isinstance(module, Parallel):
            prev = found
            for i, m in enumerate(module):
                found = _union(found, self._find_stats_modules(m, prev, f"{path}.{i}" if path else str(i)), orig=prev)
            return found

        if isinstance(module, Sequential):
            for i, m in enumerate(module):
                found = _union(found, self._find_stats_modules(m, found, f"{path}.{i}" if path else str(i)), orig=found)
            return found

        if isinstance(module, TupleAsArgs):
            return _union(
                found, self._find_stats_modules(module.inner, found, f"{path}.inner" if path else "inner"), orig=found
            )

        if hasattr(module, "calculate_stats") and hasattr(module, "combine_stats") and hasattr(module, "apply_stats"):
            return {**found, path: set(found.keys())}

        return found

    def _build_stages(self) -> Iterable[StatsPlanStage]:
        for module_paths in self._get_execution_plan_modules():
            bypass = [i for i, path in enumerate(module_paths) if self._find_input(path) == ""]
            if len(bypass) < len(module_paths):
                graph, input_mapping = self._build_graph(
                    self._find_input(path) for i, path in enumerate(module_paths) if i not in bypass
                )
            else:
                graph, input_mapping = torch.nn.Identity(), []
            for i in bypass:
                input_mapping.insert(i, STATS_PLAN_INPUT_MAPPING_BYPASS)
            yield StatsPlanStage(module_paths, input_mapping, graph)

    def _build_graph(self, input_paths: Iterable[str]) -> Tuple[torch.nn.Module, List[int]]:
        by_input = OrderedDict()
        count = 0
        for i, path in enumerate(input_paths):
            count += 1
            by_input[path] = i

        input_mods = []
        output_indices = [STATS_PLAN_INPUT_MAPPING_ALL] * count

        if len(by_input) == 1:
            input_path = next(iter(by_input))
            while input_path:
                input_mods.append(self.root.get_submodule(input_path))
                input_path = self._find_input(input_path)
            input_mods.reverse()
            if len(input_mods) > 1:
                return Sequential(*input_mods), output_indices
            return input_mods[0], output_indices

        for i, (input_path, idx) in enumerate(by_input.items()):
            mod, _ = self._build_graph([input_path])
            input_mods.append(mod)
            output_indices[idx] = i
        return Parallel(input_mods, into=tuple), output_indices

    def _get_execution_plan_modules(self) -> Iterable[List[str]]:
        resolved: Set[str] = set()
        while len(resolved) < len(self.stats_module_paths):
            resolvable = [*self._resolvable(resolved)]
            resolved.update(resolvable)
            yield resolvable

    def _resolvable(self, resolved: Set[str]) -> Iterable[str]:
        for path in self.stats_module_paths:
            if path in resolved:
                continue
            if not self.dependencies[path].issubset(resolved):
                continue
            yield path

    def _find_input(self, path: str) -> str:
        if path == "":
            return ""

        parent_path, _, name = path.rpartition(".")
        parent_path, name = (parent_path, name) if name else (name, parent_path)
        if isinstance(self.root.get_submodule(parent_path), Sequential):
            if name == "0":
                return self._find_input(parent_path)
            if not parent_path:
                return f"{int(name) - 1}"
            return f"{parent_path}.{int(name) - 1}"
        return self._find_input(parent_path)


def _union(d1: Dict[str, Set[str]], d2: Dict[str, Set[str]], orig: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    if d2 is orig:
        return d1
    return {**d1, **d2}
