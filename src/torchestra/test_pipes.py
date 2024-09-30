from dataclasses import dataclass
from typing import List

import pytest
import torch

from torchestra import Parallel, Sequential, TupleAsArgs, field_modules


@field_modules
@torch.jit.script
@dataclass
class Foo:
    a: List[str]
    b: torch.Tensor


class Add(torch.nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.n


def test_field_modules():
    class TestModule(torch.nn.Module):
        def forward(self, a: List[str], b: torch.Tensor) -> Foo:
            return Foo(a=a, b=b)

    module = Sequential(TestModule(), Parallel((Foo.a, Foo.b), into=tuple))
    compiled = torch.jit.script(module)
    inputs = (["abc", "def"], torch.tensor(1.0))

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == (["abc", "def"], torch.tensor(1.0))
    assert received_compiled == (["abc", "def"], torch.tensor(1.0))


def test_field_module_repr():
    assert repr(Foo.a) == "FieldModule(Foo, 'a', typing.List[str])"
    assert repr(Foo.b) == "FieldModule(Foo, 'b', <class 'torch.Tensor'>)"


def test_sequential_multiple_inputs():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    module = Sequential(TestModule(), Add(1.0))
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0), torch.tensor(2.0))

    received = compiled(*inputs)
    received_compiled = compiled(*inputs)

    assert received == 4.0
    assert received_compiled == 4.0


def test_sequential_single_input():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 3.0

    module = Sequential(TestModule(), Add(1.0))
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0),)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == 4.0
    assert received_compiled == 4.0


def test_sequential_multiple_inputs_in_middle():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return x + y + z

    with pytest.raises(Exception):
        _module = Sequential(Add(1.0), TestModule(), Add(1.0))


def test_sequential_non_matching_input():
    class TestModule(torch.nn.Module):
        def forward(self, x: str) -> torch.Tensor:
            return torch.tensor(len(x))

    with pytest.raises(Exception):
        _module = Sequential(Add(1.0), TestModule())


def test_sequential_identity_first():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    module = Sequential(torch.nn.Identity(), TestModule(), Add(1.0))
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0), torch.tensor(2.0))

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == 4.0
    assert received_compiled == 4.0


def test_sequential_identity_in_middle():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    module = Sequential(TestModule(), torch.nn.Identity(), Add(1.0))
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0), torch.tensor(2.0))

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == 4.0
    assert received_compiled == 4.0


def test_sequential_identity_last():
    class TestModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    module = Sequential(TestModule(), torch.nn.Identity())
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0), torch.tensor(2.0))

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == 3.0
    assert received_compiled == 3.0


def test_sequential_all_identity():
    with pytest.raises(Exception):
        _module = Sequential(torch.nn.Identity(), torch.nn.Identity())


def test_sequential_not_a_module():
    class TestModule:
        def forward(self, x):
            return x

    with pytest.raises(Exception):
        _module = Sequential(TestModule())


def test_sequential_iter():
    modules = [Add(1), torch.nn.Identity(), Add(1)]
    module = Sequential(*modules)

    received = [mod for mod in module]

    assert received == modules


def test_sequential_index():
    modules = [Add(1), torch.nn.Identity(), Add(1)]
    module = Sequential(*modules)

    assert len(module) == len(modules)
    assert module[0] == modules[0]
    assert module[1] == modules[1]
    assert module[2] == modules[2]
    assert module[-1] == modules[-1]

    with pytest.raises(IndexError):
        _ = module[3]

    with pytest.raises(IndexError):
        _ = module[-4]


def test_sequential_repr():
    module = Sequential(Add(1), torch.nn.Identity(), Add(1))

    received = repr(module)

    assert received == "Sequential(Add(), Identity(), Add())"


def test_parallel_multiple_inputs():
    class TestModule(torch.nn.Module):
        def __init__(self, z: float):
            super().__init__()
            self.z = z

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y + self.z

    module = Parallel([TestModule(1.0), TestModule(2.0)])
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0), torch.tensor(2.0))

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == [torch.tensor(4.0), torch.tensor(5.0)]
    assert received_compiled == [torch.tensor(4.0), torch.tensor(5.0)]


def test_parallel_single_input():
    module = Parallel([Add(1.0), Add(2.0)])
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0),)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == [torch.tensor(2.0), torch.tensor(3.0)]
    assert received_compiled == [torch.tensor(2.0), torch.tensor(3.0)]


def test_parallel_dict():
    module = Parallel([Add(1.0), Add(2.0)], into=dict, names=["a", "b"])
    compiled = torch.jit.script(module)
    inputs = (torch.tensor(1.0),)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
    assert received_compiled == {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}


def test_parallel_tuple():
    class TestModule(torch.nn.Module):
        def forward(self, x: str) -> int:
            return len(x)

    module = Parallel([TestModule(), torch.nn.Identity()], into=tuple)
    compiled = torch.jit.script(module)
    inputs = ("abc",)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == (3, "abc")
    assert received_compiled == (3, "abc")


def test_parallel_dataclass():
    class TestModule(torch.nn.Module):
        def forward(self, x: List[str]) -> torch.Tensor:
            return torch.tensor([len(_x) for _x in x]).sum()

    module = Parallel([torch.nn.Identity(), TestModule()], into=Foo)
    compiled = torch.jit.script(module)
    inputs = (["abc", "def"],)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == Foo(a=["abc", "def"], b=torch.tensor(6))
    assert received_compiled == Foo(a=["abc", "def"], b=torch.tensor(6))


def test_parallel_dataclass_reordered():
    class TestModule(torch.nn.Module):
        def forward(self, x: List[str]) -> torch.Tensor:
            return torch.tensor([len(_x) for _x in x]).sum()

    module = Parallel([TestModule(), torch.nn.Identity()], into=Foo, names=["b", "a"])
    compiled = torch.jit.script(module)
    inputs = (["abc", "def"],)

    received = module(*inputs)
    received_compiled = compiled(*inputs)

    assert received == Foo(a=["abc", "def"], b=torch.tensor(6))
    assert received_compiled == Foo(a=["abc", "def"], b=torch.tensor(6))


def test_parallel_dict_with_no_names():
    with pytest.raises(Exception):
        _module = Parallel([Add(1.0), Add(2.0)], into=dict)


def test_parallel_all_identity():
    with pytest.raises(Exception):
        _module = Parallel([torch.nn.Identity(), torch.nn.Identity()])


def test_parallel_not_a_module():
    class TestModule:
        def forward(self, x):
            return x

    with pytest.raises(Exception):
        _module = Parallel([TestModule()])


def test_parallel_iter():
    modules = [Add(1), torch.nn.Identity(), Add(1)]
    module = Parallel(modules)

    received = [mod for mod in module]

    assert received == modules


def test_parallel_index():
    modules = [Add(1), torch.nn.Identity(), Add(1)]
    module = Parallel(modules)

    assert len(module) == len(modules)
    assert module[0] == modules[0]
    assert module[1] == modules[1]
    assert module[2] == modules[2]
    assert module[-1] == modules[-1]

    with pytest.raises(IndexError):
        _ = module[3]

    with pytest.raises(IndexError):
        _ = module[-4]


def test_parallel_repr():
    module = Parallel([Add(1), torch.nn.Identity(), Add(1)])

    received = repr(module)

    assert received == "Parallel([Add(), Identity(), Add()], names=None, into=<class 'list'>)"


def test_tuple_as_args():
    class TestModule(torch.nn.Module):
        def forward(self, x: int, y: str) -> int:
            return x + len(y)

    module = TupleAsArgs(TestModule())
    compiled = torch.jit.script(module)
    inputs = (1, "foo")

    received = module(inputs)
    received_compiled = compiled(inputs)

    assert received == 4
    assert received_compiled == 4


def test_tuple_as_args_not_a_module():
    class TestModule:
        def forward(self, x):
            return x

    with pytest.raises(Exception):
        _module = TupleAsArgs(TestModule())


def test_tuple_as_args_inner():
    class TestModule(torch.nn.Module):
        def forward(self, x: int, y: str) -> int:
            return x + len(y)

    inner = TestModule()
    module = TupleAsArgs(inner)

    assert module.inner is inner


def test_tuple_as_args_repr():
    module = TupleAsArgs(Add(1))

    received = repr(module)

    assert received == "TupleAsArgs(Add())"
