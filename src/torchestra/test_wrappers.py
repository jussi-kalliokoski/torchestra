import torch

from torchestra import Cat, Clamp, NanToNum, Stack, ToStr, Unsqueeze


def test_stack():
    module = Stack(dim=1)
    x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

    received = module(x)

    assert torch.equal(received, torch.tensor([[1, 4], [2, 5], [3, 6]]))


def test_cat():
    module = Cat(dim=0)
    x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

    received = module(x)

    assert torch.equal(received, torch.tensor([1, 2, 3, 4, 5, 6]))


def test_nan_to_num():
    module = NanToNum()
    jit_module = torch.jit.script(module)
    x = torch.tensor([1, 2, float("nan"), 4])

    received = module(x)
    jit_received = jit_module(x)

    assert torch.equal(received, torch.tensor([1, 2, 0, 4]))
    assert torch.equal(jit_received, torch.tensor([1, 2, 0, 4]))


def test_nan_to_num_nan_override():
    module = NanToNum(nan=5.0)
    jit_module = torch.jit.script(module)
    x = torch.tensor([1, 2, float("nan"), 4])

    received = module(x)
    jit_received = jit_module(x)

    assert torch.equal(received, torch.tensor([1, 2, 5, 4]))
    assert torch.equal(jit_received, torch.tensor([1, 2, 5, 4]))


def test_nan_to_num_posinf_override():
    module = NanToNum(posinf=5.0)
    jit_module = torch.jit.script(module)
    x = torch.tensor([1, 2, float("inf"), 4])

    received = module(x)
    jit_received = jit_module(x)

    assert torch.equal(received, torch.tensor([1, 2, 5, 4]))
    assert torch.equal(jit_received, torch.tensor([1, 2, 5, 4]))


def test_nan_to_num_neginf_override():
    module = NanToNum(neginf=5.0)
    jit_module = torch.jit.script(module)
    x = torch.tensor([1, 2, float("-inf"), 4])

    received = module(x)
    jit_received = jit_module(x)

    assert torch.equal(received, torch.tensor([1, 2, 5, 4]))
    assert torch.equal(jit_received, torch.tensor([1, 2, 5, 4]))


def test_clamp():
    module = Clamp(min=1, max=3)
    x = torch.tensor([0, 1, 2, 3, 4, 5])

    received = module(x)

    assert torch.equal(received, torch.tensor([1, 1, 2, 3, 3, 3]))


def test_unsqueeze():
    module = Unsqueeze(dim=1)
    x = torch.tensor([1, 2, 3])

    received = module(x)

    assert torch.equal(received, torch.tensor([[1], [2], [3]]))


def test_to_str():
    module = ToStr()
    x = torch.tensor([1, 2, 3])

    received = module(x)

    assert received == ["1", "2", "3"]
