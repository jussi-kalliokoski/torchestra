import torch

from torchestra import CombineDicts, SplitToDict


def test_split_to_dict():
    module = SplitToDict(names=["a", "b"])
    x = torch.tensor([[1, 2], [3, 4]])

    received = module(x)

    assert len(received) == 2
    assert torch.all(received["a"] == torch.tensor([1, 3]))
    assert torch.all(received["b"] == torch.tensor([2, 4]))


def test_combine_dicts():
    module = CombineDicts()
    x = [{"a": torch.tensor([1, 2])}, {"b": torch.tensor([3, 4])}]

    received = module(x)

    assert len(received) == 2
    assert torch.all(received["a"] == torch.tensor([1, 2]))
    assert torch.all(received["b"] == torch.tensor([3, 4]))
