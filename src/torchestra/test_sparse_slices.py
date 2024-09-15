import torch

from torchestra import (
    SparseIndices,
    SparseLen,
    SparseMapSequences,
    SparseTrunc,
    SparseTruncIndices,
    SparseTruncPad,
    SparseValues,
)


def test_sparse_trunc_indices():
    module = SparseTruncIndices(3)
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])

    received = module(indices)

    assert torch.equal(received, torch.tensor([[0, 2], [2, 5], [5, 8], [9, 9], [9, 12]]))


def test_sparse_values():
    module = SparseValues()
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    received = module((values, indices))

    assert torch.equal(received, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))


def test_sparse_indices():
    module = SparseIndices()
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    received = module((values, indices))

    assert torch.equal(received, torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]]))


def test_sparse_len():
    module = SparseLen()
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])

    received = module(indices)

    assert torch.equal(received, torch.tensor([2, 3, 4, 0, 3]))


def test_sparse_trunc():
    module = SparseTrunc(3)
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    received = module((values, indices))

    assert torch.equal(received[0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    assert torch.equal(received[1], torch.tensor([[0, 2], [2, 5], [5, 8], [9, 9], [9, 12]]))


def test_sparse_trunc_pad():
    module = SparseTruncPad(3)
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    received = module((values, indices))

    assert torch.equal(received, torch.tensor([[1, 2, 0], [3, 4, 5], [6, 7, 8], [0, 0, 0], [10, 11, 12]]))


def test_sparse_map_sequences():
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x.sum()

    module = SparseMapSequences(Mod())
    indices = torch.tensor([[0, 2], [2, 5], [5, 9], [9, 9], [9, 12]])
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    received = module((values, indices))

    assert torch.equal(received, torch.tensor([3, 12, 30, 0, 33]))
