# torchestra

[![CI status](https://github.com/jussi-kalliokoski/torchestra/workflows/CI/badge.svg)](https://github.com/jussi-kalliokoski/torchestra/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)

A collection of feature engineering utilities for easily composing preprocessing pipelines for PyTorch models.

## Introduction

PyTorch is a very powerful tool for building machine learning models by mostly "just" writing Python. What's usually left as an exercise for the reader, however, is how we get the data to be in a format that the model can easily learn from, especially when you have various types of input data that need to be preprocessed in different ways. This is where `torchestra` comes in. It aims to make feature engineering a bit more like building a model, by providing a set of utilities for easily composing preprocessing pipelines for PyTorch models that can then be inspected to for example automatically build stats calculations for normalization, dictionary lookups, etc.

The following example shows how you can use `torchestra` to build a preprocessing pipeline for a model that takes in a bunch of inputs, normalizes the continuous scalar inputs, and indexes the categorical string inputs:

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
from torchestra import Sequential, Parallel, IndexLookup, StandardScore, NanToNum, Clamp, Stack, SplitToDict


@field_modules
@torch.jit.script
@dataclass
class Inputs:
    temperature: torch.Tensor
    humidity: torch.Tensor
    wind_speed: torch.Tensor
    wind_direction: List[str]
    city: List[str]
    country: List[str]


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()

        continuous_scalar_inputs = [
            Inputs.temperature,
            Inputs.humidity,
            Inputs.wind_speed,
        ]

        categorical_str_inputs = [
            Inputs.wind_direction,
            Inputs.city,
            Inputs.country,
        ]

        self.continuous_scalar_features = Sequential(
            Parallel(x for x in continuous_scalar_inputs),
            Stack(dim=1),
            StandardScore.stack([StandardScore() for _ in continuous_scalar_inputs]),
            NanToNum(),
            Clamp(min=-1.0, max=1.0),
            SplitToDict(names=[x.field for x in continuous_scalar_inputs]),
        )

        self.categorical_str_features = Parallel(
            (Sequential(x, IndexLookup()) for x in categorical_str_inputs),
            into=dict,
            names=[x.field for x in categorical_str_inputs],
        )

        self.out = Parallel([self.continuous_scalar_features, self.categorical_str_features], into=tuple)

    def forward(self, inputs: Inputs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.out(inputs)
```

This is all that's needed to define this fairly complex set of features. From here on, calculating the stats for normalization, indexing the categorical strings, etc. can be done automatically by hooking up `Features.out` into a `torchestra.StatsPlan`, executing it on your data (for example using [Ray](https://www.ray.io)), then saving up the state dict for reloading the stats into a `Features` instance at a later time.

## Installation

The recommended way of installing `torchestra` is through [`uv`](https://docs.astral.sh/uv/):

```bash
uv add torchestra
```

## Key Concepts

The main composing primitives provided currently are `Sequential` and `Parallel`. `Sequential` is a composition of modules that are applied in sequence, and `Parallel` is a composition of functions that can be applied in parallel. You can compose these to build arrangements of preprocessing steps that can not only be applied to your data, but also inspected to for example automatically build stats calculations for normalization, dictionary lookups, etc.

### Sequential

The `Sequential` class can be thought of a drop-in-replacement for `torch.nn.Sequential`, but with the modules being able to accept any types of arguments (and if the first in the chain, any number as well) and return any type of output, as long as they are compatible with the next module in the chain. This allows you to build preprocessing pipelines that can be easily composed and inspected. For example:

```python
class CombineStrs(torch.nn.Module):
    def forward(self, s1: str, s2: str) -> str:
        return s1 + s2


class StrLen(torch.nn.Module):
    def forward(self, s: str) -> torch.Tensor:
        return torch.Tensor(len(s))

arrangement = Sequential(CombineStrs(), StrLen())

print(arrangement("foo", "bar"))  # tensor(6.)
```

### Parallel

The `Parallel` class is a composition of functions that can be applied in parallel. Note that this doesn't necessarily mean that they will be applied in parallel, that's up to the graph executor, but they're independent of each other so conceptually they're parallel paths, and can be executed in parallel. Like `Sequential`, it can accept any number of any type of arguments, as long as every module accepts the same arguments. The outputs can also be of any type, excpet if the result is combined into a `list` or `dict`, where there's an additional restriction of all the outputs being of the same type. This is useful for example when you have an input that needs to be passed to multiple different modules, and you want to combine the results in some way. For example:

```python
class Add(torch.nn.Module):
    def __init__(self, y: float):
        super().__init__()
        self.y = y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.y


arrangement = Parallel([Add(1.), Add(2.)])

print(arrangement(torch.Tensor(1.)))  # [tensor(2.), tensor(3.)]
```

You can also combine the results into a dictionary:

```python
arrangement = Parallel([Add(1.), Add(2.)], into=dict, names=["a", "b"])

print(arrangement(torch.Tensor(1.)))  # {'a': tensor(2.), 'b': tensor(3.)}
```

Or a tuple of arbitrary types:

```python
class ToStr(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> str:
        return str(x.item())

arrangement = Parallel([Add(1.), ToStr()], into=tuple)

print(arrangement(torch.Tensor(1.)))  # (tensor(2.), '1.0')
```

Or even a dataclass:

```python
@torch.jit.script
@dataclass
class AB:
    a: torch.Tensor
    b: str

arrangement = Parallel([Add(1.), ToStr()], into=AB)

print(arrangement(torch.Tensor(1.)))  # AB(a=tensor(2.), b='1.0')
```

### Composing Sequential and Parallel

Sequential and parallel can be composed arbitrarily into "arrangements" that can be applied to your data. For example:

```python
arrangement = Sequential(
    Parallel([
        Sequential(Add(1.0), Add(5.0)),
        Sequential(Add(9.0), Add(12.0)),
    ]),
    Stack(),
    Parallel([
        Sequential(Add(258.0), Add(236.0)),
        Sequential(Add(1895.0), Add(1785.0)),
    ]),
    Stack(),
)
```

When you have an arrangement composed of the two, it means that the data flow can be inspected programmatically, and you can for example automatically build stats calculations for normalization, dictionary lookups, etc.

## Design Goals

### Leaky

`torchestra` should not feel like an attempt at hiding PyTorch, but rather an extension of it. If it works in PyTorch, it should definitely work in `torchestra`, and if it doesn't, it's probably a bug. You should not need to change the way you write most of your PyTorch code to use `torchestra`. Most your modules should still be plain `torch.nn.Module`s, and you should be able to use them as such.

### Composable

`torchestra` should be easy to compose. You should be able to build complex preprocessing pipelines by just composing simple building blocks. This means that the building blocks must be carefully considered to be small in scope, and often this means splitting higher level abstractions into smaller ones. For example, T-digest is an algorithm for estimating quantiles for big data, and can be useful for a variety of different kinds of statistical modules. Therefore it is exposed as `torchestra.TDigest` that contains the implementation of the algorithm in PyTorch, and there's another module `torchestra.TDigestDistribution` that uses `TDigest` to collect its statistics for converting the input data into quantile approximations.

## License

MIT license, see `LICENSE.md` for details.
