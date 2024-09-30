import functools
import inspect
import operator
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import torch

T = TypeVar("T")

# !!! HIC SUNT DRACONES !!!
#
# These modules leverage `torch.fx` to compose modules into inspectable
# pipelines. While they are extremely useful to enable building complex and
# efficient processing pipelines from simple primitives, they are also using a
# very underdocumented part of PyTorch to do that, so it can be difficult to
# understand how they work.
#
# In light of the poor documentation provided by Torch itself, here's a
# crash course of the key concepts:
# - `Graph.placeholder` creates an argument to the graph.
#   The syntax is graph.placeholder("argument_name", ArgumentType).
# - `Graph.call_function` calls a function with the given arguments.
#   The syntax is `graph.call_function(func, (arg1, arg2, ...), kwargs={"kw1": 1, "kw2": "foo"}, type_expr=OutputType)`.
#   The function can be a torch builtin or python builtin. This form is used to represent most of Python syntax,
#   e.g. `foo.bar` becomes `graph.call_function(getattr, (foo, "bar"))`.
# - `Graph.call_method` is similar to `call_function`, except it calls a method by
#   name on an object (the first argument in the args tuple).
#   The syntax is `graph.call_method("method_name", (target, arg1, arg2, ...), kwargs={"kw1": 1, "kw2": "foo"}, type_expr=OutputType)`.
# - `Graph.call_module` is used to call submodules in the module's namespace,
#   e.g. `self.nn_model(foo)` would represented as `graph.call_module("nn_model", (foo,), type_expr=torch.Tensor)`.
#   The syntax is `graph.call_module("module_name", (arg1, arg2, ...), kwargs="kw1": 1, "kw2": "foo"}, type_expr=OutputType)`.
# - `Graph.output` makes the graph return the given value,
#   e.g. `return foo` would be `graph.output(foo, type_expr=OutputType)`.
# - `GraphModule` takes a namespace and a `Graph` to construct a `torch.nn.Module` that executes the graph.
#   The namespace is usually another `torch.nn.Module` that
#   contains all the modules that would be called with Graph.call_module.
#
# Some best practices to keep in mind:
# - Always provide type hints for the arguments and return values of the functions.
#   PyTorch will just assume things are Tensors most of the time when you don't,
#   and then you will get really hard to decipher error messages.
# - Don't spend time on figuring out a way to things as graph operations if
#   it can be written as a plain `torch.nn.Module`. Just write the helper as a
#   normal module, add it to the namespace and call it. This applies especially
#   to things such as loops that can't be unrolled, things that require control
#   or stuff that's supposed to be simple but simply isn't, like inserting an
#   item to a dictionary.
# - Try to be the least amount of clever possible. This stuff is hard to read as it is.
# - UNIT TEST EXTENSIVELY. This stuff is hard to debug, and you will need to
#   make sure all your bases are covered. If the users of this code get hit by
#   graph errors, it will be really hard to figure out what they did wrong.
# - Add informative assertions every time someone stumbles into graph-related
#   errors to prevent other people from being hit by the same unhelpful message.


class FieldModule(torch.fx.GraphModule):
    """
    A module that returns a field of the input.

    This module is used to extract a field from a class.

    Args:
        cls: The class to extract the field from.
        field: The field to extract.
        field_type: The type of the field.
    """

    def __init__(self, cls: type, field: str, field_type: type):
        module_lib = torch.nn.Module()
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x", cls)
        val = graph.call_function(getattr, (placeholder, field), type_expr=field_type)
        graph.output(val, type_expr=field_type)
        super().__init__(module_lib, graph)
        self.cls = cls
        self.field = field
        self.field_type = field_type

    def __repr__(self) -> str:
        return f"FieldModule({self.cls.__name__}, {repr(self.field)}, {repr(self.field_type)})"


def field_modules(cls: type[T]) -> type[T]:
    """
    Decorator to add field modules to a class.

    What this does in practice is that when referring to the fields of a class,
    e.g. `Inputs.foo` instead of `inputs.foo`, you get a module that takes in
    the class and returns the given field.

    This means that when building pipelines, and some module in the pipeline
    returns a `field_modules` class, you can add a step to the pipeline after
    that to extract the field from the class by adding a `ThatClass.the_field`
    into the pipeline.

    Args:
        cls: The class to add field modules to.

    Returns:
        The class with field modules added.
    """
    for field, field_type in cls.__annotations__.items():
        setattr(cls, field, FieldModule(cls, field, field_type))
    return cls


class Sequential(torch.fx.GraphModule):
    """
    Composes provided modules to run in a sequence.

    This is similar to `torch.nn.Sequential`, but it can be used with modules
    that take any number and types of arguments and return any type of values.

    The combined signature takes in the arguments of the first module and
    returns the return type of the last module.

    To pass more than one argument to a module in between, combine with `Parallel`.

    The module is indexable, i.e. the `i`-th module can be accessed with `seq[i]`.

    The module is iterable, i.e. the modules can be iterated over.

    The number of modules can be accessed with `len(seq)`.

    Args:
        *args: The modules to compose.
    """

    def __init__(self, *args):
        mods = [*args]
        first_non_identity = _find_first_non_identity(mods)
        sig = _get_signature(mods[first_non_identity])
        module_lib = torch.nn.Module()
        for i, module in enumerate(mods):
            module_lib.add_module(str(i), module)
        graph = torch.fx.Graph()
        args = [graph.placeholder(p.name, p.annotation) for p in sig.parameters.values()]
        result = graph.call_module(str(first_non_identity), tuple(args), type_expr=sig.return_annotation)
        for i in range(first_non_identity + 1, len(mods)):
            if isinstance(mods[i], torch.nn.Identity):
                continue
            new_sig = _get_signature(mods[i])
            assert len(new_sig.parameters) == 1, "Only first module can accept multiple arguments"
            assert (
                sig.return_annotation == next(iter(new_sig.parameters.values())).annotation
            ), "Modules must have matching input and output types"
            sig = new_sig
            result = graph.call_module(str(i), (result,), type_expr=sig.return_annotation)
        graph.output(result, type_expr=sig.return_annotation)

        super().__init__(module_lib, graph)
        self._len = len(mods)

        # The identity modules get dropped when the graph is compiled,
        # so we need to add them back:
        for i, module in enumerate(mods):
            if isinstance(module, torch.nn.Identity):
                self.add_module(str(i), module)

    def __getitem__(self, idx) -> torch.nn.Module:
        if idx < 0:
            idx += self._len
        if idx >= self._len or idx < 0:
            raise IndexError
        return self.get_submodule(str(idx))

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[torch.nn.Module]:
        for i in range(self._len):
            yield self.get_submodule(str(i))

    def __repr__(self) -> str:
        return f"{Sequential.__name__}(" + ", ".join(repr(m) for m in self) + ")"


class Parallel(torch.fx.GraphModule):
    """
    Composes provided modules to run in "parallel".

    Similar to `Sequential`, this module allows for composing multiple modules
    into a pipeline, but instead of running them in sequence, it runs them
    independent of each other, allowing the graph to be parallelized.

    The provided modules can take any number of arguments of any type and
    return values of any type, as long as all the modules have the input
    signature, and if `into` is set to `list` or `dict`, the same return type.

    The combined signatures takes the same arguments as the provided modules,
    but returns a list, dict, tuple or provided dataclass of the return values
    of the provided modules.

    If `into` is set to `list` (default) the return type is a list of the
    return values. Additionally `names`, if provided, are only used for
    debugging purposes in this case.

    If `into` is set to `dict`, the return type is a dictionary with the
    `names` as keys and the return values as values. Names must be provided in
    this case.

    If `into` is set to `tuple`, the return type is a tuple of the return
    values. `names` are only used for debugging purposes in this case.

    If `into` is a dataclass type, the return type is an instance of the
    dataclass with the provided `names` as attributes. If `names` are not
    provided, they will default to the names of the fields defined for the
    dataclass.

    Args:
        modules: The modules to compose.
        names: The names of the modules, used for debugging purposes.
        into: The type to combine the return values into. Defaults to `list`.
    """

    def __init__(self, modules: Iterable[torch.nn.Module], names: Optional[List[str]] = None, into: type = list):
        mods = [m for m in modules]
        mod_sigs = [
            _get_signature(mod) if not isinstance(mod, torch.nn.Identity) else _identity_signature(mods) for mod in mods
        ]
        module_lib = torch.nn.Module()
        for i, module in enumerate(mods):
            module_lib.add_module(str(i), module)
        graph = torch.fx.Graph()
        args = [graph.placeholder(p.name, p.annotation) for p in mod_sigs[0].parameters.values()]
        results = []
        results_order = [i for i in range(len(mods))]
        if into not in (list, dict, tuple):
            if names is None:
                names = [k for k in inspect.signature(into).parameters]
            else:
                indices = {k: i for i, k in enumerate(inspect.signature(into).parameters)}
                results_order = [indices[k] for k in names]
        assert names is None or len(names) == len(mods), "names must be the same length as modules"
        for i, mod in enumerate(mods):
            name = names[i] if names is not None else None
            if isinstance(mod, torch.nn.Identity):
                results.append(args[0])
                continue
            results.append(
                graph.create_node(
                    "call_module", str(i), tuple(args), name=name, type_expr=mod_sigs[i].return_annotation
                )
            )
        results = [results[i] for i in results_order]

        if into is list:
            return_type = List[mod_sigs[0].return_annotation]  # type: ignore
            results_list = graph.call_function(list, tuple(), type_expr=return_type)
            for result in results:
                graph.call_method("append", (results_list, result), type_expr=None)
            graph.output(results_list, type_expr=return_type)
        elif into is dict:
            if names is None:
                raise ValueError("names must be provided when into=dict")
            entry_type = Tuple[str, mod_sigs[0].return_annotation]  # type: ignore
            entries_type = List[entry_type]  # type: ignore
            results_entries = graph.call_function(list, tuple(), type_expr=entries_type)
            for i, result in enumerate(results):
                result_tuple = graph.call_function(tuple, ((names[i], result),), type_expr=entry_type)
                graph.call_method("append", (results_entries, result_tuple), type_expr=None)
            return_type = Dict[str, mod_sigs[0].return_annotation]  # type: ignore
            results_dict = graph.call_function(dict, (results_entries,), type_expr=return_type)
            graph.output(results_dict, type_expr=return_type)
        elif into is tuple:
            tuple_type = Tuple.__getitem__(tuple([s.return_annotation for s in mod_sigs]))
            results_tuple = graph.call_function(tuple, (tuple(results),), type_expr=tuple_type)
            graph.output(results_tuple, type_expr=tuple_type)
        else:
            obj = graph.call_function(into, tuple(results), type_expr=into)
            graph.output(obj, type_expr=into)

        super().__init__(module_lib, graph)
        self._len = len(mods)
        self.into = into
        self.names = names

        # The identity modules get dropped when the graph is compiled,
        # so we need to add them back:
        for i, module in enumerate(mods):
            if isinstance(module, torch.nn.Identity):
                self.add_module(str(i), module)

    def __getitem__(self, idx) -> torch.nn.Module:
        if idx < 0:
            idx += self._len
        if idx >= self._len or idx < 0:
            raise IndexError
        return self.get_submodule(str(idx))

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[torch.nn.Module]:
        for i in range(self._len):
            yield self.get_submodule(str(i))

    def __repr__(self) -> str:
        return (
            f"{Parallel.__name__}(["
            + ", ".join(repr(m) for m in self)
            + f"], names={repr(self.names)}, into={repr(self.into)})"
        )


class TupleAsArgs(torch.fx.GraphModule):
    """
    Converts a module that takes multiple arguments into a module that takes a tuple.

    This module is used to convert a module that takes multiple arguments into a
    module that takes a single tuple argument. This can be useful when combining
    results from `Parallel` into a tuple and then have the next module expect the
    arguments with proper names.

    NOTE: Using subgraphs inside TupleAsArgs is not supported yet.

    Args:
        module: The module to convert.
    """

    def __init__(self, module: torch.nn.Module):
        sig = _get_signature(module)
        module_lib = torch.nn.Module()
        module_lib.add_module("inner", module)
        graph = torch.fx.Graph()
        arg_types = tuple(p.annotation for p in sig.parameters.values())
        all_args = graph.placeholder("x", Tuple.__getitem__(arg_types))
        args = [graph.call_function(operator.getitem, (all_args, i), type_expr=t) for i, t in enumerate(arg_types)]
        result = graph.call_module("inner", tuple(args), type_expr=sig.return_annotation)
        graph.output(result, type_expr=sig.return_annotation)

        super().__init__(module_lib, graph)
        self.inner = module

    def __repr__(self) -> str:
        return f"{TupleAsArgs.__name__}({repr(self.inner)})"


def _identity_signature(modules: List[torch.nn.Module]) -> inspect.Signature:
    sig = _get_signature(modules[_find_first_non_identity(modules)])
    assert len(sig.parameters) == 1, "Identity module can only be used when accepting a single parameter"
    return_annotation = next(iter(sig.parameters.values())).annotation
    return inspect.Signature(parameters=list(sig.parameters.values()), return_annotation=return_annotation)


def _find_first_non_identity(modules: List[torch.nn.Module]) -> int:
    for i, mod in enumerate(modules):
        if not isinstance(mod, torch.nn.Identity):
            return i
    raise ValueError("At least one module must not be Identity")


@functools.cache
def _get_signature(module: torch.nn.Module) -> inspect.Signature:
    if isinstance(module, torch.nn.Module) or isinstance(module, torch.fx.GraphModule):
        return inspect.signature(module.forward)
    raise ValueError(f"Unsupported module type: {type(module)}")
