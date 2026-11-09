"""Reading annotations: for each leaf of an argument, how many trailing axes belong to one call.

An argument's annotation is walked together with its value, once: through type aliases,
`Annotated[..., Extend(...)]`, the one arm of a union that admits the value, tuples, dicts and
dataclasses, down to jaxtyping arrays annotated `*batch ...`. The result is a PyTree of the
value's structure whose leaves are `Core` (the array's last `ndim` axes belong to each call; the
rest, at least one, are batch axes) or `Fixed` (passed to every call unchanged).
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import UnionType
from typing import Annotated, TypeAliasType, TypeIs, Union, get_args, get_origin, get_type_hints

from jax import tree
from jaxtyping import AbstractArray, PyTree

# Any runtime annotation (a class, alias, union or Annotated form); PEP 747's TypeForm, which
# Python 3.13 lacks, is the precise type.
type TypeForm = object


@dataclass(frozen=True, init=False)
class Extend:
    """Insert semantic axes immediately after every nested ``*batch`` axis."""

    axes: tuple[str, ...]

    def __init__(self, *axes: str) -> None:
        if not axes or any(not axis for axis in axes):
            raise ValueError("Extend requires one or more nonempty axis names")
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True)
class Core:
    """An array leaf whose last `ndim` axes belong to each call; the rest are batch axes."""

    ndim: int


@dataclass(frozen=True)
class Fixed:
    """A leaf passed to every call unchanged."""


type Rank = Core | Fixed


def ranks(value: PyTree, annotation: TypeForm, path: str, extended: tuple[str, ...] = ()) -> PyTree:
    """`value`'s structure with a `Rank` at every leaf, checking each array against its annotation."""
    annotation, axes = resolve(value, annotation, path)
    extended += axes
    if value is None:
        return None
    if is_array(annotation):
        return array_rank(value, annotation, path, extended)
    if isinstance(value, tuple):
        items = zip(value, tuple_annotations(annotation, len(value), path), strict=True)
        return tuple(ranks(item, inner, f"{path}[{index}]", extended) for index, (item, inner) in enumerate(items))
    if isinstance(value, dict):
        inner = mapping_annotation(annotation, path)
        return {key: ranks(item, inner, f"{path}[{key!r}]", extended) for key, item in value.items()}
    if is_dataclass(value):
        return dataclass_ranks(value, annotation, path, extended)
    if hasattr(value, "shape"):
        raise TypeError(f"{path} must use a jaxtyping annotation beginning with '*batch'")
    if any(hasattr(leaf, "shape") for leaf in tree.leaves(value)):
        raise TypeError(f"{path} holds arrays in a {type(value).__name__}, which is not traversed: use a tuple, dict or dataclass")
    return tree.map(lambda _: Fixed(), value)


def array_rank(value: PyTree, annotation: type[AbstractArray], path: str, extended: tuple[str, ...]) -> Rank:
    """An array's `Rank`: `Core` if it has batch axes, else `Fixed`."""
    if not hasattr(value, "shape"):
        raise TypeError(f"{path} is annotated as an array but received {type(value).__name__}")
    core = core_ndim(annotation, path) + len(extended)
    if value.ndim < core:
        raise ValueError(f"{path} has shape {value.shape}, which is smaller than {extended_dims(annotation, extended)!r}")
    # A leaf without batch axes reaches every call as is, never copied across the batch.
    return Core(core) if value.ndim > core else Fixed()


def dataclass_ranks(value: PyTree, annotation: TypeForm, path: str, extended: tuple[str, ...]) -> PyTree:
    """A registered dataclass with the `Rank`s of its non-static fields; an unregistered one (a
    config) is `Fixed`, unless it holds arrays."""
    if any(leaf is value for leaf in tree.leaves(value)):
        if any(hasattr(leaf, "shape") for leaf in tree.leaves([getattr(value, field.name) for field in fields(value)])):
            raise TypeError(f"{path}: register {type(value).__name__} as a PyTree (jax.tree_util.register_dataclass)")
        return Fixed()
    if not isinstance(annotation, type) or not is_dataclass(annotation):
        raise TypeError(f"{path} must have an annotated dataclass type")
    hints = get_type_hints(annotation, include_extras=True)
    mapped = (field.name for field in fields(value) if not field.metadata.get("static", False))
    return replace(value, **{name: ranks(getattr(value, name), hints.get(name), f"{path}.{name}", extended) for name in mapped})


def mapping_annotation(annotation: TypeForm, path: str) -> TypeForm:
    """The value type of a ``dict[K, V]`` or ``Mapping[K, V]`` annotation."""
    if get_origin(annotation) not in (dict, Mapping) or len(get_args(annotation)) != 2:
        raise TypeError(f"{path} must have an annotated dict[K, V] or Mapping[K, V] type")
    return get_args(annotation)[1]


def resolve(value: PyTree, annotation: TypeForm, path: str) -> tuple[TypeForm, tuple[str, ...]]:
    """Unwrap aliases and ``Extend`` metadata, taking a union's one arm that admits ``value``."""
    annotation, axes = extension(annotation)
    while get_origin(annotation) in (Union, UnionType):
        arms = tuple(arm for arm in get_args(annotation) if admits(extension(arm)[0], value))
        if len(arms) != 1:
            raise TypeError(f"{path} ({type(value).__name__}) matches {len(arms)} arms of {annotation}, not one")
        annotation, arm_axes = extension(arms[0])
        axes += arm_axes
    return annotation, axes


def extension(annotation: TypeForm) -> tuple[TypeForm, tuple[str, ...]]:
    """Separate a runtime annotation from its (possibly nested) semantic-axis metadata."""
    axes: tuple[str, ...] = ()
    seen: set[int] = set()
    while True:
        if isinstance(annotation, TypeAliasType):
            if id(annotation) in seen:
                raise TypeError(f"Cyclic type alias: {annotation.__name__}")
            seen.add(id(annotation))
            if annotation.__type_params__:
                raise TypeError(f"Generic type aliases are not supported: {annotation.__name__}")
            annotation = annotation.__value__
        elif isinstance(get_origin(annotation), TypeAliasType):
            raise TypeError(f"Generic type aliases are not supported: {annotation}")
        elif get_origin(annotation) is Annotated:
            annotation, *metadata = get_args(annotation)
            axes += tuple(axis for item in metadata if isinstance(item, Extend) for axis in item.axes)
        else:
            return annotation, axes


def admits(annotation: TypeForm, value: PyTree) -> bool:
    """Whether one union arm describes a runtime value: arrays by shape, others by class."""
    if is_array(annotation):
        return hasattr(value, "shape")
    origin = get_origin(annotation) or annotation
    return isinstance(origin, type) and isinstance(value, origin)


def tuple_annotations(annotation: TypeForm, length: int, path: str) -> tuple[TypeForm, ...]:
    """Expand a fixed or variadic tuple annotation over its runtime values."""
    if get_origin(annotation) is not tuple:
        raise TypeError(f"{path} must have an annotated tuple type")
    annotations = get_args(annotation)
    if len(annotations) == 2 and annotations[1] is Ellipsis:
        return (annotations[0],) * length
    if len(annotations) != length:
        raise TypeError(f"{path} has {length} values but {len(annotations)} annotated tuple fields")
    return annotations


def is_array(annotation: TypeForm) -> TypeIs[type[AbstractArray]]:
    """Whether an annotation is a concrete jaxtyping array type."""
    return isinstance(annotation, type) and issubclass(annotation, AbstractArray)


def core_ndim(annotation: type[AbstractArray], path: str) -> int:
    """The axes to the right of the required leading ``*batch``."""
    dimensions = annotation.dim_str.split()
    if annotation.index_variadic != 0 or not dimensions or dimensions[0] != "*batch":
        raise TypeError(f"{path} must use a jaxtyping annotation beginning with '*batch'")
    return len(annotation.dims) - 1


def extended_dims(annotation: type[AbstractArray], extended: tuple[str, ...]) -> str:
    """The effective annotation, semantic axes included, for error messages."""
    batch, *core = annotation.dim_str.split()
    return " ".join((batch, *extended, *core))
