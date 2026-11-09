"""`vectorize`: write a function for one instance, call it on any batch.

Each argument's annotation gives, for every array leaf, the trailing axes one call sees; the
leading axes left over are batch axes. Batch axes broadcast across leaves **aligned from the
left**: the shared batch is the broadcast of every leaf's first `r` batch axes, `r` the fewest
any leaf has, and a leaf's further batch axes stay inside each call (docs/adr/0001). The batch
is flattened to one axis, mapped (`placement`), and reshaped back.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
from math import prod
from typing import get_type_hints

import jax.numpy as jnp
from jax import Array, tree
from jax.tree_util import keystr
from jaxtyping import PyTree

from .annotation import Core, ranks
from .placement import Local, Placement, run

# A function whose signature vectorize preserves.
type Function[**P, R] = Callable[P, R]


def vectorize[**P, R](fn: Function[P, R]) -> Function[P, R]:
    """Vectorize a PyTree function from its jaxtyping annotations."""
    return Vectorizer()(fn)


@dataclass(frozen=True, slots=True)
class Vectorizer:
    """A configured ``vectorize``: where the batch runs. What is mapped is read from the
    annotations alone; `Extend` keeps named axes inside each call."""

    placement: Placement = field(default_factory=Local)

    def __call__[**P, R](self, fn: Function[P, R]) -> Function[P, R]:
        call = signature(fn)
        hints = get_type_hints(fn, include_extras=True)

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = call.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            layout = {name: ranks(arguments[name], hints.get(name), name) for name in arguments}
            batches = {name: batch_shapes(name, arguments[name], rank) for name, rank in layout.items()}
            batch = shared({path: shape for shapes in batches.values() for path, shape in shapes.items()})
            if not batch:
                return fn(*args, **kwargs)
            values = {name: tree.map(lambda leaf, r: flatten(leaf, r, batch), arguments[name], rank) for name, rank in layout.items()}

            def apply(inner: dict[str, PyTree]) -> R:
                one = call.bind_partial()
                one.arguments.update(arguments | inner)
                return fn(*one.args, **one.kwargs)

            result = run(self.placement, apply, values, layout, prod(batch))
            return tree.map(lambda leaf: leaf.reshape(batch + leaf.shape[1:]), result)

        return wrapped


def batch_shapes(name: str, value: PyTree, rank: PyTree) -> dict[str, tuple[int, ...]]:
    """The batch shape (axes before the core) of every batched array leaf, by path."""
    pairs = zip(tree.leaves_with_path(value), tree.leaves(rank), strict=True)
    return {name + keystr(path): leaf.shape[: leaf.ndim - r.ndim] for (path, leaf), r in pairs if isinstance(r, Core)}


def shared(shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    """The broadcast of every batched leaf's first `r` batch axes, `r` the fewest any has."""
    if not shapes:
        return ()
    rank = min(map(len, shapes.values()))
    try:
        return jnp.broadcast_shapes(*(shape[:rank] for shape in shapes.values()))
    except ValueError:
        leaves = ", ".join(f"{path} {shape}" for path, shape in shapes.items())
        raise ValueError(
            f"Batch axes do not broadcast in their first {rank}, aligned from the left: {leaves}. "
            "An axis meant to stay inside each call belongs in the annotation, or in Extend."
        ) from None


def flatten(leaf: PyTree, rank: PyTree, batch: tuple[int, ...]) -> PyTree:
    """A `Core` leaf broadcast to the shared batch, which is flattened to one leading axis; a
    leaf's batch axes beyond the shared ones stay inside the call."""
    if not isinstance(rank, Core):
        return jnp.asarray(leaf) if hasattr(leaf, "shape") else leaf
    array: Array = leaf
    rest = array.shape[len(batch) : array.ndim - rank.ndim] + array.shape[array.ndim - rank.ndim :]
    return jnp.broadcast_to(array, batch + rest).reshape((prod(batch), *rest))
