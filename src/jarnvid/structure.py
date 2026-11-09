"""Batch-aware operations on structures: `take`, `stack`, `concat`, `expand`, `batch_shape`.

A structure is a registered dataclass whose annotations say which axes of each leaf are batch
axes, as `vectorize` reads them. Only leaves with batch axes move: a leaf without them (a terrain
kept once) stays once, and plain values (Python scalars) must agree.
"""

from collections.abc import Sequence
from typing import Protocol

import jax.numpy as jnp
from jax import Array, tree
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .annotation import Core, Fixed, Rank, ranks
from .vectorizer import batch_shapes, shared


def layout(value: PyTree) -> PyTree:
    """`value`'s `Rank`s, read from its own class's annotations."""
    return ranks(value, type(value), type(value).__name__)


def batch_shape(value: PyTree) -> tuple[int, ...]:
    """The batch `vectorize` maps `value` over."""
    return shared(batch_shapes(type(value).__name__, value, layout(value)))


def take[T](value: T, index: ArrayLike) -> T:
    """The instance (or instances) at `index` along the leading batch axis."""
    size = leading(value)
    return tree.map(lambda leaf, rank: lead(leaf, rank, size)[index] if isinstance(rank, Core) else leaf, value, layout(value))


def stack[T](values: Sequence[T]) -> T:
    """`values` along a new leading batch axis."""
    return merge(values, lambda leaves, _: jnp.stack(jnp.broadcast_arrays(*leaves)))


def concat[T](values: Sequence[T]) -> T:
    """`values` joined along their leading batch axis."""
    sizes = tuple(map(leading, values))
    return merge(values, lambda leaves, ranks: jnp.concatenate([lead(*parts) for parts in zip(leaves, ranks, sizes, strict=True)]))


def expand[T](value: T, before: int = 0, after: int = 0) -> T:
    """`value` with singleton batch axes before and after its batch axes; an outer product is
    `f(expand(xs, after=1), expand(ys, before=1))`."""
    rank = len(batch_shape(value))

    def insert(leaf: PyTree, r: PyTree) -> PyTree:
        if not isinstance(r, Core):
            return leaf
        return leaf.reshape((1,) * before + leaf.shape[:rank] + (1,) * after + leaf.shape[rank:])

    return tree.map(insert, value, layout(value))


class Join(Protocol):
    """Combine one leaf from each structure into one leaf."""

    def __call__(self, leaves: tuple[Array, ...], ranks: tuple[Rank, ...], /) -> Array: ...


def merge[T](values: Sequence[T], join: Join) -> T:
    """Leaf by leaf: agreeing plain values, and one leaf without batch axes that every structure
    shares, stay once; the rest are `join`ed."""
    count = len(values)

    def leaf(*items: PyTree) -> PyTree:
        leaves, rank = items[:count], items[count:]
        if not all(hasattr(item, "shape") for item in leaves):
            if any(item != leaves[0] for item in leaves):
                raise ValueError(f"Plain values differ across the structures: {leaves}")
            return leaves[0]
        if all(isinstance(r, Fixed) and item is leaves[0] for item, r in zip(leaves, rank, strict=True)):
            return leaves[0]
        return join(leaves, rank)

    return tree.map(leaf, *values, *map(layout, values))


def leading(value: PyTree) -> int:
    """The size of `value`'s leading batch axis."""
    shape = batch_shape(value)
    if not shape:
        raise ValueError(f"{type(value).__name__} has no batch axes")
    return shape[0]


def lead(leaf: Array, rank: Rank, size: int) -> Array:
    """`leaf` with a leading batch axis of `size`, broadcast if it has one, added if not."""
    shape = leaf.shape[1:] if isinstance(rank, Core) else leaf.shape
    return jnp.broadcast_to(leaf, (size, *shape))
