"""Where the flattened batch runs: on the local devices, or split over one mesh axis."""

from dataclasses import dataclass
from typing import Protocol, assert_never

import jax
import jax.numpy as jnp
from jax import tree
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import PyTree

from .annotation import Core


@dataclass(frozen=True, slots=True)
class Local:
    """Map the flattened batch on the current devices."""


@dataclass(frozen=True, slots=True)
class Chunked:
    """Map the flattened batch `batch` instances at a time (``lax.map``): the same result as
    ``Local`` in bounded memory."""

    batch: int


@dataclass(frozen=True, slots=True)
class Sharded:
    """Split the one flattened batch axis over the active mesh's ``axis``."""

    axis: str


type Placement = Local | Chunked | Sharded


class Apply[R](Protocol):
    """One call of the wrapped function, on the mapped arguments by name."""

    def __call__(self, values: dict[str, PyTree], /) -> R: ...


def run[R](placement: Placement, apply: Apply[R], values: dict[str, PyTree], ranks: dict[str, PyTree], size: int) -> R:
    """`apply` mapped over the leading axis (of length `size`) of every `Core` leaf of `values`."""
    axes = tree.map(lambda rank: 0 if isinstance(rank, Core) else None, ranks)
    mapped = jax.vmap(apply, in_axes=(axes,))
    if size == 0:
        return mapped(values)
    match placement:
        case Local():
            return mapped(values)
        case Chunked(batch=batch):

            def at(index: jax.Array) -> R:
                return apply(tree.map(lambda leaf, rank: leaf[index] if isinstance(rank, Core) else leaf, values, ranks))

            return jax.lax.map(at, jnp.arange(size), batch_size=batch)
        case Sharded(axis=axis):
            return sharded(axis, mapped, values, ranks, size)
        case _:
            assert_never(placement)


def sharded[R](axis: str, mapped: Apply[R], values: dict[str, PyTree], ranks: dict[str, PyTree], size: int) -> R:
    """`mapped` over the batch split across the active mesh's `axis`.

    The batch is padded to a multiple of the axis with copies of its last instance, and the
    padding cut from a replicated result (a sharded axis cannot be cut unevenly). Unbatched
    arrays are replicated: shard_map cannot close over arrays sharded on Explicit axes.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if axis not in mesh.shape:
        raise ValueError(f"vectorize over Sharded({axis!r}) requires an active mesh with that axis")
    padding = -size % mesh.shape[axis]

    def spec(leaf: PyTree, rank: PyTree) -> PartitionSpec | None:
        if not hasattr(leaf, "shape"):
            return None
        return PartitionSpec(axis) if isinstance(rank, Core) else PartitionSpec()

    def place(leaf: PyTree, rank: PyTree) -> PyTree:
        if not hasattr(leaf, "shape"):
            return leaf
        if isinstance(rank, Core):
            leaf = jnp.pad(leaf, [(0, padding)] + [(0, 0)] * (leaf.ndim - 1), mode="edge")
        return jax.sharding.reshard(leaf, NamedSharding(mesh, spec(leaf, rank)))

    local = jax.shard_map(mapped, mesh=mesh, in_specs=(tree.map(spec, values, ranks),), out_specs=PartitionSpec(axis), check_vma=False)
    result = local(tree.map(place, values, ranks))
    if not padding:
        return result
    return tree.map(lambda leaf: jax.sharding.reshard(leaf, PartitionSpec())[:size], result)
