from collections.abc import Mapping
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util
from jaxtyping import Array, Float

from jarnvid import batch_shape, concat, expand, stack, take, vectorize


@tree_util.register_dataclass
@dataclass(frozen=True)
class Start:
    stack: Float[Array, "*batch unit"]
    terrain: Float[Array, "*batch cell"]
    radius: float = field(default=1.0)


@dataclass(frozen=True)
class Unregistered:
    stack: Float[Array, "*batch unit"]


def test_stack_keeps_a_shared_unbatched_leaf_once() -> None:
    terrain = jnp.ones(5)
    stacked = stack([Start(jnp.zeros(3), terrain), Start(jnp.ones(3), terrain)])
    assert stacked.terrain is terrain
    assert stacked.stack.shape == (2, 3)
    assert batch_shape(stacked) == (2,)


def test_stack_batches_unbatched_leaves_that_differ() -> None:
    stacked = stack([Start(jnp.zeros(3), jnp.zeros(5)), Start(jnp.ones(3), jnp.ones(5))])
    assert stacked.terrain.shape == (2, 5)


def test_stack_rejects_plain_values_that_differ() -> None:
    with pytest.raises(ValueError, match="Plain values differ"):
        stack([Start(jnp.zeros(3), jnp.ones(5), 1.0), Start(jnp.zeros(3), jnp.ones(5), 2.0)])


def test_take_and_concat_leave_a_shared_unbatched_leaf_alone() -> None:
    terrain = jnp.ones(5)
    starts = Start(jnp.arange(12.0).reshape(4, 3), terrain)
    joined = concat([starts, take(starts, jnp.array([0]))])
    assert joined.terrain is terrain
    np.testing.assert_array_equal(take(joined, 4).stack, starts.stack[0])


def test_concat_broadcasts_size_one_batch_axes() -> None:
    starts = Start(jnp.arange(6.0).reshape(2, 3), jnp.ones((1, 5)))
    assert concat([starts, starts]).terrain.shape == (4, 5)


def test_expand_inserts_axes_around_the_batch_only() -> None:
    expanded = expand(Start(jnp.zeros((4, 3)), jnp.ones(5)), before=1, after=2)
    assert expanded.stack.shape == (1, 4, 1, 1, 3)
    assert expanded.terrain.shape == (5,)


def test_take_rejects_a_structure_without_batch_axes() -> None:
    with pytest.raises(ValueError, match="no batch axes"):
        take(Start(jnp.zeros(3), jnp.ones(5)), 0)


def test_rejects_unregistered_dataclasses() -> None:
    with pytest.raises(TypeError, match="register Unregistered"):
        batch_shape(Unregistered(jnp.zeros((2, 3))))


def test_vectorize_traverses_annotated_dicts() -> None:
    @vectorize
    def total(starts: dict[str, Start]) -> Array:
        assert starts["blue"].stack.shape == (3,)
        return starts["blue"].stack.sum() + starts["red"].terrain.sum()

    starts = {"blue": Start(jnp.ones((2, 3)), jnp.ones(5)), "red": Start(jnp.ones(3), jnp.ones((2, 5)))}
    np.testing.assert_array_equal(total(starts), jnp.full(2, 8.0))


def test_vectorize_traverses_mappings() -> None:
    @vectorize
    def total(starts: Mapping[str, Start]) -> Array:
        return starts["blue"].stack.sum()

    assert total({"blue": Start(jnp.ones((2, 3)), jnp.ones(5))}).shape == (2,)


def test_vectorize_rejects_arrays_in_containers_it_does_not_traverse() -> None:
    @vectorize
    def total(starts: Start, extra: object) -> Array:
        return starts.stack.sum()

    with pytest.raises(TypeError, match="extra holds arrays in a list"):
        total(Start(jnp.ones((2, 3)), jnp.ones(5)), [jnp.ones(2)])
