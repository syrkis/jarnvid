from dataclasses import dataclass
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Shaped

from jarnvid import Chunked, Extend, Local, Sharded, Vectorizer, vectorize


@tree_util.register_dataclass
@dataclass(frozen=True)
class Example:
    scalar: Shaped[Array, "*batch"]
    vector: Shaped[Array, "*batch feature"]


@tree_util.register_dataclass
@dataclass(frozen=True)
class Nested:
    example: Example
    vector: Shaped[Array, "*batch feature"]


@tree_util.register_dataclass
@dataclass(frozen=True)
class Transition:
    state: Shaped[Array, "*batch unit"]
    action: Shaped[Array, "*batch unit"]


@tree_util.register_dataclass
@dataclass(frozen=True)
class Group:
    agents: tuple[Example, ...]


@tree_util.register_dataclass
@dataclass(frozen=True)
class Located:
    scalar: Shaped[Array, "*batch location"]
    vector: Shaped[Array, "*batch location feature"]


type ExampleAlias = Example
type ChainedExampleAlias = ExampleAlias
type TeamExample = Annotated[ChainedExampleAlias, Extend("team")]
type ExampleTuple = tuple[ExampleAlias, ...]
type VectorAlias = Shaped[Array, "*batch feature"]
type CycleA = CycleB
type CycleB = CycleA
type GenericAlias[T] = tuple[T, ...]


@tree_util.register_dataclass
@dataclass(frozen=True)
class AliasedFields:
    example: ExampleAlias
    vector: VectorAlias


def test_vectorize_resolves_chained_dataclass_alias_under_jit() -> None:
    @vectorize
    def total(value: ChainedExampleAlias) -> Array:
        assert value.vector.ndim == 1
        return value.scalar + value.vector.sum()

    value = Example(jnp.arange(6).reshape(2, 3), jnp.ones((2, 3, 4)))
    np.testing.assert_array_equal(jax.jit(total)(value), value.scalar + 4)


def test_vectorize_preserves_extensions_across_alias_boundaries() -> None:
    @vectorize
    def total(value: Annotated[TeamExample, Extend("time")]) -> Array:
        assert value.scalar.shape == (3, 4)
        assert value.vector.shape == (3, 4, 5)
        return value.scalar.sum() + value.vector.sum()

    value = Example(jnp.ones((2, 3, 4)), jnp.ones((2, 3, 4, 5)))
    np.testing.assert_array_equal(jax.jit(total)(value), jnp.full((2,), 72))


def test_vectorize_resolves_tuple_array_and_field_aliases() -> None:
    @vectorize
    def total(values: ExampleTuple, nested: AliasedFields, extra: VectorAlias) -> Array:
        assert extra.ndim == 1
        return sum(value.scalar for value in values) + nested.example.scalar + nested.vector.sum() + extra.sum()

    value = Example(jnp.arange(2), jnp.ones((2, 3)))
    nested = AliasedFields(value, jnp.ones((2, 3)))
    np.testing.assert_array_equal(jax.jit(total)((value, value), nested, jnp.ones((2, 3))), 3 * value.scalar + 6)


def test_vectorize_rejects_cyclic_alias() -> None:
    @vectorize
    def identity(value: CycleA):
        return value

    with pytest.raises(TypeError, match="Cyclic type alias"):
        identity(jnp.ones(2))


def test_vectorize_rejects_generic_aliases_explicitly() -> None:
    @vectorize
    def bare(value: GenericAlias):
        return value

    @vectorize
    def specialized(value: GenericAlias[Example]):
        return value

    value = Example(jnp.ones(2), jnp.ones((2, 3)))
    for function in (bare, specialized):
        with pytest.raises(TypeError, match="Generic type aliases are not supported"):
            function((value,))


def test_vectorize_checks_rank_through_extended_alias() -> None:
    @vectorize
    def identity(value: TeamExample):
        return value

    with pytest.raises(ValueError, match="smaller than"):
        identity(Example(jnp.array(1), jnp.ones(3)))


def test_infers_arbitrary_leading_batch_dimensions() -> None:
    @vectorize
    def increment(value: Example) -> Example:
        return Example(value.scalar + 1, value.vector + 1)

    value = Example(
        scalar=jnp.arange(12).reshape(2, 3, 2),
        vector=jnp.arange(48).reshape(2, 3, 2, 4),
    )
    actual = increment(value)

    np.testing.assert_array_equal(actual.scalar, value.scalar + 1)
    np.testing.assert_array_equal(actual.vector, value.vector + 1)


def test_reuses_one_function_across_different_batch_ranks() -> None:
    @vectorize
    def increment(value: Example) -> Example:
        return Example(value.scalar + 1, value.vector + 1)

    unbatched = Example(scalar=jnp.array(1), vector=jnp.arange(4))
    batched = Example(
        scalar=jnp.arange(6).reshape(3, 1, 2),
        vector=jnp.arange(24).reshape(3, 1, 2, 4),
    )

    np.testing.assert_array_equal(increment(unbatched).scalar, 2)
    np.testing.assert_array_equal(increment(batched).vector, batched.vector + 1)


def test_broadcasts_batch_dimensions_under_jit() -> None:
    @vectorize
    def add(
        offset: Shaped[Array, "*batch"],
        value: Example,
    ) -> Example:
        return Example(value.scalar + offset, value.vector + offset)

    offset = jnp.array([[10], [20]])
    value = Example(
        scalar=jnp.arange(6).reshape(2, 3),
        vector=jnp.arange(24).reshape(2, 3, 4),
    )
    actual = jax.jit(add)(offset, value)

    np.testing.assert_array_equal(actual.scalar, value.scalar + offset)
    np.testing.assert_array_equal(actual.vector, value.vector + offset[..., None])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_sharded_vectorize_broadcasts_batch_dimensions_under_jit() -> None:
    @Vectorizer(placement=Sharded("data"))
    def add(
        offset: Shaped[Array, "*batch"],
        value: Example,
    ) -> Example:
        return Example(value.scalar + offset, value.vector + offset)

    batch = jax.device_count() * 2
    offset = jnp.array([10])
    value = Example(
        scalar=jnp.arange(batch),
        vector=jnp.arange(batch * 4).reshape(batch, 4),
    )
    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    with jax.set_mesh(mesh):
        actual = jax.jit(add)(offset, value)

    np.testing.assert_array_equal(actual.scalar, value.scalar + offset)
    np.testing.assert_array_equal(actual.vector, value.vector + offset[:, None])
    assert actual.scalar.sharding.spec == PartitionSpec("data")


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_sharded_vectorize_recurses_through_variadic_tuples_under_jit() -> None:
    @Vectorizer(placement=Sharded("data"))
    def combine(group: Group) -> Array:
        return group.agents[0].vector + group.agents[1].vector + group.agents[0].scalar

    batch = jax.device_count() * 2
    left = Example(
        scalar=jnp.arange(batch).reshape(batch, 1),
        vector=jnp.arange(batch * 4).reshape(batch, 4),
    )
    right = Example(
        scalar=jnp.arange(batch, 2 * batch).reshape(batch, 1),
        vector=jnp.arange(batch * 4, batch * 8).reshape(batch, 4),
    )
    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    with jax.set_mesh(mesh):
        actual = jax.jit(combine)(Group(agents=(left, right)))

    expected = left.vector + right.vector + left.scalar
    np.testing.assert_array_equal(actual, expected)
    assert actual.sharding.spec == PartitionSpec("data", None)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_sharded_vectorize_handles_local_indexing_under_jit() -> None:
    @Vectorizer(placement=Sharded("data"))
    def lookup(index: Shaped[Array, "*batch item"]) -> Array:
        table = jnp.arange(16, dtype=jnp.int32)
        return table[index]

    batch = jax.device_count() * 2
    index = jnp.arange(batch * 4, dtype=jnp.int32).reshape(batch, 4) % 16
    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    with jax.set_mesh(mesh):
        actual = jax.jit(lookup)(index)

    np.testing.assert_array_equal(actual, jnp.arange(16, dtype=jnp.int32)[index])
    assert actual.sharding.spec == PartitionSpec("data", None)


def test_sharded_vectorize_requires_active_mesh_axis() -> None:
    @Vectorizer(placement=Sharded("data"))
    def increment(value: Example) -> Example:
        return Example(value.scalar + 1, value.vector + 1)

    value = Example(scalar=jnp.arange(4), vector=jnp.arange(16).reshape(4, 4))

    with pytest.raises(ValueError, match="active mesh"):
        increment(value)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_sharded_vectorize_pads_a_batch_not_divisible_by_the_mesh_axis() -> None:
    @Vectorizer(placement=Sharded("data"))
    def increment(value: Example) -> Example:
        return Example(value.scalar + 1, value.vector + 1)

    size = jax.device_count() + 1
    value = Example(scalar=jnp.arange(size), vector=jnp.arange(size * 4).reshape(size, 4))
    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    with jax.set_mesh(mesh):
        actual = jax.jit(increment)(value)

    np.testing.assert_array_equal(actual.scalar, value.scalar + 1)
    np.testing.assert_array_equal(actual.vector, value.vector + 1)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_sharded_vectorize_takes_unbatched_leaves_sharded_on_an_explicit_axis() -> None:
    @Vectorizer(placement=Sharded("data"))
    def total(value: Example, other: Example) -> Array:
        return value.scalar + value.vector.sum() + other.vector.sum()

    devices = jax.device_count()
    mesh = jax.make_mesh((devices,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.set_mesh(mesh):
        weights = jax.device_put(jnp.arange(2.0 * devices), NamedSharding(mesh, PartitionSpec("data")))
        actual = total(Example(jnp.arange(4.0), weights), Example(jnp.zeros(()), weights))

    np.testing.assert_array_equal(actual, jnp.arange(4.0) + 2 * weights.sum())


def test_chunked_vectorize_maps_an_empty_batch() -> None:
    @Vectorizer(placement=Chunked(2))
    def total(value: Example) -> Array:
        return value.scalar + value.vector.sum()

    assert total(Example(jnp.zeros((0,)), jnp.ones(8))).shape == (0,)


def test_passes_unbatched_numpy_leaves_as_jax_arrays() -> None:
    @vectorize
    def first(value: Example) -> Array:
        return value.scalar + value.vector.at[0].set(1.0)[0]

    np.testing.assert_array_equal(first(Example(jnp.arange(3.0), np.zeros(4))), jnp.arange(3.0) + 1)


def test_recurses_through_nested_dataclasses() -> None:
    @vectorize
    def increment(value: Nested) -> Nested:
        return Nested(
            example=Example(
                scalar=value.example.scalar + 1,
                vector=value.example.vector + 1,
            ),
            vector=value.vector + 1,
        )

    value = Nested(
        example=Example(
            scalar=jnp.arange(6).reshape(2, 3),
            vector=jnp.arange(24).reshape(2, 3, 4),
        ),
        vector=jnp.arange(24).reshape(2, 3, 4),
    )
    actual = increment(value)

    np.testing.assert_array_equal(actual.example.scalar, value.example.scalar + 1)
    np.testing.assert_array_equal(actual.vector, value.vector + 1)


def test_recurses_through_variadic_tuples_of_dataclasses_under_jit() -> None:
    @vectorize
    def combine(group: Group) -> Array:
        return group.agents[0].vector + group.agents[1].vector

    left = Example(
        scalar=jnp.arange(6).reshape(2, 3),
        vector=jnp.arange(24).reshape(2, 3, 4),
    )
    right = Example(
        scalar=jnp.arange(6).reshape(2, 3),
        vector=jnp.arange(24, 48).reshape(2, 3, 4),
    )
    group = Group(agents=(left, right))

    actual = jax.jit(combine)(group)

    np.testing.assert_array_equal(actual, left.vector + right.vector)


def test_preserves_unshared_temporal_dimensions_inside_the_scalar_call() -> None:
    @vectorize
    def transition(offset: Shaped[Array, "*batch"], trace: Transition) -> Array:
        return trace.state[:-1] + trace.action + offset

    locations = 2
    frames = 3
    units = 4
    offset = jnp.arange(locations)
    trace = Transition(
        state=jnp.arange(locations * (frames + 1) * units).reshape(locations, frames + 1, units),
        action=jnp.ones((locations, frames, units), dtype=jnp.int32),
    )

    actual = transition(offset, trace)
    expected = trace.state[:, :-1] + trace.action + offset[:, None, None]

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_extends_nested_dataclasses_with_a_semantic_time_axis(batch_shape: tuple[int, ...]) -> None:
    @vectorize
    def total(trace: Annotated[Transition, Extend("time")]) -> Array:
        return trace.state.sum(axis=0) + trace.action.sum(axis=0)

    frames = 4
    units = 5
    shape = (*batch_shape, frames, units)
    trace = Transition(
        state=jnp.arange(np.prod(shape)).reshape(shape),
        action=jnp.ones(shape, dtype=jnp.int32),
    )

    actual = total(trace)
    expected = trace.state.sum(axis=-2) + trace.action.sum(axis=-2)

    np.testing.assert_array_equal(actual, expected)


def test_extends_through_variadic_tuples() -> None:
    @vectorize
    def total(group: Annotated[Group, Extend("time")]) -> Array:
        return group.agents[0].vector.sum(axis=0) + group.agents[1].vector.sum(axis=0)

    left = Example(
        scalar=jnp.arange(6).reshape(2, 3),
        vector=jnp.arange(24).reshape(2, 3, 4),
    )
    right = Example(
        scalar=jnp.arange(6, 12).reshape(2, 3),
        vector=jnp.arange(24, 48).reshape(2, 3, 4),
    )

    actual = total(Group(agents=(left, right)))
    expected = left.vector.sum(axis=-2) + right.vector.sum(axis=-2)

    np.testing.assert_array_equal(actual, expected)


def test_extends_multiple_semantic_axes() -> None:
    @vectorize
    def total(trace: Annotated[Transition, Extend("phase", "frame")]) -> Array:
        return trace.state.sum(axis=(0, 1)) + trace.action.sum(axis=(0, 1))

    shape = (2, 3, 4, 5)
    trace = Transition(
        state=jnp.arange(np.prod(shape)).reshape(shape),
        action=jnp.ones(shape, dtype=jnp.int32),
    )

    actual = total(trace)
    expected = trace.state.sum(axis=(-3, -2)) + trace.action.sum(axis=(-3, -2))

    np.testing.assert_array_equal(actual, expected)


def test_rejects_values_smaller_than_their_extended_shape() -> None:
    @vectorize
    def total(trace: Annotated[Transition, Extend("time")]) -> Array:
        return trace.state

    trace = Transition(state=jnp.arange(4), action=jnp.arange(4))

    with pytest.raises(ValueError, match=r"\*batch time unit"):
        total(trace)


def test_preserves_named_shared_inner_dimensions() -> None:
    @vectorize
    def total(value: Located) -> Array:
        return value.vector.sum(axis=0) + value.scalar.sum()

    value = Located(
        scalar=jnp.arange(24).reshape(2, 3, 4),
        vector=jnp.arange(120).reshape(2, 3, 4, 5),
    )

    actual = jax.jit(total)(value)
    expected = value.vector.sum(axis=-2) + value.scalar.sum(axis=-1)[..., None]

    np.testing.assert_array_equal(actual, expected)


def test_extend_keeps_an_argument_whole_where_excluded_did() -> None:
    @vectorize
    def score(policies: Annotated[Example, Extend("side", "policy")], start: Shaped[Array, "*batch"]) -> Array:
        assert policies.vector.shape == (2, 3, 4)
        return policies.scalar.sum() + start

    policies = Example(jnp.ones((2, 3)), jnp.ones((2, 3, 4)))
    np.testing.assert_array_equal(score(policies, jnp.arange(5)), jnp.arange(5) + 6)


def test_passes_unregistered_config_dataclasses_unmapped_under_jit() -> None:
    @dataclass(frozen=True)
    class Auxiliary:
        offset: int

    @vectorize
    def increment(value: Example, auxiliary: Auxiliary) -> Example:
        return Example(
            scalar=value.scalar + auxiliary.offset,
            vector=value.vector + auxiliary.offset,
        )

    value = Example(
        scalar=jnp.arange(6).reshape(2, 3),
        vector=jnp.arange(24).reshape(2, 3, 4),
    )
    auxiliary = Auxiliary(offset=2)
    actual = jax.jit(increment, static_argnames=("auxiliary",))(
        value,
        auxiliary=auxiliary,
    )

    np.testing.assert_array_equal(actual.scalar, value.scalar + 2)
    np.testing.assert_array_equal(actual.vector, value.vector + 2)


def test_rejects_bare_array_annotations() -> None:
    @vectorize
    def increment(value: Array) -> Array:
        return value + 1

    with pytest.raises(TypeError, match="jaxtyping annotation beginning with '\\*batch'"):
        increment(jnp.arange(3))


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Absent:
    pass


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Present:
    vector: Shaped[Array, "*batch feature"]


type Maybe = Absent | Present


@tree_util.register_dataclass
@dataclass(frozen=True)
class Tagged:
    scalar: Shaped[Array, "*batch"]
    extra: Maybe


def test_maps_the_runtime_arm_of_a_named_sum_under_jit() -> None:
    @vectorize
    def total(value: Tagged) -> Shaped[Array, "*batch"]:
        match value.extra:
            case Present(vector=vector):
                return value.scalar + vector.sum()
            case Absent():
                return value.scalar

    scalar = jnp.arange(6).reshape(2, 3)
    vector = jnp.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(jax.jit(total)(Tagged(scalar, Present(vector))), scalar + vector.sum(-1))
    np.testing.assert_array_equal(jax.jit(total)(Tagged(scalar, Absent())), scalar)


def test_rejects_values_outside_every_arm_of_a_sum() -> None:
    @vectorize
    def identity(value: Tagged) -> Tagged:
        return value

    with pytest.raises(TypeError, match="matches 0 arms"):
        identity(Tagged(jnp.arange(3), jnp.arange(3)))


@pytest.mark.parametrize("placement", [Local(), Chunked(2)])
def test_passes_leaves_without_batch_axes_to_every_call_uncopied(placement: Local | Chunked) -> None:
    @Vectorizer(placement=placement)
    def total(value: Example) -> Array:
        assert not isinstance(value.vector, jax.core.Tracer)
        return value.scalar + value.vector.sum()

    value = Example(scalar=jnp.arange(3), vector=jnp.ones(4))
    np.testing.assert_array_equal(total(value), jnp.arange(3) + 4)


def test_names_every_leaf_when_batch_axes_do_not_broadcast() -> None:
    @vectorize
    def total(value: Example, other: Shaped[Array, "*batch"]) -> Array:
        return value.scalar + other

    with pytest.raises(ValueError, match=r"value\.scalar \(3,\), value\.vector \(3,\), other \(4,\).*Extend"):
        total(Example(jnp.arange(3), jnp.ones((3, 2))), jnp.arange(4))
