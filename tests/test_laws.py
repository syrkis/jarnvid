from dataclasses import dataclass
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import tree_util
from jaxtyping import Array, Float

from jarnvid import Chunked, Extend, Vectorizer, batch_shape, concat, expand, stack, take, vectorize

sizes = st.integers(1, 4)
batches = st.lists(sizes, min_size=0, max_size=3).map(tuple)


def pairing(x: Float[Array, "*batch dim"], y: Float[Array, "*batch dim"]) -> Float[Array, "*batch"]:
    return jnp.sin(x) @ y + x.sum()


paired = vectorize(pairing)


def nested(fn, depth):
    """`fn` under `depth` nested vmaps: the reference a vectorized call must equal."""
    for _ in range(depth):
        fn = jax.vmap(fn)
    return fn


@settings(deadline=None, max_examples=40)
@given(batches, sizes, st.integers(0, 2**31 - 1))
def test_a_vectorized_call_is_the_nested_vmap(batch, dim, seed):
    x, y = jax.random.normal(jax.random.key(seed), (2, *batch, dim))
    np.testing.assert_allclose(paired(x, y), nested(pairing, len(batch))(x, y), rtol=1e-5, atol=1e-5)


@settings(deadline=None, max_examples=40)
@given(batches, sizes, sizes, st.integers(0, 2**31 - 1))
def test_chunked_maps_the_same_values_in_any_chunk_size(batch, dim, chunk, seed):
    # Law: Chunked(c) is Local for every chunk size c, whether or not c divides the batch.
    x, y = jax.random.normal(jax.random.key(seed), (2, *batch, dim))
    chunked = Vectorizer(placement=Chunked(chunk))(pairing)
    np.testing.assert_allclose(chunked(x, y), paired(x, y), rtol=1e-5, atol=1e-5)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_batch_axes_broadcast_from_the_left(rows, columns, dim, seed):
    # Law: f(x[:, None], y[None]) is the table of f over every pair; an unbatched argument is
    # shared by every call.
    x, y = (jax.random.normal(key, (n, dim)) for key, n in zip(jax.random.split(jax.random.key(seed)), (rows, columns), strict=True))
    table = jax.vmap(lambda a: jax.vmap(lambda b: pairing(a, b))(y))(x)
    np.testing.assert_allclose(paired(x[:, None], y[None]), table, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(paired(x, y[0]), table[:, 0], rtol=1e-5, atol=1e-5)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_unshared_batch_axes_stay_inside_each_call(shared, extra, dim, seed):
    # Law (left alignment): with batches (a,) and (a, t), calls map over a, and the leaf with
    # more batch axes passes its t axis whole to each call.
    @vectorize
    def shift(offset: Float[Array, "*batch"], values: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
        assert values.shape == (extra, dim)
        return values + offset

    offset, values = jax.random.normal(jax.random.key(seed), (shared,)), jnp.ones((shared, extra, dim))
    np.testing.assert_allclose(shift(offset, values), values + offset[:, None, None], rtol=1e-6)


@settings(deadline=None, max_examples=40)
@given(batches, sizes, sizes, st.integers(0, 2**31 - 1))
def test_extended_axes_stay_inside_each_call(batch, time, dim, seed):
    # Law: an Extend axis is seen whole by every call, never mapped over.
    @vectorize
    def total(trace: Annotated[Float[Array, "*batch dim"], Extend("time")]) -> Float[Array, "*batch dim"]:
        assert trace.shape == (time, dim)
        return trace.sum(axis=0)

    trace = jax.random.normal(jax.random.key(seed), (*batch, time, dim))
    np.testing.assert_allclose(total(trace), trace.sum(axis=-2), rtol=1e-5, atol=1e-5)


@tree_util.register_dataclass
@dataclass(frozen=True)
class Side:
    plan: Float[Array, "*batch dim"]
    terrain: Float[Array, "*batch dim"]


def sides(key: jax.Array, batch: tuple[int, ...], dim: int, terrain: jax.Array) -> Side:
    return Side(jax.random.normal(key, (*batch, dim)), terrain)


@vectorize
def meet(a: Side, b: Side) -> Float[Array, "*batch"]:
    return pairing(a.plan, b.plan) + a.terrain @ b.plan


@settings(deadline=None, max_examples=40)
@given(st.lists(sizes, min_size=1, max_size=3).map(tuple), sizes, sizes, st.integers(0, 2**31 - 1))
def test_take_undoes_stack(batch, count, dim, seed):
    # Law: take(stack(xs), i) == xs[i], and a leaf every x shares without batch axes stays once.
    terrain = jnp.ones(dim)
    values = [sides(key, batch, dim, terrain) for key in jax.random.split(jax.random.key(seed), count)]
    stacked = stack(values)
    assert stacked.terrain is terrain and batch_shape(stacked) == (count, *batch)
    for index, value in enumerate(values):
        np.testing.assert_array_equal(take(stacked, index).plan, value.plan)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_take_of_concat_is_take_of_a_part(left, right, dim, seed):
    # Law: concat lays its parts end to end along the leading batch axis.
    a, b = (sides(key, (n,), dim, jnp.ones(dim)) for key, n in zip(jax.random.split(jax.random.key(seed)), (left, right), strict=True))
    joined = concat([a, b])
    np.testing.assert_array_equal(take(joined, jnp.arange(left)).plan, a.plan)
    np.testing.assert_array_equal(take(joined, left + jnp.arange(right)).plan, b.plan)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_expand_makes_the_outer_product(rows, columns, dim, seed):
    # Law: f(expand(xs, after=1), expand(ys, before=1)) is the table of f over every pair, for
    # structures with a leaf kept once.
    terrain = jnp.arange(float(dim))
    xs, ys = (sides(key, (n,), dim, terrain) for key, n in zip(jax.random.split(jax.random.key(seed)), (rows, columns), strict=True))
    table = jax.vmap(lambda a: jax.vmap(lambda b: meet(Side(a, terrain), Side(b, terrain)))(ys.plan))(xs.plan)
    np.testing.assert_allclose(meet(expand(xs, after=1), expand(ys, before=1)), table, rtol=1e-5, atol=1e-5)


@tree_util.register_dataclass
@dataclass(frozen=True)
class Pair:
    scalar: Float[Array, "*batch"]
    vector: Float[Array, "*batch dim"]


@tree_util.register_dataclass
@dataclass(frozen=True)
class Deep:
    pair: Pair
    trace: Annotated[Float[Array, "*batch dim"], Extend("time")]


type Pairs = tuple[Pair, ...]


def summarise(pair: Pair) -> Float[Array, "*batch"]:
    return pair.scalar * jnp.sin(pair.vector).sum()


def weigh(deep: Deep) -> Float[Array, "*batch dim"]:
    return deep.trace.sum(axis=0) * deep.pair.scalar + jnp.cos(deep.pair.vector)


def combine(group: Pairs) -> Float[Array, "*batch"]:
    return jnp.stack([pair.scalar + jnp.cos(pair.vector).sum() for pair in group]).sum()


def normals(key, *shapes):
    """One normal array per shape, from independent splits of `key`."""
    return [jax.random.normal(part, shape) for part, shape in zip(jax.random.split(key, len(shapes)), shapes, strict=True)]


def one_pair(batch, dim, time, arity, key):
    return (Pair(*normals(key, batch, (*batch, dim))),)


def one_deep(batch, dim, time, arity, key):
    scalar, vector, trace = normals(key, batch, (*batch, dim), (*batch, time, dim))
    return (Deep(Pair(scalar, vector), trace),)


def a_group(batch, dim, time, arity, key):
    parts = jax.random.split(key, arity)
    return (tuple(Pair(*normals(part, batch, (*batch, dim))) for part in parts),)


# Each case is a function written for one instance, and a builder giving it `batch` leading axes.
cases = {"pair": (summarise, one_pair), "deep": (weigh, one_deep), "group": (combine, a_group)}
vectorized = {name: vectorize(plain) for name, (plain, _) in cases.items()}


@settings(deadline=None, max_examples=60)
@given(st.sampled_from(sorted(cases)), batches, sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_a_vectorized_pytree_call_is_the_nested_vmap(case, batch, dim, time, arity, seed):
    # Law (the nested-vmap law over PyTrees): whatever the argument's structure (a dataclass, a
    # dataclass holding a dataclass and an Extend axis, a variadic tuple), vectorizing it over
    # `batch` leading axes equals nesting `vmap` once per batch axis.
    plain, build = cases[case]
    arguments = build(batch, dim, time, arity, jax.random.key(seed))
    np.testing.assert_allclose(vectorized[case](*arguments), nested(plain, len(batch))(*arguments), rtol=1e-5, atol=1e-5)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_batch_axes_broadcast_leafwise_inside_a_pytree(rows, columns, dim, seed):
    # Law: the shared batch is the broadcast of the leaves' batch axes, so a leaf's size-1 axis is
    # reused along it. Broadcasting must equal expanding the leaf by hand.
    scalar, vector = normals(jax.random.key(seed), (1, columns), (rows, 1, dim))
    wide = Pair(jnp.broadcast_to(scalar, (rows, columns)), jnp.broadcast_to(vector, (rows, columns, dim)))
    np.testing.assert_allclose(vectorized["pair"](Pair(scalar, vector)), nested(summarise, 2)(wide), rtol=1e-5, atol=1e-5)


@settings(deadline=None, max_examples=40)
@given(sizes, sizes, sizes, st.integers(0, 2**31 - 1))
def test_unshared_batch_axes_stay_inside_a_pytree_call(shared, extra, dim, seed):
    # Law (left alignment through a PyTree): when a dataclass's leaves carry (a,) and (a, t) batch
    # axes, calls map over a alone, and the leaf with the extra axis hands it whole to each call.
    @vectorize
    def offset(pair: Pair) -> Float[Array, "*batch dim"]:
        assert pair.scalar.shape == ()
        assert pair.vector.shape == (extra, dim)
        return pair.vector + pair.scalar

    (scalar,) = normals(jax.random.key(seed), (shared,))
    vector = jnp.ones((shared, extra, dim))
    np.testing.assert_allclose(offset(Pair(scalar, vector)), vector + scalar[:, None, None], rtol=1e-6)
