# jarnvid

Write a JAX function for one instance of a structure, and call it on any batch of them.
`vectorize` reads which axes belong to one instance from the jaxtyping annotations you
already write, and maps the function over the rest.

```sh
pip install jarnvid
```

Járnviðr is the iron wood of Norse myth. The structures it is for are complex things written
as data (a force on a map, a doctrine, a molecule): frozen dataclasses of JAX arrays, nested in
tuples and named sums.

## Example

```pycon
>>> from dataclasses import dataclass
>>> from typing import Annotated
>>> import jax.numpy as jnp
>>> from jax.tree_util import register_dataclass
>>> from jaxtyping import Array, Float
>>> from jarnvid import Extend, stack, take, vectorize
>>> @register_dataclass
... @dataclass(frozen=True)
... class Unit:
...     position: Float[Array, "*batch 2"]
...     health: Float[Array, "*batch"]
>>> type Force = Annotated[Unit, Extend("unit")]  # a side: its unit axis stays inside each call
>>> @vectorize
... def threat(force: Force, target: Float[Array, "*batch 2"]) -> Float[Array, "*batch"]:
...     distance = jnp.linalg.norm(force.position - target, axis=-1)
...     return (force.health / (1 + distance)).sum()
>>> forces = Unit(position=jnp.zeros((5, 3, 2)), health=jnp.ones((5, 3)))  # 5 scenarios, 3 units each
>>> threat(forces, jnp.ones((5, 2))).shape  # one target per scenario
(5,)
>>> threat(forces, jnp.ones(2)).shape  # one target for all: it reaches every call as is
(5,)
>>> threat(stack([forces, forces]), jnp.ones((2, 5, 2))).shape  # batch axes nest freely
(2, 5)
>>> take(forces, 0).position.shape  # the first scenario's force
(3, 2)

```

## Why not `vmap`

`jax.vmap` maps one axis, chosen per argument with `in_axes`. Several batch axes mean nested
`vmap`s, and a dataclass whose fields carry different numbers of batch axes means spelling
out an `in_axes` tree that mirrors it. `equinox.filter_vmap` chooses what to map by type, not
by shape. jarnvid reads the shapes from the annotations instead:

- One function signature gives any number of batch axes, broadcast **from the left** across
  all arguments (docs/adr/0001).
- A leaf with no batch axes (a map, a rule table) reaches every call once, never copied.
- `Extend("unit")` keeps a named axis inside each call, so a function can see a whole side.
- The same decorator maps in chunks (`Vectorizer(placement=Chunked(64))`, bounded memory) or
  shards over a device mesh (`Vectorizer(placement=Sharded("data"))`, padding a batch that does
  not divide evenly).

The price is annotation discipline: every mapped array is annotated, starting with `*batch`.

## Rules

- Arguments are PyTrees: tuples, dicts (`dict[str, T]`, `Mapping[str, T]`), registered
  dataclasses (fields marked `static` are left alone), named sums (the arm matching the value)
  and type aliases. Arrays in any other container are an error.
- What is mapped is read from the annotations alone. Plain Python values and unregistered
  dataclasses (configs) reach every call as they are.
- A leaf's batch axes beyond the ones shared by all arguments stay inside each call.

Between calls, structures are reshaped by their batch axes alone (docs/adr/0002): `take`,
`stack`, `concat`, `expand` (singleton axes, for outer products) and `batch_shape`. Leaves
without batch axes stay once.

| File | Concept |
|---|---|
| `annotation.py` | Reading annotations: which trailing axes of each leaf one call sees |
| `vectorizer.py` | `vectorize` and `Vectorizer`: broadcast, flatten, map, reshape |
| `placement.py` | Where the batch runs: `Local`, `Chunked`, or `Sharded` over a mesh axis |
| `structure.py` | `take`, `stack`, `concat`, `expand`, `batch_shape` on batched structures |

## Develop

`just test` (the tests and this README's examples), `just sharded` (four forced CPU devices),
`just check` (ruff and vulture), `just build` (the wheel and sdist). The scope is fixed: one
decorator and the structure operations around it.

## Cite

See `CITATION.cff`. MIT licence.

## Open

- JAX 0.12: jarnvid passes unchanged. With `jax>=0.12`, add a law that `Sharded` adds no
  collectives (`jax.sharding.no_collectives`), except for the result gather after padding.
