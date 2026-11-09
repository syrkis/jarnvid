# Structures are reshaped by their batch axes

Status: accepted (19 October 2026). Revises the first consequence of 0001.

## Context

Left alignment (0001) makes an outer product the caller's job: `f(xs[:, None], ys[None])`. That
works for arrays but not for the structures jarnvid is for. A dataclass has no `[:, None]`, and
`tree.map(lambda x: x[:, None], starts)` also hits leaves without batch axes (a map kept once
for every start). These leaves now go into each call unchanged and must stay that way.

A survey of downstream call sites found the same friction beyond outer products: about 49
hand-written `tree.map(lambda a: a[i])` indexings, about 20 dataclass stacks and concatenations,
and about 13 hand-written axis insertions. Only two or three sites fit a `Vectorizer(outer=...)`
declared when decorating. One payoff function is called both pairwise and as an outer product,
so the call is the place to choose, not the decoration.

## Decision

Export operations that read a structure's own annotations and move only leaves with batch axes:

- `take(value, index)`: the instances at `index` along the leading batch axis.
- `stack(values)`: a new leading batch axis.
- `concat(values)`: joined along the leading batch axis.
- `expand(value, before, after)`: singleton axes around the batch axes. An outer product is
  `f(expand(xs, after=1), expand(ys, before=1))`.
- `batch_shape(value)`: the batch `vectorize` maps over.

`stack` and `concat` keep a leaf without batch axes once when every structure holds the same
object, and give each structure its own copy otherwise. Plain values must agree.

No `Vectorizer(outer=...)`.

## Consequences

- Nested Vectorizers collapse into one call over expanded arguments, and hand-written `tree.map`
  indexing and stacking can be replaced.
- Laws (Hypothesis, `tests/test_laws.py`): `take` undoes `stack`; `take` of `concat` is `take` of
  a part; `expand` gives the table of every pair.
- A structure must be a registered dataclass. A bare array has no annotation to read, so arrays
  keep `[:, None]`.
