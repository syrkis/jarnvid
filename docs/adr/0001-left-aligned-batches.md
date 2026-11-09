# Batch axes align from the left

Status: accepted (1 October 2026).

## Context

`vectorize` maps a function written for one instance over the leading batch axes of its
arguments. When arguments carry different numbers of batch axes, they must be lined up. NumPy
aligns shapes from the right. `vectorize` has always aligned batch axes from the left: the
shared batch is the broadcast of every leaf's first `r` batch axes, where `r` is the fewest any
batched leaf has. A leaf's further batch axes stay inside each call.

Simulations rely on this: a per-location offset `(location,)` against a trace
`(location, frame, unit)` maps over locations and hands each call the whole `(frame, unit)` trace
(`test_preserves_unshared_temporal_dimensions_inside_the_scalar_call`).

## Decision

Keep left alignment. It reads as "batch axes are outermost": the axes a caller adds go in front,
and the ones a function did not ask about pass through.

## Consequences

- Outer products need an explicit axis on both sides: `f(xs[:, None], ys[None])`. Keys shaped
  `(fights, pairs)` against genomes `(pairs, dim)` need `genomes[None]`.
- `test_unshared_batch_axes_stay_inside_each_call` states the law and fails under right alignment.
- Switching to NumPy alignment would mean auditing every downstream call site.
