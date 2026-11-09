"""Járnviðr, the iron wood: complex structures as data, algebraic in JAX.

Doctrine, forces, molecules: frozen dataclasses of arrays. `vectorize` reads a function's
jaxtyping annotations (`*batch` first), broadcasts the leading batch axes of its PyTree
arguments, and maps the function over them; `Extend` keeps named semantic axes inside each call;
`Vectorizer` maps the batch in chunks, or shards it over a mesh axis. `take`, `stack`, `concat` and `expand` reshape structures
along their batch axes only.
"""

from .annotation import Extend
from .placement import Chunked, Local, Sharded
from .structure import batch_shape, concat, expand, stack, take
from .vectorizer import Vectorizer, vectorize

__all__ = ["Chunked", "Extend", "Local", "Sharded", "Vectorizer", "batch_shape", "concat", "expand", "stack", "take", "vectorize"]
