# TileCone

*A tile's dependency cone, as an annotated sub-DAG.*

Select any region of any tensor in a compute graph. TileCone draws the sub-DAG that region actually
touches - the operations on the path, and for every tensor among them, the exact index range
involved. Upstream is what the tile reads; downstream is what it feeds.

For `C[M,N] = A[M,K] @ B[K,N]`, the tile `C[64:128, 0:64]` yields a cone reaching `A[64:128, :]`
and `B[:, 0:64]` - a row band and a column band, and nothing else.

Shapes are inferred across the whole graph. No values are ever computed - the engine is integer
interval arithmetic over index sets.

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # unit and integration suite
npm run build    # typecheck + production bundle
```

Press `?` in the app for the shortcut sheet.

## Writing a graph

One statement per line, `#` comments, single assignment. Scalars bind symbolic dimensions;
`Tensor` declares an input and `Parameter` a weight. Axes may be named, and a dimension may be
arithmetic over bound symbols.

```
B = 1
H = 4
S = 128
D = 32

X  = Tensor(batch=B, seq=S, emb=H*D, dtype=fp16)
Wq = Parameter(emb=H*D, proj=H*D, dtype=fp16)

Qp = einsum("bse,ef->bsf", X, Wq)
Q4 = reshape(Qp, shape=[B, S, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
Sc = einsum("bhqd,bhkd->bhqk", Qh, Qh)
P  = softmax(Sc, axis=-1)
```

Ctrl/Cmd+Enter runs the source. Every independent error is reported in one pass, in source order,
each underlining the part it is about.

## What's supported

| | |
|---|---|
| **Contraction** | `einsum` (incl. diagonals and traces), `matmul`, `bmm`, `linear` |
| **Elementwise** | `add` `sub` `mul` `div` `pow` `maximum` `minimum`, `relu` `gelu` `silu` `sigmoid` `tanh` `exp` `log` `sqrt` `rsqrt` `neg` `abs`, with NumPy broadcasting |
| **Reduction** | `sum` `mean` `prod` `amax` `amin`, multi-axis, `keepdim` |
| **Normalisation** | `softmax`, `layernorm`, `rmsnorm` (optional weight and bias) |
| **Shape** | `reshape`, `transpose`, `slice` (strided), `pad` (constant/reflect/replicate), `concat`, `split`, `expand`, `identity`, `contiguous` |
| **Spatial** | `conv` (1–3D, grouped, strided, dilated, padded), `pool` (max/avg) |
| **Other** | `cumsum` (forward and reverse), `gather`, `cast` |

Dtypes are `fp32` `fp16` `bf16` `fp8` `int32` `int8` `bool`. Operations that compute from several
tensors promote (`fp16 + fp32 → fp32`, following PyTorch); operations that only move data require
a match, since one output buffer has no result type to infer.

Alongside the dependency cone, the app reports element and byte footprints, FLOPs and arithmetic
intensity, reuse estimates, and plain-language notes naming the constraint each operation puts on
tiling or fusion.

**Entanglement** is a third relation, next to upstream and downstream: what a tile is *combined
with*. For `C = A @ B`, the block `A[0:4, 0:4]` is multiplied only against `B[0:4, :]` - the rows a
kernel must hold resident alongside it. That is not the downstream cone read backwards, which
reaches all of `B`; entanglement asks which elements meet in the same term. Exact for `einsum`
(so `matmul`, `bmm`, `linear`), elementwise, `conv`, `concat` and `gather`; `normalize` falls back to
a marked bound.
Press `e` to show it - a stipple, next to the solid fill of what a tile needs and the ruling of what
it feeds.

**Not supported yet.** `matmul` needs matching ranks (no 3D @ 2D, no batch broadcast); einsum has
no ellipsis; `reshape` takes no `-1`; slices have no negative steps; there is no `conv_transpose`,
batch/group norm, `where`, `argmax`/`topk`, or comparison operator; a name is bound once, so
`X = relu(X)` is an error.

## The core invariant

A `Region` is a union of half-open axis-aligned boxes carrying an `exact` flag:

- `exact: true` - the region is **precisely** the dependency set.
- `exact: false` - the region is a **strict superset**, never a subset, and always carries a reason
  (`"strided conv"`, `"diagonal einsum"`, `"reshape run cap exceeded"`, …). The UI hatches these so
  an over-approximation is never presented as ground truth. Past a box cap a region is coarsened by
  merging neighbours into their hulls rather than collapsed to one bounding box.

A region that is ever a strict *subset* of the truth is a critical bug: that is the case where the
tool lies. The test suite exists mainly to make that impossible.

**Boxes may overlap, and that is the point.** In `C = matmul(A, A)` a tile of `C` reads a row band
and a column band of `A`, and they share a square. Storing the region disjointly would clip that
square out of one band and report three fragments, two of which are not regions anything reads.

The rule that makes this safe: **cardinality is measured on the set.** Elements, bytes,
percentages and FLOPs count a shared element once, so summing the boxes you can see exceeds the
element total whenever they overlap - a row states that difference as `N shared`.

## Headless API

`compileDSL` is the checked boundary for user-authored programs, and exposes the symbolic executor:

```ts
import { compileDSL } from "./src/parse/compiler";
import { box, fromBox } from "./src/core/region";

const program = compileDSL(`
  A = Tensor(256, 512, dtype=fp16)
  B = Tensor(512, 256, dtype=fp16)
  C = matmul(A, B)
`);

const cone = program.executor.upstream("C", fromBox(box([64, 128], [0, 64])));
```

Use `tryCompileDSL` to get diagnostics as data instead of an exception. The lower-level
`parseProgram` (text → AST), `lowerProgram` (AST → graph), `resolveGraph`, and the propagation
functions remain available for tooling.

`core/` is strictly headless - nothing there may import from `ui/` - so the oracle and any future
CLI run without a DOM.

## Testing

The failure mode of this app is plausible-looking wrong highlights, so correctness is checked
against a **brute-force oracle** (`src/test/oracle.ts`) rather than against the analytic rules
themselves. Every element of every tensor gets a unique id; id-sets propagate forward using each
op's *pointwise semantics* (`oracleDeps`), never its box-level backward rule. The analytic region
must then **equal** the oracle set when `exact`, and **contain** it when not.

It runs over every op in the registry - enforced, not intended: `listOps()` drives a coverage test
against a fixture table, so a newly registered op fails the suite until it has one. It also runs
over randomly generated 5–15 node graphs including diamonds, and the built-in examples at
miniature shapes. The thresholds at which an operation gives up and returns a bound are parameters
rather than constants, so those branches are checked at shapes brute force can enumerate.

A separate registry-wide law checks that `forward` and `backward` agree about whether the
dependency relation between a given input box and output box is empty, which needs no oracle and
so applies to every op at any size.

## Layout

```
src/
  core/          headless engine
    region.ts    the box algebra everything rests on
    graph.ts     IR, validation, topo sort, shape inference
    propagate.ts the backward/forward driver
    ops/         the op registry - einsum is the heart of it
    metrics.ts   flops, bytes, intensity
  parse/         lexer -> AST -> graph, JSON front end, compiler facade
  ui/            React + canvas; store.ts holds all view state
  examples/      built-in demo graphs
  test/          oracle + suites
```

See `SYSTEM.md` for the architecture and implementation invariants. `docs/README.md` classifies the
remaining design documents so historical plans are not mistaken for the live system contract.
