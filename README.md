# TileCone

*Symbolic dependence and tiling analysis for tensor programs.*

TileCone analyses a small tensor program - a layer, or a short sequence of operations - written in
a line-oriented DSL. Select a region of any tensor and it reports:

- the **Backward Cone**: every element, in every upstream tensor, that the region depends on;
- the **Forward Cone**: every downstream element the region can influence;
- the **Co-access Surface**: the elements of other operands that the region is combined with in
  the same term of a computation.

Declare a tiling instead, and it reports what each tile's task reads and computes, which producer
tasks supply it, and what the plan costs in tasks, FLOPs, recomputed work and bytes moved.

No tensor values are computed. The engine works on shapes, dtypes and integer index regions, so
every answer is either exact or a marked over-approximation with its reason.

For `C[M,N] = A[M,K] @ B[K,N]`, the tile `C[64:128, 0:64]` has a Backward Cone of `A[64:128, :]`
and `B[:, 0:64]`: a row band and a column band, and nothing else.

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # unit and integration suite
npm run build    # typecheck + production bundle
```

Press `?` in the app for the shortcut sheet.

## The three views

The inspector is split by what a figure rests on, because the three kinds of figure can be wrong in
different ways.

| View | Question | What its figures are |
|---|---|---|
| **Dependencies** | What do the drawn tiles need, feed and share? | Facts about the graph: exact, or a bound with its reason named. |
| **Execution** | What would an assumed execution do with them? | Modelled: idealised scenarios, not bounds. |
| **Plan** | How does a declared tiling divide the work? | Exact for that tiling, or a bound with its reason named. |

### Dependencies

Draw a tile on any tensor card. Tiles may sit on several tensors at once, which is how two tensors'
demands on a shared input are compared.

The canvas gives each relation its own mark: a solid fill for the Backward Cone (`U` toggles it), a
ruling for the Forward Cone (`D`), a dot stipple for the Co-access Surface (`E`), and a dashed hatch
for any region that is a conservative over-approximation. Hue identifies which tile a mark belongs
to.

The panel lists each tensor in each cone with its slice expressions, element and byte footprint,
and distance from the tile. It also reports FLOPs, duplicate graph-input demand across the drawn
tiles, and plain-language notes on the constraint each operation puts on tiling or fusion - the
contraction that must be staged, the axis a softmax needs whole. Forward Cone rows say whether the
tile **completes** a downstream region or only **partly** contributes to it: a tile covering part
of a contracted axis reaches an output without determining any element of it.

The Co-access Surface is not the Forward Cone read backwards. For `C = A @ B`, the Forward Cone of
`A[0:4, 0:4]` is `C[0:4, :]`, and that band reads all of `B`. But the block is only ever multiplied
against `B[0:4, :]`, the rows a kernel must hold resident alongside it. It is exact for `einsum`
(so `matmul`, `bmm`, `linear`), elementwise operations, `conv`, `concat` and `gather`; `normalize`
falls back to a marked bound.

### Execution

The arithmetic intensity of the tile under two scenarios: **fused** reads the cone's graph inputs
and writes its output once; **unfused** charges each operation's distinct reads and writes, with
transpose, slice, expand and row-major reshape treated as views that move nothing. Neither is a
hardware prediction.

The **reuse estimate** walks tiles of the same size across the tensor, with a fixed seed, and
estimates how many touch each input region the tile reads. Running it replays its actual sampled
tiles on the graph: each probe lights up wherever it lands across the relations switched on, with
the part it shares with the drawn tile drawn on top. Execution narrows to that one tile while it is
open and restores the other tiles on leaving.

### Plan

A plan divides produced tensors into regular tiles. Each tile is a **task**: it computes one
complete tile, including its whole reduction. Tiling a tensor is what writes it to memory. A task
reads tiled tensors and graph inputs, and computes any untiled tensor between them itself, so an
untiled intermediate is fused into the tasks that read it and recomputed by each of them.

Dragging on an untiled tensor divides it at the extents drawn; after that, clicking inspects the
task under the pointer. The first click also tiles the tensors its operation reads, so a plan
starts with one operation per task. Clearing a tensor's tiling (×) fuses it into its consumers, and
**tile** under "Computed in this task" writes it back.

For the inspected task, the panel shows its FLOPs, what it reads from each tensor, which producer
tasks supply that, and the untiled tensors it computes. A producer found only through an
over-approximated region is marked as a possible dependency rather than a definite one. For each
tiled tensor and for the whole plan it reports:

| Figure | Meaning |
|---|---|
| tasks, dependencies | task count, and producer tasks summed over tasks |
| FLOPs, recomputed | work summed over tasks, and the part of it more than one task does |
| read, distinct read | bytes each task reads, summed; and the same demand with each element once |
| written | every tiled tensor, once |
| intensity | FLOPs ÷ (read + written) |

These are logical figures for the plan as declared - each task reads its own demand, with no reuse
between tasks, and each tiled tensor is written once - not measured memory traffic. FLOPs, and so
what counts as recomputed, follow each operation's cost formula rather than a minimum over all
algorithms. Per tensor read, the panel also gives demand duplication (summed over distinct) and how
many tasks read each producer tile. For the MLP in the headless example below:

| Plan | Tasks | Dependencies | FLOPs | Recomputed | Read | Written | Intensity |
|---|---|---|---|---|---|---|---|
| `H` 64×128, `Y` 64×64 | 24 | 32 | 67.11M | 0 | 1.8 MB | 320 KB | 31.03 FLOP/B |
| `Y` 64×64, `H` fused | 8 | 0 | 100.66M | 33.55M | 1.6 MB | 64 KB | 56.89 FLOP/B |
| `Y` 64×128, `H` fused | 4 | 0 | 67.11M | 0 | 1.1 MB | 64 KB | 56.89 FLOP/B |

Fusing `H` stops writing and re-reading it, but with 64-wide `Y` tiles two tasks compute each row
band of `H`. Full-width `Y` tiles compute each band once.

Split reductions, working sets and execution order are not modelled. A plan is not included in
share links.

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
each underlining the part it is about. The source panel carries built-in examples, among them
multi-head attention, KV-cache and grouped-query decode steps, a SwiGLU feed-forward block and MoE
expert dispatch. The share button encodes the source, the tiles and the view settings in a link.

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
| **Barrier** | `opaque`: an operation whose shapes are known and whose semantics are not |

Dtypes are `fp32` `fp16` `bf16` `fp8` `int64` `int32` `uint8` `int8` `bool`. Operations that compute
from several tensors promote (`fp16 + fp32 → fp32`, following PyTorch); operations that only move
data require a match, since one output buffer has no result type to infer.

**Not supported yet.** `matmul` needs matching ranks (no 3D @ 2D, no batch broadcast); einsum has
no ellipsis; `reshape` takes no `-1`; slices have no negative steps; there is no `conv_transpose`,
batch/group norm, `where`, `argmax`/`topk`, or comparison operator; a name is bound once, so
`X = relu(X)` is an error.

Anything missing can still be carried, rather than blocking the graph it sits in:

```
H = opaque(X, S, op="BatchNormalization", shapes=[[N, C]])
```

A barrier declares its output shapes and nothing else. Every output element is assumed to read
every input element: the weakest true statement about an operation nobody has described, so it is
a superset by construction, marked inexact with the original name as its reason. Cards show that
name rather than `opaque`. A barrier's own arithmetic cannot be bounded, so a FLOP total that spans
one is reported as `unknown`; byte figures through it remain upper bounds.

## Exact answers and bounds

A `Region` is a union of half-open axis-aligned boxes carrying an `exact` flag:

- `exact: true` - the region is **precisely** the dependency set.
- `exact: false` - the region is a **superset**, never a subset, and always carries a reason
  (`"strided conv"`, `"diagonal einsum"`, `"reshape run cap exceeded"`, …). The canvas hatches
  these so an over-approximation is never shown as an exact answer. Past a box cap a region is
  coarsened by merging neighbours into their hulls rather than collapsed to one bounding box.

A region that is a strict *subset* of the true dependency set is a critical bug, because it omits
real dependencies. Most of the test suite exists to rule that out.

**Boxes may overlap.** In `C = matmul(A, A)` a tile of `C` reads a row band and a column band of
`A`, and they share a square. Storing the region disjointly would clip that square out of one band
and report three fragments, two of which no operand reads. The rule that makes overlap safe:
**cardinality is measured on the set.** Elements, bytes, percentages and FLOPs count a shared
element once, so summing the visible boxes exceeds the element total whenever they overlap - a row
states that difference as `N shared`.

Every figure carries its own status, shown beside it:

| Shown | Meaning |
|---|---|
| a plain number | exact |
| `≤` | an upper bound, from an over-approximated region |
| `~` | approximate in an unknown direction: a ratio of bounds, or a sampled estimate |
| `unknown` | no number, e.g. a FLOP total across a barrier |

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
cone.tensors.get("A"); // region 64:128, 0:512, exact
```

`downstream`, `entangled`, and the frontier-bounded `upstreamWithin` / `downstreamWithin` sit
beside it. Use `tryCompileDSL` to get diagnostics as data instead of an exception.

Plans are checked against the resolved graph:

```ts
import { tilePlan } from "./src/core/plan/plan";
import { interfaceOf, planReport, supplyOf } from "./src/core/plan/interfaces";

const mlp = compileDSL(`
  X  = Tensor(256, 128, dtype=fp16)
  W1 = Parameter(128, 512, dtype=fp16)
  W2 = Parameter(512, 128, dtype=fp16)
  H  = matmul(X, W1)
  Y  = matmul(H, W2)
`);

const plan = tilePlan(mlp.resolved, { H: [64, 128], Y: [64, 64] });
const supply = supplyOf(plan, { tensorId: "Y", coord: [1, 0] }); // four H producer tasks
const family = interfaceOf(plan, "Y"); // per tensor read: summed, distinct, duplication, fan-out
const report = planReport(plan); // each family's work, and the plan's total
```

`core/` and `parse/` import no React or browser API, so the oracle and any CLI run without a DOM.

## Testing

Correctness is checked against a **brute-force oracle** (`tests/corpus/oracle.ts`) rather than
against the analytic rules themselves. Every element of every tensor gets a unique id; id-sets
propagate using each operation's *pointwise semantics* (`oracleDeps`), never its box-level rule. The
analytic region must then **equal** the oracle set when `exact`, and **contain** it when not.

The oracle covers:

- every operation in the registry. `listOps()` drives a coverage test against a fixture table, so
  a newly registered operation fails the suite until it has one;
- random 5–15 node graphs, including diamonds, and the built-in examples at small shapes;
- the fallback branches, whose thresholds are parameters so they can be lowered to shapes brute
  force can enumerate;
- bounded cones, against an oracle that cuts at the same frontier pointwise;
- tile plans: what each task reads and computes, producer sets, and each family's summed and
  distinct bytes and fan-out, on random graphs with some tensors left untiled.

A separate registry-wide law checks that `forward` and `backward` agree about whether the dependency
relation between a given input box and output box is empty, which needs no oracle and so applies to
every operation at any size.

## Layout

```
src/
  core/          headless engine
    region.ts    the box algebra everything rests on
    graph.ts     IR, validation, topological order, shape inference
    propagate.ts backward/forward walk, transitive or stopped at a frontier
    ops/         the operation registry; einsum carries the contractions
    metrics.ts   FLOPs, bytes, intensity
    plan/        tile families, tasks, producer/consumer joins
  parse/         lexer -> AST -> graph, JSON front end, compiler facade
  examples/      built-in graphs
  view/          presentation model shared by the Worker and the store; no React
  worker/        compilation, layout and plan/reuse analysis off the UI thread
  state/         the Zustand store and its action slices
  components/    React and canvas drawing
tests/           mirrors src/; tests/corpus/ holds the oracle and whole-engine suites
```

Imports only point down that list - `core`, `parse` ← `examples` ← `view` ← `worker`, `state` ←
`components` - and `tests/corpus/layers.test.ts` fails on any that does not.

See `SYSTEM.md` for the architecture and implementation invariants. `docs/README.md` classifies the
remaining design documents so historical plans are not mistaken for the live system contract.
