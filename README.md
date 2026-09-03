# TileCone

*See what a tensor region depends on.*

Takes a DAG of tensor operations plus input shapes, infers every intermediate and output shape, and
lets you select any region of any tensor to see — highlighted in place — the exact set of upstream
regions it depends on, and the downstream regions it influences.

The canonical example: for `C[M,N] = A[M,K] @ B[K,N]`, selecting the tile `C[64:128, 0:64]`
highlights `A[64:128, :]` and `B[:, 0:64]`.

No values are ever computed. The whole engine is integer interval arithmetic over index sets.

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # unit and integration suite
npm run build    # typecheck + production bundle
```

## The core invariant

A `Region` is a union of half-open axis-aligned boxes carrying an `exact` flag:

- `exact: true` — the region is **precisely** the dependency set.
- `exact: false` — the region is a **strict superset**, never a subset, and always carries a reason
  (`"strided conv"`, `"diagonal einsum"`, `"reshape run cap exceeded"`, …). The UI hatches these so
  an over-approximation is never presented as ground truth.

A region that is ever a strict *subset* of the truth is a critical bug: that is the case where the
tool lies. The test suite exists mainly to make that impossible.

## Testing

The failure mode of this app is plausible-looking wrong highlights, so correctness is checked
against a **brute-force oracle** (`src/test/oracle.ts`) rather than against the analytic rules
themselves. Every element of every tensor gets a unique id; id-sets propagate forward through the
graph using each op's *pointwise semantics* (`oracleDeps`), never its box-level backward rule. The
analytic region must then **equal** the oracle set when `exact`, and **contain** it when not.

That oracle runs over every op in the registry, randomly generated 5–15 node graphs including
diamonds, and all the built-in examples at miniature shapes. Other suites cover the region algebra,
reshape (the one genuinely hard op), forward/backward adjointness, primitive-vs-expanded composite
equivalence, tile sizing and coverage, and determinism.

## Layout

```
src/
  core/          headless: nothing here may import from ui/
    region.ts    the box algebra everything rests on
    graph.ts     IR, validation, topo sort, shape inference
    propagate.ts the backward/forward driver
    executor.ts  checked headless query API
    ops/         the op registry — einsum is the heart of it
    metrics.ts   flops, bytes, intensity
    reuse.ts     deterministic sampled reuse estimates
    expand.ts    composite ops -> primitive subgraphs
  parse/         DSL/JSON front ends + source-aware compiler facade
  ui/            React + canvas; store.ts holds all view state
  examples/      built-in demo graphs
  test/          oracle + suites
SYSTEM.md        current architecture and implementation invariants
```

`core/` is strictly headless so the oracle and any future CLI can run without a DOM.

## Headless compiler and executor

`compileDSL` is the checked boundary for user-authored programs. It preserves an unresolved source
graph, produces a separately resolved graph, retains statement spans for line-aware parse and
semantic diagnostics, and exposes the symbolic executor:

```ts
import { compileDSL } from "./src/parse/compiler";
import { box, fromBox } from "./src/core/region";

const program = compileDSL(`
  input A [256, 512] f16
  input B [512, 256] f16
  C = matmul(A, B)
`);

const cone = program.executor.upstream(
  "C",
  fromBox(box([64, 128], [0, 64]))
);
```

Use `tryCompileDSL` when diagnostics should be returned as data instead of thrown. The lower-level
`parseDSL`, `resolveGraph`, and propagation functions remain available for tests and tooling.

## Notes on the UI

- **The inspector answers two adjacent questions.** "What it needs" lists every tensor region the
  tile reads; "What it feeds" lists every region it influences. Their headings are also the
  view controls (`u` / `d`), so collapsing a question removes the same paint from the graph. Both
  can be collapsed independently for an explicitly labelled figures-only view.
- **Downstream reach is classified, not merely highlighted.** Each downstream row says whether the
  tile completes that region or contributes only partially, and the section heading rolls those
  verdicts up. A partial verdict derived from an approximate region may over-warn, never
  under-warn. On the cards, uniform fill means **What it needs**, a diagonal ruling means **What it
  feeds**, and a `/` hatch means an approximate bound; the inspector headings carry the same
  miniature treatments as an inline legend. The spacing of a ruling is uniform: it is held for a
  supplied-share measurement the engine does not expose yet, and reading it as a quantity today
  would be reading a placeholder. Completed versus partial is stated on the row, not in the paint.
  Each selected tile rules its downstream reach at its own angle, so where two tiles feed the same
  elements the rulings cross instead of one hiding the other.
  A ruled region is delimited by a hairline, because a ruling has no crisp edge of its own. Thin
  points and lines drop both the ruling and any perimeter stroke, so emphasis cannot invent a wider
  region than the analysis returned.
- **Row-major everywhere.** Rows are the second-to-last axis, columns the last. There is no per-card
  axis remapping: a different view of a tensor is a `transpose` node in the graph, where it is part
  of the computation being explained.
- **Tiles are the reading lattice, not the resolution.** One drawn cell is one tile, but a region is
  painted at element precision as an exact rectangle — a selection that ends mid-tile shows a crisp
  edge there, not a half-lit cell. Quantising the fill would read as "partly selected" when the
  truth is "these exact elements". One global detail slider sets the tile for every tensor (default ~5% of
  the smallest axis, snapped to a power of two). Card size depends only on the tensor's shape and
  the graph's scale, never on the tile, so changing detail re-lattices cards in place without
  resizing them. **None always means 1×1**: one logical tile per element for drawing, hover,
  starter tiles, and keyboard movement. On a large tensor those boundaries may be too dense to
  display individually, but the semantics are never silently coarsened.
- **Typed ranges are exact and authoritative.** Editing a tile's range never snaps or rounds it,
  even while snap is enabled. Snap governs future canvas gestures and keyboard movement only:
  arrows move by the current grid tile while snap is on and by one element while it is off.
  Toggling snap or changing grid detail never rewrites an existing range. If the current range is
  off the new lattice, its first snapped nudge aligns one edge without resizing it; later nudges
  use the whole tile. At a tensor edge movement clamps the whole range without shrinking it.
- **One px-per-element for the whole graph.** The scale is a property of the graph, not of each
  card, so a dimension two tensors share is drawn at the same physical length in both: in
  `C[M,N] = A[M,K] @ B[K,N]`, `A`'s width and `B`'s height are equal because both are `K`. Sizing
  each card to its own budget instead makes the contraction axis read as two different lengths.
- **Hue identifies which selected box a highlight came from**, capped at three validated
  categorical colors — the largest set clearing all-pairs contrast floors on both canvas surfaces.
  Chrome deliberately sits outside those hues (amber in dark, magenta in light) so a hot edge is
  never mistaken for a cone.
- **Hovering a box in the inspector emphasises what it needs and feeds; it never hides the others.**
  The peers fade but stay legible, because comparing tiles is the point of a multi-tile selection.
  Excluding one tile from the merged analysis and dependency paint is a separate, explicit toggle
  (`h`). Its faint selection rectangle remains available so the tile can be included again.
- **Tensor placement is editable without weakening the layout.** Drag a card's header—or its dotted
  handle—to reposition it; the tensor name keeps its own click for the shape popover. Curved connectors follow live; tensor cards and operation
  nodes remain hard collision boundaries, active connector crossings stay visible over cards while
  dim context stays behind the data, and a blocked drag is visibly marked. A completed drag and
  Reset layout are chronological
  workspace undo steps, Escape cancels an in-progress move, and shared links retain the arrangement.

Press `?` in the app for the complete shortcut sheet. The main bindings are:

- `u` / `d`: toggle What it needs / What it feeds; `f`: fit the graph; `[` / `]`: scrub a hidden axis.
- Arrow keys move the focused tile; Shift+arrow moves eight steps; `h` toggles its paint; Escape cancels or unpins.
- Ctrl/Cmd+Z undoes workspace movement; Alt+1 / Alt+2 toggle the panels; Ctrl/Cmd+Enter runs source edits.

See `SYSTEM.md` for the current architecture and implementation invariants. `docs/README.md`
classifies the remaining design and decision documents so historical plans are not mistaken for
the live system contract.
