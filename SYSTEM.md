# TileCone System Architecture

This document describes TileCone as it is implemented today. It is an engineering map of the system: where source text becomes an executable graph, how tensor regions propagate through operations, and how the React UI turns those results into an interactive visualization.

For product usage and DSL examples, see `README.md`. This document focuses on boundaries, invariants, data flow, and extension points. `docs/README.md` records which supporting documents are current, historical, local-only, or disposable.

## 1. Purpose and execution model

TileCone is a symbolic tensor-dependency explorer. Given a tensor graph and a selected region of one tensor, it answers two questions:

- **Backward Cone** (upstream): which input and intermediate elements can influence this region?
- **Forward Cone** (downstream): which later elements can be influenced by this region?

TileCone does **not** execute tensor values. Its interpreter operates on shapes, dtypes, index transformations, and regions of integer index space. FLOPs and memory traffic are estimates derived from that symbolic execution.

The top-level flow is:

```text
DSL source
    │
    ▼
parser ──► unresolved Graph + source map
    │
    ▼
graph resolver
    ├── validates structure, attributes, arity, shapes, and dtypes
    ├── topologically orders nodes
    └──► ResolvedGraph
              │
              ▼
       SymbolicExecutor
          ├── backward propagation
          ├── forward propagation
          └── region-aware metrics
              │
              ▼
        Zustand UI state
          ├── graph layout and tensor canvases
          ├── selection composition
          └── inspector and diagnostics
```

The core is deliberately headless. Parsing, graph resolution, propagation, and metrics do not depend on React or browser APIs.

## 2. Source layout

```text
src/
├── core/
│   ├── graph.ts          graph IR validation and resolution
│   ├── coordinates.ts    exact mixed-radix stepping with row-major carry and borrow
│   ├── executor.ts       checked public query boundary
│   ├── propagate.ts      generic forward/backward worklist, transitive or stopped at a frontier
│   ├── entangle.ts       what a region is combined with, per operation
│   ├── region.ts         exact and conservative region algebra
│   ├── metrics.ts        FLOP and byte estimates, and the layouts that make an operation a view
│   ├── demand.ts         summed and distinct demand on one tensor from several readers, and across drawn tiles
│   ├── contribution.ts   whether a tile completes its forward cone, or only adds to it
│   ├── tally.ts          a count per element as weighted boxes, and its measurement
│   ├── multiplicity.ts   how many outputs of each operation in a cone read each operand element
│   ├── expand.ts         composite-to-primitive graph rewrites
│   ├── notes.ts          plain-language dependency constraints
│   ├── shapes.ts         symbolic dimensions and shape errors
│   ├── dim-algebra.ts    canonical forms of dimension expressions, and equality for every binding
│   ├── dtypes.ts         canonical dtypes, byte widths, and the promotion lattice
│   ├── ops/              operation semantics, registry, barriers, and fallback thresholds
│   └── plan/
│       ├── tile-family.ts  regular tilings: tails, lazy coordinates, the tiles a box meets
│       ├── plan.ts         tile families checked against a graph, and each task's demand
│       ├── interfaces.ts   producer/consumer joins, family demand, and what a plan costs
│       └── replay.ts       a family's tasks one by one, and what each shares with the inspected task
├── parse/
│   ├── lexical.ts        identifier, string, and comment rules, shared with the highlighter
│   ├── ast.ts            DSL syntax tree
│   ├── parser.ts         text -> AST, recovering per line
│   ├── sugar.ts          call-name sugar, read by lowering and the printer
│   ├── lower.ts          AST -> Graph, desugaring and name resolution
│   ├── dsl.ts            facade over the above, plus the serializer
│   ├── compiler.ts       three-phase facade and collected diagnostics
│   ├── source.ts         source spans, source maps, per-argument spans
│   └── json.ts           JSON graph import/export
├── view/               presentation model: pure functions and data, no React, no store
│   ├── tensor/
│   │   ├── tensor-view.ts    visible-axis and hidden-axis view configuration: project, slice, unfold
│   │   ├── fold.ts           unfolded panels: stride order, gaps, hit-testing
│   │   ├── fold-layout.ts    shared nesting and size budgets for layout and validation
│   │   ├── deck.ts           a slice card's deck: how many sheets lie behind the slice
│   │   ├── tiling.ts         tile-size policy and slider stops
│   │   ├── tile-spec.ts      a tensor's own per-axis tile and what a gesture selects
│   │   ├── card-size.ts      tensor-card measurement and per-card scale, usable in either realm
│   │   ├── grid.ts           grid geometry: cells, region rects, snapping, nudging, paint types
│   │   ├── raster.ts         per-card and viewport-wide backing-store budgets
│   │   ├── shape-label.ts    axis-name, symbolic, and numeric shape readings
│   │   ├── seeds.ts          default extents: drawn planes, starter tiles, first plan divisions
│   │   ├── gesture.ts        what a drag selects, and what a Plan-view gesture does
│   │   ├── movement.ts       rigid tile movement through valid snapped or element origins
│   │   ├── layers.ts         the card's paint stack as data: cones, stipples, seeds, plan paint
│   │   ├── replay-paint.ts   family-replay paint over the plan
│   │   └── menus.ts          the axis table's menu and a card's view menu
│   ├── graph/
│   │   ├── graph-scene.ts    structural layout and live routed-scene projection
│   │   ├── graph-geometry.ts collision constraints and connector routing
│   │   ├── overview-labels.ts  low-zoom label placement
│   │   └── node-layout.ts    graph-node offset contracts
│   ├── workspace.ts      shared vocabulary: selection parts, directions, views, and their pure helpers
│   ├── menu.ts           a menu as data: actions, choices and separators, bound by id
│   ├── demand.ts         a shared-demand row in words, for both views that show one
│   ├── multiplicity.ts   a read-multiplicity row in words
│   ├── reuse-lens.ts     the reuse lens: its scope, its validated ramp, and readings
│   ├── plan-work.ts      the whole-plan figures as rows, and a tiling in words
│   ├── palette.ts        validated categorical hues and canvas surfaces
│   ├── format.ts         presentation-only numeric formatting
│   ├── selection-range.ts  range syntax for editable selection parts
│   ├── shortcuts.ts      the single manifest of key bindings and help copy
│   └── dsl-highlight.ts  source-editor syntax tokens
├── worker/             the off-thread realm: imports core, parse, and view only
│   ├── analysis.worker.ts  compiler, layout, plan, and reuse host
│   ├── analysis-jobs.ts   pure jobs shared by the Worker and headless tests
│   └── analysis-protocol.ts  structured-cloneable Worker messages and artifacts
├── state/
│   ├── store.ts          the Zustand store: initial state and the action slices
│   ├── types.ts          the State type and the store's setter and getter types
│   ├── actions/          action slices: workspace, selection, view, plan, layout
│   ├── analysis.ts       per-part propagation merged into the aggregate
│   ├── selection-edit.ts selection transactions and keeping a moved tile on screen
│   ├── plan.ts           plan derivation and task inspection
│   ├── load.ts           installing a compiled graph, an example, or a shared workspace
│   ├── history.ts        workspace undo snapshots
│   ├── compile-epoch.ts  the request version that makes a build cancellable
│   ├── share.ts          workspace link encoding and decoding
│   └── analysis-worker-client.ts  compile/layout/query request correlation, cancellation, and lifecycle
├── components/         React
│   ├── graph/GraphView.tsx  graph rendering, gestures, highlighting, and viewport
│   ├── card/
│   │   ├── TensorCard.tsx    interactive tensor canvas and its layer builders
│   │   ├── draw-grid.ts      canvas drawing: fills, rulings, stipples, hatches, heat, lattice, marks
│   │   ├── useLensPaint.ts   the reuse lens's paint for one card, derived from store state
│   │   └── deal.ts           the slice deck's deal animation
│   ├── inspector/
│   │   ├── Inspector.tsx     the panel: tabs, and which sections each view shows
│   │   ├── tiles.tsx         the tile header, the tiles list, and range editing
│   │   ├── cones.tsx         the Backward and Forward Cone sections
│   │   ├── notes.tsx         dependency notes
│   │   ├── operand-reuse.tsx the Operand reuse section, for drawn tiles and for a plan task
│   │   ├── tabs.tsx          the view switch and the empty panel
│   │   ├── usePlanReplay.ts  the family replay: request and playback timers
│   │   ├── inspector-analysis.ts  memoized inspector derivation and costly-probe gating
│   │   ├── AxisEditor.tsx    every axis of the inspected tile
│   │   └── PlanPanel.tsx     the Plan view: tiled tensors and their work, one task, family demand
│   ├── chrome/           SidePanel, PanelFrame, WorkspaceHeader, GridControls,
│   │                     LayoutControls, ShortcutsDialog, OptionsMenu, DraftField, CopyButton, clipboard
│   ├── hooks/            useKeyboard, useFocusPolicy, useDismiss, useFrameThrottle, useDebounced, useDragGuard
│   └── pointer.ts        which presses start a gesture
├── examples/             built-in DSL examples
├── App.tsx               UI composition and URL-state bootstrap
└── main.tsx              entry point; applies the stored theme before paint
```

The folders are layers, and imports only point down:

```text
core, parse  <-  examples  <-  view  <-  worker, state  <-  components  <-  App, main
```

`view` is what both realms share: the Worker measures cards and lays out the graph with it, the
store derives tiles and snapping from it, and the components draw with it. It therefore imports
neither React nor the store. `tests/corpus/layers.test.ts` walks every import in `src/` and fails
on one that points up or sideways, or on a package a layer is not allowed to use (React outside
`components`, zustand outside `state`).

Tests live outside `src/`, in `tests/`, which mirrors the source tree: `tests/core/` holds the
engine's tests, `tests/view/tensor/` the tensor view model's, and so on. Many suites are about a
topic that spans several files - `figures` and `view-traffic` both exercise `metrics.ts` -
so the mirror is by folder, not file by file. `tests/corpus/` holds what exercises
the whole engine or the whole tree rather than one module: the brute-force oracle, the op
fixtures, the harness, the registry laws, the fallback, frontier, multiplicity and shared-operand
suites, and the layering check.

## 3. Core intermediate representation

The graph IR in `src/core/graph.ts` consists of three principal records:

- `Tensor`: identity, name, shape, dtype, optional axis names and semantic role, and its producer after resolution.
- `Node`: operation name, input/output tensor IDs, attributes, and an optional display label.
- `Graph`: node and tensor tables plus symbolic parameter bindings.

There are two meaningful graph phases.

### Unresolved graph

The parser creates the structural graph. Declared inputs and weights already have shapes and dtypes, while operation outputs are placeholders awaiting inference. Tensor references, attributes, and node ordering have not yet been trusted.

### Resolved graph

`resolveGraph` clones the input graph and then establishes the invariants required by execution:

1. Parameters are positive safe integers. Derived symbols evaluate in declaration order, over
   the names before them, to positive safe integers.
2. Node and tensor IDs are unique and all references exist.
3. Every operation exists in the registry.
4. Static and attribute-dependent arity constraints hold.
5. Attributes pass the operation's schema, which is **closed**: an attribute the schema does
   not name is an error, not a stripped key. A misspelling such as NumPy's `keepdims` for
   torch's `keepdim` would otherwise resolve to a different shape without a diagnostic, and
   every region and metric downstream would describe a graph the user did not write. The
   narrowing happens once at the registry boundary rather than in each `OpSpec`, so a new
   operation cannot forget it.
6. Tensor producers are unique and the node graph is acyclic.
7. Symbolic dimensions resolve to concrete shapes. A dimension may be arithmetic over
   parameters (`H*D`, `E/H`, `(H+D)*2`), so a relationship holds where it is used rather
   than being precomputed into a literal that silently stops agreeing. Division must come
   out whole: an axis is a whole number of elements, so an inexact one means the stated
   relationship does not actually hold. An authored dimension is at least 1, whether it is
   written as a literal, bound to a parameter, or falls out of arithmetic: a tensor with an
   empty axis holds no elements, so it has no regions to relate, and declaring one is a slip
   rather than a question with an answer.
8. Each operation infers the expected number of output shapes and dtypes.
9. All inferred extents are non-negative safe integers and all dtypes are supported.
10. Extents an operation requires to be equal are checked as expressions, not only as
    numbers. Each operation lists them (`symEqualities`): a contraction's two sides, a
    reshape's element counts, concatenated and broadcast axes, split sizes against the cut
    axis. `dim-algebra.ts` puts both sides into a canonical form, a polynomial over the
    parameters with exact rational coefficients and monomial divisors. A pair whose forms
    differ is a warning on `ResolvedGraph.warnings`, not an error: `M*2` against `N*2`
    resolves when `M == N` and fails for every other binding, so the graph is valid as bound
    but a parameter control can break it. A parameter against a literal is held to the same
    rule. A derived symbol is expanded into its definition once per resolution and cached,
    and each comparison visits at most a fixed number of expression nodes, past which it is
    undecided, so a long chain of definitions cannot stall compilation. Pairs are compared
    across every operand, not against the first, so an operand of unknown extent hides no
    mismatch between the others. The compiler returns warnings with a successful build, as diagnostics of severity
    `warning` with their spans, and the Worker's compile artifact carries them to the editor. The check runs over a second set of symbolic shapes in
    which every extent with no known expression is an opaque symbol. That covers
    convolution spatial axes, `slice`, `pad` and barrier outputs, and a pair that depends on
    one is left undecided rather than refuted. A division by a sum is opaque on the same
    terms.
11. Consumer indexes, producer links, and a stable topological order are built.

Resolution is referentially transparent: it operates on a clone rather than mutating the caller's graph. The resulting `ResolvedGraph` is the only graph form accepted by symbolic execution. Its `shapeParameters` lists the bindings read by declared dimension resolution and registry-owned shape inference, in declaration order. These semantic reads include operation shape attributes without relying on displayed symbolic extents or scanning unrelated strings, and the metadata crosses Worker boundaries as ordinary data.

## 4. Regions and correctness contract

A `Box` is an axis-aligned product of half-open integer intervals. A `Region` is a union of boxes plus approximation metadata:

```ts
type Region = {
  boxes: Box[];
  exact: boolean;
  reasons: string[];
};
```

The central correctness rule is:

> An exact region equals the true dependency set. An inexact region must be a conservative superset of it.

The downstream contribution verdict follows the same one-sided guarantee: **partial may
over-warn, never under-warn**. Approximation can invent missing residue and therefore a partial
warning, but it cannot make a genuinely partial contribution appear complete.

An under-approximation is therefore a correctness bug; an over-approximation is permitted only when marked `exact: false` with a reason.

`src/core/region.ts` owns region algebra and normalization. A region's boxes **may overlap**. `canonicalize` normalizes only in union-preserving ways - it drops empty boxes, drops boxes contained in another, and merges boxes that agree on every axis but one - and runs those to a fixpoint. It never splits a box on overlap. If the box count exceeds the configured cap it coarsens toward the cap by merging neighbours into their hulls, and records the loss of precision.

Normalization uses ordered sweeps rather than pairwise searches. Merging groups boxes by their other axes; containment orders lower corners ascending and upper corners descending, placing containers first and making results independent of arrival order. Reshape and strided mappings can produce thousands of boxes before the cap applies, so these algorithms must scale with that input. Containment has a work budget: stopping early retains redundant boxes without changing the represented set.

Keeping operand bands intact makes the representation meaningful to users. `matmul(A, A)` contributes a row band and a column band; disjointifying them would produce three storage fragments rather than the two regions the operation reads. `docs/OVERLAPPING_BOXES.md` records this design decision.

Overlap is therefore resolved where quantities are measured, not where they are produced. `count` computes exact union cardinality with a recursive dimensional sweep, choosing the axis with the fewest endpoints at each level. It never materializes geometric fragments and therefore does not fall back to the box-count cap while measuring the represented set. `disjointify` remains available when a consumer needs an explicit non-overlapping partition. The rule every consumer must know is one sentence:

> **Cardinality is measured on the set.** Boxes describe a region; they do not count it. Summing box volumes double-counts the overlap.

`count` supplies distinct elements, bytes, percentages, overlap totals, and uniform per-element FLOP costs. Operations with nonlocal work use region-aware formulas. `points` and canvas fills disjointify when they need explicit pieces: compositing an overlapping element twice would falsely encode multiplicity as darker paint.

Slice expressions, box counts, propagation, attribution, and canvas perimeter marks use the stored boxes. `drawGrid` builds separate rectangle lists: fills, rulings, and hatches use the disjoint form; outlines and hairline delimiters use the stored bands. Borders may cross, but tracing partition fragments would invent boundaries unrelated to the operands.

`regionOverlap` exposes the difference between the two. It is not a diagnostic: for `matmul(A, A)` the gap is exactly the square both operand bands read, so it is the count of elements genuinely read twice. Inspector rows state it as `N shared` when it is non-zero, so the listed boxes are seen to sum past a total that is nonetheless correct.

`coversAxisFully` tests the union, not whether one box spans an axis. It sweeps the other axes until their active set is constant, then tests coverage of the target intervals, with an early exit. From rank 3 upward a covering cross-section need not contain any box's corner, so corner-anchored probes are insufficient and can miss genuine full-axis pulls.

Repeated analysis is cached at two levels. Propagation resolves operation dispatch, input/output shapes, and attributes into a plan once per `ResolvedGraph`; later upstream, downstream, attribution, contribution, and reuse queries execute that plan. While a selection has at most twelve parts, the store also retains each part's two directional cones by tensor and box geometry, so adding, editing, or deleting one part only propagates the changed geometry before re-merging the results.

Set difference preserves the same contract. When the subtrahend is inexact, subtracting its represented superset could remove true elements, so the result conservatively retains the minuend and records `inexact subtraction`. Proof-oriented consumers such as full-axis detection therefore decline conclusions drawn only from an inexact bound.

Difference and intersection compute their represented sets before applying the approximation cap. Each subtrahend box can cut a fragment into up to `2 * rank` pieces, and intersection can create N² pieces. Neither operation may substitute a bound merely to shorten its own work; the cap coarsens the computed result.

The UI preserves the user's selection as **ordered parts** so individual boxes can be focused, colored, moved, or subtracted. Before execution, those parts are canonicalized into a mathematical set. This distinction prevents presentation identity from leaking into dependency semantics.

Each ordered part carries **its own tensor**, allowing tiles on several tensors to be compared. The store propagates each part from its own root and `mergeProps` unions results per tensor, retaining the shortest hop to any root. This separates presentation identity from the executor's single-root queries while preserving the conservative dependency guarantee.

## 5. DSL and compiler

The compiler boundary is `compileDSL` / `tryCompileDSL` in `src/parse/compiler.ts`.

```text
text
 └─ parseProgram          (parse/parser.ts)   → Program AST + per-line errors
      └─ lowerProgram     (parse/lower.ts)    → Graph, SourceMap, + per-statement errors
           └─ resolveGraphCollecting          → ResolvedGraph + per-node errors
                └─ SymbolicExecutor
```

Three phases, each collecting rather than throwing. `compileDSL` returns the source, unresolved graph, resolved graph, source map, and an executor as one coherent artifact; `tryCompileDSL` is the diagnostic form.

The store retains structured diagnostics and the source panel displays every reported failure with its span. It uses `tryCompileDSL`; the throwing form would expose only the first message.

**Every phase reports everything it found, and later phases still run.** A document with four unrelated mistakes returns four diagnostics in source order, not one. This is what the AST buys: a line that fails to parse becomes an `error` statement carrying whatever output names were read before the failure, so parsing continues and later statements reading that name are not reported as a second, invented error. Lowering poisons the names of any statement it rejects, for the same reason, and `resolveGraphCollecting` prunes a failed node together with everything downstream of it so a consequence is never reported as a cause.

`resolveGraphCollecting` is implemented by resolving repeatedly and pruning the subject of each error, deliberately rather than by threading an error list through `resolveGraph`. A second, collecting copy of that validation would be free to drift from the throwing one, and the two disagreeing about whether a graph is valid is a worse failure than the extra passes cost.

A `GraphError`'s subject carries the attribute it is about when it is about one, so a diagnostic can underline that attribute rather than recovering its name by parsing the sentence the error already built from it.

Diagnostics carry the narrowest span that is certainly about the error. `SourceMap.nodeArgs` records a span per input and per named attribute, so an unknown tensor underlines that tensor reference, a misspelled attribute underlines that attribute, and an unknown op underlines the call name. A statement span is the fallback, used when the error is genuinely about the statement - a shape mismatch is about the pairing of two operands, not either one.

The AST describes syntax; lowering owns operation desugaring. `parse/sugar.ts` supplies both the call-to-operation mapping and its printer inverse, including which attributes a call name already encodes. The printer omits only those attributes, preserving every other one through expansion and recompilation. Lexical rules live in `parse/lexical.ts`, shared by the parser and editor highlighter, so syntax colouring follows the grammar.

### DSL surface

The DSL is line-oriented and intentionally small:

- Numeric assignments bind symbolic dimensions, such as `B = 1` and `D = 32`.
- An assignment of a dimension expression defines a derived symbol, such as `E = H*D`. It may
  read only parameters and derived symbols defined above it, which keeps definitions acyclic.
  Lowering stores it in `Graph.derived` rather than `params`: it sizes dimensions and appears
  in symbolic shapes under its own name, the symbolic checks expand it into its definition,
  and reading it counts the parameters it is written over as used. It has no value of its
  own, so the parameter controls do not offer one.
- `Tensor(...)` declares a graph input and `Parameter(...)` declares a learned graph input.
- An axis may be labelled in place: `X = Tensor(batch=B, seq=S, emb=E, dtype=fp16)`. A constructor
  names every axis or none, because a half-named declaration is a slip rather than a
  statement; propagated names may legitimately have holes.
- Assignments create operation nodes and one or more output tensors.
- Positional arguments refer to tensors; named arguments become operation attributes.
- A dimension is an expression over parameters, not just a name: `X = Tensor(B, S, H*D)`, and
  the same language works in shape-valued attributes: `reshape(X, shape=[B, S, H, E/H])`,
  `expand(X, shape=[B, KVH, G, S, D])`, `split(X, sizes=[H*D, KVH*D, KVH*D])`.
  `Sym` remains `string | number` - the string simply holds the expression, a bare parameter
  being the one-symbol case - so nothing downstream had to learn a new shape type and the
  text the author wrote is what `notes.ts` and `toDSL` see. `toDSL` prints such an attribute
  as the expression it is rather than as a quoted string: expanding a composite writes that
  text into the editor, and `sizes=["H*D"]` is not what anyone typed. Only shape inference
  resolves these; every later pass reads the extents it produced, so `split` derives its cut
  offsets from its own output shapes rather than re-reading the attribute.
- Lists, numbers, booleans, strings, and identifiers are accepted attribute values. Numbers
  accept decimals and scientific notation, so `toDSL` cannot emit a literal the parser then
  rejects.
- Dtypes use the familiar external spellings `fp32`, `fp16`, `bf16`, `fp8`, `int64`, `int32`,
  `uint8`, `int8` and `bool`; the graph IR retains its compact internal dtype names.
- `#` introduces a comment.

Lowering normalizes convenient spellings into a smaller internal operation set. For example, `relu(...)` becomes an `elementwise` node with `fn: "relu"`, reduction names become `reduce` nodes, and layer/RMS normalization become `normalize` nodes with explicit attributes.

Every statement is an assignment and is closed at the end, so
`X = Tensor(4, 8, dtype=fp32) garbage` is a syntax error rather than a tensor whose malformed
suffix is silently discarded.

The source map connects compiler errors back to declarations or operation calls. Parsing errors are reported as DSL diagnostics; graph, shape, dtype, and arity failures become semantic diagnostics with the most specific available source span.

`src/parse/json.ts` provides a second, programmatic graph frontend. It validates the serialized structure, while deep semantic validation remains the resolver's responsibility. JSON import is currently a core capability rather than a primary UI workflow.

## 6. Operation registry

`listOps()` exposes the registry, and two suite-wide properties are driven off it rather than off a hand-kept list:

- **Coverage.** `tests/corpus/op-fixtures.ts` holds one representative instance of every op; `registry.test.ts` asserts the table covers `listOps()`, so registering an operation without a fixture fails the suite instead of shipping untested. Each fixture is also run against the brute-force oracle, so "registered" implies "oracle-checked".
- **Adjointness.** `forward` and `backward` are two implementations of one relation, and for exact regions they must agree about whether it is empty: `forward(inBox) ∩ outBox ≠ ∅` iff `backward(outBox) ∩ inBox ≠ ∅`. This needs no oracle, so it runs on every op at every fixture size, covering the ops the random-graph corpus does not compose.

### Barriers

`opaque` is the registry's escape hatch: an operation whose output shapes are declared and whose semantics are not. It exists because of what the resolver does with an operation it does not know. `resolveGraphCollecting` prunes the failing node and, transitively, everything downstream of it - the right answer for one line of hand-written DSL, where the author gets one error and no invented cascade, and the wrong one for an imported model, where a single unrecognised node silently replaces the graph with a shorter one. A cone that stops early is a **subset** of the truth, which is the one failure this engine treats as critical.

A barrier keeps the node instead. Every output element is assumed to read every input element and every input element to reach every output: the weakest true statement about an operation nobody described, a superset by construction, marked inexact with the original operation's name as its reason. Everything downstream then behaves as it does for any other approximation - hatched paint, `≈` on the rows, `≤` on the figures - and implementing the operation for real later is purely a narrowing, with nothing else to change. Its `oracleDeps` states that same total dependence, so the corpus checks it like any other op; the adjointness law has nothing to say about an operation that is never exact, and `op-fixtures.ts` marks that claim explicitly (`inexactByDesign`) so the law asserts the op really is inexact everywhere rather than being quietly skipped.

The one quantity a barrier cannot bound is its own arithmetic. `flopsFor` returns zero, and the spec declares `unknownWork` so that zero is never summed into a total beside real upper bounds as though it were a measurement. A FLOP total spanning a barrier reports `unknown` rather than a number (§8), and the dependency note says so where the reader will see it.

A barrier declares its outputs one at a time: `shapes` and a parallel `dtypes`, because real operations do not have a single output type - `TopK` returns f32 values beside i64 indices, `MaxPool` an optional index tensor, `LayerNormalization` a mean and an inverse standard deviation. Bytes are measured per tensor from that tensor's own dtype, so one declared type for all outputs would misreport the footprint of whichever it did not fit. The list may be omitted, or an entry left `null`, to take the promotion of the inputs. The two lists are parallel rather than a list of `{ shape, dtype }` records because the DSL has no record literal and adding one to the grammar to serve a single attribute is not a trade this codebase makes; their lengths are checked against the output count instead.

It also carries `domain`, `opset` and `sourceName` as provenance. None of it changes what the barrier claims; it is what a reader needs in order to decide whether implementing the operation is worth it, and `domain` qualifies the displayed name where it is not the default one, so a vendor's `Attention` is distinguishable from anyone else's on the card, in the approximation reason, and in the note.

Zero inputs is allowed, and is the one place the registry-wide "every op reads something" rule is exempted by name. A node that reads nothing and produces a value is constant-like and is usually better written as a declared tensor, but the metadata has to be able to express it: the fallback from "no representation" is dropping the node, which shortens the graph.

### Display names

An operation's registry name is its identity for dispatch and frequently not what the author wrote: `relu` and `gelu` are both `elementwise`, `sum` and `amax` are both `reduce`, and a barrier is `opaque` standing in for something with a name of its own. `OpSpec.displayName(attrs)` derives the name to show, `opLabel(node)` resolves it once for every surface that names an operation (graph cards, their layout measurement, the operations list), and an explicit `node.label` still wins for a caller that has a better name. Derived rather than stored, on the same principle as resolved shapes: an edit to the attributes cannot leave a stale label behind.

### Dtypes

Operations that *compute* from several tensors promote (`promotingDTypeOutputs`); operations that *move* data require a match (`uniformDTypeOutputs`). The lattice follows PyTorch rather than NumPy's older value-based rule: category dominates width across families, so `f16` with `i32` is `f16` and not a widening to `f64` that no inference kernel performs. Mixed precision is ordinary inference practice and has an obvious result for an add or a matmul; a `concat` of fp16 and fp32 has none, since the output is one buffer, so it stays an error that names the cast to insert. `f16` with `bf16` widens to `f32`: same width, neither contains the other, and picking either would silently discard range or precision.

`i64` and `u8` are in the lattice because index and quantisation tensors need them, not because the DSL wanted more ways to write an integer. An index, axis or shape tensor is int64 wherever it comes from, and holding one as `i32` would halve its byte count in every footprint that counted it; `u8` is the same argument for a quantised model's `zero_point`. A tensor whose source storage was narrower than its canonical dtype carries `dtypeWidening`: tensor and aggregate byte figures then include its note and report `upper`, so a two-byte source element held as four bytes cannot become an exact footprint claim. Nothing sets that marker today - it exists so a later converter cannot introduce a widening without the figures noticing.

The elementwise function table in `ops/elementwise.ts` is the definition of what `fn` may be - the attribute schema enumerates its keys, and the DSL's call-name sugar reads the same table. It also carries each function's own arity, so `relu(a, b)` and `div(a, b, c)` are rejected rather than quietly computing something else.


Every operation is described by an `OpSpec` in `src/core/ops/types.ts` and registered in `src/core/ops/index.ts`. An operation owns all semantics specific to that operator:

- attribute schema;
- static input and output cardinality;
- optional attribute-dependent arity validation;
- output shape inference;
- optional axis-name propagation, defaulting to unnamed;
- optional symbolic-extent propagation, defaulting to literal, with `null` for an extent the
  operation derives without an expression;
- optional symbolic equalities between input extents, checked as expressions during
  resolution;
- output dtype inference;
- backward region mapping from an output box to input regions;
- forward region mapping from an input box to output regions;
- optional pointwise dependency oracle used by tests;
- FLOP estimation, optionally over a whole region when work is shared or nonlocal;
- an optional plain-language note about the constraint it imposes on a cone (§8a);
- optional read multiplicity, or the `readsOnce` declaration, counting how many outputs read each
  operand element (§8c).

This is the main extensibility seam. The graph resolver and propagation engine do not switch on operation names; they dispatch through the registry.

Operations currently cover LLM-oriented primitives such as matrix multiplication, linear layers, elementwise functions, reductions, normalization, softmax, reshape/transpose, indexing, gather, concatenation/splitting, expansion, scans, and layout operations.

Elementwise operations broadcast under the usual trailing-aligned rule, rank 0 included, and
the region mappings broadcast with them: an axis of extent 1 maps back to `[0, 1)` and forward
to the full output extent, so a bias or a mask shows the fan-out it really has. Both directions
are checked against the brute-force oracle rather than argued for. Convolution and pooling are supported but treated as secondary/experimental surfaces.

Some high-level operations are expandable. `src/core/expand.ts` rewrites `softmax` and supported normalization forms into primitive subgraphs, preserves the externally visible output tensor, then resolves the rewritten graph again. Expansion is a graph transformation, not a special execution mode.

## 7. Symbolic execution

`SymbolicExecutor` in `src/core/executor.ts` is the checked public API. It validates that the selected tensor exists and that every region interval has the correct rank, integer bounds, and in-bounds coordinates. It then delegates to the propagation engine.

`src/core/propagate.ts` implements one generic traversal with two directions:

- **Backward:** visit nodes in reverse topological order and map affected output regions to their input regions.
- **Forward:** visit nodes in topological order and map affected input regions to their output regions.

For each node, the engine calls the registered operation once per relevant source box, accumulates contributions for each destination tensor, and canonicalizes the union. It also carries approximation reasons and records the shortest propagation depth used by the visualization.

Queries may request upstream, downstream, or both directions. Low-level propagation functions remain available for operation development and testing, but UI and external callers should normally use the executor so malformed selections fail at the boundary.

### Bounded cones

A transitive cone answers "what does this tile ultimately depend on". Planning a stage asks something narrower: what does this tile demand of the stage's own boundary. `propagateWithin(graph, seed, direction, frontier)` - checked as `executor.upstreamWithin` and `downstreamWithin` - answers it by stopping at a set of frontier tensors.

The definition is one sentence: a bounded cone relates the seed to the elements joined to it by a dependency path whose *interior* avoids the frontier. The endpoints of a path - the seed, and the tensor a region is reported on - may lie on it. A frontier tensor is therefore reached and carries its demand, but nothing is followed through it. A tensor also reachable *around* the frontier keeps exactly the part that route supplies: for `D = add(transpose(A), A)` bounded at the transpose's output, `D[0,1]` demands `A[0,1]` and not `A[1,0]`. The frontier cuts paths, not tensors.

Two consequences fall out of the definition rather than being added to it. The seed is never blocked, even when it is on the frontier, because it is where every path starts. And so one boundary serves both directions: a stage's inputs and outputs together bound a backward walk from one of its outputs and a forward walk from one of its inputs, with no need to say which side the seed is on. `stoppedAt` lists the frontier tensors a cone reached other than its seed - where its demand lands on the boundary.

`crossings` splits that demand by operand slot. It has one entry per node and slot through which the cone reaches a frontier tensor, with the region carried through that slot. A tensor read through two slots, as in `matmul(X, X)`, has two entries, and their union is the tensor's region in `tensors`. The walk computes each slot's contribution before merging it into the tensor's region, and a bounded walk keeps that contribution for tensors on the frontier.

It is the same walk as the transitive cone, with a set of tensors that may be reached but are never read as a source. The blocking is applied in the loop rather than in the memoized propagation plan, which is shared by every query on the graph and must not change with one caller's frontier. An empty frontier gives the transitive regions exactly.

`BoundedCone` is deliberately not a `PropResult`, and the two differ in shape so the compiler keeps them apart. Everything that consumes a `PropResult` - metrics, notes, contribution, input sharing, the canvas - reads a missing tensor as no dependency and a tensor without a producer as a graph input. A cone that stops early breaks both readings: the graph inputs past the frontier are absent, and the frontier tensors standing in for them have producers. Passed to `computeMetrics` it would understate input bytes while presenting them as a bound. A test holds a `@ts-expect-error` on exactly that call, so a change that made the types compatible fails the build.

The checked boundary refuses a frontier naming a tensor that does not exist (`EXEC_FRONTIER`), because an ignored name stops nothing and the cone would run past the boundary it was given. It also refuses a frontier that is not a list: a string is iterable, and `"CW"` must not be read as two tensors. The direction is required and is one of the two, never `both`, since a frontier is usually a stage's whole boundary and a default direction is how a caller gets the other half of it.

## 7a. Entanglement

A third relation alongside upstream and downstream: given a block of one operand, which elements of the *other* operands meet it in the same term of the computation. `executor.entangled(tensorId, region)` returns one entry per (node, other-slot) pair.

In the UI it is a separate toggle (`e`), not a fourth `Direction`: what a tile reads and what it is multiplied against are independent questions, and folding them into one control would make them alternatives. Its paint is a stipple — solid fill means needs, a ruling means feeds, and a third relation needs a mark that is not a line, so it cannot be read as another cone or as the approximation hatch. Hue still follows the tile's colour slot, and hiding a tile hides its entanglement too. A tensor carrying a visible entangled region returns to full graph opacity even when it is outside the directional cone; that visibility does not heat unrelated connectors or turn entanglement into a graph path. Independently of this view, every incoming link of an operation with more than one input is marked `arg0`, `arg1`, and so on near the operation endpoint, preserving operand order even when the same tensor occupies several slots.

It is not a view over the other two, and the difference is the point. For `C = A @ B` and a block `A[0:4, 0:4]`, the downstream cone is `C[0:4, :]` and what that cone reads of `B` is *all* of `B` - correctly, since every `C[m,n]` in the band reads every row. But that block of `A` is only ever multiplied against `B[0:4, :]`. Composing `forward` with `backward` cannot recover this: the correlation between which input element produced which output element is discarded at the region boundary, where a set of elements becomes a shape.

`einsum` (so `matmul`, `bmm`, `linear`), elementwise, `conv`, `concat` and `gather` answer exactly. `concat` answers *empty*, which is information rather than a gap: its operands are placed side by side and never combined, so a kernel may write the pieces independently. `normalize` deliberately does not answer - its output is not a sum of products, so "the same term" has no single reading there, and an honest bound is better than a hook with mushy semantics.

Operations answer through `OpSpec.coaccess`, with `OpSpec.oracleTerms` as its ground truth - the terms an output element evaluates, each naming what every input contributes. That is strictly finer than `oracleDeps`, which flattens terms away and says only that a whole row of `A` and a whole column of `B` reach one output. `registry.test.ts` requires the two hooks together: `coaccess` without `oracleTerms` would be an analytic claim nothing can check.

An operation without the hooks falls back to composing `forward` and `backward`, marked inexact with the reason. That is a genuine superset - every term combining two elements does land in some output the block reaches - so a missing implementation degrades rather than lies.

Scope is one operation. Tensors meeting several hops apart relate through a chain of these; following the chain would mean carrying the correlation across propagation, which is a materially harder problem.

## 8. Metrics

Metrics are derived from the backward dependency result in `src/core/metrics.ts`:

- **FLOPs:** estimated from affected operation outputs. Operators may provide box-local or whole-region cost functions.
- **Input bytes:** bytes read from leaf tensors.
- **Intermediate bytes:** unique footprint of contributing produced tensors, not read/write traffic.
- **Output bytes:** bytes in the selected producer output.

These are what the drawn tiles need, not a claim about how they are executed. Which tensors are
written and which are computed where they are needed is a plan's decision (§8b), and the Plan view
is where it is made and costed.

`OpSpec.layout` identifies views: transpose, slice, and expand always, and reshape when its input is row-major. Layouts track dense axis order, allowing composed transposes to become row-major again; slice and expand produce `strided` layouts. A reshape of any other layout is conservatively treated as a copy. `viewLayouts` derives this in topological order. `flopsOver` skips views, but a plan charges a tiled view as a written copy.

`analysisGroup` names the tensor whose drawn tiles the readout describes. A draw, pin, or group-header choice sets it; hover does not. `groupFocus` narrows within that group, while hovering another group's tile only emphasises its canvas mark. Drawing releases the pin, and deleting a group's last tile retires it and falls back to the anchor tensor.

Footprint attribution and entanglement retain global colour indices but exclude other groups. Above the per-tile cap, group analysis still works while per-tile figures are unavailable. Movement and hidden-axis keys use the same target. The header counts the union of enabled group tiles, excluding hidden tiles and duplicate overlap; above the attribution cap it counts all group tiles. Range editing and deletion belong to the tile list, including when it contains only one tile.

Symbolic extents travel that path too, and are a **separate** propagation from axis names -
`inferSymShapes` rather than `inferAxisNames`. A name says what an axis *is* and survives a change
of extent; a symbol says how *wide* it is and does not. A convolution's height axis is still `h`
after striding but is no longer `H` elements, so conv names it while leaving the derived spatial
extent literal. It can still carry unchanged batch and output-channel symbols; pooling similarly
carries batch/channel, and gather takes the replacement-axis extent from its indices input.
A contraction carries its free axes (`[M, K] × [K, N] → [M, N]`), a transpose permutes symbols with
their axes, a reduction drops the symbol of an axis it collapses, a concatenation states the joined
axis as the sum that met there (`P+T`, which the dimension grammar can read back), and a reshape
takes its split axes from the target shape the author wrote.

Nothing here is taken on trust. `resolveGraph` evaluates every proposed symbol against the bound
parameters and keeps it only if it comes out at the extent inference actually produced; anything
else falls back to the literal. `Tensor.symShape` is therefore always full length and always
evaluates to `resolved`, so a wrong mapping degrades to a number rather than asserting a shape the
graph does not have.

Axis names travel the same path as shapes and dtypes. A declaration names its axes, and each
operation decides what survives it: an einsum label carries a name across a contraction, a
transpose permutes names with their axes, a reduction drops the name of an axis it collapses,
and a reshape names only the axes a group maps one-to-one - splitting an embedding into heads
produces axes the source never named. `inferAxisNames` is optional and defaults to unnamed,
because the inspector and the notes layer present a name as the source's own word for an axis,
so a wrong name misinforms where no name merely omits.

Byte estimates use the dtype inferred during graph resolution, so dtype propagation and metrics share one source of truth. Per-tensor readouts include element count, bytes, boxes, exactness, approximation reasons, slice expressions, and propagation depth.


einsum FLOPs count the fused loop: `(operands - 1)` multiplies plus one add per contracted position, each term dropping out where it should. An outer product only multiplies, a reduction only adds, a transpose does neither, and the ordinary two-operand contraction comes out at the conventional `2K` per element.

Every figure in `AggregateReadout` is a `Figure` - `{ value, status, reasons }` - because a number alone cannot say which direction it is wrong in, and these are wrong in three different ways:

- `exact` is a count.
- `upper` is "no more than this". A byte figure measured over a widened region is one: a superset contributes bytes that are not really needed, and a region is never a subset of the truth, so the figure is never understated.
- `approximate` is a number that moved in an unknown direction. Only ratios are this. Widening a region raises an intensity's numerator *and* its denominator, so the quotient can land either side of the truth, and `≤` would claim a side it does not have.
- `unknown` has no number at all, and `value` is `null` rather than a partial total. A barrier contributes zero FLOPs while the operations around it contribute upper bounds, so their sum is neither a ceiling nor a floor. It used to be printed with `≤`, which was wrong rather than merely incomplete.

Statuses combine through `addFigures` and `ratioFigure`, which keep the weaker claim: a sum containing an unknown loses its number, and a ratio of an unknown quantity is not a looser ratio but no ratio. Each figure is qualified only by the regions it actually summed, so a widened weight no longer puts `≤` on an output-byte count that is exact. `exact` and `reasons` remain on the readout, derived from the figures rather than tracked beside them, for callers that want the one-line answer.

The inspector writes what each figure has earned - nothing on a count, `≤`, `~`, or the word `unknown` - and says beside it that only the arithmetic is missing while the byte figures still hold.

## 8a. Dependency notes

`src/core/notes.ts` turns a backward result into short statements about what constrains the cone - the contraction that must be staged, the axis a softmax needs whole, the halo a convolution re-reads. These answer the question the numbers do not: not how much, but why the footprint has this shape, and what it implies for tiling or fusing across it.

The text belongs to the operation, not the UI, because only the operation knows the reason. `OpSpec.dependencyNote` receives a `NoteCtx` - shapes, attributes, tensor names, the verified
symbolic dimensions, the propagated axis names, and the regions the current query
actually produced - and returns a draft or null. A note names an axis the way the source does,
preferring the axis's own name over the dimension it was declared with: `emb` says what the axis
*is* where `H*D` only says how wide it is, and the name is the one of the two that survives onto a
produced tensor. Without it a note about an intermediate falls back to a bare position or an
operation-internal einsum label - which is exactly where a reader most needs the word. Three rules govern it:

- **A note may only claim what the cone did.** A contraction note is withheld unless the region demonstrably spans the whole contracted axis, so a degenerate selection can never produce a false statement.
- **Notes carry a `key`.** Drafts sharing a key describe one constraint about different tensors and are merged rather than repeated, which stops a family of look-alike notes from crowding a distinct one past the display cap.
- **A note is never worth failing an analysis over.** A throwing `dependencyNote` is skipped at the presentation boundary; propagation, metrics, and every other finding remain available.

A draft also carries a `severity` and `flags` naming the tensors the constraint is visible on. Both are written where the note is - the operation - rather than derived in the view: only the operation knows how hard its constraint binds and which of its operands shows it. Severity is 3 for an axis consumed in full (a contraction, a reduction), 2 for an axis closed by a normalisation, 1 for ordering, halo re-reads, or strided access. There is no zero, because a constraint not worth stating is `null`.

Severity decides **what survives the display cap**, not what order notes are read in. The cap has to drop something, and dropping by graph order drops whichever the topological walk reached last - which can be the constraint that forces staging or accumulation. Selection is therefore by severity, then by nearness to the tile; the survivors are displayed in graph order, because that is the order the computation happens in. The inspector renders the severity weight and, when capped, the surviving and total counts. Flags are collected over every constraint found rather than the surviving ones, since a flag is true of its tensor whether or not its note had room.

`coneIsFullyElementwise` separates "every step fused" - a real finding - from "the cone reached no operation at all", so the two empty states can say different things.

`src/core/contribution.ts` answers the downstream half. Being in the forward cone means a tile *influences* a tensor, not that it produces it: a tile spanning part of a contracted axis reaches an output without determining a single element of it. The test runs backwards - what does that downstream region actually read? - and subtracts the tile; anything left is what the tile does not supply. The asymmetry is deliberate and matches the region contract: an over-approximated backward region can make the residue phantom, so the flag may over-warn, but an empty residue proves the true residue is empty, so it can never under-warn. Rows carry `exact` so an over-warning is shown as one, and the probe count is capped up front because each probe is a full propagation.

Shared demand across the drawn tiles is a separate derived analysis, `inputSharing` in `src/core/demand.ts`, rather than part of `computeMetrics`. Over the enabled tiles' backward cones it compares summed graph-input element demand with the distinct union of that demand, recording both how many tiles were selected and how many actually reach each input. It is `sharedDemand` with the tiles as readers, the computation a plan's boundary rows make with tasks, so both report the same figures and `view/demand.ts` words both rows the same way. Their difference is duplicate, potentially shareable demand - not a claim that the hardware reloads it or that a kernel can avoid all of it. This difference remains an upper bound when contributing regions were widened: at every element, widening can only increase how many tile regions contain it, and duplicate membership is `max(membership - 1, 0)`. The duplicate byte count therefore uses `≤` in that case, while its ratio to distinct demand uses `~` because both sides of the ratio moved.

## 8b. Tile plans

`src/core/plan/` divides a computation into tasks and reports what each task reads from the others.

A **tile family** (`tile-family.ts`) covers a tensor with boxes of one extent per axis. On an axis the extent does not divide, the last tile is shorter, and every element lies in exactly one tile. A family is separate from the canvas lattice in `view/tensor/tiling.ts`, which follows the display detail; a family changes only when the plan does. Coordinates are enumerated lazily in row-major order, which is a listing order and not an execution order. The tiles a box meets are a range of coordinates on each axis, so a lookup costs the number of tiles met.

A **plan** (`plan.ts`) maps produced tensors to tile extents and is checked against a resolved graph, which it keeps; every query on the plan runs against that graph. A graph input cannot be planned because no task computes it.

Tiling a tensor is what writes it to memory. A **task** computes one complete tile of one tiled tensor, including its whole reduction. It reads tiled tensors and graph inputs, and computes every untiled tensor between them and its tile itself, so an untiled intermediate is recomputed by each task that needs it. The plan's `frontier` is therefore every tiled tensor and every graph input, and a task's demand is the bounded cone from its tile at that frontier (§7). The cone's `crossings` give what it reads per operation and operand slot; its other produced tensors are what it computes. Leaving a tensor untiled is how a plan fuses it into its consumers, and a plan with every produced tensor tiled has one operation per task.

`supplyOf(plan, task)` in `interfaces.ts` joins each slot's demand with the family of the tensor it reads. A producer task is needed when the demand meets its tile. The intersection is the witness, recorded per slot. A tile met through two slots is listed once with two witnesses, so a repeated operand does not repeat a dependency. Each demand is classified by its supplier:

- `input`: a graph input, read from memory and produced by no task. A graph input read by two operations gives neither a producer.
- `tasks`: a tiled tensor, supplied by its family's tasks.

Since a task stops only at tiled tensors and graph inputs, every read has one of the two, and the producer list is always complete. The supply also lists `computes`, the untiled tensors the task computes with the region of each, and `flops`, the work of its tile and of `computes`. FLOPs come from `flopsOver` in `metrics.ts`, the same count the cone metrics use: it charges each (operation, output) pair over a region, each element once, and reports `unknown` through a barrier.

A producer found through exact demand is `definite`. A producer found only through over-approximated demand is possible, since the true demand may not meet that tile. No true producer is omitted, because an over-approximated region contains the true one. `used` is the number of distinct elements of the tile the consumer reads, as a figure that is `upper` when a witness is over-approximated.

`interfaceOf(plan, tensorId)` evaluates every task of a family and reports, per tensor the family reads, a `SharedDemand` (`core/demand.ts`) with the tasks as readers - the computation the Dependencies view's input sharing makes with the drawn tiles as readers:

- `summed`: bytes each task reads, summed over tasks. A task counts an element once however many slots read it.
- `distinct`: bytes in the union of all tasks' demand.
- `duplication`: `summed / distinct`, the mean number of tasks that read a demanded element. This is demand duplication. It is not a cache hit rate or a count of memory transfers.
- `duplicate`: `summed - distinct`, the bytes more than one task reads.
- `fanOut`, for a planned tensor: how many consumer tasks read each producer tile.

These use the figure statuses of §8, through the same `byteFigure` as the cone rows. `summed` and `distinct` are `upper` when a demand is over-approximated or a dtype was widened, and `duplication` is then `approximate`. `exact` records whether every demand on the tensor was exact, and `fanOut` counts only definite dependencies when it is.

Each family's report also carries a `Work`, and `planReport(plan)` gives one per family and one for the whole plan. It is the plan's cost in the figures two plans are compared by, and nothing beyond them:

- `tasks` and `dependencies`: task count, and producer tasks summed over tasks - the edges of the task graph. `upper` when some edge is only possible.
- `flops`: work summed over tasks. `recomputed`: the part of it that is the same work done by more than one task, `flops` less the FLOPs of the union of everything the tasks compute. An untiled intermediate that several tasks compute counts here, and so does a row statistic every tile across a normalised axis computes again, since the region-aware FLOP formulas charge a split tile for the whole row.
- `read`: bytes the tasks read, each task counted separately, so no reuse between tasks is assumed. `readDistinct`: the union, each element once. `written`: every tiled tensor, once.
- `intensity`: `flops / (read + written)`.

These are logical quantities of the declared plan under stated assumptions - one read of each
distinct element per task, no reuse between tasks, each tiled tensor written once - and the panel
says so beside them. They are not measured or predicted memory traffic. `recomputed` is relative to
the operations' own cost formulas (§8), not to a minimum over every algorithm for the same result.

The plan total measures `readDistinct` and `recomputed` over unions across families, so an intermediate that two families both compute is recomputed at the plan level although neither family recomputes it. `differenceFigure` keeps the direction a difference still has: an upper bound less an exact figure is an upper bound, and a difference whose subtrahend is inexact is only `approximate`. The report names the graph outputs the plan does not tile as `unwritten`, because a plan that never writes an output leaves that output's work out of its totals.

Each task costs one bounded query plus one intersection per producer tile it meets. A family with more tasks than the budget, 4096 by default, is declined with status `over-budget` and no figures, and the plan then has no total. A sum over part of a family or a plan is a lower bound on its total, and no figure in the app is reported as a lower bound. In the browser, the plan evaluation runs in the query Worker; only the finished report crosses back to React.

`familyReplay(plan, task)` in `replay.ts` walks the inspected task's family for the canvas, in
listing order - row-major over the grid, which is a listing and not an execution order. The family
is cut into at most `REPLAY_FRAMES` (48) contiguous runs of near-equal length (`replayRuns`), one
task per run when the family is no longer than that. Each frame is a run: its tiles as a region on
the family's tensor, laid out by the same row-major decomposition reshape uses, and what the bounded
cone of those tiles reaches on every other tensor (`regionDemand`). A cone relates each element on
its own, so a run's reach is the union of its tasks' reach. Beside it is the part the inspected
task's cone reaches too, which is the element-level picture of what the family's demand rows count:
the elements two tasks both read are their duplicate demand. Runs rather than a sample keep every
task in the replay and the trail on the canvas without gaps, while a long family still replays in
the time a short one does. Every frame is exact for the plan, or bounded with its cone's reasons.

A replay can follow the plan upstream, and deeper than 1 its step changes. At depth 1 a step is a family task and its frame stops at the plan's frontier, as a task does. At depth `d > 1` a step is a path of tasks (`steppedReplay`): a family task, one producer task it requires, one producer of that, down to `d - 1` levels, enumerated depth first in listing and operand order (`readOrder`), and stopping early at a task with no producers. Its frame is each task on the path, the elements of the other operands each is combined with inside the task it feeds, and what the last task reads one level further. A task's tile is carried forward through the feeding task's own computation - from the operations reading it across the frontier, through every fused operation its values pass - so in `D = relu(C); Y = D @ W` a C step shows the part of D it becomes and the W block that part meets, two operations later. Pairings come from `coaccessWithin` (`core/entangle.ts`): entanglement restricted to that task's outputs, taken box by box and joined, exact where the operation is `coaccessSeparable` - contractions and elementwise ops - and marked as a bound elsewhere. The whole producer tile is drawn on its own tensor; only the part the consumer uses is carried forward. Task cones are memoized, and a depth needing more than `STEP_QUERY_BUDGET` of them is declined with a message rather than computed. Each step is its own frame up to `STEP_FRAMES` (2048), far above the depth-1 cap of 48, because a run of steps paints their union - a whole row of producer tiles - and loses the granularity the depth exists for; past it, steps are cut into runs. Stepped playback paces frames down to 32 ms, against 84 ms at depth 1. A frame also records the tiles its steps walk below the family task (`walked`), and those are drawn as the family's are: a trail of the tiles walked before the active frame, each once, less the active frame's (`trailOf`), since several steps share one family task and stacking their fills made brightness count steps, and the active step's tiles at full strength. At depth 2 of `D2 = add(D, D)` the walk is along D's tiles, so the unit a step moves is what the card shows moving. Reach the inspected task does not share is outlined in the hue as well as filled, and an outline too thin to stroke fills its rectangle in the border's ink, so a level-2 read one row tall stays visible. The text below describes the level walk the lineage and the overlap use. Each further level takes the tiled tensors the last level read and follows the tasks that
produce them: those tasks compute whole tiles, so the next level is the bounded cone of the tiles
the demand meets (`tilesMet`), and a tensor reached by several levels keeps the nearest. A Y task
reading eight rows of C whose tasks compute sixteen-row tiles therefore reaches sixteen rows of
C's inputs - what the plan computes, not the element-level cone of what was read. Each tiled tensor
a frame reaches also carries its producer tiles. The overlap is measured against the inspected task
followed to the same depth. `replayDepthLimit` gives how many levels a family has: one more than the
deepest tiled tensor its tasks read through untiled intermediates. The depth is chosen per family
(`State.replayDepth`) and clamped to that limit. All limits are computed in one iterative
pass over the graph's topological order, with untiled tensors carrying ancestor levels and tiled
outputs adding one. The index is cached by plan in a `WeakMap`: first use costs O(nodes + input
edges), subsequent family and playback queries cost O(1), and retiling creates a new index.
Depth discovery never recurses or re-walks shared untiled prefixes.

Plan and task errors are `PlanError`s with stable codes: `PLAN_UNKNOWN_TENSOR`, `PLAN_GRAPH_INPUT`, `PLAN_TILE`, `PLAN_SIZE`, `PLAN_UNPLANNED` and `PLAN_TASK`.

## 8c. Read multiplicity

A footprint says which elements a tile needs and drops how often each one is used. For `C = A @ B`
and an `m x n` tile of `C`, the footprint of `A` is `m x K` and each of those elements is read by
`n` outputs. That count is the reuse inside the tile, and a region cannot hold it. A union records
whether an element is reached, not how many times.

The multiplicity of an operand element is the number of output elements of the operation that read
it: `|{o in S : a in deps(o)}|`, the relation `oracleDeps` states, counted instead of unioned. It
counts outputs, not terms. An output that meets an element in several terms of its own computation
reads it once, as `oracleDeps` lists it once. Counted this way it applies to every operation:
- a contraction reads an `A` element once per output column of the tile, and a `B` element once per
  output row;
- a softmax row is read once per selected output in it;
- a reduction reads each element once;
- a broadcast bias is read once per broadcast position;
- a convolution's halo is read once per window that covers it;
- a barrier is bounded by the box.

`core/tally.ts` holds the representation. A `Tally` is a list of boxes with positive counts, and
the value at an element is the **sum** of the counts of the pieces containing it. Pieces may overlap,
and overlap is how counts add: a convolution's halo is one shifted run per kernel tap, and the
positions two taps reach are the ones read twice. This inverts the rule in §4. There, summing box
volumes double-counts; here, summing `count * volume` gives exactly the total number of reads. It is
a separate type for the same reason `BoundedCone` is not a `PropResult`: `canonicalize` drops
contained boxes and merges neighbours, which preserves a union and would lose a count.
`measureTally` measures the distribution with the dimensional sweep `count` uses, adding up the
active counts where `count` only asks whether any box is active. It never builds the level sets it
measures.

Operations declare their count through `OpSpec.multiplicity(outSlot, outBox, inSlot)`. Most have
one of two forms:
- **The same count on every element `backward` names** (`uniformTally`): einsum and its sugar
  (the box's extent over the output labels the operand does not carry), elementwise and expand (the
  broadcast fan-out), softmax, normalize, conv's weights, and gather's indices.
- **A product of one count per axis** (`separableTally`): pad, conv and pool activations, gather's
  data, and cumsum. Each axis is a list of steps that add where they overlap.

An operation that reads each input element into at most one output element declares `readsOnce`
instead of writing a hook. These are reduce, transpose, slice, concat, split, reshape, and the
identity family.

Without either, `operandTally` falls back to the box's element count on `backward`'s region, marked
inexact. No element can be read by more outputs than the box holds. The barrier relies on this.

An inexact tally is an upper bound at every element, so its support contains the true support. This
is the region contract stated per element. It is where a widened `backward` region, a strided
enumeration past `stridedEnum`, a gather without index values, or a count past
`Limits.tallyPieces` lands. A separable count over that cap merges neighbouring steps at the larger
count. The cap is separate from `maxBoxes` and larger (4096 against 256), because a tally is
measured but never drawn or propagated, and a strided window's count alternates along each axis.
Scans build their triangular count directly as disjoint steps. Past the cap, they group the ramp
at each group's maximum count before allocating, so their work depends on the cap rather than
the number of selected positions.

`operandReads(graph, regions)` takes what is computed of each tensor, the same input `flopsOver`
takes: a backward cone's tensors under Dependencies, or a plan task's `Supply.work` - its tile and the
untiled tensors it computes - under Plan. For each operation computing one of those tensors it takes
that region as the outputs needed. It splits the region into disjoint boxes, since an output in two
boxes is still one reader, and counts every operand. A tensor read through several slots gives
one row whose counts add across them, so `matmul(A, A)` reads its shared corner from both bands.

The count is per operation, and that is deliberate. An operation's output region in a backward cone
is what the tile needs of it, so its operands' counts follow from the operation alone, exactly where
the regions are. Composing counts along a chain is a different question. Multiplying them along paths
counts a residual branch twice, and how often an intermediate is recomputed depends on what an
execution stores. Neither is answered here.

The figures keep the statuses of §8:
- **Exact tally:** figures within the safe integer range are exact. Overflowing measurements
  report `unknown`, as does a mean that depends on them; independently safe counts remain available.
- **Inexact tally:** these move only up, and are upper bounds:
  - reads, the sum of the counts;
  - elements read;
  - elements read more than once;
  - the largest count.
- **Inexact tally:** these move in no fixed direction, and are approximate:
  - elements read exactly once. An element with a true count of 1 can be raised to 2, and one with a
    true count of 0 can be raised to 1.
  - the mean, which is a ratio.

## 9. UI architecture

The React layer is a projection over the headless compiler and executor.

### Application state

`src/state/store.ts` is the single Zustand store. It holds construction only - the initial state and one spread per action slice in `state/actions/`; the pure derivations those actions share live beside it in `state/`. Its state is divided conceptually into:

- **Source:** DSL text, selected example, unresolved/resolved graph, and load errors.
- **Selection:** ordered parts, each naming its own tensor, and the independently enabled **Backward Cone** / **Forward Cone** views. Their controls live on the matching inspector section heads; both may collapse into the explicitly labelled figures-only state without disabling analysis. Direct canvas gestures add by default and Alt subtracts; both compose against the drawn tensor's parts only, so a gesture on one tensor can never edit or discard a part on another. Replacement is reserved for controlled internal transitions. Composition retains object identity for untouched parts, allowing their index-based UI metadata to be remapped safely if another tensor loses parts.
- **Attribution:** which part is focused (emphasised, one at a time) and which parts are enabled in the merged analysis (any number, sticky). Disabled parts retain a faint selection rectangle and an inspector row so they can be included again, but their cached propagation is excluded from directional rows, metrics, notes, contribution verdicts, segmented footprints, and graph dependency highlights. Disabling the focused part clears its focus. Metadata is cleared for edited parts and follows untouched parts by identity when their indices shift. Alongside these sits `analysisGroup`, the tensor whose tiles the readout describes; it is set by a draw, a pin, or a group header and never by hovering (§8).
- **Analysis:** aggregate backward/forward results, bounded per-box results, focus/pin state, and hover previews. **Both directions are always computed.** `direction` is a view filter over the analysis, not a gate on producing it: the inspector answers "what does this tile need" and "what does it feed" from one result, and a view choice must not decide whether a number exists. The filter is applied where highlights and directional rows are drawn, so switching it changes the picture without discarding the underlying analysis. `none` is the named figures-only combination: both section heads remain available to reopen either view while rows and graph highlights are hidden.
- **Plan:** `planTiles` holds the tile extents of each planned tensor and `planTask` the task being inspected. `plan` and `planSupply` are derived from them against the resolved graph and held in the store beside the cone results, so the cards and the panel read one answer. A plan names tensors and tile coordinates in one graph, so a graph replacement clears it exactly as it clears the selection. A plan edit appends a workspace history entry as a tile edit does, and every snapshot carries the plan, so one undo stack covers both. `keptPlans` holds up to two plans kept for comparison, each a tiling, the whole-plan `Work` it was evaluated to, and the graph outputs it does not write. The totals are captured when the plan is kept rather than recomputed, which is sound because the graph cannot change under them: `loadResolvedGraph` clears them with the plan. Clearing the current tiling leaves them, so a kept plan can be restored into an empty one. Keeping and dropping are not workspace edits; restoring a kept plan is an ordinary plan edit and undoes like one. A share link carries the tiling, the inspected task, the kept tilings and whether the Plan view was open (`planToLink`). It carries no figures. Restoring checks every tiling and the task against the compiled graph, and any one that does not fit refuses the whole link, as an invalid tile or offset does. Kept plans are then evaluated again (`restoredKeptPlans`) before anything is installed: alongside the inspected task in one compile-Worker restore transaction, or in place without a Worker. The graph is published only after these measurements succeed, so inspector query retries cannot cancel restoration. A build started in the meantime wins by compile epoch, and the restore installs nothing.
- **View:** per-tensor project/slice/unfold settings, tile scale, gesture snapping, the shape reading, the graph's px-per-element scale, metric options, graph focus, panel layout, node offsets, whether operation nodes answer a drag, and the transient family replay. `axisMode` chooses whether a compact card states semantic labels or numeric extents. `view/tensor/shape-label.ts` owns the label fallback - axis name, then verified symbolic extent, then number - while the tensor details keep all three readings separate (`emb`, `H*D`, and `512`), dropping any that a less symbolic reading below it already states, so the surviving row is named after what it actually is. A new workspace opens on labels; a shared link restores the reading and node offsets it was written against rather than being reinterpreted by a later default. Offsets are keyed by scene node key (`t:<tensor>`, `n:<node>`), so cards and operations move on the same terms and a link written before operations could move reads its bare tensor ids as `t:` keys. Whether operations answer a drag at all is a remembered workspace preference like the theme, not part of a link: what a link reproduces is the arrangement, not the state of someone else's canvas. Cards reserve the widest reading during base layout, so changing the mode cannot create overlap or move the graph. Selection edits, committed node moves, and resetting every node to generated placement share one chronological workspace history. The family replay is deliberately not an edit: it overlays the plan, holds its last run until cleared, then fades to reveal the plan unchanged.

Applying DSL recompiles the entire source into a new resolved graph. In a browser, parsing, lowering, resolution, graph-scale measurement, and immutable Dagre placement are one compile-Worker job. The result is structured-cloneable graph data plus its base layout; the main realm reattaches only the small `shapesOf` convenience function and commits the artifact atomically. Compilation and family/reuse analysis occupy separate latest-wins Worker lanes: a long query cannot queue in front of a build, and starting replacement work terminates the superseded Worker rather than merely ignoring its eventual answer. A successful build registers its graph in a fresh query Worker before subsequent analyses are posted. Requests carry monotonically increasing identities, and the store also keeps a compile epoch, so an older response cannot replace a newer edit. The synchronous action remains as a headless/test fallback when `Worker` is unavailable. Changing a selection runs one bidirectional query per part, which serves both the per-part attribution and, merged, the aggregate the panels read. Past the per-part cap attribution is dropped and the queries are grouped by tensor instead, bounding the cost by the number of tensors drawn on rather than the number of tiles; the grouped result can only be equal or coarser, never tighter and never an under-approximation.

Actions defined against one tensor's axes - arrow-key moves, hidden-axis sliders, the axis legend - follow the active analysis group. A focused tile narrows the action within that group; when an action requires one concrete tile and none is focused, the most recently drawn enabled tile in the group is the anchor. Those that *edit* the selection - the arrow moves, the removal key, and the include/exclude key - additionally require the inspector to be on Dependencies, the view whose list shows which tile they act on. Removal asks for more than the others: it acts only on a *focused* tile, hovered or pinned, never on the anchor fallback, because Escape drops the focus precisely to say the reader is done pointing at it and a destructive key must not outlive that; view toggles stay global, because the canvas shows what they did. Dependencies and Plan are an ordinary button group, not ARIA tabs: every button is a tab stop, Enter and Space activate them, and none claims the arrow navigation that a tab pattern would require. That choice preserves arrow keys for moving the tile even after the view switch has been clicked. An axis index means different things on tensors of different rank, so there is no single axis such an action could apply everywhere.

Expanding a composite node first resolves shape context, so composites fed by inferred intermediate tensors have their true rank available. The rewrite is serialized back to primitive DSL and compiled through the same Worker boundary, keeping the displayed source, share links, unresolved graph, and resolved graph synchronized. Selection state that could no longer be valid is cleared. Its undo entry retains the previous resolved graph and base layout, so undo restores them directly instead of recompiling and laying out on the UI thread.

### Main components

Replay frames paint over the plan without rerunning any query. Display
subtraction is cached by geometry and limited to 2,048 box pairs and 256
fragments; exceeding either budget keeps the full reach visible with an
approximation mark. The plan's figures are unaffected by this rendering
fallback.

- `App.tsx` composes the identity header, source panel, graph, and tile inspector. URL state crosses one asynchronous transactional store boundary: the DSL is compiled in the compile Worker and every selected part is validated before any source, graph, settings, or selection state is committed. Failure falls back to a built-in example without exposing a half-restored graph.
- `WorkspaceHeader.tsx` contains product identity, description, the shortcut-sheet entry point, and the global theme control.
- `SidePanel.tsx` owns the DSL draft, share action, examples, and operation list. An example with an `experiment` is built through the workspace restore path (`experimentWorkspace`): its exact source opens the Plan view with its tiles, task and kept tilings, evaluated as a shared link's are; `tests/state/experiments.test.ts` pins the figures its comments quote. Draft and last-built source are separate: graph-code edits retain a dimmed, inert snapshot; parameter-only edits keep the last successful graph interactive while resized source builds. `sameParameterizedSource` determines this distinction and caches the latest comparison. Inactive snapshots retain pan, zoom, and fit, but block node movement, selection, inspection, and workspace edits; menus close and live gestures cancel. Reverting the draft restores interaction. The native textarea has a synchronized, pointer-transparent syntax layer and accessible build status and diagnostics. `ParamControls` and editor hints use resolved `shapeParameters`, showing only parameters used by the built graph. Boxes belong to that build: graph-code changes hide them, and typing in the editor does not change their displayed values. Parameter edits rewrite only the named number through `view/params.ts`; stepper granularity is chosen at the start of a stepping run and held until that run ends. A size-only rebuild identified by `sameStructure` clamps card views with `carriedViewCfg` and preserves arrangement through `placementCarry`/`carriedOffsets`: node centres and viewport stay fixed, with collision correction only for nodes that grew. Selection, plan, and undo history reset because their coordinates belong to the old shapes. Rebuilding identical text is a fresh reload.
- `GraphView.tsx` consumes the Worker's base layout and manages highlighting, viewport, and node gestures; layout and routing algorithms remain in `view/graph/`. Background surfaces pan, while nodes and HUD controls retain their own gestures. All gestures require a primary press (`components/pointer.ts`); secondary presses open context menus. Under `select`, canvases draw tiles and operations select themselves. Under `move`, or Ctrl/Cmd for one gesture (`useMoveModifier`), node capture handlers start a drag before children receive the press, suppressing previews and popovers while retaining the card menu. Node borders change colour by state: accent for inspection, grey for movement, warning for blocked motion. Escape leaves move mode after any active drag is cancelled. Plan always uses select: move/reset, node-arrow movement, and the temporary modifier are disabled. The remembered tool returns in Dependencies. The world permits negative positions; reset layout is undoable and separate from viewport controls. Focused movable nodes own arrows and use the drag collision sweep, with one undo entry per press; global tile movement stands down over `[data-node-move]`. Connector SVGs put dim context behind plates and ordinary/hot links above them, including during dragging.
- The operations list carries a persistent `selectedOp`, distinct from the one-shot `focusNode` glide request. It is set from either end - a row click, or any selection edit, via `operationForTensor` (a tensor's producer, else its single consumer, else nothing) - and cleared by Escape, a press outside the graph, or a click on empty canvas. Panning, zooming and fit keep it, so a selected operation survives the reader moving the view to look around it. The outside-press listener leaves the canvas background and its controls to `GraphView`, which clears only when a background press is released without travelling further than a click.
- `TensorCard.tsx` renders canvas grids and converts pointer gestures into element-space boxes. Pure helpers build its paint: `buildLayers` for cones, `buildPlanPaint` for tasks, and `buildReplayPaint` for family replay (§8b). Replay leaves a faint trail only on the family's tensor; elsewhere it shows only the active run's reads/computations, emphasising overlap with the inspected task and outlining needed producer tiles. Upstream levels fade progressively. The plan recedes during playback and returns as the overlay fades; no lattice is added and neither selection nor plan is replaced. A finished replay holds its last run, paused; play/pause, previous/next and a scrubber stop on any run, and **clear** or a change of view, task, or plan retires the trail together over 240 ms. Reduced motion shows the full trail at once, held for stepping by hand. Hover probes compute both directions under `MAX_PREVIEW_NODES`; changing the view filter does not re-query. Per-tensor subscriptions avoid unrelated repainting, movement within one logical cell is ignored, and `useFrameThrottle` coalesces probes to one per frame. Clears use the same throttle so a pending move cannot resurrect a retired preview.
- `Inspector.tsx` presents tile identity and the tile list, followed by Backward/Forward Cone, approximation, cost, sharing, operand reuse, and dependency notes. Each tensor appears once per direction with box count, segmented attribution, slice expressions, and copy action. Directional headings are also view toggles and paint legends; Forward Cone summarises completed/partial contributions. Operand reuse lists operations by their output tensors, avoiding ambiguous repeated names, and reports reused bytes, largest count of uses per element, and bytes used once. Uniformly single-use operands share one line. A reused row's figures switch the reuse lens (`reuseLens`, `view/reuse-lens.ts`): the operand's card replaces its cone paint with that row's count per element. The card derives the paint itself (`useLensPaint`) from the same scope rules as the row (`lensRegions`), and draws it from `tallyLevels` - the tally split into disjoint pieces with summed counts - in a dedicated opaque pass, lowest count first, so a projected cell shows the largest count among its hidden indices. It bypasses the layer fills, which composite by alpha and coarsen past a box budget. Past 16,384 pieces nothing is drawn and the legend says so. Inexact counts are hatched. Nothing composites over the counts (`composeCardLayers`): the card's tiles and the gesture in progress keep only outlines and corner marks, and a replay is not drawn on that card. The shaded row stays in the list, with its switch and legend, even when its count falls to one. The lens is cleared by Escape, by leaving its view, and by a graph replacement; a scope change redraws it. Its heading states the active group/focus scope; empty states distinguish graph inputs from operations that read nothing for the selected tile. The panel keeps canvas settings on the HUD and editing in the tile list. With no selection it explains the three relations and offers a starter tile.

- **Dependencies** reports quantities derived from the graph and drawn region: cones, co-access, footprints, FLOPs/bytes, duplicate input demand, operand reuse, and notes. **Plan** reports quantities derived from a declared tiling. Neither assumes an unstated execution: written tensors and recomputed intermediates are explicit plan choices. Drawn-tile sharing belongs to Dependencies; whole-family sharing belongs to Plan, reached through "plan with this tile". The inspector view is store-owned and survives redraws; when tiles exist their identity header remains above the switch.
- `shortcuts.ts` is the single manifest for key matching and help copy. `useKeyboard.ts` dispatches global bindings, local owners such as the source editor match against the same manifest, and `ShortcutsDialog.tsx` renders its grouped inventory. The shortcut sheet traps focus, makes the workspace inert, restores the control that opened it, and suspends every global binding except Escape while open.
- `useFocusPolicy.ts` preserves keyboard ownership after gestures that prevent the browser's normal focus change. Pressing outside the focused element releases it; pointer-used buttons, sliders, and separators release focus at the end of the press, while Tab-reached controls retain focus and a visible ring. Text fields, menus, list boxes, and dialogs retain their own focus behaviour. Programmatic restoration follows the same policy: menus and the example picker refocus triggers for keyboard use, and panel collapse moves focus only if it was inside that panel.
- `inspector-analysis.ts` derives the inspector's scoped metrics, notes, rows, selection seeds, and contribution report with independent memoization. It re-merges the cached per-tile propagation using only enabled tiles, so a visibility toggle synchronizes every readout without rerunning the executor. Contribution classification can launch many checked backward probes, so it runs only while downstream rows are visible; note findings are collected once and shared by the row flags and notes section.
- `PlanPanel.tsx` presents tiled tensors and family work, whole-plan `Work`, the inspected task's FLOPs, reads/producers, computed intermediates, operand reuse, and family demand (§8b–8c). It is available without a drawn tile. **fuse** beside a tiled intermediate fuses it into consumers, and reads **stop writing** on a graph output; **materialize** beside a computed intermediate writes it back to memory. **keep** adds comparison columns, oldest first and current last; up to two kept plans can be restored or dropped. Under each kept plan, `describePlanChange` (`view/plan-work.ts`) states what the current tiling changes from it (fuses, materializes, retiles, starts or stops writing) and which rows are lower and higher, comparing only figures exact on both sides. Percentage differences use `ratioFigure` and are approximate when either value is bounded. A "not written" row exposes omitted graph outputs across every compared plan. Reads are grouped by tensor in operand order (`readOrder`: the task's own slots, then fused reads nearest first), graph inputs badged, with computing-output labels distinguishing repeated operation kinds. Following a producer (`followPlanTask`) keeps `planTrail`, read through `activePlanTrail` only while its `to` is the inspected task object itself - identity, not equal coordinates - so any other way of reaching a task, even the same one, retires it for good, while an undo restores the object and the trail with it; the trail renders as crumbs and Alt+← steps back. **Upstream, to the inputs** is `taskLineage` (`core/plan/replay.ts`): the replay's level walk for one task at the plan's full depth, giving per upstream tensor the nearest level and its producer tasks, input region, or fused region; pointing at a row sets `lineageFocus`, which outlines those tiles on the card, solid where reached through exact demand and dashed where only possible. The walk keeps the exact part of each reach apart from its widened union, so a row merging an exact path with a widened one still names its definite tasks. Producer sets are exact unions of tile-coordinate boxes with separate definite subsets for widened demands; detail rows use arithmetic paging and report the full count. Small sets get individual outlines, large sets tile-aligned range outlines. Interactive inspection uses `TaskSupply`; `supplyOf` explicitly materializes the full join for headless callers. Escape releases the task without removing tilings, and undoing a plan edit brings Plan forward. Plans of at most 256 tasks evaluate automatically; larger ones evaluate on request. Both paths use one latest-wins query-Worker job. The dependency matrix appears only when both families have at most 64 tiles.
- In Plan, `buildPlanPaint` draws the inspected tile, its reads, and its computations in solid needs paint, hatched when widened. Only tiled tensors have a lattice. Needed producer tiles use neutral outlines, dashed for possible dependencies; their unfilled portions show what the consumer does not read. The graph highlights every tensor the task reads or computes. `planGesture` divides an untiled produced tensor on its first gesture and thereafter inspects the task under the press. Retiling requires explicit extents or clearing the tiling. Plan drags always snap; on unfolded cards they include the hidden-axis spans previewed across panels. A first click uses the display tile on visible axes and one element on other axes unless the tensor owns a tile, and also tiles the produced operands, initially giving one operation per task. Clearing an operand's tiling then fuses it. Untiled cards show that proposed cover faintly. Display detail seeds a plan but never changes it afterward; arrows step by plan tiles.
- `PanelFrame.tsx` owns side-panel width, the mouse- and keyboard-adjustable drag strip, and collapse-to-rail for both panels. Collapsing unmounts transient panel UI while store-owned drafts and analysis state survive; when focus was inside the panel it moves to the replacement rail or collapse control instead of falling onto the document, and when it was elsewhere - typing in the other panel when `Alt + 1` collapses this one, or nowhere after a click - it is left where it was. The family replay is the one piece of store state that does not survive it: the timers that advance and retire the overlay live in the inspector, so leaving it behind would freeze the frames on every card with nothing able to clear them.
- `reuse-lens.ts` owns the lens's violet ramp, validated as an ordinal ramp against the card surfaces, and a scale from zero to the largest count. Violet is the one hue with no other meaning on a card.
- `palette.ts` owns the six categorical hues, their recorded CVD/contrast validation, and the canvas surfaces they were validated against. A tile takes a colour slot when it is created (`SelPart.hue`, the lowest slot no tile holds, else the least used) and keeps it while it lives, so removing a tile does not repaint the others; slots travel in share links and workspace files. Two rules live there: identity is never carried by colour alone, and chrome must sit outside the data hues, because hue on a card means *which selection box* and a hot graph edge must not be mistakable for a cone.
- `share.ts` owns link encoding and decoding. Decoding rejects unknown settings, malformed parts, offsets, and plan fields as one transaction; absent legacy fields receive documented defaults. Links without plan fields retain pre-plan behaviour. The store then validates graph-dependent bounds, tensor identities, and tilings before installation.
- `graph-scene.ts` owns the two-stage graph rendering pipeline. In the browser's first stage, the compile Worker combines a resolved graph with the pure card-measurement policy and produces the immutable Dagre base layout. The main realm combines that base with persistent node offsets to produce the cheap live scene and routed connectors. Query results, highlighting, and viewport state cannot invalidate structural placement. Links are rebuilt from ordered operation operands rather than Dagre's endpoint-keyed edge set, so repeated operands such as `matmul(X, X)` remain distinct.
- `graph-geometry.ts` contains the lower-level pure geometry for collision-safe motion and connector curves; large pointer jumps cannot tunnel through another node, but there is no artificial world wall. A neighbour a node already overlaps neither stops nor vetoes its motion, so a move can end an overlap but never start one. A relayout of the same graph - a card's footprint changing - carries the drawn placement across with `carriedOffsets`, as a size-only run does, so a grown card is nudged clear instead of laid over its neighbour. Connectors sharing a pair fan their anchors and stagger their flow marks to remain readable. A connector ends on a card's grid, not its box (`anchorRect`): `cardSize` returns the grid's rect inside the card (`port`), centred under the header on the bottom edge, and it travels with the node through layout and resized-card sizes. The box is as wide as the header label and as tall as the header rows, so ending there left a vector's connectors short of its grid. `.tensor-card` is a column filling its slot with the grid at the bottom, so the drawn card matches that rect. Collision and placement still use the whole box.
- `useDragGuard.ts` suppresses text selection for the duration of any drag. It is a hook rather than part of `setDragging` so that store actions stay free of DOM access and remain testable headlessly.

Fit may go below the manual legibility floor to show the whole scene. It centres both axes with at least 20 screen pixels of margin. `lowZoomBound` uses the lower of the legibility floor and fitted scale, independent of the current gesture, so manual zoom can return to the fitted overview but cannot go farther out.

The left panel's width defines the canvas origin. A `useLayoutEffect` compensates its delta in the same commit, before paint, keeping the world stationary when that panel changes. `ResizeObserver` handles window-driven refits of an untouched overview; panel changes do not refit. Measuring the origin correction there would happen after paint and visibly shift the graph for a frame.

Selecting an operation row centres the operator, not the tensor it writes: the row names the operation, while the starter tile it also sets still lands on the first output because that is where a cone begins. `focusNode` therefore carries a node kind, and is a one-shot request the viewport clears once it has acted, so asking twice acts twice. The viewport slides to the target over `GLIDE_MS` with an ease-out rather than jumping, because a view that changes without showing the movement leaves the reader to work out what moved and which way. Any deliberate gesture - wheel, zoom button, pan, or fit - cancels an animation still in flight, and a reduced-motion preference skips it. Hover coordinate tooltips counter-scale to their normal screen size, and `⊞` distinguishes the lattice span from tensor shape.

Below 75% zoom, compact headers hide shape, tile span, and role. `overview-labels.ts` checks a 10px name against neighboring node footprints and previously placed names before allowing it to borrow space above its own header. Width is bounded by the projected card and 180 screen pixels; long names ellipsize, with the full name and shape details still available on the name. If that space is occupied, the ordinary smaller name remains. This pass changes presentation only: zoom never reruns Dagre or changes the canvas position.

Operation labels use the same placement pass, trying the node, the gap above, then below; no label is drawn if all three are occupied. Tensor names take priority. The solver is memoized on `paintScale(tf.k)`, avoiding work on every wheel event. Counter-scaling stops at `LABEL_FLOOR_SCALE` (20%), below which labels shrink with the graph; text below `MIN_LEGIBLE_PX` is omitted. Operand labels (`arg0`, `arg1`) remain in world coordinates and follow the same legibility floor. Hover readouts and name popovers use the floored counter-scale too.

### Tensor rendering and tiling

Tensor cards use a row-major projection by default:

- the last tensor axis is horizontal;
- the second-last axis is vertical;
- higher axes can be projected, sliced, or unfolded;
- rank-zero and rank-one tensors use corresponding reduced layouts.

`ViewCfg.axes` chooses the drawn plane through `cardViewMenu`: right-click the card, or use Menu/Shift+F10 on its focused name. Rows and columns must be distinct; assigning an axis already in the other role swaps them (`axesWith`). The default pair is stored as absent, and a remapped card labels its pair. This changes presentation only: stored element-space selections and graph semantics remain unchanged. `setViewAxes` derives the tensor's anchor tile (`viewAnchorIndex`: its focused tile, else its last drawn tile) and reveals its newly hidden positions.

The menu also selects how hidden axes are read:

- **Project:** each cell represents the union over hidden axes. Their coverage is accumulated before layer compositing, since disjoint N-D regions can overlap in the plane.
- **Slice:** sliders choose one index per hidden axis. Up to five blank deck sheets indicate slices still to come in row-major order. `deal.ts` animates the previous canvas snapshot forward or backward, respects reduced motion, and releases temporary rasters on completion, cancellation, or culling.
- **Unfold:** every hidden index gets a panel. `fold.ts` nests panels in stride order with wider gaps between outer groups: all axes across (`→`), all down (`↓`), or the fastest across and the rest down (`▦`, for at least two hidden axes). Painting and hit-testing share this geometry. Validation rejects more than 256 panels and directions whose minimum panels/gaps exceed the size budget. An unfold that no longer fits after a shape or plane change falls back to slice.

`cardShapeView` extracts footprint-affecting settings for the layout Worker: plane, hidden mode, and presence of projected tile-position controls. Only these rerun placement; position edits and tile extents that retain the same controls do not. The previous layout stays usable while the latest request runs. `cardScaleFor` shrinks remapped or unfolded planes to fit actual rounded panels and gaps. A card's header is its name and its shape, symbolic or numeric, and its footprint is as wide as the longest reading of the shape. Its right-click menu states the rest: the grid it draws (`grid 4×8`, or the plan tile and task count in the Plan view) and its ratio to the graph's scale (`display scale ÷4`, `×1.5`), since a card off that scale has lengths that do not compare with other cards.

**Resize card** in that menu selects Move and exposes top-right and bottom-right handles for one tensor. Dragging or using arrows on a focused handle changes `ViewCfg.scale`, preserving canvas proportions, tensor dimensions, gesture tile extents, and analysis. Manual size overrides feed `buildGraphScene` directly, so connectors reroute without Dagre; the opposite corner stays anchored. A resize makes its own room: each frame places every other node again from where the gesture began (`roomForNode`), moving the ones the card covers aside, so neighbours follow the card's size both ways and return as it shrinks back. A card is always drawn at its stored scale. Placement never reserves room for a stored scale, so a later layout can leave a resized card overlapping a neighbour; the card's menu then offers **make room for this size**. Scales are stored to three decimals. Each completed gesture is one undo step covering size and every node it moved. Escape cancels an active resize, putting every node back, and exits its mode; clicking elsewhere also exits, leaving Move selected. Mode is transient, while size travels in autosave, files, and share links. The Plan view retains its fixed-layout policy.

Fixed canvases cannot enter resize mode; a scale stored before a view became fixed is kept for when it can apply again, but the menu does not state it. Live geometry updates size and positions atomically through `setCardGeometry`, and cancellation through `restoreCardGeometry`. The menu's **set display size** opens a validated percentage editor (100% is the automatic fit), committing once on Apply. **Make room for this size**, offered while the card overlaps another node, keeps the chosen card anchored and clears overlapping nodes as one undoable arrangement; its viewport fits the result. Both actions respect the Plan view's fixed-layout policy.

One canvas cell represents one **tile**, not necessarily one element. `src/view/tensor/tiling.ts` gives each axis its own display tile (`axisTile`): about `20 / 2^tileScale` tiles across it, the tile a power of two, at most half the axis. It depends on the axis's extent and the setting only. An axis of one length is therefore gridded alike on every card - `seq = 128` takes one tile on Q and on the scores, so a snapped click selects the same rows on both - and no edit elsewhere in the graph re-grids a tensor. Tiles are rectangular where the drawn axes differ in length. Card dimensions remain stable while detail changes.

The tile is chosen without regard to pixels. An earlier rule coarsened tiles until a cell filled 5 canvas px, which tied every tile to `graphPx` and so to the largest tensor: editing `K` re-gridded tensors without a `K`. Density is now the renderer's concern alone (`latticeStride`), and the card outlines the box a click would take under the pointer, so snapping stays legible when drawn lines thin out.

The detail control reads as what it asks of every axis - `≈20 per axis`, `≈40 per axis` (`tileScaleLabel`) - which is one statement for the whole graph; each card's header prints the tile it settled on, rows × columns. `tileScale` remains the serialized power-of-two shift, so share links keep parsing. `effectiveTileScaleStops` removes adjacent settings that tile every axis the graph draws identically, keeping the representative nearest the default. The leftmost **None** stop is a semantic exception: `axisTile` returns 1 unconditionally, so it means one logical tile per element for snapping, hover, starter tiles, and keyboard movement even when a large card cannot display every boundary. None and the default remain separate stops when both happen to tile by one element, and `effectiveTileScaleIndex` preserves the exact stored intent before considering equivalent legacy plateaus.

A tensor's own tile (`ViewCfg.tile`, the axis table's "grid") replaces the display tile on every axis. An extent shorter than a tile clips it rather than pretending the tensor has elements it does not.

`ViewCfg.tile` stores a tensor's own per-axis tile and travels in share links (`view/tensor/tile-spec.ts`). The inspector's `AxisEditor` accepts extents or takes them from a drawn selection through "use as card grid"; the UI calls these extents the card grid, keeping "tile" for the selection and "task" for the plan. These extents replace the visible lattice, snapped gesture spans, arrow steps, starter tile, and first plan division. In slice/project modes a hidden axis takes the tile containing its stored position; projected axes narrower than the tensor get independent numeric tile-position controls, while sliders remain exclusive to slice. `[`/`]` step the first positioned hidden axis. Unfolded gestures snap to tile extents using their panel coordinates.

Setting a tile is a gesture setting, not an undo step or stored-box edit; the axis editor may explicitly refit the inspected selection as one edit. Rows show axis extent, grid extent, selected range, and shortened tails, followed by the full selection. They shade hidden axes without choosing the plane and omit coordinates that would be meaningless off-lattice. `axisTableMenu` holds "use as card grid" and "reset card grid". "plan with this tile" is a button beside the Axes heading (`planWithTile`): it divides the tensor and inspects the task containing the selection in the Plan view.

Menu builders in `view` return actions, choices, and separators (`view/menu.ts`). Actions execute and close; choices apply immediately and stay open for multi-part settings. Inapplicable entries remain disabled with an explanatory tooltip. `OptionsMenu.tsx` binds handlers by entry id and opens at the pointer, or below the focused owner for Menu/Shift+F10; owners with room also expose a ⋯ button. Body portals use fixed coordinates to avoid panel clipping and graph scaling. The menu owns arrows and contains pointer, wheel, key, and context-menu events, preventing React portal bubbling from starting graph gestures.

Shared links preserve per-tensor project/slice/unfold settings, chosen axes, tile extents, and positions; legacy links default to projection. Restoration validates identity, rank, bounds, and unfold limits transactionally. View changes clear the hover label/preview and discard queued probes for the previous view.

`selectionBoxFromDrag` defines selection and hover consistently. Without an owned tile, projection selects the full hidden-axis extents and slicing selects the slider indices. An unfolded drag names hidden coordinates through its endpoint panels and selects the N-D box between its corners; snapped gestures expand every axis to tile boundaries. Hover readouts and cone previews use the same single-cell gesture, ensuring the following click selects what they describe.

**Snapping (`snapToGrid`) is a property of future gestures and movement, not stored selection geometry or analysis.** A drag is tracked in elements throughout and expanded to whole current-grid tiles only when it commits. The inspector's text range is an explicit element-space edit and is never rounded, whether snap is on or off. Toggling snap or changing detail leaves every existing box and the workspace undo depth untouched. A keyboard nudge uses the current tile while snapping and one element when not. If the box starts off the current lattice - after a typed edit or grid change - its first snapped nudge aligns its leading edge in the requested direction without changing its extent; later nudges advance by whole tiles. A boundary clamp may use a shorter final delta so the whole box lands flush without shrinking. Changing detail changes the next snapped gesture and nudge, never the current box.

Along a row-major stride-one axis, `nudgeSelection` carries through valid tile origins: columns finish before rows, and rows before hidden panels. Alt+arrow grows the keyboard tile's right or bottom edge one grid step, or shrinks it back (`growSelection`, `view/growth.ts`). The box before the first step is kept as a baseline in `growth`, and each step stores the difference between the grown tile's backward cone and the baseline's per tensor; `activeGrowth` reads it only while the keyboard tile is still the part object the last step installed, so a move, edit, undo or release retires it for good with nothing to clear - moving the tile away and back does not revive it. The cards draw the added elements over the cones, and the inspector's Growth section lists them, tensors with nothing new included. Alt+arrow is claimed throughout the workspace, as it is Back and Forward in some browsers. The arrows move one tile, the keyboard's tile (`keyboardTile`): a tile hovered in the list, else `editBox`, the tile just drawn or clicked in the list, which a press outside its card and the inspector releases (`useEditTileRelease`, and a background click in `GraphView`). With none, nothing moves; no group moves as a whole. `editBox` is separate from the pin, so selecting a tile to move it leaves the analysis merged; the card outlines it at the emphasis weight. `movement.ts` moves through valid origins as a rigid frame, includes flush-edge and off-grid alignment stops, and preserves extents. Other axes retain clamped translation. `core/coordinates.ts` shares mixed-radix stepping with Plan task movement; BigInt ordinals prevent lost steps on huge tensors. Both directions stop at the outer domain ends. Slice positions follow a carried tile or task, including on undo; unfolded cards need no position update.

Every typed value in the inspector - a tile's range, a tensor's tile extents, a plan's extents - is a `DraftField`: typing changes a draft, Enter or leaving the field applies it, and Escape abandons it and leaves the field. A draft that does not parse is marked and kept for correction. Only Enter and Escape stop at the field, so the panel shortcuts reach the app from inside it, as they do from the source editor. Every edit records its undo entry through `recordWorkspace`, which snapshots the selection, the node offsets and the plan together, so no edit can record a snapshot that leaves part of the workspace out.

Text edits are normal selection transactions: they repropagate immediately and add one workspace undo entry. An edited pinned tile retains its pin. An edited disabled tile retains its disabled state and its new cached propagation remains excluded from merged analysis until the tile is enabled again. Snap and detail are view/gesture settings rather than workspace geometry, so undoing a later movement restores the box without reverting those settings.

Card *size* is set by a single px-per-element figure for the whole graph. `graphScale` derives it from every tensor's drawn plane - the largest tensor sets the budget, the smallest non-degenerate side raises the scale if that budget would make it invisible - and the store holds the result as `graphPx`, computed once per resolved graph in `loadResolvedGraph`. This is what makes a shared dimension render at one length everywhere it appears, so a matmul's contraction axis is visually comparable across its operands. `graphPx` is deliberately not recomputed for view state. Degenerate axes (extent 1) are excluded from the scale and drawn at a fixed width, because they have no length to preserve and would otherwise pin the whole graph at the per-element cap.

`src/view/tensor/grid.ts` is the pure geometry and `src/components/card/draw-grid.ts` the canvas drawing that consumes it. Regions are painted at **element** precision: `regionRects` maps each box straight to canvas pixels, so a region that ends mid-tile draws a crisp edge rather than a partially shaded cell. The tile lattice is drawn over the top as a reading aid and does not quantise the marks. A thin region is widened to a one-screen-pixel minimum (limited by the whole canvas), expanding inward at the far edge to retain the floor, which over-states extent rather than letting it disappear - the same conservative direction the region algebra takes. Perimeter emphasis is suppressed when its stroke would be wider than the mark, so a one-element point or line never grows false area. Stroke widths are measured in screen pixels, and so is the lattice threshold: `grid.MIN_LATTICE_PX` is a screen-space stride threshold and the only density control, since the tile itself ignores pixels. `latticeStride` draws every 2^m-th boundary, with `m` the smallest power of two clearing the threshold, so every drawn line is a snapping boundary and the lattice coarsens instead of disappearing. The converse does not hold: a snapped edge can land between two drawn lines. The snap unit stays independent of the viewport, so the same drag selects the same range at any zoom, and the hover outline shows the box a click takes. Canvas backing resolution follows DPR and coarse zoom buckets, and unchanged backing stores are reused; CSS uses ordinary interpolation rather than re-pixelating that supersampled result.

`src/view/tensor/raster.ts` bounds raster storage independently of element-space geometry. A card receives at most 4 × 1024² pixels (16 MiB of RGBA storage), and cards near the viewport share a 16 × 1024² pixel budget (64 MiB), with an 8192-pixel side ceiling. Allocation also respects requested DPR and zoom resolution. Cards beyond a small screen-space prepaint margin release their backing stores and skip painting; their DOM and CSS footprints remain available for keyboard navigation, gestures, and placement. These limits cover backing-store pixels after effects settle, not browser or GPU overhead. Oversized cards trade raster sharpness for bounded memory without changing selections or counts.

Committed selection parts additionally carry short corner marks outside their filled rectangles, drawn after cone paint and the lattice. Their gap, length and stroke weight use screen pixels; a surface-coloured under-stroke keeps them distinct from nearby ink. These marks identify the seed without enlarging its fill, retain the part’s hue and focus/hidden alpha, and are clipped at the canvas boundary. A seed occupying the whole canvas therefore relies on the existing selected-card outline where external corners cannot fit.

The canvas assigns one perceptual channel to each fact. Hue identifies the selected tile. Uniform fill means **Backward Cone**; a `\` ruling at a CSS-pixel pitch means **Forward Cone**; a dot stipple means **What it is combined with**; the opposite dashed `/` hatch means a conservative over-approximation.

Entanglement uses a stipple to remain distinct from cone rulings and approximation hatches. `buildLayers` splits it against the tile's painted needs fill; covered portions use card-surface dots (`knockout`) so same-hue overlap remains visible. `patternFloorFor` and `patternFitsRect` share each pattern's visibility floor with the renderer. Below that floor stipples become dotted hairline delimiters, never the solid fill that already means needs.

Rulings are diagonal to avoid beating against the axis-aligned lattice. The downstream slope rotates through twelve slots matching the attribution cap, avoiding the hatch's 45° and near-horizontal/vertical angles. Different slopes expose overlapping cones even when tiles past the sixth share a hue. Approximation uses a broken opposite-slope stroke: surface-coloured over solid upstream paint and layer-coloured over downstream rulings. The header's `≈` covers marks too small to hatch.

Rulings are stroked directly at full raster resolution, with phase anchored to the canvas origin so adjacent rectangles share a continuous pattern. This permits arbitrary angles without bitmap-tile seams. Spacing carries density at constant line weight; current ruling density stays fixed pending an honest supplied-share measure.

Persistent cone alpha represents hidden-axis coverage, never graph depth or recency. Exact depth is shown by `dN`. Two transient overlays may modulate alpha: hover probes by depth, because they have no inspector row, and family replay by recency as the active run sits above its trail and then retires. Neither changes committed analysis.

`paintScale` counteracts the CSS transform in 32 logarithmic steps per octave, rounding down to preserve minimum mark size. Below a pattern's visibility floor, or beyond the renderer's safety cap, rulings and hatches use a quieter solid; stipples use their dotted delimiter. Density degrades before direction. Ruled rectangles also receive a hairline delimiter, restoring the precise region edge where the last stroke stops short; solid fills need none.

When supplied-share density is supported, named perceptual steps will encode it and inspector rows will carry the exact value or conservative bound beside the contribution verdict. Density remains reserved for that quantity rather than reused for focus or depth.

## 10. End-to-end interaction flow

For a typical edit-and-select interaction:

1. The user edits DSL in the side panel and runs it.
2. The compile Worker parses the text, creates source mappings, resolves the graph, and runs Dagre without waiting behind family or reuse work.
3. The store accepts the response only if it is still current, then atomically replaces the graph and base layout or retains compilation diagnostics for display.
4. Tensor cards derive their grids from resolved shapes while `GraphView` projects the returned base layout through the current per-node offsets.
5. A pointer gesture creates or composes an ordered selection box.
6. The store submits the canonical selection to `SymbolicExecutor`.
7. Operation specifications map the region through the graph in the requested direction(s).
8. The active inspector view derives its analyses: Dependencies needs merged cost, operand reads, sharing, notes, and contribution probes; Plan uses its own model. Hidden views skip these derivations.
9. Tensor canvases and the inspector render aggregate and per-box results, filtered by the current direction.

The core jobs remain deterministic and usable synchronously in headless code. Browser compilation includes the default structural layout in its Worker lane; shared-workspace restoration also evaluates the current task and kept plans there before installation; card footprint changes use an independent layout lane; whole-family and reuse analysis share a separately cancellable query lane. Layout and queries cannot cancel one another or block a build behind stale work. Each lane owns its graph cache, which is cleared on worker replacement and successful compilation. Selection propagation and visible inspector measurements still run synchronously. React components do not implement operation semantics; they only create queries and render results.

## 11. Testing strategy

The suite verifies both local contracts and the relations they form across the system:

- **Region algebra:** canonicalization preserves the point set and operand bands; disjointification partitions the same set; counts measure unions once; conservative caps never omit true elements. Shared-operand cases such as `matmul(A, A)` check reporting and costs end to end.
- **Compiler and registry:** shapes, dtypes, attributes, ranks, arity, and executor boundaries reject malformed input. Diagnostics collect independent failures in source order without downstream cascades and use the narrowest reliable span. Dtype promotion obeys the join laws; sugar and printing round-trip to a fixpoint. Every registered op must have a fixture checked against its oracle and the forward/backward adjointness law.
- **Dependency semantics:** pointwise `oracleDeps` independently checks exact propagation for equality and inexact propagation for containment. Lowered limits force fallback branches on small tensors, with assertions proving those branches ran. Random graphs exercise transitive cones and frontiers that actually shorten walks. `oracleTerms` checks co-access separately from the broader forward/backward fallback.
- **Figures and findings:** sums retain the weakest status, unknown work loses its number, and ratios of bounds stay approximate. Barriers retain all declared dependencies and output dtypes. Notes claim only proven constraints, merge by key, retain severe findings under the cap, and use source axis names. Contribution may over-warn but never under-warn.
- **Plans and multiplicity:** independent task oracles walk through untiled tensors and stop at tiled tensors/inputs, checking slot demand, computed regions, complete producer sets, and family/plan totals. Widened demand may add possible producers but never omit true ones or call uncertain ones definite. Tile families cover each element once, including tails. Read tallies match pointwise output-reader counts, with capped tallies never below truth; hook-free fallback paths are checked too. Fixed two-matmul chains exercise fusion, sharing, misaligned tilings, and recomputation.
- **State and interaction:** selection composition, active groups, focus/enablement, plan edits, undo, transactional links, and invalid payload rejection. UI tests cover pointer-task mapping, plan retile/fuse behaviour, view-dependent footprints, projection/slice/unfold controls, cross-panel commits, and slice-raster cleanup. Frame coalescing keeps the latest call and trailing clears.
- **Rendering and replay:** element-precise marks, distinct needs/feeds/co-access textures, approximation hatching, and focus fading. Replay runs cover every task once; each run's reach equals the union of its tasks' bounded cones. Trails remain only on the family's tensor and retire together. Geometry, raster limits, layout workers, and layering rules are checked independently of a DOM rasterizer.

`tests/corpus/` owns the oracle, fixtures, randomized graph checks, and layering laws. `docs/UI_CONTRACTS.md` supplies the manual scenarios for behaviour that depends on pixels, timing, and gestures.

Pure implementation seams such as reshape decomposition, diagonal mapping, layer construction, and grid rectangles carry `@internal` markers when direct invariant tests are stronger than a large graph or React test. These are not embedding APIs; helpers that merely duplicate production behaviour are not retained.

Normal verification:

```bash
npm test
npm run build
```

`npm run build` includes `tsc --noEmit`; that command can also be run separately for a typecheck.

## 12. How to extend the system

### Add an operation

1. Implement an `OpSpec` under `src/core/ops/`.
2. Define its attribute schema and static or dependent arity.
3. Implement shape and dtype inference.
3a. Map axis names where an output axis is genuinely the input's own axis, and leave it
    unnamed otherwise. A borrowed name is worse than none: the inspector prints it as the
    source's own word for that axis.
4. Implement conservative backward and forward region mappings.
5. Add a pointwise oracle where practical and define FLOP semantics.
5a. State how many outputs read each operand element (`multiplicity`), or declare `readsOnce` when
    each input element reaches at most one output element. Without either, the operation's reads
    are bounded by its box and marked inexact. The multiplicity corpus checks whichever applies
    against the oracle.
6. Register the spec in `src/core/ops/index.ts`.
7. Add an entry to `tests/corpus/op-fixtures.ts`. This is not optional: `registry.test.ts` checks the
   table against `listOps()`, so a registered operation without a fixture fails the suite. The
   fixture is then run against the oracle and against the adjointness law automatically.
8. Add exact cases, edge cases, and malformed-input cases of its own beyond the fixture.

No executor or UI dispatch changes should be necessary unless the operation requires a genuinely new visualization concept.

### Add DSL syntax

Prefer lowering syntax sugar into an existing canonical operation. Syntax belongs in `parser.ts` and the tree it builds; the mapping from a call name to an operation and its attributes belongs in `lower.ts`. Keeping that split is what lets the parser stay ignorant of the operation registry, and it is where the printer's inverse mapping should be checked against. If syntax introduces new semantics, first add those semantics to the operation registry, then update lowering, serialization, and diagnostic source mapping.

A new lexical rule - a character an identifier may contain, a new escape - goes in `parse/lexical.ts`, never in the parser or the highlighter alone. Both derive from it so that colours cannot drift from the grammar.

### Add a visualization feature

Keep shape/index logic in the core or pure UI geometry helpers. The store should orchestrate queries, while React components should remain consumers of resolved graphs and propagation results.

## 13. Current boundaries and deliberate limitations

- TileCone models dependencies and costs, not tensor values, numeric stability, or runtime scheduling.
- Regions are unions of axis-aligned boxes that may overlap; highly fragmented mappings may conservatively collapse at the box cap. Because normalization no longer splits, the cap is reached later than it used to be, but a box count is no longer an upper bound on how many disjoint pieces a region has.
- Parsed operation outputs use an empty shape as an unresolved placeholder before graph resolution. Consumers should not execute or render that intermediate form.
- Family replay covers every task in at most 48 contiguous runs in listing order. These frames do not affect plan cost or demand figures, which are evaluated separately.
- Past a cap, a region is coarsened by merging neighbours into their hulls rather than collapsed to one bounding box, so the shape of the set survives. What remains abrupt is only the transition to `exact: false`.
- Coarsening recovers an asymptotic loss, not a constant one. A diagonal's fallback was quadratic in the selection and is now linear; a strided access is already linear and stays within its stride factor, which no box decomposition improves without one box per element.
- `disjointify` degrades the same way, after a work budget rather than only a fragment cap: splitting is exponential in the rank, so a crossing family could spend seconds approaching the cap. A rank-4 crossing family still reaches the bounding box, marked.
- einsum FLOPs are the cost of the equation as written, as one fused contraction. A pairwise decomposition of a three-operand contraction costs less, and writing it as separate einsums is what reports that.
- The partial-contribution flag can over-warn on an over-approximated region and never under-warns; past its probe cap, downstream rows are simply unflagged.
- Per-part attribution is intentionally capped to keep interaction responsive; aggregate propagation remains complete.
- Expanding a composite can expose intermediate traffic, so dependency and FLOP semantics may remain equivalent while displayed intermediate-byte estimates change.
- A barrier contributes zero FLOPs, so a FLOP total spanning one reports `unknown` rather than a number. What is lost is the arithmetic only: byte figures through a barrier remain upper bounds, because its output shapes are declared.
- A bounded cone has no metrics, notes, or contribution report; those are defined over transitive cones. A plan counts FLOPs over each task's own regions with `flopsOver`, and otherwise reports demand, duplication, fan-out and the `Work` figures. Staged reads, partial work, working sets and live storage are not modelled.
- A task computes its whole reduction, and its fusion is decided per tensor: an untiled tensor is recomputed by every task that reads it. Split reductions, a tensor that is both written and recomputed, worker placement and execution order do not exist yet. A tiled view (a transpose, say) is charged as a written copy.
- A plan is not part of the DSL. A share link carries it (§9), but a payload too long for a URL is copied as raw JSON, and nothing in the app reads that back yet.
- Plans are compared on whole-plan `Work` only. A per-family or per-tensor comparison, and any ranking of plans, is left to the reader.
- Read multiplicity is counted per operation, over the outputs the cone needs of it. Counts are not composed along a chain, and the number of times an intermediate would be recomputed is not modelled. Term counts, which would charge an output once per term that meets an element, are not reported.
- JSON graph support exists below the UI, but the main interactive authoring path is the DSL.

These boundaries are useful when deciding where new work belongs: semantic truth should live in the graph, region, operation, and executor layers; orchestration belongs in the store; presentation belongs in React and the canvas helpers.
