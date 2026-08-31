# TileCone System Architecture

This document describes TileCone as it is implemented today. It is an engineering map of the system: where source text becomes an executable graph, how tensor regions propagate through operations, and how the React UI turns those results into an interactive visualization.

For product usage and DSL examples, see `README.md`. This document focuses on boundaries, invariants, data flow, and extension points. `docs/README.md` records which supporting documents are current, historical, local-only, or disposable.

## 1. Purpose and execution model

TileCone is a symbolic tensor-dependency explorer. Given a tensor graph and a selected region of one tensor, it answers two questions:

- **Upstream:** which input and intermediate elements can influence this region?
- **Downstream:** which later elements can be influenced by this region?

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
│   ├── executor.ts       checked public query boundary
│   ├── propagate.ts      generic forward/backward worklist
│   ├── region.ts         exact and conservative region algebra
│   ├── metrics.ts        FLOP, byte, and intensity estimates
│   ├── reuse.ts          seeded dependency-reuse estimator
│   ├── expand.ts         composite-to-primitive graph rewrites
│   ├── notes.ts          plain-language dependency constraints
│   ├── shapes.ts         symbolic dimensions and shape errors
│   ├── dtypes.ts         canonical dtypes and byte widths
│   └── ops/              operation semantics and registry
├── parse/
│   ├── dsl.ts            DSL parser and serializer
│   ├── compiler.ts       parse + resolve facade and diagnostics
│   ├── source.ts         source spans and source maps
│   └── json.ts           JSON graph import/export
├── ui/
│   ├── store.ts          application state and analysis orchestration
│   ├── GraphView.tsx     graph rendering, gestures, highlighting, and viewport
│   ├── TensorCard.tsx    interactive tensor canvas
│   ├── Inspector.tsx     tile grid, selection parts, footprint, cost, notes
│   ├── SidePanel.tsx     source, cone direction, examples, and operations
│   ├── PanelFrame.tsx    side-panel width, drag strip, collapse-to-rail
│   ├── WorkspaceHeader.tsx  product identity and theme control
│   ├── grid.ts           pure grid geometry and drawing
│   ├── graph-geometry.ts collision constraints and connector routing
│   ├── graph-scene.ts    structural layout and live routed-scene projection
│   ├── tiling.ts         tile-size policy and slider stops
│   ├── palette.ts        validated categorical hues and canvas surfaces
│   ├── share.ts          workspace link encoding and decoding
│   ├── selection-range.ts  range syntax for editable selection parts
│   ├── useKeyboard.ts    global key bindings
│   └── useDragGuard.ts   text-selection suppression during drags
├── examples/             built-in DSL examples
├── test/                 unit, property-style, and integration tests
├── App.tsx               UI composition and URL-state bootstrap
└── main.tsx              entry point; applies the stored theme before paint
```

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

1. Parameters are positive safe integers.
2. Node and tensor IDs are unique and all references exist.
3. Every operation exists in the registry.
4. Static and attribute-dependent arity constraints hold.
5. Attributes pass the operation's schema.
6. Tensor producers are unique and the node graph is acyclic.
7. Symbolic dimensions resolve to concrete shapes.
8. Each operation infers the expected number of output shapes and dtypes.
9. All inferred extents are non-negative safe integers and all dtypes are supported.
10. Consumer indexes, producer links, and a stable topological order are built.

Resolution is referentially transparent: it operates on a clone rather than mutating the caller's graph. The resulting `ResolvedGraph` is the only graph form accepted by symbolic execution.

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

An under-approximation is therefore a correctness bug; an over-approximation is permitted only when marked `exact: false` with a reason.

`src/core/region.ts` owns region algebra and normalization. Canonicalization removes empty boxes, disjointifies overlap, merges compatible boxes, and keeps results deterministic. If fragmentation exceeds the configured box cap, it falls back to a bounding box and records the loss of precision.

The UI preserves the user's selection as **ordered parts** so individual boxes can be focused, colored, moved, or subtracted. Before execution, those parts are canonicalized into a mathematical set. This distinction prevents presentation identity from leaking into dependency semantics.

Each part carries **its own tensor**, so parts may span several tensors at once. Comparing what two tensors pull from a shared input is a primary use of the tool, and it is impossible if drawing on one discards the tile on the other. The executor is unaffected: a cone is still defined from a single root, and multiplicity lives in the store, which runs one propagation per part and merges the results per tensor (`mergeProps`). Merging takes the union of regions and the *shortest* hop to any root, so a tensor reachable from two selections reports the distance it actually has. Because union never under-approximates, a merged result is a valid dependency claim whenever its inputs were.

## 5. DSL and compiler

The compiler boundary is `compileDSL` / `tryCompileDSL` in `src/parse/compiler.ts`.

```text
text
 └─ parseDSLWithSource
      ├─ Graph                 unresolved IR
      └─ SourceMap             declarations and nodes → source spans
          └─ resolveGraph
               └─ ResolvedGraph
                    └─ SymbolicExecutor
```

`compileDSL` returns the source, unresolved graph, resolved graph, source map, and an executor as one coherent artifact. `tryCompileDSL` is the diagnostic form used when callers need structured parse or semantic failures instead of an exception.

### DSL surface

The DSL is line-oriented and intentionally small:

- `input`, `weight`, and `param` declare typed tensors with symbolic or literal dimensions.
- Assignments create operation nodes and one or more output tensors.
- Positional arguments refer to tensors; named arguments become operation attributes.
- Lists, numbers, booleans, strings, and identifiers are accepted attribute values.
- `#` introduces a comment.

The parser normalizes convenient spellings into a smaller internal operation set. For example, `relu(...)` becomes an `elementwise` node with `fn: "relu"`, reduction names become `reduce` nodes, and layer/RMS normalization become `normalize` nodes with explicit attributes.

The source map connects compiler errors back to declarations or operation calls. Parsing errors are reported as DSL diagnostics; graph, shape, dtype, and arity failures become semantic diagnostics with the most specific available source span.

`src/parse/json.ts` provides a second, programmatic graph frontend. It validates the serialized structure, while deep semantic validation remains the resolver's responsibility. JSON import is currently a core capability rather than a primary UI workflow.

## 6. Operation registry

Every operation is described by an `OpSpec` in `src/core/ops/types.ts` and registered in `src/core/ops/index.ts`. An operation owns all semantics specific to that operator:

- attribute schema;
- static input and output cardinality;
- optional attribute-dependent arity validation;
- output shape inference;
- output dtype inference;
- backward region mapping from an output box to input regions;
- forward region mapping from an input box to output regions;
- optional pointwise dependency oracle used by tests;
- FLOP estimation, optionally over a whole region when work is shared or nonlocal;
- an optional plain-language note about the constraint it imposes on a cone (§8a).

This is the main extensibility seam. The graph resolver and propagation engine do not switch on operation names; they dispatch through the registry.

Operations currently cover LLM-oriented primitives such as matrix multiplication, linear layers, elementwise functions, reductions, normalization, softmax, reshape/transpose, indexing, gather, concatenation/splitting, expansion, scans, and layout operations. Convolution and pooling are supported but treated as secondary/experimental surfaces.

Some high-level operations are expandable. `src/core/expand.ts` rewrites `softmax` and supported normalization forms into primitive subgraphs, preserves the externally visible output tensor, then resolves the rewritten graph again. Expansion is a graph transformation, not a special execution mode.

## 7. Symbolic execution

`SymbolicExecutor` in `src/core/executor.ts` is the checked public API. It validates that the selected tensor exists and that every region interval has the correct rank, integer bounds, and in-bounds coordinates. It then delegates to the propagation engine.

`src/core/propagate.ts` implements one generic traversal with two directions:

- **Backward:** visit nodes in reverse topological order and map affected output regions to their input regions.
- **Forward:** visit nodes in topological order and map affected input regions to their output regions.

For each node, the engine calls the registered operation once per relevant source box, accumulates contributions for each destination tensor, and canonicalizes the union. It also carries approximation reasons and records the shortest propagation depth used by the visualization.

Queries may request upstream, downstream, or both directions. Low-level propagation functions remain available for operation development and testing, but UI and external callers should normally use the executor so malformed selections fail at the boundary.

## 8. Metrics

Metrics are derived from the backward dependency result in `src/core/metrics.ts`:

- **FLOPs:** estimated from affected operation outputs. Operators may provide box-local or whole-region cost functions.
- **Input bytes:** bytes read from leaf tensors.
- **Intermediate bytes:** bytes associated with contributing produced tensors, optionally included in arithmetic intensity.
- **Output bytes:** bytes in the selected producer output.
- **Arithmetic intensity:** FLOPs divided by the configured byte denominator.

Byte estimates use the dtype inferred during graph resolution, so dtype propagation and metrics share one source of truth. Per-tensor readouts include element count, bytes, boxes, exactness, approximation reasons, slice expressions, and propagation depth.

## 8a. Dependency notes

`src/core/notes.ts` turns a backward result into short statements about what constrains the cone — the contraction that must be staged, the axis a softmax needs whole, the halo a convolution re-reads. These answer the question the numbers do not: not how much, but why the footprint has this shape, and what it implies for tiling or fusing across it.

The text belongs to the operation, not the UI, because only the operation knows the reason. `OpSpec.dependencyNote` receives a `NoteCtx` — shapes, attributes, tensor names, the declared (possibly symbolic) dimensions, and the regions the current query actually produced — and returns a draft or null. Two rules govern it:

- **A note may only claim what the cone did.** A contraction note is withheld unless the region demonstrably spans the whole contracted axis, so a degenerate selection can never produce a false statement.
- **Notes carry a `key`.** Drafts sharing a key describe one constraint about different tensors and are merged rather than repeated, which stops a family of look-alike notes from crowding a distinct one past the display cap.

`coneIsFullyElementwise` separates "every step fused" — a real finding — from "the cone reached no operation at all", so the two empty states can say different things.

Reuse is a separate derived analysis in `src/core/reuse.ts`, rather than part of `computeMetrics`. It takes one anchored selection, deterministically samples same-sized tiles across that tensor with a fixed seed, and uses checked backward queries to estimate how many tiles touch each input region in the anchor's footprint. Keeping it pure and seeded makes repeated estimates reproducible without conflating a sampled statistic with the exact cost metrics.

## 9. UI architecture

The React layer is a projection over the headless compiler and executor.

### Application state

`src/ui/store.ts` is the single Zustand store. Its state is divided conceptually into:

- **Source:** DSL text, selected example, unresolved/resolved graph, and load errors.
- **Selection:** ordered parts, each naming its own tensor, and the independently enabled upstream/downstream cones. Direct canvas gestures add by default and Alt subtracts; both compose against the drawn tensor's parts only, so a gesture on one tensor can never edit or discard a part on another. Replacement is reserved for controlled internal transitions. Composition retains object identity for untouched parts, allowing their index-based UI metadata to be remapped safely if another tensor loses parts.
- **Attribution:** which part is focused (emphasised, one at a time) and which parts are hidden (excluded from painting, any number). These are separate axes: focus is transient and drives emphasis, visibility is sticky and drives whether a cone contributes paint. Neither affects metrics, so a hidden part still reports its footprint. Metadata is cleared for edited parts and follows untouched parts by identity when their indices shift.
- **Analysis:** aggregate backward/forward results, bounded per-box results, focus/pin state, and hover previews.
- **View:** per-tensor projection settings, tile scale, gesture snapping, the graph's px-per-element scale, metric options, graph focus, panel layout, and tensor offsets. Selection edits and committed tensor moves share one chronological workspace history.

Applying DSL recompiles the entire source into a new resolved graph. Changing a selection runs one query per part, which serves both the per-part attribution and, merged, the aggregate the panels read. Past the per-part cap attribution is dropped and the queries are grouped by tensor instead, bounding the cost by the number of tensors drawn on rather than the number of tiles; the grouped result can only be equal or coarser, never tighter and never an under-approximation.

Actions defined against one tensor's axes — arrow-key moves, hidden-axis sliders, the reuse estimate, the axis legend — follow the **anchor**: the focused part's tensor, else the last one drawn on. An axis index means different things on tensors of different rank, so there is no single axis such an action could apply everywhere.

Expanding a composite node rewrites the unresolved graph, resolves the result, and clears selection state that could no longer be valid.

### Main components

- `App.tsx` composes the identity header, source panel, graph, and tile inspector. URL state crosses one transactional store boundary: the DSL is compiled and every selected part is validated before any source, graph, settings, or selection state is committed. Failure falls back to a built-in example without exposing a half-restored graph.
- `WorkspaceHeader.tsx` contains only product identity, description, and the global theme control.
- `SidePanel.tsx` owns the editable DSL draft, independent upstream/downstream toggles, built-in examples, and operation list. Running the draft updates the store only after compilation succeeds.
- `GraphView.tsx` is the graph controller and React projection: it derives highlighting, manages pan/zoom/focus and tensor gestures, and renders the current scene. It does not own layout or routing algorithms.
- `TensorCard.tsx` renders an interactive tensor grid on canvas and converts pointer gestures into element-space boxes. Its paint stack is built by a pure `buildLayers`, so the encoding rules — which cone is filled and which outlined, what fades under focus, what a hidden box still shows — are asserted directly instead of inferred from pixels.
- `Inspector.tsx` owns global tile-grid detail, editable selection parts, exactness, tensor slices, metrics, and the reuse estimate.
- `PanelFrame.tsx` owns side-panel width, the drag strip, and collapse-to-rail for both panels.
- `palette.ts` owns the categorical hues, their recorded CVD/contrast validation, and the canvas surfaces they were validated against. Two rules live there: identity is never carried by colour alone, and chrome must sit outside the data hues, because hue on a card means *which selection box* and a hot graph edge must not be mistakable for a cone.
- `share.ts` holds link encoding and decoding together. They were split across a toolbar and `App` once, and when the toolbar was removed the encoder went with it, leaving the app able to restore links nothing could produce. Decoding is all-or-nothing: an unknown setting or any malformed part rejects the payload, while genuinely absent legacy fields receive their documented defaults. The store then validates graph-dependent selection bounds transactionally.
- `graph-scene.ts` owns the two-stage graph rendering pipeline. A resolved graph plus a tensor-measurement policy produces the immutable Dagre base layout; that base plus persistent tensor offsets produces the cheap live scene and routed connectors. Query results, highlighting, and viewport state cannot invalidate structural placement. Links are rebuilt from ordered operation operands rather than Dagre's endpoint-keyed edge set, so repeated operands such as `matmul(X, X)` remain distinct.
- `graph-geometry.ts` contains the lower-level pure geometry for collision-safe motion and connector curves; large pointer jumps cannot tunnel through another node. Connectors sharing a pair fan their anchors and stagger their flow marks to remain readable.
- `useDragGuard.ts` suppresses text selection for the duration of any drag. It is a hook rather than part of `setDragging` so that store actions stay free of DOM access and remain testable headlessly.

### Tensor rendering and tiling

Tensor cards use a fixed row-major projection:

- the last tensor axis is horizontal;
- the second-last axis is vertical;
- higher axes are controlled by slice or projection settings;
- rank-zero and rank-one tensors use corresponding reduced layouts.

One canvas cell represents one **tile**, not necessarily one element. `src/ui/tiling.ts` chooses a shape-aware base tile and applies the global detail scale. Card dimensions remain stable while detail changes.

The detail control is global but its outcome is per tensor, and two things follow from that. `effectiveTileScaleStops` offers only the scales that actually change the graph-wide lattice, because snapping tiles to powers of two inside a bounded card leaves roughly half the raw range inert. The leftmost stop is always the explicit "no tiling" request, one cell per element, and `settledTiles` reports what the graph landed on beside it — a single size when every tensor agrees, a spread when the fit rule coarsens some and not others. That readout is what keeps the request honest: on a graph whose widest tensor cannot draw a cell per element, "none" settles on the same lattice as its neighbour, and displaying the settled size is what makes that visible rather than leaving the stop silently inert.

**Snapping (`snapToGrid`) is a property of the gesture, not of the analysis.** A drag is tracked in elements throughout and expanded to whole tiles only when the box commits, so turning snapping off costs no precision that was ever available — regions have always been element-precise, and it was only the gesture that rounded. A keyboard nudge steps by the same unit the pointer works in (`nudgeUnit`): one tile while snapping, one element when not, so the keyboard cannot place a box at offsets a drag cannot reach.

Card *size* is set by a single px-per-element figure for the whole graph. `graphScale` derives it from every tensor's drawn plane — the largest tensor sets the budget, the smallest non-degenerate side raises the scale if that budget would make it invisible — and the store holds the result as `graphPx`, computed once per resolved graph in `loadResolvedGraph`. This is what makes a shared dimension render at one length everywhere it appears, so a matmul's contraction axis is visually comparable across its operands. `graphPx` is deliberately not recomputed for view state. Degenerate axes (extent 1) are excluded from the scale and drawn at a fixed width, because they have no length to preserve and would otherwise pin the whole graph at the per-element cap.

`src/ui/grid.ts` is a pure geometry and drawing layer. Regions are painted at **element** precision: `regionRects` maps each box straight to canvas pixels, so a region that ends mid-tile draws a crisp edge rather than a partially shaded cell. The tile lattice is drawn over the top as a reading aid and does not quantise the marks. A thin region is widened to a one-pixel minimum, which over-states extent rather than letting it disappear — the same conservative direction the region algebra takes.

The one genuinely fractional quantity is hidden-axis coverage: in projection mode a box covering part of an axis that is not on screen really does represent a fraction of what a drawn cell stands for, so that survives as an alpha multiplier. It cannot be expressed geometrically, because the axis it concerns is not drawn. Individual selection boxes retain stable hues, upstream and downstream cones are independently toggled in the left panel, and inexact regions are hatched rather than displayed as exact.

## 10. End-to-end interaction flow

For a typical edit-and-select interaction:

1. The user edits DSL in the side panel and runs it.
2. The compiler parses the text, creates source mappings, and resolves the graph.
3. The store atomically replaces the graph or retains a compilation error for display.
4. Dagre lays out the resolved graph and tensor cards derive their grids from resolved shapes.
5. A pointer gesture creates or composes an ordered selection box.
6. The store submits the canonical selection to `SymbolicExecutor`.
7. Operation specifications map the region through the graph in the requested direction(s).
8. Metrics are calculated from the upstream result.
9. Tensor canvases and the inspector render aggregate and per-box results.

This flow keeps parsing and execution synchronous and deterministic. React components do not implement operation semantics; they only create queries and render results.

## 11. Testing strategy

The test suite checks the architecture at several levels:

- region algebra and canonicalization;
- shape, dtype, attribute, rank, and arity validation;
- operation-specific forward and backward mappings;
- compiler diagnostics and source spans;
- executor boundary validation;
- composite expansion equivalence;
- store behavior, tiling geometry, and slider-stop policy;
- dependency notes: that a note claims only what the cone did, names axes as the source does, and merges look-alikes rather than crowding out distinct ones;
- canvas encoding: filled versus outlined per direction, focus fading rather than hiding, and regions drawn at element precision rather than rounded to the lattice;
- workspace link round-trips, including refusal of malformed payloads;
- randomized graph and propagation cases.

Some tests deliberately import pure implementation seams: reshape decomposition, the einsum
diagonal mapper, canvas layer construction, grid rectangles, and the exhaustive region point
enumerator. Those exports carry an `@internal` marker. They are not application or embedding APIs;
they exist because testing the invariant directly is materially stronger than inferring it through
React or a large graph. Helpers that merely duplicated production behavior are not retained for
tests.

For operations with a pointwise `oracleDeps`, the test harness can enumerate small tensors and compare analytic propagation against brute-force dependencies. Exact results must equal the oracle; inexact results must contain it. This directly enforces the core no-under-approximation rule.

The normal verification commands are:

```bash
npm test
npm run build
npx tsc --noEmit
```

## 12. How to extend the system

### Add an operation

1. Implement an `OpSpec` under `src/core/ops/`.
2. Define its attribute schema and static or dependent arity.
3. Implement shape and dtype inference.
4. Implement conservative backward and forward region mappings.
5. Add a pointwise oracle where practical and define FLOP semantics.
6. Register the spec in `src/core/ops/index.ts`.
7. Add exact cases, edge cases, malformed-input cases, and oracle comparisons.

No executor or UI dispatch changes should be necessary unless the operation requires a genuinely new visualization concept.

### Add DSL syntax

Prefer lowering syntax sugar into an existing canonical operation. If syntax introduces new semantics, first add those semantics to the operation registry, then update parsing/serialization and diagnostic source mapping.

### Add a visualization feature

Keep shape/index logic in the core or pure UI geometry helpers. The store should orchestrate queries, while React components should remain consumers of resolved graphs and propagation results.

## 13. Current boundaries and deliberate limitations

- TileCone models dependencies and costs, not tensor values, numeric stability, or runtime scheduling.
- Regions are unions of axis-aligned boxes; highly fragmented mappings may conservatively collapse at the box cap.
- Parsed operation outputs use an empty shape as an unresolved placeholder before graph resolution. Consumers should not execute or render that intermediate form.
- The DSL compiler exposes structured diagnostics, while parts of the current UI reduce them to a display string.
- Reuse estimation is deterministic for a given seed but sampled; exact FLOP/byte metrics remain a separate contract.
- Per-part attribution is intentionally capped to keep interaction responsive; aggregate propagation remains complete.
- Expanding a composite can expose intermediate traffic, so dependency and FLOP semantics may remain equivalent while displayed intermediate-byte estimates change.
- JSON graph support exists below the UI, but the main interactive authoring path is the DSL.

These boundaries are useful when deciding where new work belongs: semantic truth should live in the graph, region, operation, and executor layers; orchestration belongs in the store; presentation belongs in React and the canvas helpers.
