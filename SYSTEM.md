# TileCone System Architecture

This document describes TileCone as it is implemented today. It is an engineering map of the system: where source text becomes an executable graph, how tensor regions propagate through operations, and how the React UI turns those results into an interactive visualization.

For product usage and DSL examples, see `README.md`. This document focuses on boundaries, invariants, data flow, and extension points. `docs/README.md` records which supporting documents are current, historical, local-only, or disposable.

## 1. Purpose and execution model

TileCone is a symbolic tensor-dependency explorer. Given a tensor graph and a selected region of one tensor, it answers two questions:

- **What it needs** (upstream): which input and intermediate elements can influence this region?
- **What it feeds** (downstream): which later elements can be influenced by this region?

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
│   ├── contribution.ts   whether a tile completes what it feeds, or only adds to it
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
│   ├── inspector-analysis.ts  memoized inspector derivation and costly-probe gating
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
│   ├── ShortcutsDialog.tsx  grouped inventory of global bindings
│   ├── clipboard.ts      copy helper with success/failure feedback
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
   relationship does not actually hold.
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

The downstream contribution verdict follows the same one-sided guarantee: **partial may
over-warn, never under-warn**. Approximation can invent missing residue and therefore a partial
warning, but it cannot make a genuinely partial contribution appear complete.

An under-approximation is therefore a correctness bug; an over-approximation is permitted only when marked `exact: false` with a reason.

`src/core/region.ts` owns region algebra and normalization. Canonicalization removes empty boxes, disjointifies overlap, merges compatible boxes, and keeps results deterministic. If fragmentation exceeds the configured box cap, it falls back to a bounding box and records the loss of precision.

Set difference preserves the same contract. When the subtrahend is inexact, subtracting its represented superset could remove true elements, so the result conservatively retains the minuend and records `inexact subtraction`. Proof-oriented consumers such as full-axis detection therefore decline conclusions drawn only from an inexact bound.

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
- An axis may be labelled in place: `input X [batch: B, seq: S, emb: E] f16`. A declaration
  names every axis or none, because a half-named declaration is a slip rather than a
  statement; propagated names may legitimately have holes.
- Assignments create operation nodes and one or more output tensors.
- Positional arguments refer to tensors; named arguments become operation attributes.
- A dimension is an expression over parameters, not just a name: `input X [B, S, H*D]`, and
  the same language works in shape-valued attributes such as `reshape(X, shape=[B, S, H, E/H])`.
  `Sym` remains `string | number` — the string simply holds the expression, a bare parameter
  being the one-symbol case — so nothing downstream had to learn a new shape type and the
  text the author wrote is what `notes.ts` and `toDSL` see.
- Lists, numbers, booleans, strings, and identifiers are accepted attribute values. Numbers
  accept decimals and scientific notation, so `toDSL` cannot emit a literal the parser then
  rejects.
- `#` introduces a comment.

The parser normalizes convenient spellings into a smaller internal operation set. For example, `relu(...)` becomes an `elementwise` node with `fn: "relu"`, reduction names become `reduce` nodes, and layer/RMS normalization become `normalize` nodes with explicit attributes.

Both statement forms are closed at the end: an assignment and a declaration each refuse
trailing input, so `input X [4, 8] 32` is a syntax error rather than an f32 tensor whose
width silently sizes every byte estimate.

The source map connects compiler errors back to declarations or operation calls. Parsing errors are reported as DSL diagnostics; graph, shape, dtype, and arity failures become semantic diagnostics with the most specific available source span.

`src/parse/json.ts` provides a second, programmatic graph frontend. It validates the serialized structure, while deep semantic validation remains the resolver's responsibility. JSON import is currently a core capability rather than a primary UI workflow.

## 6. Operation registry

Every operation is described by an `OpSpec` in `src/core/ops/types.ts` and registered in `src/core/ops/index.ts`. An operation owns all semantics specific to that operator:

- attribute schema;
- static input and output cardinality;
- optional attribute-dependent arity validation;
- output shape inference;
- optional axis-name propagation, defaulting to unnamed;
- optional symbolic-extent propagation, defaulting to literal;
- output dtype inference;
- backward region mapping from an output box to input regions;
- forward region mapping from an input box to output regions;
- optional pointwise dependency oracle used by tests;
- FLOP estimation, optionally over a whole region when work is shared or nonlocal;
- an optional plain-language note about the constraint it imposes on a cone (§8a).

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

## 8. Metrics

Metrics are derived from the backward dependency result in `src/core/metrics.ts`:

- **FLOPs:** estimated from affected operation outputs. Operators may provide box-local or whole-region cost functions.
- **Input bytes:** bytes read from leaf tensors.
- **Intermediate bytes:** bytes associated with contributing produced tensors, optionally included in arithmetic intensity.
- **Output bytes:** bytes in the selected producer output.
- **Arithmetic intensity:** FLOPs divided by the configured byte denominator.

Symbolic extents travel that path too, and are a **separate** propagation from axis names —
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
and a reshape names only the axes a group maps one-to-one — splitting an embedding into heads
produces axes the source never named. `inferAxisNames` is optional and defaults to unnamed,
because the inspector and the notes layer present a name as the source's own word for an axis,
so a wrong name misinforms where no name merely omits.

Byte estimates use the dtype inferred during graph resolution, so dtype propagation and metrics share one source of truth. Per-tensor readouts include element count, bytes, boxes, exactness, approximation reasons, slice expressions, and propagation depth.

## 8a. Dependency notes

`src/core/notes.ts` turns a backward result into short statements about what constrains the cone — the contraction that must be staged, the axis a softmax needs whole, the halo a convolution re-reads. These answer the question the numbers do not: not how much, but why the footprint has this shape, and what it implies for tiling or fusing across it.

The text belongs to the operation, not the UI, because only the operation knows the reason. `OpSpec.dependencyNote` receives a `NoteCtx` — shapes, attributes, tensor names, the verified
symbolic dimensions, the propagated axis names, and the regions the current query
actually produced — and returns a draft or null. A note names an axis the way the source does,
preferring the axis's own name over the dimension it was declared with: `emb` says what the axis
*is* where `H*D` only says how wide it is, and the name is the one of the two that survives onto a
produced tensor. Without it a note about an intermediate falls back to a bare position or an
operation-internal einsum label — which is exactly where a reader most needs the word. Two rules govern it:

- **A note may only claim what the cone did.** A contraction note is withheld unless the region demonstrably spans the whole contracted axis, so a degenerate selection can never produce a false statement.
- **Notes carry a `key`.** Drafts sharing a key describe one constraint about different tensors and are merged rather than repeated, which stops a family of look-alike notes from crowding a distinct one past the display cap.
- **A note is never worth failing an analysis over.** A throwing `dependencyNote` is skipped at the presentation boundary; propagation, metrics, and every other finding remain available.

A draft also carries a `severity` and `flags` naming the tensors the constraint is visible on. Both are written where the note is — the operation — rather than derived in the view: only the operation knows how hard its constraint binds and which of its operands shows it. Severity is 3 for an axis consumed in full (a contraction, a reduction), 2 for an axis closed by a normalisation, 1 for ordering, halo re-reads, or strided access. There is no zero, because a constraint not worth stating is `null`.

Severity decides **what survives the display cap**, not what order notes are read in. The cap has to drop something, and dropping by graph order drops whichever the topological walk reached last — which can be the constraint that forces staging or accumulation. Selection is therefore by severity, then by nearness to the tile; the survivors are displayed in graph order, because that is the order the computation happens in. The inspector renders the severity weight and, when capped, the surviving and total counts. Flags are collected over every constraint found rather than the surviving ones, since a flag is true of its tensor whether or not its note had room.

`coneIsFullyElementwise` separates "every step fused" — a real finding — from "the cone reached no operation at all", so the two empty states can say different things.

`src/core/contribution.ts` answers the downstream half. Being in the forward cone means a tile *influences* a tensor, not that it produces it: a tile spanning part of a contracted axis reaches an output without determining a single element of it. The test runs backwards — what does that downstream region actually read? — and subtracts the tile; anything left is what the tile does not supply. The asymmetry is deliberate and matches the region contract: an over-approximated backward region can make the residue phantom, so the flag may over-warn, but an empty residue proves the true residue is empty, so it can never under-warn. Rows carry `exact` so an over-warning is shown as one, and the probe count is capped up front because each probe is a full propagation.

Reuse is a separate derived analysis in `src/core/reuse.ts`, rather than part of `computeMetrics`. It takes one anchored selection, deterministically samples same-sized tiles across that tensor with a fixed seed, and uses checked backward queries to estimate how many tiles touch each input region in the anchor's footprint. Keeping it pure and seeded makes repeated estimates reproducible without conflating a sampled statistic with the exact cost metrics.

## 9. UI architecture

The React layer is a projection over the headless compiler and executor.

### Application state

`src/ui/store.ts` is the single Zustand store. Its state is divided conceptually into:

- **Source:** DSL text, selected example, unresolved/resolved graph, and load errors.
- **Selection:** ordered parts, each naming its own tensor, and the independently enabled **What it needs** / **What it feeds** views. Their controls live on the matching inspector section heads; both may collapse into the explicitly labelled figures-only state without disabling analysis. Direct canvas gestures add by default and Alt subtracts; both compose against the drawn tensor's parts only, so a gesture on one tensor can never edit or discard a part on another. Replacement is reserved for controlled internal transitions. Composition retains object identity for untouched parts, allowing their index-based UI metadata to be remapped safely if another tensor loses parts.
- **Attribution:** which part is focused (emphasised, one at a time) and which parts are enabled in the merged analysis (any number, sticky). Disabled parts retain a faint selection rectangle and an inspector row so they can be included again, but their cached propagation is excluded from directional rows, metrics, notes, contribution verdicts, segmented footprints, and graph dependency highlights. Disabling the focused part clears its focus. Metadata is cleared for edited parts and follows untouched parts by identity when their indices shift.
- **Analysis:** aggregate backward/forward results, bounded per-box results, focus/pin state, and hover previews. **Both directions are always computed.** `direction` is a view filter over the analysis, not a gate on producing it: the inspector answers "what does this tile need" and "what does it feed" from one result, and a view choice must not decide whether a number exists. The filter is applied where highlights and directional rows are drawn, so switching it changes the picture without discarding the underlying analysis. `none` is the named figures-only combination: both section heads remain available to reopen either view while rows and graph highlights are hidden.
- **View:** per-tensor projection settings, tile scale, gesture snapping, the shape reading, the graph's px-per-element scale, metric options, graph focus, panel layout, and tensor offsets. `axisMode` chooses whether a compact card states semantic labels or numeric extents. `ui/shape-label.ts` owns the label fallback — axis name, then verified symbolic extent, then number — while the tensor details keep all three readings separate (`emb`, `H*D`, and `512`), dropping any that a less symbolic reading below it already states, so the surviving row is named after what it actually is. A new workspace opens on labels; a shared link restores the reading and tensor offsets it was written against rather than being reinterpreted by a later default. Cards reserve the widest reading during base layout, so changing the mode cannot create overlap or move the graph. Selection edits, committed tensor moves, and resetting every tensor to generated placement share one chronological workspace history.

Applying DSL recompiles the entire source into a new resolved graph. Changing a selection runs one bidirectional query per part, which serves both the per-part attribution and, merged, the aggregate the panels read. Past the per-part cap attribution is dropped and the queries are grouped by tensor instead, bounding the cost by the number of tensors drawn on rather than the number of tiles; the grouped result can only be equal or coarser, never tighter and never an under-approximation.

Actions defined against one tensor's axes — arrow-key moves, hidden-axis sliders, the reuse estimate, the axis legend — follow the **anchor**: the focused part's tensor, else the last one drawn on. An axis index means different things on tensors of different rank, so there is no single axis such an action could apply everywhere.

Expanding a composite node first resolves shape context, so composites fed by inferred intermediate tensors have their true rank available. The rewrite is serialized back to primitive DSL and compiled through the normal boundary, keeping the displayed source, share links, unresolved graph, and resolved graph synchronized. Selection state that could no longer be valid is cleared.

### Main components

- `App.tsx` composes the identity header, source panel, graph, and tile inspector. URL state crosses one transactional store boundary: the DSL is compiled and every selected part is validated before any source, graph, settings, or selection state is committed. Failure falls back to a built-in example without exposing a half-restored graph.
- `WorkspaceHeader.tsx` contains product identity, description, the shortcut-sheet entry point, and the global theme control.
- `SidePanel.tsx` owns the editable DSL draft, share action, built-in examples, and operation list. Running the draft updates the store only after compilation succeeds.
- `GraphView.tsx` is the graph controller and React projection: it derives highlighting, manages pan/zoom/focus and tensor gestures, and renders the current scene. Viewport panning starts on every background surface, including the transformed inner layout behind connectors; tensor cards, operation nodes, and zoom controls retain their own gestures. The card header is the full-width drag surface and the dotted button remains its visible and keyboard-focusable affordance; the tensor name is carved back out of it, because it is the focus target for the shape popover and a drag there would swallow the click that opens it. The world is unbounded, scene bounds include negative card positions, useful zoom is bounded by tensor legibility and canvas supersampling, and reset layout is an undoable recovery action. Connector SVGs are split by meaning: dim context runs behind the plates, while ordinary and hot connectors cross above them, including during a card drag. It does not own layout or routing algorithms.
- `TensorCard.tsx` renders an interactive tensor grid on canvas and converts pointer gestures into element-space boxes. Its paint stack is built by a pure `buildLayers`, so the encoding rules — uniform upstream, ruled downstream, dashed diagonal hatch for approximation, focus fading, and what a hidden box still shows — are asserted directly instead of inferred from pixels. Hover previews are selected per tensor and repeated pointer frames inside one logical cell are ignored; the canvas effect declares its paint dependencies, so a hover does not re-rasterise unrelated cards.
- `Inspector.tsx` is ordered by the question the product asks: the tile's identity, the tiles list, then **What it needs** and **What it feeds**, followed by approximation, cost, reuse, and dependency notes. Each directional heading is also its adjacent view toggle and its fill chip is the legend for the matching canvas treatment; each tensor appears once per direction with its box count, segmented per-tile footprint, slice expressions, and copy action. The What it feeds rollup carries completed-versus-partial counts so the product's distinguishing verdict is visible before rows are scanned. Dependency severity uses progressively heavier rules in muted, warning, and error ink; accent remains reserved for active UI. Setup is pinned to a strip at the bottom because the tile lattice is chosen once and then stops being read. With nothing drawn the panel names the two questions it will answer and offers a starting tile rather than rendering null sections.
- `shortcuts.ts` is the single manifest for key matching, labels, and help copy. `useKeyboard.ts` dispatches global bindings, local owners such as the source editor match against the same manifest, and `ShortcutsDialog.tsx` renders its grouped inventory. An open dialog gets first refusal on Escape.
- `inspector-analysis.ts` derives the inspector's scoped metrics, notes, rows, selection seeds, and contribution report with independent memoization. It re-merges the cached per-tile propagation using only enabled tiles, so a visibility toggle synchronizes every readout without rerunning the executor. Contribution classification can launch many checked backward probes, so it runs only while downstream rows are visible; note findings are collected once and shared by the row flags and notes section.
- `PanelFrame.tsx` owns side-panel width, the drag strip, and collapse-to-rail for both panels.
- `palette.ts` owns the categorical hues, their recorded CVD/contrast validation, and the canvas surfaces they were validated against. Two rules live there: identity is never carried by colour alone, and chrome must sit outside the data hues, because hue on a card means *which selection box* and a hot graph edge must not be mistakable for a cone.
- `share.ts` holds link encoding and decoding together. They were split across a toolbar and `App` once, and when the toolbar was removed the encoder went with it, leaving the app able to restore links nothing could produce. Decoding is all-or-nothing: an unknown setting, malformed part, or invalid layout offset rejects the payload, while genuinely absent legacy fields receive their documented defaults. The store then validates graph-dependent selection bounds and tensor identities transactionally.
- `graph-scene.ts` owns the two-stage graph rendering pipeline. A resolved graph plus a tensor-measurement policy produces the immutable Dagre base layout; that base plus persistent tensor offsets produces the cheap live scene and routed connectors. Query results, highlighting, and viewport state cannot invalidate structural placement. Links are rebuilt from ordered operation operands rather than Dagre's endpoint-keyed edge set, so repeated operands such as `matmul(X, X)` remain distinct.
- `graph-geometry.ts` contains the lower-level pure geometry for collision-safe motion and connector curves; large pointer jumps cannot tunnel through another node, but there is no artificial world wall. Connectors sharing a pair fan their anchors and stagger their flow marks to remain readable.
- `useDragGuard.ts` suppresses text selection for the duration of any drag. It is a hook rather than part of `setDragging` so that store actions stay free of DOM access and remain testable headlessly.

### Tensor rendering and tiling

Tensor cards use a fixed row-major projection:

- the last tensor axis is horizontal;
- the second-last axis is vertical;
- higher axes are controlled by slice or projection settings;
- rank-zero and rank-one tensors use corresponding reduced layouts.

One canvas cell represents one **tile**, not necessarily one element. `src/ui/tiling.ts` chooses a shape-aware base tile and applies the global detail scale. Card dimensions remain stable while detail changes.

The detail control is global but its ordinary scale values may settle differently per tensor. `effectiveTileScaleStops` removes adjacent scale values that produce the same graph-wide lattice, because power-of-two fitting inside bounded cards otherwise leaves much of the raw range inert. Its accessible value and tooltip report those settled lattices rather than the serialized power-of-two shift. The leftmost **None** stop is a semantic exception: `tileFor` returns 1 unconditionally, so it means one logical tile per element for snapping, hover, starter tiles, and keyboard movement even when a large card cannot display every boundary. Dense boundary lines are omitted by the renderer; the underlying grid is never coarsened. None and Auto remain separate stops when Auto also happens to yield 1×1, and `effectiveTileScaleIndex` preserves the exact stored intent before considering equivalent legacy plateaus.

Tiles remain square in element space: one `tile` value governs both visible axes because kernel tiling and the snap/nudge contract need one unit. An extent shorter than that value clips the tile rather than pretending the tensor has elements it does not. Each card therefore prints its effective visible-plane span (for example `4×128`, not `128×128` on a four-row tensor), while the setup strip summarizes the graph-wide square lattice.

A gesture only ever names two axes, so the card has one rule for the rest and
`selectionBoxFromDrag` is the only place that applies it: in projection mode a drag selects the
**full extent** of every hidden axis, because projection draws the union across them and a gesture
must select what it appears to touch; in slice mode it stays pinned to the sliders. The hover
readout and the hover cone preview are the degenerate single-cell case of that same call, not a
parallel derivation — a hover is the click that has not happened yet, and when the two were
computed separately the tooltip named a slider slice that the following click did not select.

**Snapping (`snapToGrid`) is a property of future gestures and movement, not stored selection geometry or analysis.** A drag is tracked in elements throughout and expanded to whole current-grid tiles only when it commits. The inspector's text range is an explicit element-space edit and is never rounded, whether snap is on or off. Toggling snap or changing detail leaves every existing box and the workspace undo depth untouched. A keyboard nudge uses the current tile while snapping and one element when not. If the box starts off the current lattice — after a typed edit or grid change — its first snapped nudge aligns its leading edge in the requested direction without changing its extent; later nudges advance by whole tiles. A boundary clamp may use a shorter final delta so the whole box lands flush without shrinking. Changing detail changes the next snapped gesture and nudge, never the current box.

Text edits are normal selection transactions: they repropagate immediately and add one workspace undo entry. An edited pinned tile retains its pin. An edited disabled tile retains its disabled state and its new cached propagation remains excluded from merged analysis until the tile is enabled again. Snap and detail are view/gesture settings rather than workspace geometry, so undoing a later movement restores the box without reverting those settings.

Card *size* is set by a single px-per-element figure for the whole graph. `graphScale` derives it from every tensor's drawn plane — the largest tensor sets the budget, the smallest non-degenerate side raises the scale if that budget would make it invisible — and the store holds the result as `graphPx`, computed once per resolved graph in `loadResolvedGraph`. This is what makes a shared dimension render at one length everywhere it appears, so a matmul's contraction axis is visually comparable across its operands. `graphPx` is deliberately not recomputed for view state. Degenerate axes (extent 1) are excluded from the scale and drawn at a fixed width, because they have no length to preserve and would otherwise pin the whole graph at the per-element cap.

`src/ui/grid.ts` is a pure geometry and drawing layer. Regions are painted at **element** precision: `regionRects` maps each box straight to canvas pixels, so a region that ends mid-tile draws a crisp edge rather than a partially shaded cell. The tile lattice is drawn over the top as a reading aid and does not quantise the marks. A thin region is widened to a one-pixel minimum, which over-states extent rather than letting it disappear — the same conservative direction the region algebra takes. Perimeter emphasis is suppressed when its stroke would be wider than the mark, so a one-element point or line never grows false area. Canvas backing resolution follows DPR and coarse zoom buckets; CSS uses ordinary interpolation rather than re-pixelating that supersampled result.

The canvas assigns one perceptual channel to each fact. Hue identifies the selected tile. Uniform fill means **What it needs**; a `\` ruling at a CSS-pixel pitch means **What it feeds**; the opposite dashed `/` hatch means a conservative over-approximation. Every ruling is diagonal because the tile lattice is axis-aligned, so an upright one at a comparable pitch would beat against it and read as banding that shifts with zoom. The downstream slope rotates with the selection box index, skipping the 45° the hatch owns and staying clear of 0°/90°. Its twelve unique slots match the per-box attribution cap. That rotation is what makes overlap visible: two boxes reaching the same elements used to paint identical rulings in identical phase, so the later cone covered the earlier exactly and the shared area read as one region — hue cannot separate them either, since the hues are painted over each other. At different angles the overlap crosses itself. It matters most past the third box, where the categorical hues run out and every further box shares one neutral colour, leaving the slope as the only thing still telling them apart. Approximation differs by texture kind rather than slope alone: its broken stroke remains distinguishable when it crosses a downstream ruling instead of merely reading as denser ink.

Rulings are stroked directly rather than tiled as a repeating bitmap, because a bitmap tile only repeats seamlessly at angles commensurate with its own edges — which is what limited the fill to 45° and made its spacing drift between zoom buckets. Stroking accepts any angle and rasterises at full resolution. Phase is anchored to the canvas origin, not to each rect, so neighbouring regions lie on one continuous ruling instead of restarting the pattern at every boundary. Hidden-axis coverage owns persistent cone alpha in both directions, because in projection mode it is genuinely a fraction of what a drawn cell stands for. Graph depth does not fade either direction: the exact `dN` row badge owns that integer instead. The transient hover preview remains the deliberate exception because it has no row and must sit below committed analysis. Density is carried by *spacing* at a constant line weight, so it reads as ink and survives greyscale and thumbnails. The exact graph scale counteracts the outer CSS transform, keeping ruling geometry stable on screen without the jumps caused by a second set of zoom buckets; therefore zoom alone never removes direction. Only a region whose resulting screen extent is below three pixels falls back to solid fill. Every requested pattern that cannot be drawn uses the same quieter solid, whether extent or the renderer's safety cap caused the degradation. This deliberately borrows alpha because no geometric direction channel survives in such a mark: density degrades before direction, and visibility before either. A ruled rect carries a hairline delimiter at a fraction of the weight of the perimeter that used to encode direction — a ruling stops at its last line rather than at the region bound, so without one the extent reads as ragged; solid fills are already crisp and get none.

Ruling density is deliberately constant until the engine exposes an honest supplied-share measure. When that quantity lands, paint will use a small set of named perceptual steps rather than pretending the roughly six-pixel pitch range supports percentage precision; the inspector row will own the exact value or conservative bound alongside its `partial`/`completed` verdict. Density is reserved for that future quantity rather than reused for focus or depth.

## 10. End-to-end interaction flow

For a typical edit-and-select interaction:

1. The user edits DSL in the side panel and runs it.
2. The compiler parses the text, creates source mappings, and resolves the graph.
3. The store atomically replaces the graph or retains a compilation error for display.
4. Dagre lays out the resolved graph and tensor cards derive their grids from resolved shapes.
5. A pointer gesture creates or composes an ordered selection box.
6. The store submits the canonical selection to `SymbolicExecutor`.
7. Operation specifications map the region through the graph in the requested direction(s).
8. Metrics, notes, and the partial-contribution report are derived from those results.
9. Tensor canvases and the inspector render aggregate and per-box results, filtered by the current direction.

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
- note severity, cap selection, and the per-tensor flags, including that the hardest constraint survives the cap and that a full-axis pull is still seen once its region is split into disjoint boxes;
- partial versus complete downstream contribution, including the over-approximated case;
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
3a. Map axis names where an output axis is genuinely the input's own axis, and leave it
    unnamed otherwise. A borrowed name is worse than none: the inspector prints it as the
    source's own word for that axis.
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
- The partial-contribution flag can over-warn on an over-approximated region and never under-warns; past its probe cap, downstream rows are simply unflagged.
- Per-part attribution is intentionally capped to keep interaction responsive; aggregate propagation remains complete.
- Expanding a composite can expose intermediate traffic, so dependency and FLOP semantics may remain equivalent while displayed intermediate-byte estimates change.
- JSON graph support exists below the UI, but the main interactive authoring path is the DSL.

These boundaries are useful when deciding where new work belongs: semantic truth should live in the graph, region, operation, and executor layers; orchestration belongs in the store; presentation belongs in React and the canvas helpers.
