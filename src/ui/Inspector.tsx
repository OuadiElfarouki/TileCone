import React, { useEffect, useMemo, useState } from "react";
import { Contribution, MAX_CONTRIBUTION_PROBES } from "../core/contribution";
import { TensorReadout } from "../core/metrics";
import type { ConeFindings } from "../core/notes";
import { estimateInputReuse, ReuseEstimate } from "../core/reuse";
import {
  Box,
  Region,
  count,
  fromBox,
  partsOverlap,
  subtract,
  union,
} from "../core/region";
import { formatBytes } from "./TensorCard";
import { useInspectorAnalysis } from "./inspector-analysis";
import { aggregateColors, boxColor, MAX_DISTINCT_HUES, rgbCss } from "./palette";
import {
  anchorTensorId,
  MAX_PER_BOX_PROPS,
  partsOn,
  planesOf,
  selectedTensorIds,
  startingTiles,
  useStore,
  viewAxes,
} from "./store";
import { formatSelectionBox, parseSelectionBox } from "./selection-range";
import {
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  settledTiles,
  TILE_SCALE_NONE,
} from "./tiling";

function fmt(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(n < 10 && !Number.isInteger(n) ? 2 : 0);
}

const scaleLabel = (scale: number) =>
  scale === TILE_SCALE_NONE ? "none" : scale === 0 ? "auto" : `${scale > 0 ? "×" : "÷"}${2 ** Math.abs(scale)}`;

/** What the graph actually settled on, which the request only asks for. */
function settledLabel(min: number, max: number): string {
  return min === max ? `${min} × ${min}` : `${min} × ${min} – ${max} × ${max}`;
}

/**
 * Setup, pinned below the reading path. The lattice is chosen once and then
 * stops being read, so it earns a strip rather than the top of the panel — but
 * it stays visible, because the slider changes what a drawn tile means.
 */
function SetupStrip(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const setSnapToGrid = useStore((s) => s.setSnapToGrid);
  const selection = useStore((s) => s.selection);
  const clearSelection = useStore((s) => s.clearSelection);
  const detail = useMemo(() => {
    const planes = planesOf(resolved);
    const stops = effectiveTileScaleStops(planes, graphPx);
    const index = effectiveTileScaleIndex(planes, graphPx, stops, tileScale);
    return { stops, index, settled: settledTiles(planes, stops[index], graphPx) };
  }, [resolved, graphPx, tileScale]);
  const requested = detail.stops[detail.index];
  const { min, max } = detail.settled;
  // "none" that could not be honoured is worth saying out loud; every other
  // request either lands or is a plain power-of-two shift of the auto base.
  const overridden = requested === TILE_SCALE_NONE && min > 1;

  return (
    <section className="inspector-setup">
      <div className="setup-row">
        <span className="setup-kicker">grid</span>
        <span
          className={`tile-settled${overridden ? " overridden" : ""}`}
          title={
            overridden
              ? `one cell per element does not fit this graph's largest tensor, so the finest drawable lattice is ${settledLabel(min, max)}`
              : min === max
                ? "the tile every tensor settles on"
                : "tiles differ per tensor: the fit rule coarsens the largest ones"
          }
        >
          {settledLabel(min, max)}
        </span>
        <button
          className={`mini snap-btn${snapToGrid ? " on" : ""}`}
          aria-pressed={snapToGrid}
          onClick={() => setSnapToGrid(!snapToGrid)}
          title="snap a drawn box out to whole tiles; off cuts an exact element range"
        >
          snap
        </button>
        <button className="mini clear-all" onClick={clearSelection} disabled={!selection}>
          clear all
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, detail.stops.length - 1)}
        step={1}
        value={detail.index}
        onChange={(event) => setTileScale(detail.stops[Number(event.target.value)])}
        aria-label="tile grid detail"
        aria-valuetext={scaleLabel(detail.stops[detail.index])}
        title={`global tile detail — ${detail.stops.map(scaleLabel).join(" · ")}`}
      />
    </section>
  );
}

function SelectionRangeInput({
  box,
  shape,
  onCommit,
}: {
  box: Box;
  shape: number[];
  onCommit: (box: Box) => void;
}): React.ReactElement {
  const formatted = formatSelectionBox(box);
  const [draft, setDraft] = useState(formatted);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    setDraft(formatted);
    setInvalid(false);
  }, [formatted]);

  const commit = () => {
    const parsed = parseSelectionBox(draft, shape);
    if (!parsed) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    const unchanged = parsed.every(
      (interval, axis) => interval.lo === box[axis].lo && interval.hi === box[axis].hi
    );
    if (!unchanged) onCommit(parsed);
  };

  return (
    <input
      className={`box-range${invalid ? " invalid" : ""}`}
      value={draft}
      aria-label="selection range"
      aria-invalid={invalid}
      title={invalid ? `expected ${shape.length} in-bounds index or lo:hi fields` : "edit range; Enter or blur applies"}
      spellCheck={false}
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") commit();
        if (event.key === "Escape") {
          setDraft(formatted);
          setInvalid(false);
        }
      }}
    />
  );
}

/**
 * Why the footprint has the shape it has. The numbers above say how much; this
 * says what constrains it, which is the part that transfers to writing a kernel.
 */
function DependencyNotes({
  findings,
  hasSelection,
  focusedBox,
  attributed,
}: {
  findings: ConeFindings | null;
  hasSelection: boolean;
  focusedBox: number | null;
  attributed: boolean;
}): React.ReactElement {
  return (
    <section className="ins-section notes-section">
      <div className="ins-title">
        Dependency notes
        {focusedBox !== null && attributed && <span className="muted"> · tile {focusedBox + 1}</span>}
      </div>
      {!hasSelection || !findings ? (
        <p className="hint">
          Draw a tile to read its cone. Notes here call out the reductions and contractions
          that constrain a fused kernel.
        </p>
      ) : findings.notes.length ? (
        <ul className="notes-list">
          {findings.notes.map((note) => (
            <li key={`${note.nodeId}:${note.text}`}>
              <b>{note.op}</b>
              {note.text}
            </li>
          ))}
        </ul>
      ) : findings.elementwise ? (
        <p className="hint">
          This cone is elementwise all the way through: any tiling of the selection fuses
          without cross-tile traffic.
        </p>
      ) : (
        <p className="hint">
          Nothing upstream of this selection — it reaches no operation, so no dependency
          constrains it.
        </p>
      )}
    </section>
  );
}

const EMPTY_REGION: Region = { boxes: [], exact: true, reasons: [] };

function FootprintBar({
  tensorId,
  elements,
  totalElements,
  perBox,
  focusedBox,
  dark,
}: {
  tensorId: string;
  elements: number;
  totalElements: number;
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  focusedBox: number | null;
  dark: boolean;
}): React.ReactElement {
  if (totalElements <= 0) return <span className="footprint-bar" />;

  const regions = perBox?.map(
    (prop) => prop.backward?.tensors.get(tensorId)?.region ?? EMPTY_REGION
  );
  const segments: { elements: number; color: string; title: string; shared?: boolean }[] = [];

  if (regions && focusedBox !== null && regions[focusedBox]) {
    segments.push({
      elements,
      color: rgbCss(boxColor(focusedBox, dark)),
      title: `tile ${focusedBox + 1}: ${fmt(elements)} elements`,
    });
  } else if (regions) {
    let exclusiveTotal = 0;
    regions.forEach((region, index) => {
      let others = EMPTY_REGION;
      regions.forEach((other, otherIndex) => {
        if (otherIndex !== index) others = union(others, other);
      });
      const exclusive = count(subtract(region, others));
      exclusiveTotal += exclusive;
      if (exclusive > 0)
        segments.push({
          elements: exclusive,
          color: rgbCss(boxColor(index, dark)),
          title: `tile ${index + 1} only: ${fmt(exclusive)} elements`,
        });
    });
    const shared = Math.max(0, elements - exclusiveTotal);
    if (shared > 0)
      segments.push({
        elements: shared,
        color: "",
        title: `shared by multiple tiles: ${fmt(shared)} elements`,
        shared: true,
      });
  } else if (elements > 0) {
    segments.push({ elements, color: "var(--muted)", title: `${fmt(elements)} elements touched` });
  }

  return (
    <span
      className="footprint-bar"
      title={`whole bar = ${fmt(totalElements)} tensor elements`}
      aria-label={`${fmt(elements)} of ${fmt(totalElements)} elements touched`}
    >
      {segments.map((segment, index) => (
        <i
          key={index}
          className={segment.shared ? "shared" : ""}
          title={segment.title}
          style={{
            width: `${Math.min(100, (segment.elements / totalElements) * 100)}%`,
            background: segment.color || undefined,
          }}
        />
      ))}
    </span>
  );
}

/** Slice expressions minus the over-approximation comment the readout appends. */
const sliceLines = (row: TensorReadout) =>
  row.sliceExprsNumpy.filter((line) => !line.startsWith("#"));

/**
 * One tensor in one direction: how much of it the tile touches, why it has that
 * shape, and — downstream — whether the tile finishes it or only feeds it.
 */
function ConeRow({
  row,
  hue,
  flags,
  contribution,
  onCopy,
  copied,
}: {
  row: TensorReadout;
  hue: string;
  flags: string[];
  contribution?: Contribution;
  onCopy: () => void;
  copied: boolean;
}): React.ReactElement {
  const exprs = sliceLines(row);
  const share = row.totalElements > 0 ? (row.elements / row.totalElements) * 100 : 0;

  return (
    <div className="cone-row">
      <div className="cone-row-head">
        <b>{row.name}</b>
        {/* One box is the common case and reads best inline. More than one is
            exactly when a single truncated expression would hide the answer, so
            those move to their own block below. */}
        {exprs.length === 1 && (
          <code className="cone-expr">{exprs[0].slice(row.name.length)}</code>
        )}
        {!row.exact && <span className="badge approx" title={row.reasons.join("; ")}>≈</span>}
        {row.isInput && <span className="badge input">in</span>}
        <span className="muted d-tag" title={`${row.depth} step${row.depth === 1 ? "" : "s"} along the cone`}>
          d{row.depth}
        </span>
        <span className="row-stats">
          {share.toFixed(1)}% · {formatBytes(row.bytes)}
        </span>
      </div>
      <span
        className="cone-bar"
        title={`${fmt(row.elements)} of ${fmt(row.totalElements)} elements`}
        aria-label={`${share.toFixed(1)} percent of ${row.name}`}
      >
        <i style={{ width: `${Math.min(100, share)}%`, background: hue }} />
      </span>
      {flags.map((flag) => (
        <span className="cone-flag" key={flag}>
          {flag}
        </span>
      ))}
      {contribution?.partial && (
        <span className="cone-flag">
          partial — {contribution.detail}
          {!contribution.exact && " (from an over-approximated region)"}
        </span>
      )}
      {exprs.length > 1 && (
        <div className="slice-exprs">
          {exprs.slice(0, 4).map((line, index) => (
            <code key={index}>{line}</code>
          ))}
          {exprs.length > 4 && (
            <code className="muted"># … {exprs.length - 4} more boxes</code>
          )}
        </div>
      )}
      <button className="mini copy-exprs" title="copy every slice expression for this tensor" onClick={onCopy}>
        {copied ? "copied ✓" : "copy"}
      </button>
    </div>
  );
}

function ConeSection({
  arrow,
  title,
  rows,
  hue,
  flags,
  contributions: contrib,
  showRows,
  empty,
  copiedKey,
  onCopy,
}: {
  arrow: string;
  title: string;
  rows: TensorReadout[];
  hue: string;
  flags: Map<string, string[]>;
  contributions?: Map<string, Contribution>;
  showRows: boolean;
  empty: string;
  copiedKey: string | null;
  onCopy: (row: TensorReadout, key: string) => void;
}): React.ReactElement {
  const bytes = rows.reduce((total, row) => total + row.bytes, 0);

  return (
    <section className="cone-section">
      <div className="cone-head">
        <span className="cone-arrow" aria-hidden>
          {arrow}
        </span>
        <h2 className="panel-title">{title}</h2>
        <span className="rollup">
          {rows.length} tensor{rows.length === 1 ? "" : "s"} · {formatBytes(bytes)}
        </span>
      </div>
      {showRows &&
        (rows.length ? (
          rows.map((row) => {
            const key = `${title}:${row.tensorId}`;
            return (
              <ConeRow
                key={row.tensorId}
                row={row}
                hue={hue}
                flags={flags.get(row.tensorId) ?? []}
                contribution={contrib?.get(row.tensorId)}
                copied={copiedKey === key}
                onCopy={() => onCopy(row, key)}
              />
            );
          })
        ) : (
          <p className="hint">{empty}</p>
        ))}
    </section>
  );
}

/**
 * The tile itself: which one it is, where it sits, and how big it is. At one
 * tile this is also where the range is edited, because a list of one restates
 * the header and adds nothing to compare it against.
 */
function TileIdentity({
  onCopyAll,
  copied,
}: {
  onCopyAll: () => void;
  copied: boolean;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved)!;
  const selection = useStore((s) => s.selection);
  const focusedBox = useStore((s) => s.focusedBox);
  const replaceBox = useStore((s) => s.replaceBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const theme = useStore((s) => s.theme);

  if (!selection) return null;
  const parts = selection.parts;
  const anchorId = anchorTensorId(selection, focusedBox);
  const index = focusedBox ?? parts.length - 1;
  const part = parts[index];
  if (!anchorId || !part) return null;

  // With several tiles and none focused, everything below describes the merged
  // cone. Naming one of them here would put the wrong tile at the top of a
  // readout about all of them, so the header describes the selection instead.
  const merged = parts.length > 1 && focusedBox === null;
  const tensor = resolved.tensors[anchorId];
  const shape = tensor.resolved!;
  const { rowAxis, colAxis } = viewAxes(shape);
  const axisLabel = (axis: number) => tensor.axisNames?.[axis] ?? `ax${axis}`;
  const volume = (box: Box) =>
    box.reduce((total, interval) => total * (interval.hi - interval.lo), 1);
  const elements = merged
    ? parts.reduce((total, p) => total + volume(p.box), 0)
    : volume(part.box);

  return (
    <header className="tile-identity">
      <div className="tile-ord">
        {/* The swatch is the tile's hue on the canvas: the header and the cone
            it describes have to be the same colour, so it follows the anchor's
            own index rather than the first one. A merged readout belongs to no
            single hue, so it carries none. */}
        {!merged && (
          <i className="swatch" style={{ background: rgbCss(boxColor(index, theme === "dark")) }} />
        )}
        <span>
          {merged ? `${parts.length} tiles · merged` : `tile ${index + 1} of ${parts.length}`}
        </span>
        <button className="mini copy-cone" onClick={onCopyAll} title="copy every slice expression in this cone, both directions">
          {copied ? "copied ✓" : "copy"}
        </button>
        {parts.length === 1 && (
          <button className="mini" title="remove this tile from the selection" onClick={() => deleteBox(index)}>
            ×
          </button>
        )}
      </div>
      <div className="tile-name">
        <b
          title={
            merged
              ? "every tensor the selection draws on"
              : `rows ${axisLabel(rowAxis >= 0 ? rowAxis : 0)} · cols ${axisLabel(colAxis >= 0 ? colAxis : 0)} · shape [${shape.join("×")}]`
          }
        >
          {merged
            ? selectedTensorIds(selection)
                .map((id) => resolved.tensors[id].name)
                .join(" · ")
            : tensor.name}
        </b>
        <span className="tile-count">{fmt(elements)} el</span>
      </div>
      {/* Only at one tile. From two upwards the list below carries every range,
          and repeating the focused one here made the header change height as
          the pointer moved across the list. */}
      {parts.length === 1 && (
        <SelectionRangeInput box={part.box} shape={shape} onCommit={(next) => replaceBox(index, next)} />
      )}
    </header>
  );
}

/**
 * The selection's tiles, one row each, from two upwards. Hovering a row
 * emphasises that tile's cone across the whole graph; clicking pins it.
 */
function RegionEditor(): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const replaceBox = useStore((s) => s.replaceBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const pinned = useStore((s) => s.pinnedBox);
  const hoverBox = useStore((s) => s.hoverBox);
  const togglePinBox = useStore((s) => s.togglePinBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const toggleBoxHidden = useStore((s) => s.toggleBoxHidden);
  const theme = useStore((s) => s.theme);

  if (!resolved || !selection || selection.parts.length < 2) return null;
  const parts = selection.parts;
  const dark = theme === "dark";

  // Overlap is a within-tensor question: two boxes on different tensors index
  // different things and cannot double-count each other.
  const overlap = { unique: 0, summed: 0 };
  for (const id of selectedTensorIds(selection)) {
    const one = partsOverlap(partsOn(selection, id).map((p) => p.box));
    overlap.unique += one.unique;
    overlap.summed += one.summed;
  }

  return (
    <div className="ins-section">
      <div className="ins-title">Tiles</div>
      <p className="hint">
        {perBox
          ? "hover a tile to emphasise its cone; click to pin it, then arrow keys move that tile alone"
          : `too many tiles to trace individually (over ${MAX_PER_BOX_PROPS}) — showing the merged cone`}
      </p>
      {overlap.summed > overlap.unique && (
        <p className="hint overlap">
          tiles overlap: {fmt(overlap.summed)} counted across parts,{" "}
          <b>{fmt(overlap.unique)}</b> distinct elements — upstream and downstream use the
          deduplicated set
        </p>
      )}

      <div className="box-list" onMouseLeave={() => hoverBox(null)}>
        {parts.map(({ tensorId, box: b }, i) => {
          const active = focusedBox === i;
          const hidden = hiddenBoxes.has(i);
          const t = resolved.tensors[tensorId];
          const shape = t.resolved!;
          return (
            <div
              className={`box-row${active ? " active" : ""}${pinned === i ? " pinned" : ""}${hidden ? " hidden-cone" : ""}`}
              key={i}
              onMouseEnter={() => hoverBox(i)}
              onClick={() => perBox && togglePinBox(i)}
              title={perBox ? "hover to emphasise this tile's dependencies; click to pin (esc unpins)" : undefined}
            >
              {/* One control, not two: the swatch *is* the visibility toggle, so
                  the row states the tile's hue and its shown/hidden state once. */}
              <button
                className="vis-toggle"
                disabled={!perBox}
                style={{
                  borderColor: rgbCss(boxColor(i, dark)),
                  background: hidden ? "transparent" : rgbCss(boxColor(i, dark)),
                }}
                title={
                  perBox
                    ? `${hidden ? "show" : "hide"} this tile's cone (h) — its numbers stay in the table either way`
                    : "per-tile cones are not traced at this many tiles"
                }
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBoxHidden(i);
                }}
                aria-label={`${hidden ? "show" : "hide"} tile ${i + 1}`}
              />
              <span className="box-body">
                <span className="box-label">
                  <b>{t.name}</b> · {fmt(b.reduce((a, I) => a * (I.hi - I.lo), 1))} elements
                </span>
                <SelectionRangeInput
                  box={b}
                  shape={shape}
                  onCommit={(next) => replaceBox(i, next)}
                />
              </span>
              <button
                className="mini"
                title="remove this tile from the selection"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteBox(i);
                }}
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
      {parts.length > MAX_DISTINCT_HUES && (
        <p className="hint">
          tiles past the {MAX_DISTINCT_HUES}rd share a neutral color — only {MAX_DISTINCT_HUES} hues stay
          distinguishable side by side, so use hover to tell the rest apart.
        </p>
      )}
    </div>
  );
}

/** Nothing drawn yet: teach the two questions instead of rendering empty tables. */
function EmptyPanel(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const setSelection = useStore((s) => s.setSelection);
  const starts = useMemo(
    () => startingTiles(resolved, tileScale, graphPx),
    [resolved, tileScale, graphPx]
  );

  return (
    <div className="empty-panel">
      <h2 className="panel-title">No tile drawn</h2>
      <p className="hint">
        Drag a rectangle on any tensor to cut a tile. Shift adds another, Alt subtracts. This
        panel then answers two questions about it.
      </p>
      <dl className="empty-questions">
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↑
            </span>
            What it needs
          </dt>
          <dd>Every upstream tensor the tile reads, and how much of each.</dd>
        </div>
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↓
            </span>
            What it feeds
          </dt>
          <dd>Everything downstream it reaches, and whether that contribution is partial.</dd>
        </div>
      </dl>
      {starts.length > 0 && (
        <>
          <div className="setup-kicker">or start from</div>
          <div className="empty-starts">
            {starts.map((start) => (
              <button
                key={start.tensorId}
                className="mini"
                onClick={() => setSelection(start.tensorId, fromBox(start.box), "replace")}
              >
                {resolved.tensors[start.tensorId].name} — {start.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export function Inspector(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const direction = useStore((s) => s.direction);
  const countIntermediates = useStore((s) => s.countIntermediates);
  const setCountIntermediates = useStore((s) => s.setCountIntermediates);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);

  const [reuse, setReuse] = useState<ReuseEstimate[] | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const {
    metrics,
    findings,
    seeds,
    contribution: contrib,
    upstream,
    downstream,
  } = useInspectorAnalysis({
    resolved,
    selection,
    backward: backwardRes,
    forward: forwardRes,
    perBox,
    focusedBox,
    countIntermediates,
    direction,
  });

  useEffect(() => setReuse(null), [resolved, selection, focusedBox]);

  if (!resolved) return <aside className="inspector" />;

  const selBoxes = selection?.parts.length ?? 0;
  const dark = theme === "dark";
  /** Past the cap the cones are still correct, but merged rather than attributed. */
  const merged = selBoxes > MAX_PER_BOX_PROPS;
  const aggregate = aggregateColors(dark);
  /** A bar carries the hue of the tile it belongs to, or the aggregate when the
   *  rows describe several tiles at once. */
  const coneHue = (cone: "upstream" | "downstream") => {
    if (focusedBox !== null) return rgbCss(boxColor(focusedBox, dark));
    if (selBoxes === 1 && !merged) return rgbCss(boxColor(0, dark));
    return rgbCss(cone === "upstream" ? aggregate.upstream : aggregate.downstream);
  };

  // `direction` chooses which question is on screen; it never decides whether
  // the panel works. Both cones are computed either way.
  const showUpstream = direction !== "forward";
  const showDownstream = direction !== "backward";
  const showRows = direction !== "none";

  // An empty section has two causes, and they are different findings: the
  // selection sits at the edge of the graph, or what it reaches is selected too
  // and is therefore not something it reads or feeds.
  const seedIds = [...seeds.keys()];
  const upstreamEmpty = seedIds.some((id) => resolved.tensors[id].producer)
    ? "Everything upstream of the selection is itself selected."
    : "Every selected tensor is a graph input — there is nothing upstream to read.";
  const downstreamEmpty = seedIds.some((id) => resolved.consumers[id]?.length)
    ? "Everything the selection feeds is itself selected."
    : "Nothing consumes this selection — it is a graph output.";

  const visibleRows = [...(showUpstream ? upstream : []), ...(showDownstream ? downstream : [])];
  const approxReasons = [
    ...new Set(visibleRows.filter((row) => !row.exact).flatMap((row) => row.reasons)),
  ];

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. The sweep
   * is defined by one tile on one tensor, so it follows the anchor part — the
   * focused one, else the last drawn — rather than mixing tensors. */
  const computeReuse = () => {
    if (!selection || !resolved) return;
    const probe =
      selection.parts[focusedBox !== null ? focusedBox : selection.parts.length - 1];
    if (!probe) return;
    setReuse(estimateInputReuse(resolved, { tensorId: probe.tensorId, region: fromBox(probe.box) }));
  };

  const copy = (text: string, key: string) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(key);
      setTimeout(() => setCopied(null), 1200);
    });
  };
  const copyRow = (row: TensorReadout, key: string) => copy(sliceLines(row).join("\n"), key);
  const copyCone = () =>
    copy(
      [...upstream, ...downstream].flatMap((row) => sliceLines(row)).join("\n"),
      "cone"
    );

  return (
    <aside className="inspector">
      <div className="inspector-scroll">
        {!selection ? (
          <EmptyPanel />
        ) : (
          <>
            <TileIdentity onCopyAll={copyCone} copied={copied === "cone"} />
            {/* The tiles list is the selector for everything below it: it picks
                which cone the two sections describe, so it sits above them. */}
            <RegionEditor />

            {showUpstream && (
              <ConeSection
                arrow="↑"
                title="Needs upstream"
                rows={upstream}
                hue={coneHue("upstream")}
                flags={findings?.flags ?? new Map()}
                showRows={showRows}
                empty={upstreamEmpty}
                copiedKey={copied}
                onCopy={copyRow}
              />
            )}
            {showDownstream && (
              <ConeSection
                arrow="↓"
                title="Feeds downstream"
                rows={downstream}
                hue={coneHue("downstream")}
                flags={new Map()}
                contributions={contrib?.byTensor}
                showRows={showRows}
                empty={downstreamEmpty}
                copiedKey={copied}
                onCopy={copyRow}
              />
            )}
            {showDownstream && showRows && contrib?.capped && (
              <p className="hint">
                more than {MAX_CONTRIBUTION_PROBES} tensors downstream — rows below do not say
                whether this tile completes them or only feeds them
              </p>
            )}

            {approxReasons.length > 0 && (
              <div className="ins-section warn">
                ⚠ some regions are conservative over-approximations: {approxReasons.join("; ")}
              </div>
            )}

            {metrics && (
              <>
                <div className="ins-section">
                  <div className="ins-title">
                    {direction === "forward"
                      ? "Cost of the upstream cone"
                      : focusedBox !== null && perBox
                        ? `Cost of tile ${focusedBox + 1}`
                        : "Cost of this cone"}
                  </div>
                  <div className="kv">
                    <span>FLOPs</span><span>{fmt(metrics.flops)}</span>
                    <span>input bytes</span><span>{formatBytes(metrics.inputBytes)}</span>
                    <span>intermediate</span><span>{formatBytes(metrics.intermediateBytes)}</span>
                    <span>output bytes</span><span>{formatBytes(metrics.outputBytes)}</span>
                    <span>cone working set</span>
                    <span>{formatBytes(metrics.inputBytes + metrics.intermediateBytes + metrics.outputBytes)}</span>
                    <span>intensity</span><span>{metrics.intensity.toFixed(2)} FLOP/B</span>
                  </div>
                  <label className="chk">
                    <input type="checkbox" checked={countIntermediates} onChange={(e) => setCountIntermediates(e.target.checked)} />
                    count intermediates in bytes
                  </label>
                </div>
                <div className="ins-section">
                  <div className="ins-title">
                    Reuse <button className="mini" onClick={computeReuse}>estimate</button>
                  </div>
                  {reuse ? (
                    <div className="kv">
                      {reuse.map((estimate) => (
                        <React.Fragment key={estimate.tensorId}>
                          <span>{resolved.tensors[estimate.tensorId].name}</span>
                          <span>
                            {estimate.estimatedTiles.toFixed(estimate.estimatedTiles < 10 ? 1 : 0)}×
                            {` (of ${estimate.totalTiles} tiles)`}
                          </span>
                        </React.Fragment>
                      ))}
                    </div>
                  ) : (
                    <p className="hint">selection-sized output tiles touching each input's current footprint (sampled)</p>
                  )}
                </div>
                {/* At one tile the two directional sections already cover every
                    tensor this table covers. From two upwards it is the only
                    place the shared segment between tiles appears. */}
                {selBoxes > 1 && (
                  <div className="ins-section">
                    <div className="ins-title">Footprint per tensor</div>
                    {metrics.tensors.map((t) => (
                      <div key={t.tensorId} className="tensor-row">
                        <div className="tensor-row-head">
                          <b>{t.name}</b>
                          <span className="role-tag">{t.depth === 0 ? "selected" : "required"}</span>
                          <span className="muted" title={`${t.depth} step${t.depth === 1 ? "" : "s"} upstream of the selection`}>d{t.depth}</span>
                          {!t.exact && <span className="badge approx">≈</span>}
                          {t.isInput && <span className="badge input">in</span>}
                          <span className="row-stats">
                            {fmt(t.elements)} el ({((t.elements / t.totalElements) * 100).toFixed(1)}%) · {formatBytes(t.bytes)} · {t.boxCount} box{t.boxCount === 1 ? "" : "es"}
                          </span>
                        </div>
                        <FootprintBar
                          tensorId={t.tensorId}
                          elements={t.elements}
                          totalElements={t.totalElements}
                          perBox={perBox}
                          focusedBox={focusedBox}
                          dark={dark}
                        />
                        <div className="slice-exprs">
                          {t.sliceExprsNumpy.slice(0, 4).map((line, i) => (
                            <code key={i}>{line}</code>
                          ))}
                          {t.sliceExprsNumpy.length > 4 && (
                            <code className="muted"># … {t.sliceExprsNumpy.length - 4} more boxes</code>
                          )}
                          <button
                            className="mini copy-exprs"
                            title="copy every slice expression for this tensor"
                            onClick={() => copy(t.sliceExprsNumpy.join("\n"), t.tensorId)}
                          >
                            {copied === t.tensorId ? "copied ✓" : "copy"}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
            <DependencyNotes
              findings={findings}
              hasSelection={!!selection}
              focusedBox={focusedBox}
              attributed={!!perBox}
            />
          </>
        )}
      </div>
      <SetupStrip />
    </aside>
  );
}
