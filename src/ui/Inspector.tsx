import React, { useEffect, useMemo, useState } from "react";
import { Contribution, MAX_CONTRIBUTION_PROBES } from "../core/contribution";
import { TensorReadout } from "../core/metrics";
import type { ConeFindings } from "../core/notes";
import { estimateInputReuse, ReuseEstimate } from "../core/reuse";
import {
  Box,
  Region,
  count,
  formatBoxIndices,
  fromBox,
  partsOverlap,
  subtract,
  union,
} from "../core/region";
import { formatBytes } from "./format";
import { analysisTensorId, groupAttribution, groupFocus, measuredParts, measuredElements, useInspectorAnalysis } from "./inspector-analysis";
import { aggregateColors, boxColor, MAX_DISTINCT_HUES, rgbCss } from "./palette";
import {
  ConeDirection,
  MAX_PER_BOX_PROPS,
  partsOn,
  planesOf,
  selectedTensorIds,
  startingTiles,
  useStore,
} from "./store";
import { formatSelectionBox, parseSelectionBox } from "./selection-range";
import { copyText } from "./clipboard";
import { hasSymbolicShape } from "./shape-label";
import {
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  settledTiles,
  TILE_SCALE_NONE,
} from "./tiling";
import { viewAxes } from "./tensor-view";

function fmt(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(n < 10 && !Number.isInteger(n) ? 2 : 0);
}

/** What the graph actually settled on, which the request only asks for. */
function settledLabel(min: number, max: number): string {
  return min === max ? `${min} × ${min}` : `${min} × ${min} – ${max} × ${max}`;
}

function stopLabel(
  scale: number,
  settled: { min: number; max: number }
): string {
  const lattice = settledLabel(settled.min, settled.max);
  return scale === TILE_SCALE_NONE ? `none → ${lattice}` : lattice;
}

/**
 * Setup, pinned below the reading path. The lattice is chosen once and then
 * stops being read, so it earns a strip rather than the top of the panel, but
 * it stays visible, because the slider changes what a drawn tile means.
 */
function SetupStrip(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const setSnapToGrid = useStore((s) => s.setSnapToGrid);
  const axisMode = useStore((s) => s.axisMode);
  const setAxisMode = useStore((s) => s.setAxisMode);
  const selection = useStore((s) => s.selection);
  const clearSelection = useStore((s) => s.clearSelection);
  const detail = useMemo(() => {
    const planes = planesOf(resolved);
    const stops = effectiveTileScaleStops(planes, graphPx);
    const index = effectiveTileScaleIndex(planes, graphPx, stops, tileScale);
    const settled = stops.map((scale) => settledTiles(planes, scale, graphPx));
    return { stops, index, settled, labels: stops.map((scale, i) => stopLabel(scale, settled[i])) };
  }, [resolved, graphPx, tileScale]);
  const requested = detail.stops[detail.index];
  const { min, max } = detail.settled[detail.index];
  const hasSemanticLabels = Object.values(resolved.tensors).some(hasSymbolicShape);

  return (
    <section className="inspector-setup">
      <div className="setup-row">
        <span className="setup-kicker">grid · all tensors</span>
        <span
          className="tile-settled"
          title={
            requested === TILE_SCALE_NONE
              ? "one logical tile per element; boundaries are omitted where they are too dense to draw"
              : min === max
                ? "the tile every tensor settles on"
                : "tiles differ per tensor: the fit rule coarsens the largest ones"
          }
        >
          {settledLabel(min, max)}
        </span>
        <button
          className={`mini toggle${snapToGrid ? " on" : ""}`}
          aria-pressed={snapToGrid}
          onClick={() => setSnapToGrid(!snapToGrid)}
          title="snap a drawn box out to whole tiles; off cuts an exact element range"
        >
          snap
        </button>
        <button
          className={`mini toggle${axisMode === "symbolic" ? " on" : ""}`}
          aria-pressed={axisMode === "symbolic"}
          onClick={() => setAxisMode(axisMode === "symbolic" ? "numeric" : "symbolic")}
          disabled={!hasSemanticLabels}
          title="read compact shapes as semantic labels or numeric extents; hover a tensor name for all readings"
        >
          {axisMode === "symbolic" ? "labels" : "extents"}
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
        aria-valuetext={detail.labels[detail.index]}
        title={`global tile detail - ${detail.labels.join(" · ")}`}
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
const NOTE_SEVERITY = {
  1: { word: "advisory" },
  2: { word: "moderate" },
  3: { word: "strong" },
} as const;

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
}): React.ReactElement | null {
  // The directional sections already explain an absent/disabled selection.
  // Repeating that state here would make three parts of one panel teach the
  // same two questions in different words.
  if (!hasSelection || !findings) return null;
  const capped = findings.constraintCount > findings.notes.length;
  return (
    <section className="ins-section notes-section">
      <div className="ins-title">
        Dependency notes
        {focusedBox !== null && attributed && <span className="muted"> · tile {focusedBox + 1}</span>}
        {capped && (
          <span className="muted"> · {findings.notes.length} of {findings.constraintCount} constraints</span>
        )}
      </div>
      {findings.notes.length ? (
        <ul className="notes-list">
          {findings.notes.map((note) => (
            <li
              key={`${note.nodeId}:${note.text}`}
              className={`severity-${note.severity}`}
              title={`${NOTE_SEVERITY[note.severity].word} constraint · severity ${note.severity} of 3`}
            >
              <span className="sr-only">
                {NOTE_SEVERITY[note.severity].word} constraint, severity {note.severity} of 3. {" "}
              </span>
              <b>{note.op}</b>
              {note.text}
            </li>
          ))}
        </ul>
      ) : findings.elementwise ? (
        <p className="hint">
          Everything this tile needs is elementwise: any tiling of the selection fuses
          without cross-tile traffic.
        </p>
      ) : (
        <p className="hint">
          Nothing this tile needs reaches an operation, so no dependency constrains it.
        </p>
      )}
    </section>
  );
}

const EMPTY_REGION: Region = { boxes: [], exact: true, reasons: [] };

function FootprintBar({
  tensorId,
  direction,
  elements,
  totalElements,
  perBox,
  hiddenBoxes,
  focusedBox,
  dark,
}: {
  tensorId: string;
  direction: ConeDirection;
  elements: number;
  totalElements: number;
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  dark: boolean;
}): React.ReactElement {
  if (totalElements <= 0) return <span className="footprint-bar" />;

  const regions = perBox?.map(
    (prop, index) => hiddenBoxes.has(index)
      ? EMPTY_REGION
      : prop[direction]?.tensors.get(tensorId)?.region ?? EMPTY_REGION
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
  row.sliceExprs.filter((line) => !line.startsWith("#"));
type CopyFeedback = { key: string; state: "copied" | "failed" } | null;

/**
 * One tensor in one direction: how much of it the tile touches, why it has that
 * shape, and -downstream- whether the tile finishes it or only feeds it.
 */
function ConeRow({
  row,
  direction,
  hue,
  flags,
  contribution,
  perBox,
  hiddenBoxes,
  focusedBox,
  dark,
  onCopy,
  copyState,
}: {
  row: TensorReadout;
  direction: ConeDirection;
  hue: string;
  flags: string[];
  contribution?: Contribution;
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  dark: boolean;
  onCopy: () => void;
  copyState: "copied" | "failed" | null;
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
        <span className="badge depth" title={`${row.depth} step${row.depth === 1 ? "" : "s"} along this direction`}>
          d{row.depth}
        </span>
        <span className="row-stats">
          {share.toFixed(1)}% · {formatBytes(row.bytes)} · {row.boxCount} box{row.boxCount === 1 ? "" : "es"}
          {/* Boxes may overlap - two operand slots reading one tensor give two
              bands sharing a corner. Without this the listed boxes visibly sum
              past the element total and the row looks like it is miscounting,
              when in fact the total is the union and the shared elements are
              read twice. Stated only when there are some. */}
          {row.overlap > 0 && (
            <span
              className="row-shared"
              title={`${row.overlap.toLocaleString()} element${row.overlap === 1 ? "" : "s"} lie in more than one box, counted once in the total`}
            >
              {" "}· {row.overlap.toLocaleString()} shared
            </span>
          )}
        </span>
      </div>
      {perBox ? (
        <FootprintBar
          tensorId={row.tensorId}
          direction={direction}
          elements={row.elements}
          totalElements={row.totalElements}
          perBox={perBox}
          hiddenBoxes={hiddenBoxes}
          focusedBox={focusedBox}
          dark={dark}
        />
      ) : (
        <span
          className="cone-bar"
          title={`${fmt(row.elements)} of ${fmt(row.totalElements)} elements`}
          aria-label={`${share.toFixed(1)} percent of ${row.name}`}
        >
          <i style={{ width: `${Math.min(100, share)}%`, background: hue }} />
        </span>
      )}
      {flags.map((flag) => (
        <span className="cone-flag" key={flag}>
          {flag}
        </span>
      ))}
      {contribution?.partial && (
        <span className="cone-flag">
          partial : {contribution.detail}
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
      <button
        className={`mini copy-exprs${copyState === "failed" ? " copy-failed" : ""}`}
        title="copy this tensor's slice expressions"
        onClick={onCopy}
        aria-live="polite"
      >
        {copyState === "copied" ? "copied ✓" : copyState === "failed" ? "copy failed" : "copy"}
      </button>
    </div>
  );
}

function ConeSection({
  direction,
  arrow,
  title,
  rows,
  hue,
  flags,
  contributions: contrib,
  enabled,
  onToggle,
  empty,
  copiedKey,
  copyFeedback,
  perBox,
  hiddenBoxes,
  focusedBox,
  dark,
  onCopy,
}: {
  direction: ConeDirection;
  arrow: string;
  title: string;
  rows: TensorReadout[];
  hue: string;
  flags: Map<string, string[]>;
  contributions?: Map<string, Contribution>;
  enabled: boolean;
  onToggle: () => void;
  empty: string;
  copiedKey: string | null;
  copyFeedback: "copied" | "failed" | null;
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  dark: boolean;
  onCopy: (row: TensorReadout, key: string) => void;
}): React.ReactElement {
  const bytes = rows.reduce((total, row) => total + row.bytes, 0);
  const verdicts = contrib ? [...contrib.values()] : [];
  const partial = verdicts.filter((item) => item.partial).length;
  const completed = verdicts.length - partial;

  return (
    <section className="cone-section">
      <div className="cone-head">
        <button
          className={`cone-toggle${enabled ? " on" : ""}`}
          onClick={onToggle}
          aria-pressed={enabled}
          aria-expanded={enabled}
          title={`${enabled ? "hide" : "show"} ${title.toLowerCase()} (${direction === "backward" ? "u" : "d"})`}
        >
          <span
            className={`cone-key ${direction === "backward" ? "needs" : "feeds"}`}
            aria-hidden
          />
          <span className="cone-arrow" aria-hidden>{arrow}</span>
          <span>{title}</span>
        </button>
        <span className="rollup">
          {rows.length} tensor{rows.length === 1 ? "" : "s"}
          {verdicts.length > 0 && ` · ${completed} completed, ${partial} partial`}
          {` · ${formatBytes(bytes)}`}
        </span>
      </div>
      {enabled &&
        (rows.length ? (
          rows.map((row) => {
            const key = `${title}:${row.tensorId}`;
            return (
              <ConeRow
                key={row.tensorId}
                row={row}
                direction={direction}
                hue={hue}
                flags={flags.get(row.tensorId) ?? []}
                contribution={contrib?.get(row.tensorId)}
                perBox={perBox}
                hiddenBoxes={hiddenBoxes}
                focusedBox={focusedBox}
                dark={dark}
                copyState={copiedKey === key ? copyFeedback : null}
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
 * The analysed tile set and its distinct element count. Ranges are edited only
 * in the tile list, including when the workspace contains a single tile.
 */
function TileIdentity({
  onCopyAll,
  copyState,
  activeTensorId,
  focusedBox,
}: {
  onCopyAll: () => void;
  copyState: "copied" | "failed" | null;
  activeTensorId: string | null;
  focusedBox: number | null;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved)!;
  const selection = useStore((s) => s.selection);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const perBox = useStore((s) => s.perBox);
  const theme = useStore((s) => s.theme);

  if (!selection) return null;
  const anchorId = activeTensorId;
  if (!anchorId) return null;
  /**
   * The header describes the same group as everything under it, so its counts
   * come from the active tensor's tiles rather than from the whole selection.
   * A tile on another tensor is in the list above, not in this answer.
   */
  const group = partsOn(selection, anchorId);
  const index = focusedBox ?? group[group.length - 1]?.index;
  const part = selection.parts[index];
  if (part === undefined || index === undefined) return null;
  const ordinal = group.findIndex((row) => row.index === index);

  // With several tiles in the group and none focused, everything below
  // describes their merged cone. Naming one of them here would put the wrong
  // tile at the top of a readout about all of them.
  const merged = group.length > 1 && focusedBox === null;
  const tensor = resolved.tensors[anchorId];
  const shape = tensor.resolved!;
  const { rowAxis, colAxis } = viewAxes(shape);
  const axisLabel = (axis: number) => tensor.axisNames?.[axis] ?? `ax${axis}`;
  const measured = measuredParts(selection.parts, anchorId, hiddenBoxes, focusedBox, !!perBox);
  const elements = measuredElements(measured);

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
          {measured.length === 0 ? "no enabled tiles" : merged
            ? `${measured.length} of ${group.length} tiles · merged`
            : `tile ${ordinal + 1} of ${group.length}`}
        </span>
        <button
          className={`mini copy-cone${copyState === "failed" ? " copy-failed" : ""}`}
          onClick={onCopyAll}
          title="copy slice expressions for everything this tile needs and feeds"
          aria-live="polite"
        >
          {copyState === "copied" ? "copied ✓" : copyState === "failed" ? "copy failed" : "copy all"}
        </button>
      </div>
      <div className="tile-name">
        <b
          title={`rows ${axisLabel(rowAxis >= 0 ? rowAxis : 0)} · cols ${axisLabel(colAxis >= 0 ? colAxis : 0)} · shape [${shape.join("×")}]`}
        >
          {tensor.name}
        </b>
        <span className="tile-count">{fmt(elements)} el</span>
      </div>
    </header>
  );
}

/**
 * The selection's tiles, one row each, including a single tile. Hovering a row
 * emphasises that tile's cone across the whole graph; clicking pins it.
 */
function RegionEditor({ activeTensorId, onSelectGroup }: {
  activeTensorId: string | null;
  onSelectGroup: (tensorId: string) => void;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const replaceBox = useStore((s) => s.replaceBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const focusedBox = useStore((s) => s.focusedBox);
  const pinned = useStore((s) => s.pinnedBox);
  const hoverBox = useStore((s) => s.hoverBox);
  const togglePinBox = useStore((s) => s.togglePinBox);
  const toggleBoxHidden = useStore((s) => s.toggleBoxHidden);
  const theme = useStore((s) => s.theme);

  if (!resolved || !selection || selection.parts.length === 0) return null;
  const parts = selection.parts;
  const dark = theme === "dark";
  /**
   * Tiles in drawn order, split into one group per tensor.
   *
   * The split is the semantics made visible: everything below this list
   * analyses one group, so the list has to show where a group ends. Order
   * follows first appearance, which keeps a row from jumping to a new place
   * when a second tile lands on a tensor already in the list.
   */
  const groups = selectedTensorIds(selection).map((tensorId) => ({
    tensorId,
    rows: partsOn(selection, tensorId),
  }));

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
      {!perBox && <p className="hint">too many tiles to trace individually (over {MAX_PER_BOX_PROPS}), showing merged needs and feeds</p>}
      {overlap.summed > overlap.unique && (
        <p className="hint overlap">
          tiles overlap: {fmt(overlap.summed)} counted across parts,{" "}
          <b>{fmt(overlap.unique)}</b> distinct elements; both analyses use the deduplicated set
        </p>
      )}

      <div className="box-list" onMouseLeave={() => hoverBox(null)}>
        {groups.map((group) => (
          <div
            className={`tile-group${group.tensorId === activeTensorId ? " active" : ""}`}
            key={group.tensorId}
          >
            {groups.length > 1 && (
              <div className="tile-group-head">
                <button className="mini" aria-pressed={group.tensorId === activeTensorId}
                  onClick={() => onSelectGroup(group.tensorId)}
                  title="analyse this tensor's enabled tiles together">
                  <b>{resolved.tensors[group.tensorId].name}</b>
                </button>
                <span className="muted">
                  {group.rows.length} tile{group.rows.length === 1 ? "" : "s"}
                </span>
                {group.tensorId === activeTensorId ? (
                  <span className="tile-group-mark">analysed below</span>
                ) : (
                  <span className="tile-group-mark muted">select group or tile</span>
                )}
              </div>
            )}
        {group.rows.map(({ index: i, box: b }) => {
          const tensorId = group.tensorId;
          const active = focusedBox === i;
          const hidden = hiddenBoxes.has(i);
          const t = resolved.tensors[tensorId];
          const shape = t.resolved!;
          return (
            <div
              className={`box-row${active ? " active" : ""}${pinned === i ? " pinned" : ""}${hidden ? " hidden-cone" : ""}`}
              key={i}
              onMouseEnter={() => hidden ? hoverBox(null) : hoverBox(i)}
              onClick={() => perBox && !hidden && togglePinBox(i)}
              title={perBox && !hidden ? "hover to emphasise this tile's analysis; click to pin (Esc unpins)" : undefined}
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
                    ? hidden
                      ? "include this tile in the merged analysis and graph highlights"
                      : "exclude this tile from the merged analysis and graph highlights (h)"
                    : "per-tile needs and feeds are not traced at this many tiles"
                }
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBoxHidden(i);
                }}
                aria-label={`${hidden ? "include" : "exclude"} tile ${i + 1}`}
                aria-pressed={!hidden}
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
                className="mini danger"
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
        ))}
      </div>
      {parts.length > MAX_DISTINCT_HUES && (
        <p className="hint">
          tiles past the {MAX_DISTINCT_HUES}rd share a neutral color, only {MAX_DISTINCT_HUES} hues stay
          distinguishable side by side, so use hover to tell the rest apart.
        </p>
      )}
    </div>
  );
}

/** Nothing drawn yet: teach the three relations instead of rendering empty tables. */
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
        panel then answers three questions about it.
      </p>
      <dl className="empty-questions">
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↑
            </span>
            Backward Cone
          </dt>
          <dd>Everything upstream the tile requires.</dd>
        </div>
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↓
            </span>
            Forward Cone
          </dt>
          <dd>Everything downstream the tile affects.</dd>
        </div>
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ×
            </span>
            Co-access Surface
          </dt>
          <dd>
            Everything on the same level as the tile that is co-accessed with it. 
            Not a hop along the graph, press <kbd>e</kbd>.
          </dd>
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
                {resolved.tensors[start.tensorId].name} - {start.label}
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
  const byTensorRes = useStore((s) => s.byTensorRes);
  const direction = useStore((s) => s.direction);
  const toggleDirection = useStore((s) => s.toggleDirection);
  const showEntangled = useStore((s) => s.showEntangled);
  const toggleEntangled = useStore((s) => s.toggleEntangled);
  const entangled = useStore((s) => s.entangled);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const tileFocus = useStore((s) => s.focusedBox);
  /**
   * The tile group everything below the tiles list describes, and the focus
   * that is allowed to narrow it.
   *
   * The group is store state named by a draw, a pin, or the group header, so a
   * tile drawn on another tensor moves the readout with it. Focus narrows
   * within that group and stops there: `groupFocus` is null while the pointer
   * is over another group's row, which leaves that row lit on the canvas
   * without re-scoping the panel under it.
   */
  const selectedGroup = useStore((s) => s.analysisGroup);
  // Above the attribution cap, tile hover must not masquerade as a single-tile readout.
  const rawFocus = perBox ? tileFocus : null;
  // Stable across renders, so the memos below key off the selection itself
  // rather than off a fresh empty array.
  const parts = useMemo(() => selection?.parts ?? [], [selection]);
  const activeTensorId = analysisTensorId(parts, selectedGroup);
  /** Everything below the tiles list reads the narrowed focus, never the raw
   *  one, so a hover outside the group cannot reach any of it. */
  const focusedBox = groupFocus(parts, rawFocus, activeTensorId);
  const scopedAttribution = useMemo(() => groupAttribution(perBox, parts, activeTensorId),
    [perBox, parts, activeTensorId]);
  const selectGroup = useStore((s) => s.selectAnalysisGroup);

  const [reuse, setReuse] = useState<ReuseEstimate[] | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<CopyFeedback>(null);

  const {
    metrics,
    bounds,
    findings,
    seeds,
    contribution: contrib,
    upstream,
    downstream,
  } = useInspectorAnalysis({
    resolved,
    selection,
    perBox,
    byTensorRes,
    hiddenBoxes,
    focusedBox,
    activeTensorId,
    direction,
  });

  useEffect(() => setReuse(null), [resolved, selection, focusedBox, hiddenBoxes, activeTensorId]);

  if (!resolved) return <aside className="inspector" />;

  /**
   * Everything below the tiles list is counted over the active group, not over
   * the whole selection: a tile on another tensor is not part of this answer
   * and must not appear in its totals or its "N tiles" wording.
   */
  const groupParts = activeTensorId ? partsOn(selection, activeTensorId) : [];
  const selBoxes = groupParts.length;
  const enabledBoxes = groupParts.filter((part) => !hiddenBoxes.has(part.index)).length;
  const dark = theme === "dark";
  /** Past the cap the cones are still correct, but merged rather than attributed.
   *  The cap counts the whole selection, because that is what `perBox` tracks. */
  const merged = !perBox;
  const aggregate = aggregateColors(dark);
  /** A bar carries the hue of the tile it belongs to, or the aggregate when the
   *  rows describe several tiles at once. */
  const coneHue = (cone: "upstream" | "downstream") => {
    if (focusedBox !== null) return rgbCss(boxColor(focusedBox, dark));
    if (selBoxes === 1 && !merged) return rgbCss(boxColor(groupParts[0].index, dark));
    return rgbCss(cone === "upstream" ? aggregate.upstream : aggregate.downstream);
  };

  // Both questions keep their heading in the inspector; the filter collapses
  // its rows and canvas paint. The store refuses to collapse the final one.
  const showUpstream = direction === "backward" || direction === "both";
  const showDownstream = direction === "forward" || direction === "both";

  // An empty section has two causes, and they are different findings: the
  // selection sits at the edge of the graph, or what it reaches is selected too
  // and is therefore not something it reads or feeds.
  const seedIds = [...seeds.keys()];
  const upstreamEmpty = enabledBoxes === 0
    ? "No tiles are enabled, include one above to analyse it."
    : seedIds.some((id) => resolved.tensors[id].producer)
      ? "Everything the selection needs is itself selected."
      : "Every selected tensor is a graph input, it needs nothing earlier.";
  const downstreamEmpty = enabledBoxes === 0
    ? "No tiles are enabled, include one above to analyse it."
    : seedIds.some((id) => resolved.consumers[id]?.length)
      ? "Everything the selection feeds is itself selected."
      : "Nothing consumes this selection, it feeds no later tensor.";

  const visibleRows = [...(showUpstream ? upstream : []), ...(showDownstream ? downstream : [])];
  const approxRows = visibleRows.filter((row) => !row.exact).length;
  const approxReasons = [
    ...new Set(visibleRows.filter((row) => !row.exact).flatMap((row) => row.reasons)),
  ];
  /* One row per entangled region, skipping tiles the user hid so the list and
     the paint describe the same set. */
  const entangledRows = (entangled ?? []).flatMap((forPart, index) =>
    parts[index]?.tensorId !== activeTensorId || hiddenBoxes.has(index) || (focusedBox !== null && focusedBox !== index)
      ? []
      : forPart.map((e) => ({
          name: resolved?.tensors[e.tensorId]?.name ?? e.tensorId,
          op: e.op,
          region: e.region,
        }))
  );

  /**
   * What the Cost figures actually cover, named in the heading.
   *
   * Merged is the common case and it was the one the heading got wrong: with
   * several tiles and no focus every figure describes their *union*, and the
   * heading still read "this tile". Past the attribution cap `hiddenBoxes` is
   * not honoured either (`enabledPropResult` returns the whole aggregate), so
   * the count has to come from what was measured rather than from what is
   * switched on.
   */
  const measuredBoxes = perBox ? enabledBoxes : selBoxes;
  const costScope =
    focusedBox !== null && perBox
      ? `tile ${focusedBox + 1}`
      : measuredBoxes > 1
        ? `${measuredBoxes === selBoxes ? measuredBoxes : `${measuredBoxes} of ${selBoxes}`} tiles · merged`
        : "this tile";

  /** Prefix for a cost figure measured over an over-approximated region. */
  const bound = metrics && !metrics.exact ? "\u2264\u202f" : "";
  /**
   * Intensity takes `~`, not `\u2264`. Every other figure is a sum over the cone's
   * regions, so widening a region can only raise it. Intensity is a ratio of
   * two such sums, and widening moves both: extra elements that carry no work
   * pull it down, extra work over bytes already counted pushes it up. The
   * figure is disturbed in an unknown direction, which is what `~` says and
   * `\u2264` would misstate.
   */
  const ratioBound = metrics && !metrics.exact ? "~\u202f" : "";

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. The sweep
   * is defined by one tile on one tensor, so it follows the anchor part (the
   * focused one) else the last drawn rather than mixing tensors. */
  const computeReuse = () => {
    if (!selection || !resolved) return;
    const fallback = selection.parts
      .map((_, index) => index)
      .filter((index) => selection.parts[index].tensorId === activeTensorId && !hiddenBoxes.has(index))
      .pop();
    const probeIndex = focusedBox !== null && !hiddenBoxes.has(focusedBox)
      ? focusedBox
      : fallback;
    const probe = probeIndex === undefined ? null : selection.parts[probeIndex];
    if (!probe) return;
    setReuse(estimateInputReuse(resolved, { tensorId: probe.tensorId, region: fromBox(probe.box) }));
  };

  const copy = async (text: string, key: string) => {
    setCopyFeedback({ key, state: (await copyText(text)) ? "copied" : "failed" });
    setTimeout(() => setCopyFeedback(null), 1600);
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
            <TileIdentity
              activeTensorId={activeTensorId}
              focusedBox={focusedBox}
              onCopyAll={copyCone}
              copyState={copyFeedback?.key === "cone" ? copyFeedback.state : null}
            />
            {/* The tiles list is the selector for everything below it: it picks
                which cone the two sections describe, so it sits above them. */}
            <RegionEditor activeTensorId={activeTensorId} onSelectGroup={selectGroup} />

            <ConeSection
              direction="backward"
              arrow="↑"
              title="Backward Cone"
              rows={upstream}
              hue={coneHue("upstream")}
              flags={findings?.flags ?? new Map()}
              enabled={showUpstream}
              onToggle={() => toggleDirection("backward")}
              empty={upstreamEmpty}
              copiedKey={copyFeedback?.key ?? null}
              copyFeedback={copyFeedback?.state ?? null}
              perBox={scopedAttribution}
              hiddenBoxes={hiddenBoxes}
              focusedBox={focusedBox}
              dark={dark}
              onCopy={copyRow}
            />
            <ConeSection
              direction="forward"
              arrow="↓"
              title="Forward Cone"
              rows={downstream}
              hue={coneHue("downstream")}
              flags={new Map()}
              contributions={contrib?.byTensor}
              enabled={showDownstream}
              onToggle={() => toggleDirection("forward")}
              empty={downstreamEmpty}
              copiedKey={copyFeedback?.key ?? null}
              copyFeedback={copyFeedback?.state ?? null}
              perBox={scopedAttribution}
              hiddenBoxes={hiddenBoxes}
              focusedBox={focusedBox}
              dark={dark}
              onCopy={copyRow}
            />
            {direction === "none" && (
              <p className="view-mode-note" role="status">
                Figures only · Backward & Forward cones are hidden.
              </p>
            )}
            {showDownstream && contrib?.capped && (
              <p className="hint">
                more than {MAX_CONTRIBUTION_PROBES} tensors are fed: rows below do not say
                whether this tile completes them or only feeds them
              </p>
            )}

            {/* A third relation, so a third section rather than rows folded into
                one of the cones: "what this tile is multiplied against" is not
                a hop along the graph and does not belong under a direction. */}
            <section className="cone-section">
              <div className="cone-head">
                <button
                  className={`cone-toggle${showEntangled ? " on" : ""}`}
                  onClick={toggleEntangled}
                  aria-pressed={showEntangled}
                  aria-expanded={showEntangled}
                  title={`${showEntangled ? "hide" : "show"} what it is combined with (e)`}
                >
                  <span className="cone-key combined" aria-hidden="true" />
                  Co-access Surface
                </button>
              </div>
              {showEntangled &&
                (merged ? (
                  <p className="hint">
                    too many tiles to trace individually (over {MAX_PER_BOX_PROPS})
                  </p>
                ) : entangledRows.length ? (
                  <ul className="ent-list">
                    {entangledRows.map((row, i) => (
                      <li key={i}>
                        <code>
                          {row.name}[{formatBoxIndices(row.region.boxes[0])}
                          {row.region.boxes.length > 1 ? ", …" : ""}]
                        </code>
                        <span className="muted"> via {row.op}</span>
                        {!row.region.exact && (
                          <span className="badge approx" title={row.region.reasons.join("; ")}>
                            ≈
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">
                    Nothing.
                  </p>
                ))}
            </section>

            {approxReasons.length > 0 && (
              /* Contract A1 asks this section to name a *reason*. Counting the
                 badges restated something already on screen and left the
                 reasons in a title, which is the one place a mouse-less reader
                 never reaches. The count stays because it says how far to
                 look; the reasons are now the content. */
              <div className="ins-section warn">
                ≈ {approxRows} row{approxRows === 1 ? "" : "s"} over-approximated:{" "}
                {approxReasons.join(", ")}
              </div>
            )}

            {metrics && (
              <>
                {/* These figures are bounds when any region they were measured
                    on was widened, and the relation is exactly "no more than" -
                    a region is never a subset of the truth. `<=` says that;
                    `~` would suggest the number could also be an overestimate
                    in the other direction, which it cannot. */}
                <div className="ins-section">
                  <div className="ins-title">
                    {`Cost to compute ${costScope} `}
                    <span
                      className="muted"
                      role="img"
                      tabIndex={0}
                      aria-label="Idealized estimates, not hardware bounds. Fused assumes perfect sharing; unfused counts separate per-op reads and writes."
                      title="Idealized estimates, not hardware bounds. Fused assumes perfect sharing; unfused counts separate per-op reads and writes."
                    > ⓘ</span>
                    {/* Every figure below is measured over the cone's regions,
                        so an over-approximated region makes all of them upper
                        bounds. The rows already say so individually; without
                        this the totals were the one place a bound was printed
                        as a count. */}
                    {!metrics.exact && (
                      <span className="badge approx" title={metrics.reasons.join("; ")}>≈</span>
                    )}
                  </div>
                  <div className="kv">
                    <span>FLOPs</span><span>{bound}{fmt(metrics.flops)}</span>
                    <span>input bytes</span><span>{bound}{formatBytes(metrics.inputBytes)}</span>
                    <span>intermediate</span><span>{bound}{formatBytes(metrics.intermediateBytes)}</span>
                    <span>output bytes</span><span>{bound}{formatBytes(metrics.outputBytes)}</span>
                    <span>working set</span>
                    <span>{bound}{formatBytes(metrics.inputBytes + metrics.intermediateBytes + metrics.outputBytes)}</span>
                    {/* Two scenarios across both operations and tiles, not a
                        guaranteed interval for a real kernel. */}
                    <span title={
                      bounds
                        ? `one kernel over every op and every tile: intermediates never reach memory and a shared operand band is fetched once - ${fmt(bounds.fused.flops)} FLOP / ${formatBytes(bounds.fused.bytes)}`
                        : undefined
                    }>
                      intensity · fused
                    </span>
                    <span>
                      {bounds && bounds.fused.bytes > 0
                        ? `${ratioBound}${(bounds.fused.flops / bounds.fused.bytes).toFixed(2)} FLOP/B`
                        : "—"}
                    </span>
                    <span title={
                      bounds?.unfused
                        ? `every op of every tile as its own job: separate input reads and output writes, materialized views, and no cross-op cache reuse - ${fmt(bounds.unfused.flops)} FLOP / ${formatBytes(bounds.unfused.bytes)}`
                        : `per-tile cones are not traced past ${MAX_PER_BOX_PROPS} tiles, and the merged result cannot be taken apart again`
                    }>
                      intensity · unfused
                    </span>
                    <span>
                      {bounds?.unfused && bounds.unfused.bytes > 0
                        ? `${ratioBound}${(bounds.unfused.flops / bounds.unfused.bytes).toFixed(2)} FLOP/B`
                        : "—"}
                    </span>
                  </div>
                  {!metrics.exact && (
                    <p className="hint overlap">
                      Bounds, not counts: {metrics.reasons.join(", ")} widened a
                      region these figures were measured on. Read every ≤ as “no
                      more than” - those are never understated. The two ~
                      intensities are ratios of such figures, so they are
                      disturbed in an unknown direction.
                    </p>
                  )}
                </div>
                <div className="ins-section">
                  <div className="ins-title">
                    Reuse{" "}
                    <button
                      className="mini"
                      onClick={computeReuse}
                      title="sample selection-sized output tiles across the graph and count how many touch each input's current footprint"
                    >
                      estimate
                    </button>
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
                    <p className="hint">sampled sweep · run on demand</p>
                  )}
                </div>
              </>
            )}
            <DependencyNotes
              findings={findings}
              hasSelection={enabledBoxes > 0}
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
