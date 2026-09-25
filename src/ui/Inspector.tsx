import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Contribution, MAX_CONTRIBUTION_PROBES } from "../core/contribution";
import type { ResolvedGraph } from "../core/graph";
import { ratioFigure, sumFigures } from "../core/metrics";
import { TensorReadout } from "../core/metrics";
import type { ConeFindings } from "../core/notes";
import { estimateInputReuseSweep, ReuseEstimate, type ReuseSweep } from "../core/reuse";
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
import { formatBytes, formatFigure } from "./format";
import type { ConeCost } from "./inspector-analysis";
import { analysisTensorId, groupAttribution, groupFocus, measuredParts, measuredElements, useInspectorAnalysis } from "./inspector-analysis";
import { aggregateColors, boxColor, MAX_DISTINCT_HUES, rgbCss } from "./palette";
import { PlanPanel } from "./PlanPanel";
import {
  ConeDirection,
  InspectorTab,
  MAX_PER_BOX_PROPS,
  partsOn,
  selectedTensorIds,
  startingTiles,
  useDark,
  useStore,
} from "./store";
import { formatSelectionBox, parseSelectionBox } from "./selection-range";
import { SHORTCUTS } from "./shortcuts";
import { CopyButton } from "./CopyButton";
import { viewAxes } from "./tensor-view";
import {
  analysisWorkerAvailable,
  isAnalysisCancelled,
  reuseInWorker,
} from "./analysis-worker-client";

function fmt(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(n < 10 && !Number.isInteger(n) ? 2 : 0);
}

function SelectionRangeInput({
  box,
  shape,
  label,
  onCommit,
}: {
  box: Box;
  shape: number[];
  label: string;
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
      aria-label={label}
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

/** Past the cap there is no per-tile propagation left to read, and three places
 *  used to say so in three different ways. */
const MERGED_AT_CAP = `over ${MAX_PER_BOX_PROPS} tiles: traced as one merged region, not per tile`;
/** Both cones report an empty selection the same way. */
const NO_ENABLED_TILES = "No tiles are enabled, include one above to analyse it.";

const EMPTY_REGION: Region = { boxes: [], exact: true, reasons: [] };

/**
 * How a readout is attributed to individual tiles: the per-tile propagation
 * when there is one, which tiles are switched off, which one is focused, and
 * the theme the tile hues are mixed for. The footprint bar, the row around it
 * and the section around that all need the same four, so they travel together
 * rather than being restated at every level.
 */
type TileAttribution = {
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  dark: boolean;
};

function FootprintBar({
  tensorId,
  direction,
  elements,
  totalElements,
  attr: { perBox, hiddenBoxes, focusedBox, dark },
}: {
  tensorId: string;
  direction: ConeDirection;
  elements: number;
  totalElements: number;
  attr: TileAttribution;
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

/**
 * The local half of the reuse answer: what one tile's step along each axis still
 * shares with this tile. Both directions usually agree, and printing "50/50%"
 * for that would be noise, so equal shares collapse to one figure.
 */
export function neighbourShares(
  neighbors: ReuseEstimate["neighbors"],
  label: (axis: number) => string
): { text: string; reasons: string[] } {
  const byAxis = new Map<number, ReuseEstimate["neighbors"]>();
  const reasons = new Set<string>();
  for (const probe of neighbors) {
    const shares = byAxis.get(probe.axis) ?? [];
    shares.push(probe);
    byAxis.set(probe.axis, shares);
    if (!probe.exact) probe.reasons.forEach((reason) => reasons.add(reason));
  }
  const text = [...byAxis]
    .map(([axis, probes]) => {
      const values = probes.map((probe) =>
        `${probe.exact ? "" : "~"}${Math.round(probe.sharedFraction * 100)}%`
      );
      const distinct = [...new Set(values)];
      if (distinct.length === 1) return `${label(axis)} ${distinct[0]}`;
      return `${label(axis)} ${probes.map((probe, index) =>
        `${probe.delta < 0 ? "−" : "+"}${values[index]}`
      ).join(" / ")}`;
    })
    .join(" · ");
  return { text, reasons: [...reasons] };
}

/** Sampling and conservative geometry are independent sources of uncertainty. */
export function reuseQualifiers(
  estimate: Pick<ReuseEstimate, "exhaustive" | "geometryExact">
): { count: string; fraction: string } {
  return {
    count: estimate.exhaustive ? (estimate.geometryExact ? "" : "≤ ") : "~ ",
    // A ratio of widened regions has no one-sided bound. Sampling likewise
    // makes the reported mean an estimate even when every region is exact.
    fraction: estimate.exhaustive && estimate.geometryExact ? "" : "~ ",
  };
}

type ReuseProbe = { tensorId: string; box: Box; colorIndex?: number };
type ReuseRun = {
  graph: ResolvedGraph;
  probe: ReuseProbe;
  rows: ReuseEstimate[];
  sweep?: ReuseSweep;
};

const sameBox = (left: Box, right: Box) =>
  left.length === right.length && left.every(
    (interval, axis) => interval.lo === right[axis].lo && interval.hi === right[axis].hi
  );

/**
 * A reuse sweep is a cache keyed by the graph and the exact tile that seeded
 * it. Effect-based clearing is too late: React renders once with the new graph
 * before an effect runs, and an old input ID can crash that render.
 */
export function currentReuseRows(
  run: ReuseRun | null,
  graph: ResolvedGraph | null,
  probe: ReuseProbe | null
): ReuseEstimate[] | null {
  return run && graph === run.graph && probe &&
    probe.tensorId === run.probe.tensorId && sameBox(probe.box, run.probe.box)
    ? run.rows
    : null;
}

/** Slice expressions minus the over-approximation comment the readout appends. */
const sliceLines = (row: TensorReadout) =>
  row.sliceExprs.filter((line) => !line.startsWith("#"));

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
  attr,
}: {
  row: TensorReadout;
  direction: ConeDirection;
  hue: string;
  flags: string[];
  contribution?: Contribution;
  attr: TileAttribution;
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
          {share.toFixed(1)}% · {formatFigure(row.byteFigure, formatBytes)} · {row.boxCount} box{row.boxCount === 1 ? "" : "es"}
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
      {attr.perBox ? (
        <FootprintBar
          tensorId={row.tensorId}
          direction={direction}
          elements={row.elements}
          totalElements={row.totalElements}
          attr={attr}
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
      <CopyButton
        className="mini copy-exprs"
        title="copy this tensor's slice expressions"
        text={() => exprs.join("\n")}
        label="copy"
      />
    </div>
  );
}

/* One direction, one identity: the heading, the glyph, the paint and the key
   that toggles it all follow from `direction`, so a call site names only the
   direction. The shortcut letters come from the manifest that binds them. */
const CONE = {
  backward: { title: "Backward Cone", arrow: "↑", paint: "needs", key: SHORTCUTS.needs },
  forward: { title: "Forward Cone", arrow: "↓", paint: "feeds", key: SHORTCUTS.feeds },
} as const satisfies Record<ConeDirection, unknown>;

/* The two classes of question the panel answers, and the promise each one
   makes. A figure under Dependencies is a function of the graph and the drawn
   region - exact, or a bound that names why. A figure under Execution exists
   only once an execution is assumed, so it is modelled and could be wrong in
   either direction; mixing the two in one column would lend the models the
   others' credibility. */
const TABS = [
  {
    id: "dependencies",
    label: "Dependencies",
    hint: "what the drawn tiles need, feed and share: exact, or bounded with its reason named",
  },
  {
    id: "execution",
    label: "Execution",
    hint: "what an assumed execution would do with them: modelled, not bounded",
  },
  /* A third class, and a third promise. A plan figure rests on a declared
     tiling rather than on the drawn region, and it is exact for that tiling or
     bounded with its reason - so it belongs with neither the figures that need
     no plan nor the ones that are modelled. */
  {
    id: "plan",
    label: "Plan",
    hint: "how a declared tiling divides the work: exact for that tiling, or bounded",
  },
] as const satisfies readonly { id: InspectorTab; label: string; hint: string }[];

/**
 * A two-button view switch, deliberately not the ARIA tab pattern. Tabs would
 * promise roving focus and arrow navigation, while arrows already move the
 * tile this panel describes. Both buttons remain ordinary tab stops and expose
 * their current state through `aria-pressed`.
 */
function InspectorTabs(): React.ReactElement {
  const tab = useStore((s) => s.inspectorTab);
  const setTab = useStore((s) => s.setInspectorTab);
  return (
    <div className="ins-tabs" role="group" aria-label="analysis class">
      {TABS.map(({ id, label, hint }) => (
        <button
          key={id}
          id={`ins-tab-${id}`}
          className="ins-tab"
          aria-pressed={tab === id}
          aria-controls={`ins-panel-${id}`}
          title={hint}
          onClick={() => setTab(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function ConeSection({
  direction,
  rows,
  hue,
  flags,
  contributions: contrib,
  enabled,
  onToggle,
  empty,
  attr,
}: {
  direction: ConeDirection;
  rows: TensorReadout[];
  hue: string;
  flags: Map<string, string[]>;
  contributions?: Map<string, Contribution>;
  enabled: boolean;
  onToggle: () => void;
  empty: string;
  attr: TileAttribution;
}): React.ReactElement {
  const { title, arrow, paint, key } = CONE[direction];
  const bytes = sumFigures(rows.map((row) => row.byteFigure));
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
          title={`${enabled ? "hide" : "show"} ${title.toLowerCase()} (${key.keys[0]})`}
        >
          <span className={`cone-key ${paint}`} aria-hidden />
          <span className="cone-arrow" aria-hidden>{arrow}</span>
          <span>{title}</span>
        </button>
        <span className="rollup">
          {rows.length} tensor{rows.length === 1 ? "" : "s"}
          {verdicts.length > 0 && ` · ${completed} completed, ${partial} partial`}
          {` · ${formatFigure(bytes, formatBytes)}`}
        </span>
      </div>
      {enabled &&
        (rows.length ? (
          rows.map((row) => (
            <ConeRow
              key={row.tensorId}
              row={row}
              direction={direction}
              hue={hue}
              flags={flags.get(row.tensorId) ?? []}
              contribution={contrib?.get(row.tensorId)}
              attr={attr}
            />
          ))
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
  coneExprs,
  activeTensorId,
  focusedBox,
}: {
  coneExprs: () => string;
  activeTensorId: string | null;
  focusedBox: number | null;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved)!;
  const selection = useStore((s) => s.selection);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const perBox = useStore((s) => s.perBox);
  const dark = useDark();

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
          <i className="swatch" style={{ background: rgbCss(boxColor(index, dark)) }} />
        )}
        <span>
          {measured.length === 0 ? "no enabled tiles" : merged
            ? `${measured.length} of ${group.length} tiles · merged`
            : `tile ${ordinal + 1} of ${group.length}`}
        </span>
        <CopyButton
          className="mini copy-cone"
          title="copy slice expressions for everything this tile needs and feeds"
          text={coneExprs}
          label="copy all"
        />
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
  const clearSelection = useStore((s) => s.clearSelection);
  const toggleBoxHidden = useStore((s) => s.toggleBoxHidden);
  const dark = useDark();

  if (!resolved || !selection || selection.parts.length === 0) return null;
  const parts = selection.parts;
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
      <div className="ins-title with-action">
        Tiles
        <button className="mini" onClick={clearSelection} title="remove every tile from the selection">
          clear all
        </button>
      </div>
      {!perBox && <p className="hint">{MERGED_AT_CAP}</p>}
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
                {group.tensorId === activeTensorId && (
                  <span className="tile-group-mark">analysed below</span>
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
                <button
                  className="box-label box-pin"
                  disabled={!perBox || hidden}
                  aria-label={`${pinned === i ? "unpin" : "pin"} tile ${i + 1} on ${t.name}`}
                  aria-pressed={pinned === i}
                  title={pinned === i
                    ? "unpin this tile so the group is analysed together"
                    : "pin this tile so the readout follows it"}
                  onClick={(event) => {
                    event.stopPropagation();
                    togglePinBox(i);
                  }}
                >
                  <b>{t.name}</b> · {fmt(b.reduce((a, I) => a * (I.hi - I.lo), 1))} elements
                </button>
                <SelectionRangeInput
                  box={b}
                  shape={shape}
                  label={`selection range for tile ${i + 1} on ${t.name}`}
                  onCommit={(next) => replaceBox(i, next)}
                />
              </span>
              <button
                className="mini danger"
                aria-label={`remove tile ${i + 1} from ${t.name}`}
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
            Not a hop along the graph; press <kbd>e</kbd> to show it.
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
  const dark = useDark();
  const selection = useStore((s) => s.selection);
  const byTensorRes = useStore((s) => s.byTensorRes);
  const direction = useStore((s) => s.direction);
  const toggleDirection = useStore((s) => s.toggleDirection);
  const showEntangled = useStore((s) => s.showEntangled);
  const toggleEntangled = useStore((s) => s.toggleEntangled);
  const entangled = useStore((s) => s.entangled);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const tab = useStore((s) => s.inspectorTab);
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
  const selectGroup = useStore((s) => s.selectAnalysisGroup);

  /** The one enabled tile that defines a sweep. Keeping this derivation beside
   *  the cache key prevents its button and its displayed result from choosing
   *  subtly different fallbacks. */
  const reuseProbe = useMemo<ReuseProbe | null>(() => {
    if (!selection || !activeTensorId) return null;
    const fallback = selection.parts
      .map((_, index) => index)
      .filter((index) =>
        selection.parts[index].tensorId === activeTensorId && !hiddenBoxes.has(index)
      )
      .pop();
    const probeIndex = focusedBox !== null && !hiddenBoxes.has(focusedBox)
      ? focusedBox
      : fallback;
    const probe = probeIndex === undefined ? null : selection.parts[probeIndex];
    return probe ? { tensorId: probe.tensorId, box: probe.box, colorIndex: probeIndex! } : null;
  }, [selection, activeTensorId, hiddenBoxes, focusedBox]);
  const [reuseRun, setReuseRun] = useState<ReuseRun | null>(null);
  const [reusePending, setReusePending] = useState(false);
  const [reuseError, setReuseError] = useState<string | null>(null);
  const executionPlayback = useStore((s) => s.executionPlayback);
  const setExecutionPlayback = useStore((s) => s.setExecutionPlayback);
  const updateExecutionPlayback = useStore((s) => s.updateExecutionPlayback);
  const playbackTimer = useRef<number | null>(null);
  const fadeTimer = useRef<number | null>(null);
  const reuseRequest = useRef(0);
  const reuseProbeRef = useRef(reuseProbe);
  reuseProbeRef.current = reuseProbe;

  const stopPlaybackTimers = useCallback(() => {
    if (playbackTimer.current !== null) window.clearInterval(playbackTimer.current);
    if (fadeTimer.current !== null) window.clearInterval(fadeTimer.current);
    playbackTimer.current = null;
    fadeTimer.current = null;
  }, []);

  const startPlayback = useCallback((sweep: ReuseSweep, probe: ReuseProbe) => {
    stopPlaybackTimers();
    const tile = probe.box.map((interval) => interval.hi - interval.lo);
    const reduced = typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches;
    setExecutionPlayback({
      tensorId: probe.tensorId,
      anchorBox: probe.box,
      tile,
      colorIndex: probe.colorIndex ?? 0,
      frames: sweep.frames,
      visited: reduced ? sweep.frames.length : Math.min(1, sweep.frames.length),
      phase: reduced ? "settled" : "playing",
      exiting: false,
      opacity: reduced ? 0.72 : 1,
    });
    if (reduced || sweep.frames.length <= 1) {
      updateExecutionPlayback({
        visited: sweep.frames.length,
        phase: "settled",
        opacity: 0.72,
      });
      return;
    }
    let visited = 1;
    const delay = Math.max(42, Math.min(110, Math.round(2100 / sweep.frames.length)));
    playbackTimer.current = window.setInterval(() => {
      visited++;
      if (visited > sweep.frames.length) {
        window.clearInterval(playbackTimer.current!);
        playbackTimer.current = null;
        updateExecutionPlayback({
          visited: sweep.frames.length,
          phase: "settled",
          opacity: 0.72,
        });
      } else {
        updateExecutionPlayback({ visited });
      }
    }, delay);
  }, [setExecutionPlayback, stopPlaybackTimers, updateExecutionPlayback]);

  const fadePlayback = useCallback(() => {
    const current = useStore.getState().executionPlayback;
    if (!current || current.exiting) return;
    stopPlaybackTimers();
    const initial = current.opacity;
    const started = performance.now();
    updateExecutionPlayback({ exiting: true });
    fadeTimer.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - started) / 240);
      updateExecutionPlayback({ opacity: initial * (1 - progress) });
      if (progress >= 1) {
        window.clearInterval(fadeTimer.current!);
        fadeTimer.current = null;
        setExecutionPlayback(null);
      }
    }, 30);
  }, [setExecutionPlayback, stopPlaybackTimers, updateExecutionPlayback]);

  /** Coming back interrupts a departure. Settle rather than resume from where
   *  the fade froze it: the animation explains the estimate once, and picking
   *  a half-watched sweep back up mid-stride explains nothing. `replay` runs
   *  the whole thing again for a reader who wants it. */
  const cancelFade = useCallback(() => {
    const current = useStore.getState().executionPlayback;
    if (!current?.exiting) return;
    stopPlaybackTimers();
    updateExecutionPlayback({
      exiting: false,
      phase: "settled",
      visited: current.frames.length,
      opacity: 0.72,
    });
  }, [stopPlaybackTimers, updateExecutionPlayback]);

  useEffect(() => {
    if (tab === "execution") cancelFade();
    else fadePlayback();
  }, [cancelFade, fadePlayback, tab]);

  useEffect(() => {
    if (!executionPlayback) return;
    if (
      !reuseProbe ||
      executionPlayback.tensorId !== reuseProbe.tensorId ||
      !sameBox(executionPlayback.anchorBox, reuseProbe.box)
    ) fadePlayback();
  }, [executionPlayback, fadePlayback, reuseProbe]);

  useEffect(() => {
    reuseRequest.current++;
    setReusePending(false);
    setReuseError(null);
  }, [reuseProbe]);

  /* The timers that advance and retire a playback live here, so an unmounted
     panel - collapse-to-rail unmounts it - would leave the overlay frozen on
     every card with nothing left able to clear it. The state goes with them,
     and goes at once rather than fading: a fade explains a departure, and
     there is no longer a panel on screen to have departed from. */
  useEffect(() => () => {
    stopPlaybackTimers();
    setExecutionPlayback(null);
  }, [setExecutionPlayback, stopPlaybackTimers]);

  const {
    metrics,
    bounds,
    sharing,
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

  if (!resolved) return <aside className="inspector" aria-label="Tile inspector" />;
  const reuse = currentReuseRows(reuseRun, resolved, reuseProbe);
  const visiblePlayback = tab === "execution" && executionPlayback && reuseProbe &&
    executionPlayback.tensorId === reuseProbe.tensorId &&
    sameBox(executionPlayback.anchorBox, reuseProbe.box)
      ? executionPlayback
      : null;

  /**
   * Everything below the tiles list is counted over the active group, not over
   * the whole selection: a tile on another tensor is not part of this answer
   * and must not appear in its totals or its "N tiles" wording.
   */
  const groupParts = activeTensorId ? partsOn(selection, activeTensorId) : [];
  const selBoxes = groupParts.length;
  const enabledBoxes = groupParts.filter((part) => !hiddenBoxes.has(part.index)).length;
  /** What the cone sections need to paint a readout per tile, in one piece. */
  const attribution: TileAttribution = {
    perBox: groupAttribution(perBox, parts, activeTensorId),
    hiddenBoxes,
    focusedBox,
    dark,
  };
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
    ? NO_ENABLED_TILES
    : seedIds.some((id) => resolved.tensors[id].producer)
      ? "Everything the selection needs is itself selected."
      : "Every selected tensor is a graph input, it needs nothing earlier.";
  const downstreamEmpty = enabledBoxes === 0
    ? NO_ENABLED_TILES
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

  /* Each figure now carries its own status, so its mark comes from the figure
     rather than from whether anything in the readout was approximate. One
     widened weight used to put `\u2264` on an output-byte count that was exact, and
     - worse - the same `\u2264` on a FLOP total spanning a barrier, which is not a
     bound in that direction or any other. `formatFigure` writes what each one
     has earned, including the word `unknown` where there is no number. */

  /**
   * A scenario's intensity, or the reason there isn't one.
   *
   * Built through `ratioFigure` so the quotient inherits what its parts know:
   * `~` where both moved over a widened region, and `unknown` where the
   * numerator crossed an operation nobody described - a ratio of an unknown
   * quantity is not a looser ratio, it is not a ratio.
   */
  const intensityOf = (cost: ConeCost | null | undefined): string => {
    if (!cost) return "—";
    const ratio = ratioFigure(cost.flops, cost.bytes);
    if (ratio.value === null) return "unknown";
    if (cost.bytes.value === 0) return "—";
    return formatFigure(ratio, (v) => `${v.toFixed(2)} FLOP/B`);
  };

  /* Summed through the figure algebra rather than by adding three numbers, so
     the total inherits the weakest of the three claims instead of presenting
     itself as whatever the last one was. */
  const workingSet = metrics
    ? sumFigures([metrics.inputBytes, metrics.intermediateBytes, metrics.outputBytes])
    : null;

  /** Reuse neighbours step along the anchor tensor's axes, so it names them. */
  const axisLabelOf = (axis: number) =>
    (activeTensorId ? resolved.tensors[activeTensorId]?.axisNames?.[axis] : null) ?? `ax${axis}`;

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. The sweep
   * is defined by one tile on one tensor, so it follows the anchor part (the
   * focused one) else the last drawn rather than mixing tensors. */
  const runReuse = (probe: ReuseProbe, graph: ResolvedGraph, request: number, retried: boolean) => {
    /** Whether the answer would still be about the tile and the graph it was
     *  asked about. Read fresh: the sweep outlives the click that started it. */
    const stale = () => {
      const current = reuseProbeRef.current;
      return request !== reuseRequest.current ||
        useStore.getState().resolved !== graph ||
        !current ||
        current.tensorId !== probe.tensorId ||
        !sameBox(current.box, probe.box);
    };
    const work = analysisWorkerAvailable()
      ? reuseInWorker({
          graphId: useStore.getState().workerGraphId,
          graph,
          tensorId: probe.tensorId,
          box: probe.box,
        })
      : Promise.resolve(estimateInputReuseSweep(graph, {
          tensorId: probe.tensorId,
          region: fromBox(probe.box),
        }));
    void work.then((sweep) => {
      if (stale()) return;
      setReusePending(false);
      setReuseRun({ graph, probe, rows: sweep.estimates, sweep });
      if (useStore.getState().inspectorTab === "execution") startPlayback(sweep, probe);
    }).catch((error) => {
      if (request !== reuseRequest.current) return;
      /* A cancellation discarded the work, not the question, so re-ask it
         while the tile and the graph are still the ones it was about. Once:
         a second cancellation is something contending for the lane rather
         than the one build that takes it, and a silent button beats a loop. */
      if (isAnalysisCancelled(error)) {
        if (!retried && !stale()) return runReuse(probe, graph, request, true);
        setReusePending(false);
        return;
      }
      setReusePending(false);
      setReuseError(error instanceof Error ? error.message : String(error));
    });
  };

  const computeReuse = () => {
    if (!reuseProbe) return;
    const cached = reuseRun && currentReuseRows(reuseRun, resolved, reuseProbe)
      ? reuseRun.sweep
      : null;
    if (cached) {
      startPlayback(cached, reuseProbe);
      return;
    }
    setReusePending(true);
    setReuseError(null);
    runReuse(reuseProbe, resolved, ++reuseRequest.current, false);
  };

  return (
    <aside className="inspector" aria-label="Tile inspector">
      <div className="inspector-scroll">
        {/* The Plan view describes tasks rather than drawn tiles, so it is
            reachable with nothing drawn, and the switch sits above the empty
            panel because it is the only route to it. */}
        {tab === "plan" ? (
          <>
            <InspectorTabs />
            <PlanPanel />
          </>
        ) : !selection ? (
          <>
            <InspectorTabs />
            <EmptyPanel />
          </>
        ) : (
          <>
            <TileIdentity
              activeTensorId={activeTensorId}
              focusedBox={focusedBox}
              coneExprs={() =>
                [...upstream, ...downstream].flatMap((row) => sliceLines(row)).join("\n")
              }
            />
            {/* The tile header sits above the strip: both classes of question
                are about the same tile, and it is the tiles list below that
                picks which one. */}
            <InspectorTabs />
            {tab === "dependencies" ? (
              <div
                className="ins-tabpanel"
                role="region"
                id="ins-panel-dependencies"
                aria-labelledby="ins-tab-dependencies"
              >
            {/* The tiles list is the selector for everything below it: it picks
                which cone the two sections describe, so it sits above them. */}
            <RegionEditor activeTensorId={activeTensorId} onSelectGroup={selectGroup} />

            <ConeSection
              direction="backward"
              rows={upstream}
              hue={coneHue("upstream")}
              flags={findings?.flags ?? new Map()}
              enabled={showUpstream}
              onToggle={() => toggleDirection("backward")}
              empty={upstreamEmpty}
              attr={attribution}
            />
            <ConeSection
              direction="forward"
              rows={downstream}
              hue={coneHue("downstream")}
              flags={new Map()}
              contributions={contrib?.byTensor}
              enabled={showDownstream}
              onToggle={() => toggleDirection("forward")}
              empty={downstreamEmpty}
              attr={attribution}
            />
            {direction === "none" && (
              <p className="view-mode-note" role="status">
                Figures only · Backward & Forward cones are hidden.
              </p>
            )}
            {showDownstream && contrib?.capped && (
              <p className="hint">
                more than {MAX_CONTRIBUTION_PROBES} tensors are fed: the rows above do not
                say whether this tile completes them or only feeds them
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
                  title={`${showEntangled ? "hide" : "show"} the co-access surface (${SHORTCUTS.entangled.keys[0]})`}
                >
                  <span className="cone-key combined" aria-hidden="true" />
                  Co-access Surface
                </button>
              </div>
              {showEntangled &&
                (merged ? (
                  <p className="hint">{MERGED_AT_CAP}</p>
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
                  <p className="hint">Nothing meets this tile in a shared term.</p>
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
                {/* Each figure says for itself what may be read into it: a
                    bare number is a count, `<=` is "no more than", and a figure
                    whose arithmetic crosses an operation nobody described says
                    `unknown` instead of offering a number that is neither a
                    ceiling nor a floor. */}
                <div className="ins-section">
                  <div className="ins-title">
                    {`Cost to compute ${costScope} `}
                    {!metrics.exact && (
                      <span className="badge approx" title={metrics.reasons.join("; ")}>≈</span>
                    )}
                  </div>
                  <div className="kv">
                    <span>FLOPs</span>
                    <span title={metrics.flops.reasons.join("; ")}>
                      {formatFigure(metrics.flops, fmt)}
                    </span>
                    <span>input bytes</span>
                    <span>{formatFigure(metrics.inputBytes, formatBytes)}</span>
                    <span>intermediate</span>
                    <span>{formatFigure(metrics.intermediateBytes, formatBytes)}</span>
                    <span>output bytes</span>
                    <span>{formatFigure(metrics.outputBytes, formatBytes)}</span>
                    <span>working set</span>
                    <span>{workingSet && formatFigure(workingSet, formatBytes)}</span>
                  </div>
                  {metrics.flops.status === "unknown" && (
                    /* The one qualification a reader cannot act on without
                       being told: every other figure here is still a bound,
                       and only the arithmetic is missing. Without this the
                       word `unknown` beside a full set of byte counts reads
                       as a glitch rather than as the answer. */
                    <p className="hint overlap">
                      No FLOP total through {metrics.unknownOperations} unmodelled
                      {metrics.unknownOperations === 1 ? " operation" : " operations"}:
                      its arithmetic is not described, so a sum across it would be
                      neither an upper nor a lower bound. The byte figures still hold.
                    </p>
                  )}
                  {metrics.exact || metrics.flops.status === "unknown" ? null : (
                    <p className="hint overlap">
                      Bounds, not counts: read ≤ as “no more than”, never understated.
                    </p>
                  )}
                </div>
                {/* A function of the tiles on screen: duplicate element demand
                    at graph inputs. It is a shareable footprint, not a claim
                    about cache hits, transactions, or bytes actually loaded.
                    The sweep that asks the same question of tiles nobody drew
                    is modelled, and lives under Execution. */}
                <div className="ins-section">
                  <div className="ins-title">Shared graph-input demand</div>
                  {sharing && sharing.length > 0 ? (
                    <ul className="reuse-list">
                      {sharing.map((row) => (
                        <li key={row.tensorId}>
                          <code>{resolved.tensors[row.tensorId].name}</code>
                          <span
                            className="muted"
                            title={row.geometryExact
                              ? `${row.contributingTiles} of ${row.selectedTiles} tiles demand ${formatBytes(row.summedDemandBytes)} in total; ${formatBytes(row.distinctDemandBytes)} distinct`
                              : `${row.contributingTiles} of ${row.selectedTiles} tiles have widened footprints; duplicate demand is no more than ${formatBytes(row.duplicateDemandBytes)}`}
                          >
                            {row.duplicateDemandBytes > 0
                              ? `${row.geometryExact ? "" : "~ "}${(row.summedDemandBytes / row.distinctDemandBytes).toFixed(2)}× demand · ${row.geometryExact ? "" : "≤ "}${formatBytes(row.duplicateDemandBytes)} duplicate`
                              : "no duplicate demand"}
                          </span>
                          {!row.geometryExact && (
                            <span className="badge approx" title={row.reasons.join("; ")}>≈</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : !perBox && parts.length > MAX_PER_BOX_PROPS ? (
                    <p className="hint">
                      Per-tile sharing is unavailable above {MAX_PER_BOX_PROPS} total tiles.
                    </p>
                  ) : enabledBoxes < 2 || focusedBox !== null ? (
                    <p className="hint">
                      Enable and analyse at least two tiles together to compare their demand.
                    </p>
                  ) : (
                    <p className="hint">
                      These tiles have no graph-input demand.
                    </p>
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
              </div>
            ) : (
              <div
                className="ins-tabpanel"
                role="region"
                id="ins-panel-execution"
                aria-labelledby="ins-tab-execution"
              >
                <p className="tab-note">
                  Everything here assumes an execution the graph does not fix: a tiling of the
                  whole tensor, an order to run it in, what fuses with what. These are models,
                  not bounds - a wrong assumption moves them in either direction.
                </p>
                {/* Both scenarios assume a fusion decision the graph does not
                    make, which is what moved them off the cost table: the
                    counts there are measured, these are argued. */}
                <div className="ins-section">
                  <div className="ins-title">Arithmetic intensity</div>
                  <div className="kv">
                    <span title={
                      bounds
                        ? `one kernel over every op and every tile: intermediates never reach memory and a shared operand band is fetched once - ${formatFigure(bounds.fused.flops, fmt)} FLOP / ${formatFigure(bounds.fused.bytes, formatBytes)}`
                        : undefined
                    }>
                      fused
                    </span>
                    <span>{intensityOf(bounds?.fused)}</span>
                    <span title={
                      bounds?.unfused
                        ? `every op of every tile as its own job: separate input reads and output writes, materialized views, and no cross-op cache reuse - ${formatFigure(bounds.unfused.flops, fmt)} FLOP / ${formatFigure(bounds.unfused.bytes, formatBytes)}`
                        : `per-tile cones are not traced past ${MAX_PER_BOX_PROPS} tiles, and the merged result cannot be taken apart again`
                    }>
                      unfused
                    </span>
                    <span>{intensityOf(bounds?.unfused)}</span>
                  </div>
                  {metrics?.flops.status === "unknown" ? (
                    <p className="hint overlap">
                      Unavailable: the numerator is a FLOP total across an operation whose
                      arithmetic is not modelled. A ratio of an unknown quantity is not a
                      looser ratio, it is not one.
                    </p>
                  ) : metrics && !metrics.exact ? (
                    <p className="hint overlap">
                      Both are ratios of figures measured over a widened region, which moves
                      the numerator and the denominator at once: ~ says disturbed in an
                      unknown direction, where ≤ would claim a side.
                    </p>
                  ) : null}
                </div>
                <div className="ins-section">
                  <div className="ins-title with-action">
                    Reuse sweep
                    <button
                      className="mini"
                      onClick={computeReuse}
                      disabled={!reuseProbe || reusePending}
                      title="sample tiles of the selection's size across the anchor tensor and estimate how many demand part of each graph-input footprint"
                    >
                      {reusePending ? "estimating…" : reuse ? "replay" : "estimate"}
                    </button>
                  </div>
                  <p className="hint">
                    Tiles of this tile's size, laid over the whole tensor: how many of them demand
                    part of the same graph-input footprint.
                  </p>
                  {visiblePlayback?.phase === "playing" && (
                    <p className="hint">
                      Probe {visiblePlayback.visited} of {visiblePlayback.frames.length}
                    </p>
                  )}
                  {reuseError ? (
                    <p className="hint overlap">Reuse estimate failed: {reuseError}</p>
                  ) : reusePending ? (
                    <p className="hint">Evaluating sampled tiles off the UI thread…</p>
                  ) : !reuse ? (
                    <p className="hint">sampled sweep · run on demand</p>
                  ) : reuse.length === 0 ? (
                    <p className="hint">This tile has no graph-input demand.</p>
                  ) : (
                    <ul className="reuse-list">
                      {reuse.map((estimate) => {
                        const neighbours = neighbourShares(estimate.neighbors, axisLabelOf);
                        const qualifier = reuseQualifiers(estimate);
                        return (
                          <li key={estimate.tensorId}>
                            <code>{resolved.tensors[estimate.tensorId].name}</code>
                            <span
                              className="muted"
                              title={
                                estimate.exhaustive
                                  ? "every tile of this size was probed"
                                  : `${estimate.probes} of ${estimate.totalTiles} tiles probed, one per stratum`
                              }
                            >
                              {qualifier.count}
                              {estimate.estimatedTiles} of {estimate.totalTiles} tiles
                              {estimate.meanSharedFraction !== null &&
                                ` · ${qualifier.fraction}${Math.round(estimate.meanSharedFraction * 100)}% of the footprint each`}
                            </span>
                            {!estimate.geometryExact && (
                              <span className="badge approx" title={estimate.reasons.join("; ")}>≈</span>
                            )}
                            {(!estimate.exhaustive || neighbours.text) && (
                              <span
                                className="reuse-detail"
                                title={neighbours.reasons.length
                                  ? `approximate neighbours: ${neighbours.reasons.join("; ")}`
                                  : undefined}
                              >
                                {!estimate.exhaustive && `${estimate.probes} sampled`}
                                {!estimate.exhaustive && neighbours.text && " · "}
                                {neighbours.text && `neighbouring tile shares ${neighbours.text}`}
                              </span>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
