import React, { useEffect, useMemo, useState } from "react";
import { computeMetrics } from "../core/metrics";
import { coneIsFullyElementwise, dependencyNotes } from "../core/notes";
import { propagateBackward } from "../core/propagate";
import {
  Box,
  Region,
  count,
  fromBox,
  intersect,
  isEmpty,
  partsOverlap,
  subtract,
  union,
} from "../core/region";
import { formatBytes } from "./TensorCard";
import { boxColor, MAX_DISTINCT_HUES, rgbCss } from "./palette";
import {
  anchorTensorId,
  MAX_PER_BOX_PROPS,
  partsOn,
  planesOf,
  selectedTensorIds,
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

function TileGridControl(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const setSnapToGrid = useStore((s) => s.setSnapToGrid);
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
    <section className="tile-grid-control">
      <div className="tile-grid-head">
        <span>tile grid</span>
        <b>{scaleLabel(requested)}</b>
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
        title="global tile detail; every stop changes the rendered lattice"
      />
      <div className="tile-grid-stops">
        {detail.stops.map((stop) => <span key={stop}>{scaleLabel(stop)}</span>)}
      </div>
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
function DependencyNotes(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);

  const scoped = focusedBox !== null ? perBox?.[focusedBox]?.backward ?? backwardRes : backwardRes;

  const result = useMemo(() => {
    if (!resolved || !scoped) return null;
    return {
      notes: dependencyNotes(resolved, scoped),
      elementwise: coneIsFullyElementwise(resolved, scoped),
    };
  }, [resolved, scoped]);

  return (
    <section className="ins-section notes-section">
      <div className="ins-title">
        Dependency notes
        {focusedBox !== null && perBox && <span className="muted"> · box {focusedBox + 1}</span>}
      </div>
      {!selection || !result ? (
        <p className="hint">
          Draw a tile to read its cone. Notes here call out the reductions and contractions
          that constrain a fused kernel.
        </p>
      ) : result.notes.length ? (
        <ul className="notes-list">
          {result.notes.map((note) => (
            <li key={`${note.nodeId}:${note.text}`}>
              <b>{note.op}</b>
              {note.text}
            </li>
          ))}
        </ul>
      ) : result.elementwise ? (
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
      title: `box ${focusedBox + 1}: ${fmt(elements)} elements`,
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
          title: `box ${index + 1} only: ${fmt(exclusive)} elements`,
        });
    });
    const shared = Math.max(0, elements - exclusiveTotal);
    if (shared > 0)
      segments.push({
        elements: shared,
        color: "",
        title: `shared by multiple boxes: ${fmt(shared)} elements`,
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
 * The selection's boxes, one row each. Hovering a row emphasises that box's
 * dependency cone across the whole graph; clicking pins the emphasis.
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

  if (!resolved || !selection) return null;
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

  // The trailing axis legend describes one tensor, so it follows the anchor --
  // the same tensor the arrow keys move.
  const anchor = anchorTensorId(selection, focusedBox);
  const anchorT = anchor ? resolved.tensors[anchor] : null;
  const anchorShape = anchorT?.resolved ?? [];
  const { rowAxis: rowAx, colAxis: colAx } = viewAxes(anchorShape);
  const axisLabel = (ax: number) => anchorT?.axisNames?.[ax] ?? `ax${ax}`;

  return (
    <div className="ins-section">
      {parts.length > 1 && (
        <p className="hint">
          {perBox
            ? "hover a box to emphasise its cone; click to pin it, then arrow keys move that box alone"
            : `too many boxes to trace individually (over ${MAX_PER_BOX_PROPS}) — showing the merged cone`}
        </p>
      )}
      {overlap.summed > overlap.unique && (
        <p className="hint overlap">
          boxes overlap: {fmt(overlap.summed)} counted across parts,{" "}
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
              title={perBox ? "hover to emphasise this box's dependencies; click to pin (esc unpins)" : undefined}
            >
              {/* One control, not two: the swatch *is* the visibility toggle, so
                  the row states the box's hue and its shown/hidden state once. */}
              <button
                className="vis-toggle"
                disabled={!perBox}
                style={{
                  borderColor: rgbCss(boxColor(i, dark)),
                  background: hidden ? "transparent" : rgbCss(boxColor(i, dark)),
                }}
                title={
                  perBox
                    ? `${hidden ? "show" : "hide"} this box's cone (h) — its numbers stay in the table either way`
                    : "per-box cones are not traced at this many boxes"
                }
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBoxHidden(i);
                }}
                aria-label={`${hidden ? "show" : "hide"} box ${i + 1}`}
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
                title="remove this box from the selection"
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
          boxes past the {MAX_DISTINCT_HUES}rd share a neutral color — only {MAX_DISTINCT_HUES} hues stay
          distinguishable side by side, so use hover to tell the rest apart.
        </p>
      )}
      {anchorT && (
        <p className="hint">
          <b>{anchorT.name}</b> · rows {axisLabel(rowAx >= 0 ? rowAx : 0)} · cols{" "}
          {axisLabel(colAx >= 0 ? colAx : 0)} · shape [{anchorShape.join("×")}]
        </p>
      )}
    </div>
  );
}

export function Inspector(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const direction = useStore((s) => s.direction);
  const countIntermediates = useStore((s) => s.countIntermediates);
  const setCountIntermediates = useStore((s) => s.setCountIntermediates);
  const clearSelection = useStore((s) => s.clearSelection);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);

  const [reuse, setReuse] = useState<Record<string, string> | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  // When a box is focused, every readout below scopes to that box alone.
  const scopedRes = focusedBox !== null ? perBox?.[focusedBox]?.backward ?? backwardRes : backwardRes;

  const metrics = useMemo(() => {
    setReuse(null);
    if (!resolved || !scopedRes) return null;
    return computeMetrics(resolved, scopedRes, countIntermediates);
  }, [resolved, scopedRes, countIntermediates]);

  if (!resolved) return <aside className="inspector" />;

  const selBoxes = selection?.parts.length ?? 0;
  const hue = rgbCss(boxColor(0, theme === "dark"));
  /** Past the cap the cones are still correct, but merged rather than attributed. */
  const merged = selBoxes > MAX_PER_BOX_PROPS;

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. The sweep
   * is defined by one tile on one tensor, so it follows the anchor part — the
   * focused one, else the last drawn — rather than mixing tensors. */
  const computeReuse = () => {
    if (!selection || !resolved || !scopedRes) return;
    const probe =
      selection.parts[focusedBox !== null ? focusedBox : selection.parts.length - 1];
    if (!probe) return;
    const shape = resolved.tensors[probe.tensorId].resolved!;
    const bb = probe.box;
    const tileExt = bb.map((I) => I.hi - I.lo);
    const gridDims = shape.map((e, ax) => Math.ceil(e / Math.max(1, tileExt[ax])));
    const totalTiles = gridDims.reduce((a, b) => a * b, 1);
    const probeCount = Math.min(48, totalTiles);
    const picked = new Set<number>();
    const results: Record<string, { touch: number }> = {};
    const inputs = Object.values(resolved.tensors).filter((t) => !t.producer);
    for (const t of inputs) results[t.id] = { touch: 0 };
    for (let k = 0; k < probeCount; k++) {
      let flat = totalTiles <= 48 ? k : Math.floor(Math.random() * totalTiles);
      while (totalTiles > 48 && picked.has(flat)) flat = Math.floor(Math.random() * totalTiles);
      picked.add(flat);
      const tIdx: number[] = [];
      let rest = flat;
      for (let ax = shape.length - 1; ax >= 0; ax--) {
        tIdx.unshift(rest % gridDims[ax]);
        rest = Math.floor(rest / gridDims[ax]);
      }
      const probeBox = shape.map((e, ax) => ({
        lo: Math.min(tIdx[ax] * tileExt[ax], e - 1),
        hi: Math.min((tIdx[ax] + 1) * tileExt[ax], e),
      }));
      const res = propagateBackward(resolved, { tensorId: probe.tensorId, region: fromBox(probeBox) });
      for (const t of inputs) {
        const mine = scopedRes.tensors.get(t.id)?.region;
        const theirs = res.tensors.get(t.id)?.region;
        if (mine && theirs && !isEmpty(intersect(mine, theirs))) results[t.id].touch++;
      }
    }
    const out: Record<string, string> = {};
    for (const t of inputs) {
      const est = (results[t.id].touch / probeCount) * totalTiles;
      out[t.id] = `${est.toFixed(est < 10 ? 1 : 0)}× (of ${totalTiles} tiles)`;
    }
    setReuse(out);
  };

  const copy = (text: string, key: string) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(key);
      setTimeout(() => setCopied(null), 1200);
    });
  };

  return (
    <aside className="inspector">
      <TileGridControl />
      <div className="tiles-heading">
        <h2 className="panel-title">
          Tiles{" "}
          <small title={merged ? `per-box cones stop at ${MAX_PER_BOX_PROPS} selections; above that the cone is correct but merged` : undefined}>
            {selBoxes ? `${selBoxes} selected${merged ? " · cones merged" : ""}` : "none selected"}
          </small>
        </h2>
        <button className="mini" onClick={clearSelection} disabled={!selection}>clear all</button>
      </div>
      {/* The legend has to track the canvas: direction is drawn as fill vs
          outline, but only in "both" mode — with one direction active there is
          nothing to disambiguate and the region is simply filled. */}
      <div className="tile-legend">
        <span><i className="selected" style={{ background: hue }} /> selected tile</span>
        {direction === "both" ? (
          <>
            <span><i className="required" style={{ background: hue }} /> upstream (filled)</span>
            <span><i className="produced" style={{ borderColor: hue }} /> downstream (outlined)</span>
          </>
        ) : (
          direction !== "none" && (
            <span>
              <i className="required" style={{ background: hue }} />{" "}
              {direction === "backward" ? "upstream" : "downstream"}
            </span>
          )
        )}
        <span><i className="approx" /> approximate</span>
      </div>
      {!selection && <p className="hint">Click or drag on any tensor grid to select a region. Shift adds, Alt subtracts.</p>}
      <RegionEditor />
      {metrics && direction !== "forward" && (
        <>
          <div className="ins-section">
            <div className="ins-title">
              {focusedBox !== null && perBox ? `Cost of box ${focusedBox + 1}` : "Cost of this cone"}
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
                {Object.entries(reuse).map(([id, v]) => (
                  <React.Fragment key={id}>
                    <span>{resolved.tensors[id].name}</span>
                    <span>{v}</span>
                  </React.Fragment>
                ))}
              </div>
            ) : (
              <p className="hint">selection-sized output tiles touching each input's current footprint (sampled)</p>
            )}
          </div>
          {metrics.tensors.some((t) => !t.exact) && (
            <div className="ins-section warn">
              ⚠ some regions are conservative over-approximations:{" "}
              {[...new Set(metrics.tensors.flatMap((t) => t.reasons))].join("; ")}
            </div>
          )}
          <div className="ins-section">
            <div className="ins-title">Footprint per tensor</div>
            {/* Flat rather than a disclosure: the slice expressions are the
                answer this panel exists to give, and hiding them behind a
                triangle means the cone cannot be read in one pass. */}
            {metrics.tensors.map((t) => (
              <div key={t.tensorId} className="tensor-row">
                <div className="tensor-row-head">
                  <b>{t.name}</b>
                  {/* The role is what a reader wants first; depth is the finer
                      detail and stays available beside it. */}
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
                  dark={theme === "dark"}
                />
                <div className="slice-exprs">
                  {t.sliceExprsNumpy.slice(0, 4).map((line, i) => (
                    <code key={i}>{line}</code>
                  ))}
                  {t.sliceExprsNumpy.length > 4 && (
                    <code className="muted"># … {t.sliceExprsNumpy.length - 4} more boxes</code>
                  )}
                  {/* Revealed on hover: one per row would otherwise put twenty
                      buttons between the reader and the expressions. */}
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
        </>
      )}
      <DependencyNotes />
      {direction === "forward" && selection && (
        <p className="hint">Downstream mode shows influence; switch to Upstream for cost metrics.</p>
      )}
      {direction === "none" && selection && (
        <p className="hint">Enable Upstream or Downstream to show this selection's cone.</p>
      )}
    </aside>
  );
}
