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
import { MAX_PER_BOX_PROPS, planesOf, useStore, viewAxes } from "./store";
import { tileOf } from "./grid";
import { formatSelectionBox, parseSelectionBox } from "./selection-range";
import { effectiveTileScaleIndex, effectiveTileScaleStops } from "./tiling";

function fmt(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(n < 10 && !Number.isInteger(n) ? 2 : 0);
}

const scaleLabel = (scale: number) =>
  scale === 0 ? "auto" : `${scale > 0 ? "×" : "÷"}${2 ** Math.abs(scale)}`;

function TileGridControl(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const detail = useMemo(() => {
    const planes = planesOf(resolved);
    const stops = effectiveTileScaleStops(planes, graphPx);
    return { stops, index: effectiveTileScaleIndex(planes, graphPx, stops, tileScale) };
  }, [resolved, graphPx, tileScale]);

  return (
    <section className="tile-grid-control">
      <div className="tile-grid-head">
        <span>tile grid</span>
        <b>{scaleLabel(detail.stops[detail.index])}</b>
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
  const moveSelection = useStore((s) => s.moveSelection);
  const replaceBox = useStore((s) => s.replaceBox);
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const deleteBox = useStore((s) => s.deleteBox);
  const undoWorkspace = useStore((s) => s.undoWorkspace);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const pinned = useStore((s) => s.pinnedBox);
  const hoverBox = useStore((s) => s.hoverBox);
  const togglePinBox = useStore((s) => s.togglePinBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const toggleBoxHidden = useStore((s) => s.toggleBoxHidden);
  const theme = useStore((s) => s.theme);

  if (!resolved || !selection) return null;
  const t = resolved.tensors[selection.tensorId];
  const shape = t.resolved!;
  const { rowAxis: rowAx, colAxis: colAx } = viewAxes(shape);
  const tile = tileOf(shape, tileScale, graphPx); // the nudge pad moves by whole tiles
  const axisLabel = (ax: number) => t.axisNames?.[ax] ?? `ax${ax}`;
  const boxes = selection.region.boxes;
  const overlap = partsOverlap(boxes);
  const dark = theme === "dark";

  return (
    <div className="ins-section">
      <div className="ins-title">
        Region <span className="muted">{boxes.length} box{boxes.length === 1 ? "" : "es"}</span>
      </div>

      <div
        className="nudge-pad"
        title={
          focusedBox !== null
            ? `move box ${focusedBox + 1} alone by one ${tile}-element tile (arrow keys; Shift = 8 tiles)`
            : `move the whole selection by one ${tile}-element tile — pin a box below to move it alone`
        }
      >
        <button onClick={() => moveSelection(rowAx, -tile)} disabled={rowAx < 0} style={{ gridArea: "up" }}>↑</button>
        <button onClick={() => moveSelection(colAx, -tile)} disabled={colAx < 0} style={{ gridArea: "left" }}>←</button>
        <span className="nudge-mid" style={{ gridArea: "mid" }}>
          {focusedBox !== null ? `#${focusedBox + 1}` : "all"}
        </span>
        <button onClick={() => moveSelection(colAx, tile)} disabled={colAx < 0} style={{ gridArea: "right" }}>→</button>
        <button onClick={() => moveSelection(rowAx, tile)} disabled={rowAx < 0} style={{ gridArea: "down" }}>↓</button>
        <button className="undo-btn" style={{ gridArea: "undo" }} onClick={undoWorkspace} title="undo the last selection or layout change (ctrl+z)">undo</button>
      </div>

      {boxes.length > 1 && (
        <p className="hint">
          {perBox
            ? "hover a box to emphasise its cone; click to pin it, then the arrows move that box alone"
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
        {boxes.map((b, i) => {
          const active = focusedBox === i;
          const hidden = hiddenBoxes.has(i);
          return (
            <div
              className={`box-row${active ? " active" : ""}${pinned === i ? " pinned" : ""}${hidden ? " hidden-cone" : ""}`}
              key={i}
              onMouseEnter={() => hoverBox(i)}
              onClick={() => perBox && togglePinBox(i)}
              title={perBox ? "hover to emphasise this box's dependencies; click to pin (esc unpins)" : undefined}
            >
              <button
                className="vis-toggle"
                disabled={!perBox}
                title={
                  perBox
                    ? `${hidden ? "show" : "hide"} this box's cone (h) — its numbers stay in the table either way`
                    : "per-box cones are not traced at this many boxes"
                }
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBoxHidden(i);
                }}
              >
                {hidden ? "○" : "●"}
              </button>
              <i className="swatch" style={{ background: rgbCss(boxColor(i, dark)) }} />
              <SelectionRangeInput
                box={b}
                shape={shape}
                onCommit={(next) => replaceBox(i, next)}
              />
              <span className="muted">{fmt(b.reduce((a, I) => a * (I.hi - I.lo), 1))}</span>
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
      {boxes.length > MAX_DISTINCT_HUES && (
        <p className="hint">
          boxes past the {MAX_DISTINCT_HUES}rd share a neutral color — only {MAX_DISTINCT_HUES} hues stay
          distinguishable side by side, so use hover to tell the rest apart.
        </p>
      )}
      <p className="hint">
        rows {axisLabel(rowAx >= 0 ? rowAx : 0)} · cols {axisLabel(colAx >= 0 ? colAx : 0)} · shape [{shape.join("×")}]
      </p>
      <details className="keys">
        <summary className="hint">keyboard</summary>
        <div className="kv">
          <span>arrows</span><span>move by one tile</span>
          <span>shift+arrows</span><span>move by 8 tiles</span>
          <span>[ / ]</span><span>scrub hidden axis</span>
          <span>h</span><span>hide/show focused cone</span>
          <span>ctrl+z</span><span>undo selection / layout</span>
          <span>u / d</span><span>toggle upstream / downstream</span>
          <span>f</span><span>fit to view</span>
          <span>esc</span><span>unpin / leave editing</span>
        </div>
      </details>
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

  const selBoxes = selection?.region.boxes.length ?? 0;

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. */
  const computeReuse = () => {
    if (!selection || !resolved || !scopedRes) return;
    const shape = resolved.tensors[selection.tensorId].resolved!;
    const bb = selection.region.boxes[0];
    if (!bb) return;
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
      const res = propagateBackward(resolved, { tensorId: selection.tensorId, region: fromBox(probeBox) });
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
        <h2>Tiles <small>{selBoxes ? `${selBoxes} selected` : "none selected"}</small></h2>
        <button className="mini" onClick={clearSelection} disabled={!selection}>clear all</button>
      </div>
      <div className="tile-legend">
        <span><i className="selected" style={{ background: rgbCss(boxColor(0, theme === "dark")) }} /> selected tile</span>
        <span><i className="required" style={{ background: rgbCss(boxColor(0, theme === "dark")) }} /> dependency</span>
        <span><i className="approx" /> approximate</span>
      </div>
      {!selection && <p className="hint">Click or drag on any tensor grid to select a region. Shift adds, Alt subtracts.</p>}
      {selection && (
        <div className="ins-section">
          <div className="ins-title">
            Selection: <b>{resolved.tensors[selection.tensorId].name}</b>
          </div>
          <div className="kv">
            <span>elements</span><span>{fmt(count(selection.region))}</span>
            <span>boxes</span><span>{selBoxes}</span>
          </div>
        </div>
      )}
      <RegionEditor />
      {metrics && direction !== "forward" && (
        <>
          <div className="ins-section">
            <div className="ins-title">
              {focusedBox !== null && perBox ? `Box ${focusedBox + 1} (upstream)` : "Aggregate (upstream)"}
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
            <div className="ins-title">Contributing tensors</div>
            {metrics.tensors.map((t) => (
              <details key={t.tensorId} className="tensor-row" open={t.isInput}>
                <summary>
                  <b>{t.name}</b>
                  <span className="muted"> d{t.depth}</span>
                  {!t.exact && <span className="badge approx">≈</span>}
                  {t.isInput && <span className="badge input">in</span>}
                  <span className="row-stats">
                    {fmt(t.elements)} el ({((t.elements / t.totalElements) * 100).toFixed(1)}%) · {formatBytes(t.bytes)} · {t.boxCount} box{t.boxCount === 1 ? "" : "es"}
                  </span>
                  <FootprintBar
                    tensorId={t.tensorId}
                    elements={t.elements}
                    totalElements={t.totalElements}
                    perBox={perBox}
                    focusedBox={focusedBox}
                    dark={theme === "dark"}
                  />
                </summary>
                <div className="slice-exprs">
                  {t.sliceExprsNumpy.slice(0, 12).map((line, i) => (
                    <code key={i}>{line}</code>
                  ))}
                  {t.sliceExprsNumpy.length > 12 && <code># … {t.sliceExprsNumpy.length - 12} more boxes</code>}
                  <button
                    className="mini"
                    onClick={() => copy(t.sliceExprsNumpy.join("\n"), t.tensorId)}
                  >
                    {copied === t.tensorId ? "copied ✓" : "copy numpy/torch"}
                  </button>
                </div>
              </details>
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
