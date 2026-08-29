import React, { useMemo, useState } from "react";
import { computeMetrics } from "../core/metrics";
import { propagateBackward } from "../core/propagate";
import { count, fromBox, intersect, isEmpty } from "../core/region";
import { formatBytes } from "./TensorCard";
import { boxColor, isDarkTheme, MAX_DISTINCT_HUES, rgbCss } from "./palette";
import { MAX_PER_BOX_PROPS, useStore, viewAxes } from "./store";
import { tileOf } from "./grid";

function fmt(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(n < 10 && !Number.isInteger(n) ? 2 : 0);
}

/**
 * The selection's boxes, one row each. Hovering a row isolates that box's
 * dependency cone across the whole graph; clicking pins the isolation.
 */
function RegionEditor(): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const moveSelection = useStore((s) => s.moveSelection);
  const tileScale = useStore((s) => s.tileScale);
  const deleteBox = useStore((s) => s.deleteBox);
  const undoSelection = useStore((s) => s.undoSelection);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const setFocusedBox = useStore((s) => s.setFocusedBox);
  const [pinned, setPinned] = useState<number | null>(null);

  if (!resolved || !selection) return null;
  const t = resolved.tensors[selection.tensorId];
  const shape = t.resolved!;
  const { rowAxis: rowAx, colAxis: colAx } = viewAxes(shape);
  const tile = tileOf(shape, tileScale); // the nudge pad moves by whole tiles
  const axisLabel = (ax: number) => t.axisNames?.[ax] ?? `ax${ax}`;
  const boxes = selection.region.boxes;
  const dark = isDarkTheme();

  const focus = (i: number | null) => {
    if (pinned !== null) return; // a pinned row wins over hover
    setFocusedBox(i);
  };
  const togglePin = (i: number) => {
    if (pinned === i) {
      setPinned(null);
      setFocusedBox(null);
    } else {
      setPinned(i);
      setFocusedBox(i);
    }
  };

  return (
    <div className="ins-section">
      <div className="ins-title">
        Region <span className="muted">{boxes.length} box{boxes.length === 1 ? "" : "es"}</span>
      </div>

      <div className="nudge-pad" title={`move the selection by one ${tile}-element tile (arrow keys; Shift = 8 tiles)`}>
        <button onClick={() => moveSelection(rowAx, -tile)} disabled={rowAx < 0} style={{ gridArea: "up" }}>↑</button>
        <button onClick={() => moveSelection(colAx, -tile)} disabled={colAx < 0} style={{ gridArea: "left" }}>←</button>
        <span className="nudge-mid" style={{ gridArea: "mid" }}>move</span>
        <button onClick={() => moveSelection(colAx, tile)} disabled={colAx < 0} style={{ gridArea: "right" }}>→</button>
        <button onClick={() => moveSelection(rowAx, tile)} disabled={rowAx < 0} style={{ gridArea: "down" }}>↓</button>
        <button className="undo-btn" style={{ gridArea: "undo" }} onClick={undoSelection} title="undo the last selection change (ctrl+z)">undo</button>
      </div>

      {boxes.length > 1 && (
        <p className="hint">
          {perBox
            ? "hover a box to isolate its cone; click to pin it"
            : `too many boxes to trace individually (over ${MAX_PER_BOX_PROPS}) — showing the merged cone`}
        </p>
      )}

      <div className="box-list" onMouseLeave={() => focus(null)}>
        {boxes.map((b, i) => {
          const active = focusedBox === i;
          return (
            <div
              className={`box-row${active ? " active" : ""}${pinned === i ? " pinned" : ""}`}
              key={i}
              onMouseEnter={() => focus(i)}
              onClick={() => perBox && togglePin(i)}
              title={perBox ? "hover to isolate this box's dependencies; click to pin" : undefined}
            >
              <i className="swatch" style={{ background: rgbCss(boxColor(i, dark)) }} />
              <code>
                [{b.map((I) => (I.hi - I.lo === 1 ? `${I.lo}` : `${I.lo}:${I.hi}`)).join(", ")}]
              </code>
              <span className="muted">{fmt(b.reduce((a, I) => a * (I.hi - I.lo), 1))}</span>
              <button
                className="mini"
                title="remove this box from the selection"
                onClick={(e) => {
                  e.stopPropagation();
                  setPinned(null);
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
          <span>ctrl+z</span><span>undo selection</span>
          <span>u / d / b</span><span>up / down / both</span>
          <span>f · esc</span><span>fit · clear</span>
        </div>
      </details>
    </div>
  );
}

export function Inspector(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const direction = useStore((s) => s.direction);
  const countIntermediates = useStore((s) => s.countIntermediates);
  const setCountIntermediates = useStore((s) => s.setCountIntermediates);
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
      <h3>Inspector</h3>
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
      {direction === "forward" && selection && (
        <p className="hint">Forward mode shows downstream influence; switch to Upstream/Both for cost metrics.</p>
      )}
    </aside>
  );
}
