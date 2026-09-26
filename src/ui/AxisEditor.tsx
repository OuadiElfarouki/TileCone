import React, { useEffect, useState } from "react";
import type { Box } from "../core/region";
import { tileOf } from "./grid";
import { useStore } from "./store";
import { remapped, viewAxes } from "./tensor-view";
import { gestureTile, lastTileExtent, tilePosition } from "./tile-spec";

/**
 * The inspected tile, one row per axis.
 *
 * The canvas names two axes; this names all of them. Each row states the
 * axis, its extent, the tensor's tile on it, and where the inspected tile sits,
 * so a rectangle on a card never stands for more than the reader can see
 * written down. The tile column is editable: typing an extent gives the tensor
 * a tile of its own (`ViewCfg.tile`) and refits the inspected tile to it, so
 * `[1, 2, 64, 128]` on `[B, H, S, D]` is four entries rather than a gesture
 * nobody can make. The step buttons move the tile by one tile along that axis,
 * hidden axes included, and the card's slice follows it.
 */
export function AxisEditor({ index }: { index: number }): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const part = useStore((s) => s.selection?.parts[index]);
  const cfg = useStore((s) => (part ? s.viewCfgs[part.tensorId] : undefined));
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const setTensorTile = useStore((s) => s.setTensorTile);
  const stepTile = useStore((s) => s.stepTile);
  const setPlanTileAt = useStore((s) => s.setPlanTileAt);
  const setInspectorTab = useStore((s) => s.setInspectorTab);
  const setViewAxes = useStore((s) => s.setViewAxes);

  if (!resolved || !part) return null;
  const tensor = resolved.tensors[part.tensorId];
  const shape = tensor.resolved!;
  if (!shape.length) return null;
  const box = part.box;
  const tile = gestureTile(shape, cfg, tileOf(shape, tileScale, graphPx, cfg));
  const own = !!cfg?.tile;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const name = (axis: number) => tensor.axisNames?.[axis] ?? `ax${axis}`;
  const extents = box.map((interval) => interval.hi - interval.lo);
  // A tile of the lattice, including a shortened last one. Anything else was
  // drawn at other extents or offsets and is described by its ranges alone.
  const matchesTile = box.every((interval, axis) =>
    tilePosition(interval, tile[axis], shape[axis]).aligned
  );
  // What "plan with it" divides at: the tile itself when this is one, since a
  // shortened boundary tile's extents are not the tiling's.
  const planExtents = matchesTile ? tile : extents;
  const planeRemapped = remapped(shape, cfg);
  /** Draw `axis` as the card's rows or columns. Taking the other role's axis
   *  swaps the two, so the card always draws two distinct axes. */
  const drawAs = (axis: number, role: "rows" | "cols") => {
    const pair: [number, number] = role === "rows"
      ? [axis, axis === colAxis ? rowAxis : colAxis]
      : [axis === rowAxis ? colAxis : rowAxis, axis];
    setViewAxes(part.tensorId, pair, index);
  };
  const setAxis = (axis: number, extent: number) => {
    const next = tile.slice();
    next[axis] = extent;
    setTensorTile(part.tensorId, next, index);
  };

  return (
    <div className="ins-section axis-editor">
      <div className="ins-title with-action">
        Axes
        <span className="axis-actions">
          {!matchesTile && (
            <button
              className="mini"
              title="make this tile's extents the tensor's tile, so later gestures and steps use them"
              onClick={() => setTensorTile(part.tensorId, extents, index)}
            >
              use as tile
            </button>
          )}
          {own && (
            <button
              className="mini"
              title="return this tensor to the canvas tile: the detail setting on the visible axes and the view mode on the others"
              onClick={() => setTensorTile(part.tensorId, null)}
            >
              reset
            </button>
          )}
          {planeRemapped && (
            <button
              className="mini"
              title="draw the last two axes again, the default plane"
              onClick={() => setViewAxes(part.tensorId, null, index)}
            >
              default plane
            </button>
          )}
          <button
            className="mini"
            disabled={!tensor.producer}
            title={tensor.producer
              ? "divide this tensor into tasks of these extents and inspect the task holding this tile"
              : "a graph input has no tasks: nothing computes it"}
            onClick={() => {
              setPlanTileAt(part.tensorId, planExtents, box.map((interval) => interval.lo));
              setInspectorTab("plan");
            }}
          >
            plan with it
          </button>
        </span>
      </div>
      <table className="axis-table">
        <thead>
          <tr>
            <th scope="col">axis</th>
            <th scope="col" className="num">size</th>
            <th scope="col" className="num" title={own ? "this tensor's tile" : "the canvas tile; type to give the tensor its own"}>
              tile
            </th>
            <th scope="col">range</th>
            <th scope="col" className="num">tile #</th>
          </tr>
        </thead>
        <tbody>
          {shape.map((extent, axis) => {
            const position = tilePosition(box[axis], tile[axis], extent);
            const last = lastTileExtent(extent, tile[axis]);
            const onLast = position.coord === position.count - 1 && last !== tile[axis];
            const visible = axis === rowAxis ? "rows" : axis === colAxis ? "cols" : null;
            return (
              <tr key={axis} className={visible ? "visible" : "hidden-axis"}>
                <th scope="row" title={visible ? `drawn as the card's ${visible}` : "not drawn: the card shows a slice or the union"}>
                  {name(axis)}
                  {shape.length >= 2 ? (
                    <span className="axis-draw" role="group" aria-label={`draw ${name(axis)} as`}>
                      {(["rows", "cols"] as const).map((role) => (
                        <button
                          key={role}
                          className={`axis-role${visible === role ? " on" : ""}`}
                          aria-pressed={visible === role}
                          aria-label={`draw ${name(axis)} as ${role}`}
                          title={visible === role
                            ? `${name(axis)} is drawn as the card's ${role}`
                            : `draw ${name(axis)} as the card's ${role} - display only, the graph is unchanged`}
                          onClick={() => visible !== role && drawAs(axis, role)}
                        >
                          {role === "rows" ? "↕" : "↔"}
                        </button>
                      ))}
                      {!visible && <small>{cfg?.projection ? "proj" : "slice"}</small>}
                    </span>
                  ) : (
                    <small>{visible ?? ""}</small>
                  )}
                </th>
                <td className="num">{extent}</td>
                <td className="num">
                  <ExtentInput
                    value={tile[axis]}
                    max={extent}
                    own={own}
                    label={`${name(axis)} tile extent on ${tensor.name}`}
                    onCommit={(value) => setAxis(axis, value)}
                  />
                </td>
                <td>
                  <code>{box[axis].lo}:{box[axis].hi}</code>
                  {extent % tile[axis] !== 0 && (
                    <small
                      className={onLast ? "boundary on" : "boundary"}
                      title={`${extent} is not a multiple of ${tile[axis]}: the last tile is ${last} wide`}
                    >
                      last {last}
                    </small>
                  )}
                </td>
                <td className="num axis-step">
                  <button
                    className="mini"
                    aria-label={`previous tile along ${name(axis)}`}
                    disabled={box[axis].lo === 0}
                    onClick={() => stepTile(index, axis, -1)}
                  >
                    ‹
                  </button>
                  <span title={position.whole ? undefined : "not on this axis's tile lattice"}>
                    {!position.whole ? "–" : position.aligned ? position.coord : `${position.coord}–${position.last}`}
                    /{position.count}
                  </span>
                  <button
                    className="mini"
                    aria-label={`next tile along ${name(axis)}`}
                    disabled={box[axis].hi === extent}
                    onClick={() => stepTile(index, axis, 1)}
                  >
                    ›
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="axis-selection" aria-label="complete selection">
        {selectionReading(box, name)}
      </p>
    </div>
  );
}

/** `B[0:1] H[2:4] S[128:192] D[0:128]`: every axis, named, so none is implied. */
export function selectionReading(box: Box, name: (axis: number) => string): string {
  return box.map((interval, axis) => `${name(axis)}[${interval.lo}:${interval.hi}]`).join(" ");
}

function ExtentInput({
  value,
  max,
  own,
  label,
  onCommit,
}: {
  value: number;
  max: number;
  own: boolean;
  label: string;
  onCommit: (value: number) => void;
}): React.ReactElement {
  const [draft, setDraft] = useState(String(value));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => {
    setDraft(String(value));
    setInvalid(false);
  }, [value]);
  const commit = () => {
    const parsed = Number(draft.trim());
    if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > max) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    if (parsed !== value) onCommit(parsed);
  };
  return (
    <input
      className={`axis-extent${own ? " own" : ""}${invalid ? " invalid" : ""}`}
      inputMode="numeric"
      value={draft}
      aria-label={label}
      aria-invalid={invalid}
      title={invalid ? `a whole number from 1 to ${max}` : "Enter or blur applies"}
      spellCheck={false}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") commit();
        if (event.key === "Escape") {
          setDraft(String(value));
          setInvalid(false);
        }
      }}
    />
  );
}
