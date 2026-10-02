import React, { useEffect, useState } from "react";
import type { Box } from "../../core/region";
import { tileOf } from "../../view/tensor/grid";
import { useStore } from "../../state/store";
import { axisTableMenu, type AxisTableAction } from "../../view/tensor/menus";
import { viewAxes } from "../../view/tensor/tensor-view";
import { gestureTile, isLatticeTile, lastTileExtent, tilePosition } from "../../view/tensor/tile-spec";
import { OptionsMenu, OptionsMenuButton, useOptionsMenu } from "../chrome/OptionsMenu";

/**
 * The inspected tile, one row per axis.
 *
 * The canvas names two axes; this names all of them. Each row states the
 * axis, its extent, the tensor's tile on it, and the inspected tile's range,
 * so a rectangle on a card never stands for more than the reader can see
 * written down. The tile column is editable: typing an extent gives the tensor
 * a tile of its own (`ViewCfg.tile`) and refits the inspected tile to it, so
 * `[1, 2, 64, 128]` on `[B, H, S, D]` is four entries rather than a gesture
 * nobody can make. Actions on the tile - keeping it as the tensor's tile,
 * resetting that, planning with it - are occasional, and sit in a menu behind
 * ⋯ or a right-click. Which axes the card draws is the tensor's own setting,
 * whatever tile is studied, and is chosen on the card (`cardViewMenu`); the
 * table only shades the axes the card does not draw.
 */
export function AxisEditor({ index }: { index: number }): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const part = useStore((s) => s.selection?.parts[index]);
  const cfg = useStore((s) => (part ? s.viewCfgs[part.tensorId] : undefined));
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const setTensorTile = useStore((s) => s.setTensorTile);
  const setPlanTileAt = useStore((s) => s.setPlanTileAt);
  const setInspectorTab = useStore((s) => s.setInspectorTab);

  // Before the early returns: a hook's position in the call order is fixed.
  const menu = useOptionsMenu();

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
  const matchesTile = isLatticeTile(box, tile, shape);
  // What "use as tile" and "plan with this tile" take: the tile itself when
  // this is one, since a shortened boundary tile's extents are not the tiling's.
  const tileExtents = matchesTile ? tile : extents;
  const setAxis = (axis: number, extent: number) => {
    const next = tile.slice();
    next[axis] = extent;
    setTensorTile(part.tensorId, next, index);
  };
  const actions: Record<AxisTableAction, () => void> = {
    own: () => setTensorTile(part.tensorId, tileExtents, index),
    reset: () => setTensorTile(part.tensorId, null),
    plan: () => {
      setPlanTileAt(part.tensorId, tileExtents, box.map((interval) => interval.lo));
      setInspectorTab("plan");
    },
  };

  return (
    <div className="ins-section axis-editor" onContextMenu={menu.onContextMenu}>
      <div className="ins-title with-action">
        Axes
        <OptionsMenuButton menu={menu} label="axis options" />
      </div>
      <OptionsMenu
        menu={menu}
        label="axis options"
        spec={axisTableMenu({ own, matchesTile, input: !tensor.producer })}
        handlers={{ action: actions, choice: {} }}
      />
      <table className="axis-table">
        <thead>
          <tr>
            <th scope="col">axis</th>
            <th scope="col" className="num">size</th>
            <th scope="col" className="num" title={own ? "this tensor's tile" : "the canvas tile; type to give the tensor its own"}>
              tile
            </th>
            <th scope="col">range</th>
          </tr>
        </thead>
        <tbody>
          {shape.map((extent, axis) => {
            const position = tilePosition(box[axis], tile[axis], extent);
            const last = lastTileExtent(extent, tile[axis]);
            const onLast = position.coord === position.count - 1 && last !== tile[axis];
            const drawn = axis === rowAxis ? "rows" : axis === colAxis ? "columns" : null;
            return (
              <tr key={axis} className={drawn ? undefined : "hidden-axis"}>
                <th
                  scope="row"
                  title={drawn
                    ? `drawn as the card's ${drawn}`
                    : "not drawn: the card shows a slice or the union; right-click the card to draw it"}
                >
                  {name(axis)}
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
  /** Enter is a deliberate act, so on a tensor without a tile of its own it
   *  commits even an unchanged value, which makes the canvas tile its own.
   *  Blur is not: tabbing through the fields must not change anything. */
  const commit = (explicit: boolean) => {
    const parsed = Number(draft.trim());
    if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > max) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    if (parsed !== value || (explicit && !own)) onCommit(parsed);
  };
  return (
    <input
      className={`axis-extent${own ? " own" : ""}${invalid ? " invalid" : ""}`}
      inputMode="numeric"
      value={draft}
      aria-label={label}
      aria-invalid={invalid}
      title={invalid
        ? `a whole number from 1 to ${max}`
        : own ? "Enter or blur applies" : "Enter or blur applies; Enter keeps the canvas tile as this tensor's own"}
      spellCheck={false}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={() => commit(false)}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") commit(true);
        if (event.key === "Escape") {
          setDraft(String(value));
          setInvalid(false);
        }
      }}
    />
  );
}
