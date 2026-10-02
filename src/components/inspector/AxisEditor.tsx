import React, { useEffect, useState } from "react";
import type { Box } from "../../core/region";
import { tileOf } from "../../view/tensor/grid";
import { useStore } from "../../state/store";
import { remapped, viewAxes } from "../../view/tensor/tensor-view";
import { gestureTile, isLatticeTile, lastTileExtent, tilePosition } from "../../view/tensor/tile-spec";
import { OptionsMenu, useOptionsMenu, type MenuItem } from "../chrome/OptionsMenu";

/** The axis table's occasional actions, and whether each applies to this tile. */
export type AxisOption = {
  id: "own" | "reset" | "plane" | "plan";
  label: string;
  title: string;
  disabled: boolean;
};

export function axisOptions({
  own,
  matchesTile,
  planeRemapped,
  input,
}: {
  /** The tensor has a tile of its own. */
  own: boolean;
  /** The inspected tile is one tile of the tensor's lattice. */
  matchesTile: boolean;
  /** The card draws a pair of axes other than the last two. */
  planeRemapped: boolean;
  /** The tensor is a graph input, which no task computes. */
  input: boolean;
}): AxisOption[] {
  return [
    {
      id: "own",
      label: "use as tile",
      // Offered whenever the tensor has no tile of its own, even when this
      // tile already matches the canvas lattice: owning it is still a change,
      // because a detail change would otherwise resize it.
      disabled: own && matchesTile,
      title: own && matchesTile
        ? "this tile is already the tensor's tile"
        : matchesTile
          ? "keep this tile as the tensor's own, so later detail changes no longer resize it"
          : "make this tile's extents the tensor's tile, so later gestures and steps use them",
    },
    {
      id: "reset",
      label: "reset tile",
      disabled: !own,
      title: own
        ? "return this tensor to the canvas tile: the detail setting on the visible axes and the view mode on the others"
        : "the tensor already follows the canvas tile",
    },
    {
      id: "plane",
      label: "draw default plane",
      disabled: !planeRemapped,
      title: planeRemapped
        ? "draw the last two axes again, the default plane"
        : "the card already draws the last two axes",
    },
    {
      id: "plan",
      label: "plan with this tile",
      disabled: input,
      title: input
        ? "a graph input has no tasks: nothing computes it"
        : "divide this tensor into tasks of these extents and inspect the task holding this tile",
    },
  ];
}

/**
 * The inspected tile, one row per axis.
 *
 * The canvas names two axes; this names all of them. Each row states the
 * axis, its extent, the tensor's tile on it, and the inspected tile's range,
 * so a rectangle on a card never stands for more than the reader can see
 * written down. The tile column is editable: typing an extent gives the tensor
 * a tile of its own (`ViewCfg.tile`) and refits the inspected tile to it, so
 * `[1, 2, 64, 128]` on `[B, H, S, D]` is four entries rather than a gesture
 * nobody can make. Actions that change the tensor's tile, its drawn plane or
 * the plan are occasional, and sit in a menu behind ⋯ or a right-click.
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
  const setViewAxes = useStore((s) => s.setViewAxes);

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
  const run: Record<AxisOption["id"], () => void> = {
    own: () => setTensorTile(part.tensorId, tileExtents, index),
    reset: () => setTensorTile(part.tensorId, null),
    plane: () => setViewAxes(part.tensorId, null, index),
    plan: () => {
      setPlanTileAt(part.tensorId, tileExtents, box.map((interval) => interval.lo));
      setInspectorTab("plan");
    },
  };
  const items: MenuItem[] = axisOptions({
    own,
    matchesTile,
    planeRemapped,
    input: !tensor.producer,
  }).map((option) => ({ ...option, onSelect: run[option.id] }));

  return (
    <div className="ins-section axis-editor" onContextMenu={menu.onContextMenu}>
      <div className="ins-title with-action">
        Axes
        <OptionsMenu menu={menu} label="axis options" items={items} />
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
