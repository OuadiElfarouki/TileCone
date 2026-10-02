/**
 * The menus a tensor offers, as data (`view/menu.ts`).
 *
 * Two owners, kept apart by what their settings belong to. The axis table's
 * menu acts on the inspected tile and the tensor's tile, which a gesture uses.
 * A card's view menu acts on how the tensor is drawn, which belongs to the
 * tensor whatever tile is being studied, and so is reached from the card.
 */

import type { MenuSpec } from "../menu";
import { remapped, ViewCfg, viewAxes } from "./tensor-view";

export type AxisTableAction = "own" | "reset" | "plan";

export function axisTableMenu({
  own,
  matchesTile,
  input,
}: {
  /** The tensor has a tile of its own. */
  own: boolean;
  /** The inspected tile is one tile of the tensor's lattice. */
  matchesTile: boolean;
  /** The tensor is a graph input, which no task computes. */
  input: boolean;
}): MenuSpec<AxisTableAction> {
  return [
    {
      kind: "action",
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
      kind: "action",
      id: "reset",
      label: "reset tile",
      disabled: !own,
      title: own
        ? "return this tensor to the canvas tile: the detail setting on the visible axes and the view mode on the others"
        : "the tensor already follows the canvas tile",
    },
    {
      kind: "action",
      id: "plan",
      label: "plan with this tile",
      disabled: input,
      title: input
        ? "a graph input has no tasks: nothing computes it"
        : "divide this tensor into tasks of these extents and inspect the task holding this tile",
    },
  ];
}

export type CardViewAction = "swap" | "default";
export type CardViewChoice = "rows" | "cols";

/**
 * Which axes a card draws. Display only: the graph, every selection and every
 * figure are unchanged. Empty for a tensor of rank below two, which has no
 * pair to choose.
 */
export function cardViewMenu(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes"> | undefined,
  axisName: (axis: number) => string
): MenuSpec<CardViewAction, CardViewChoice> {
  if (shape.length < 2) return [];
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const role = (id: CardViewChoice, label: string, drawn: number, way: string) => ({
    kind: "choice" as const,
    id,
    label,
    options: shape.map((_, axis) => ({
      value: axis,
      label: axisName(axis),
      checked: axis === drawn,
      title: axis === drawn
        ? `${axisName(axis)} is drawn ${way}`
        : `draw ${axisName(axis)} ${way} - display only, the graph is unchanged`,
    })),
  });
  const plain = viewAxes(shape);
  return [
    role("rows", "rows", rowAxis, "down the card"),
    role("cols", "columns", colAxis, "across the card"),
    { kind: "separator" },
    {
      kind: "action",
      id: "swap",
      label: "swap rows and columns",
      disabled: false,
      title: `draw ${axisName(colAxis)} down and ${axisName(rowAxis)} across`,
    },
    {
      kind: "action",
      id: "default",
      label: "draw default plane",
      disabled: !remapped(shape, cfg),
      title: remapped(shape, cfg)
        ? `draw the last two axes again: ${axisName(plain.rowAxis)} down, ${axisName(plain.colAxis)} across`
        : "the card already draws the last two axes",
    },
  ];
}
