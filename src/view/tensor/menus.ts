/**
 * The menus a tensor offers, as data (`view/menu.ts`).
 *
 * Two owners, kept apart by what their settings belong to. The axis table's
 * menu acts on the inspected tile and the tensor's tile, which a gesture uses.
 * A card's view menu acts on how the tensor is drawn, which belongs to the
 * tensor whatever tile is being studied, and so is reached from the card.
 */

import type { MenuChoice, MenuSpec } from "../menu";
import { scaleRatioLabel } from "./card-size";
import {
  hiddenAxes,
  hiddenMode,
  remapped,
  UNFOLD_MARK,
  unfoldRefusal,
  type UnfoldDir,
  ViewCfg,
  viewAxes,
} from "./tensor-view";

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

export type CardViewAction = "swap" | "default" | "resize" | "resetSize";
export type CardViewChoice = "rows" | "cols" | "hidden";

/** `cfg` is the view as drawn; `stored` is the one the card was resized to,
 *  which differs while its neighbours leave it less room. */
export function cardDisplayMenu(shape: readonly number[], cfg: ViewCfg, graphPx: number, movable: boolean, resizing: boolean,
  stored: ViewCfg = cfg): MenuSpec<CardViewAction, CardViewChoice> {
  const text = scaleRatioLabel(shape, cfg, graphPx) ?? "×1";
  const limited = stored.scale !== cfg.scale;
  return [
    { kind: "info", label: `display scale ${text}${limited ? ", limited by neighbours" : ""}`,
      title: "Relative to the graph's scale; display size changes no tensor dimensions or analysis." +
        (limited ? ` Resized to ${scaleRatioLabel(shape, stored, graphPx) ?? "×1"}, it is drawn smaller until its neighbours leave room.` : "") },
    { kind: "action", id: "resize", label: resizing ? "finish resizing" : "resize card", disabled: !movable,
      title: movable ? "enter Move and drag a right corner to resize; Escape or clicking elsewhere finishes resizing" : "the Plan view keeps graph placement fixed" },
    { kind: "action", id: "resetSize", label: "reset display size", disabled: !movable || cfg.scale === undefined,
      title: "return to this card's automatically fitted size (undoable)" },
  ];
}

/**
 * The readings a card offers for its hidden axes, in menu order. A choice's
 * value is an index into this list.
 */
export const HIDDEN_VIEWS: readonly (
  | { mode: "project" }
  | { mode: "slice" }
  | { mode: "unfold"; dir: UnfoldDir }
)[] = [
  { mode: "project" },
  { mode: "slice" },
  { mode: "unfold", dir: "across" },
  { mode: "unfold", dir: "down" },
  { mode: "unfold", dir: "grid" },
];

/** The view settings that select `HIDDEN_VIEWS[value]`. */
export function hiddenViewCfg(value: number): Pick<ViewCfg, "projection" | "unfold"> {
  const view = HIDDEN_VIEWS[value];
  if (view.mode === "unfold") return { projection: false, unfold: view.dir };
  return { projection: view.mode === "project", unfold: undefined };
}

function hiddenChoice(
  shape: readonly number[],
  cfg: Partial<Pick<ViewCfg, "axes" | "projection" | "unfold">> | undefined,
  axisName: (axis: number) => string
): MenuChoice<"hidden"> {
  const hidden = hiddenAxes(shape, cfg);
  const names = hidden.map(axisName).join(", ");
  const fastest = axisName(hidden[hidden.length - 1]);
  const slowest = hidden.slice(0, -1).map(axisName).join(", ");
  const mode = hiddenMode(cfg);
  const titles = {
    project: `draw each cell as the union over ${names}; a drag takes all of them`,
    slice: `draw one index of ${names} at a time, the rest stacked behind it; sliders pick the index`,
    across: `lay every index of ${names} side by side across the card, slowest axis outermost`,
    down: `stack every index of ${names} down the card, slowest axis outermost: the row-major flattening`,
    grid: `lay ${fastest} across the card and ${slowest} down it`,
  };
  const options = HIDDEN_VIEWS.flatMap((view, value) => {
    // A grid differs from across only when there are two hidden axes to split.
    if (view.mode === "unfold" && view.dir === "grid" && hidden.length < 2) return [];
    const key = view.mode === "unfold" ? view.dir : view.mode;
    const checked = view.mode === mode && (view.mode !== "unfold" || view.dir === cfg?.unfold);
    const refusal = view.mode === "unfold" ? unfoldRefusal(shape, { ...cfg, unfold: view.dir }) : null;
    const disabled = refusal !== null;
    return [{
      value,
      label: view.mode === "unfold" ? `unfold ${UNFOLD_MARK[view.dir]}` : view.mode,
      title: disabled ? `cannot unfold: ${refusal}` : titles[key],
      checked,
      disabled,
    }];
  });
  return { kind: "choice", id: "hidden", label: "hidden", options };
}

/**
 * How a card draws its tensor: how it reads the axes it does not draw, for a
 * tensor that has any, and which pair it draws. Display only: the graph, every
 * selection and every figure are unchanged. Empty for a tensor of rank below
 * two, which has no pair to choose.
 */
export function cardViewMenu(
  shape: readonly number[],
  cfg: Partial<Pick<ViewCfg, "axes" | "projection" | "unfold">> | undefined,
  axisName: (axis: number) => string
): MenuSpec<CardViewAction, CardViewChoice> {
  if (shape.length < 2) return [];
  const hidden: MenuSpec<CardViewAction, CardViewChoice> = shape.length > 2
    ? [hiddenChoice(shape, cfg, axisName), { kind: "separator" }]
    : [];
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
    ...hidden,
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
