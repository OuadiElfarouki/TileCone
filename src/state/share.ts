import { Box } from "../core/region";
import { AxisMode } from "../view/tensor/shape-label";
import { isViewCfg, type ViewCfg } from "../view/tensor/tensor-view";
import { nodeKey, readNodeKey } from "../view/graph/graph-scene";
import { Direction, InspectorTab, MAX_NODE_OFFSET } from "../view/workspace";
import type { TaskRef } from "../core/plan/plan";
import { MAX_KEPT_PLANS } from "./plan";
import type { State, WorkspaceRestore } from "./types";

/**
 * Shareable workspace state.
 *
 * Encoding and decoding live together on purpose: they were split across a
 * toolbar and `App` before, and when the toolbar was removed the encoder went
 * with it, leaving the app able to restore links nobody could produce. Keeping
 * the pair in one module makes that failure impossible to repeat and lets the
 * round trip be tested directly.
 */
export type WorkspaceLink = {
  dsl: string;
  dir: Direction;
  /** Optional: links written before the combined-with view existed default off. */
  ent?: boolean;
  tile: number;
  /** Optional: links written before snapping existed simply default to on. */
  snap?: boolean;
  /** Optional: links written before the shape view existed keep numeric cards. */
  axes?: AxisMode;
  /** Optional: user displacement from the generated graph layout, keyed by
   * scene node key. Links written before operations could move carry bare
   * tensor ids, which `decodeWorkspace` reads as the `t:` keys they were. */
  pos?: Record<string, [number, number]>;
  /** Missing entries use the default projected view at index zero. */
  views?: Record<string, ViewCfg>;
  /**
   * One entry per drawn part, in order, each naming its own tensor. The order
   * is load-bearing: it is what assigns hues and footprint rows, so parts are
   * listed flat rather than grouped by tensor.
   *
   * Links written before tiles could span tensors carry the older
   * `{ t, boxes }` object; `decodeWorkspace` accepts both.
   */
  sel: { t: string; box: [number, number][] }[] | null;
  /**
   * Optional: the tile plan, as extents per tiled tensor. Links written before
   * plans could be shared carry none and open with nothing tiled.
   */
  plan?: Record<string, number[]>;
  /** Optional: the inspected task, as its tensor and tile coordinate. */
  task?: { t: string; at: number[] };
  /**
   * Optional: the tilings kept for comparison, oldest first. Only the tilings
   * travel. Their figures are measured again when the link is opened, so a
   * link cannot assert figures the graph does not produce.
   */
  kept?: Record<string, number[]>[];
  /** Optional: present only when the link was made in the Plan view. */
  tab?: "plan";
};

type LegacySel = { t: string; boxes: [number, number][][] };

const isBox = (box: unknown): box is [number, number][] =>
  Array.isArray(box) &&
  box.length > 0 &&
  box.every(
    (interval) =>
      Array.isArray(interval) &&
      interval.length === 2 &&
      interval.every((v) => Number.isSafeInteger(v)) &&
      interval[0] < interval[1]
  );

/** Tile extents: one positive whole number per axis. */
const isExtents = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every((v) => Number.isSafeInteger(v) && v > 0);

/** A tiling: at least one tensor, each with extents. Whether the extents fit
 *  the tensor is a question about the graph, answered on restore. */
const isTiling = (value: unknown): value is Record<string, number[]> =>
  !!value && typeof value === "object" && !Array.isArray(value) &&
  Object.keys(value).length > 0 &&
  Object.entries(value).every(([id, extents]) => !!id && isExtents(extents));

const DIRECTIONS: Direction[] = ["none", "backward", "forward", "both"];
const AXIS_MODES: AxisMode[] = ["symbolic", "numeric"];

/** Hash payloads above this are impractical as URLs; callers fall back to JSON. */
/** @internal Exported so the URL/raw-payload boundary is tested directly. */
export const MAX_HASH_LENGTH = 8000;

export function encodeWorkspace(state: WorkspaceLink): string {
  return btoa(unescape(encodeURIComponent(JSON.stringify(state))));
}

/**
 * Parse a `#s=` payload. Returns null for anything malformed rather than
 * throwing or repairing: a link that cannot be trusted should fall back to the
 * default example, not to a half-restored workspace.
 */
export function decodeWorkspace(hash: string): WorkspaceLink | null {
  const m = /^#?s=(.+)$/.exec(hash);
  if (!m) return null;
  // The encoder never emits a URL payload beyond this boundary. Enforce the
  // same limit before base64 decoding so an untrusted location hash cannot make
  // startup allocate and parse an arbitrarily large document.
  if (m[1].length > MAX_HASH_LENGTH) return null;
  try {
    return parseWorkspaceLink(JSON.parse(decodeURIComponent(escape(atob(m[1])))));
  } catch {
    return null;
  }
}

/** The same structural validation for URL payloads and workspace files. */
export function parseWorkspaceLink(value: unknown): WorkspaceLink | null {
  try {
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    const raw = value as Partial<WorkspaceLink>;
    if (typeof raw.dsl !== "string" || !raw.dsl) return null;
    const dir = raw.dir === undefined
      ? "backward"
      : DIRECTIONS.includes(raw.dir as Direction) ? (raw.dir as Direction) : null;
    const tile = raw.tile === undefined
      ? 0
      : typeof raw.tile === "number" && Number.isFinite(raw.tile) ? raw.tile : null;
    const ent = raw.ent === undefined
      ? undefined
      : typeof raw.ent === "boolean" ? raw.ent : null;
    const snap = raw.snap === undefined
      ? true
      : typeof raw.snap === "boolean" ? raw.snap : null;
    // Absent is a legacy link taking its documented default; present-but-wrong
    // is a payload that cannot be trusted, and is refused like any other.
    //
    // This default is `numeric` while a fresh workspace opens on `symbolic`
    // (`store.ts`), and the divergence is the point: a link should restore the
    // cards it was written against, not be reinterpreted by a later default.
    const axes = raw.axes === undefined
      ? "numeric"
      : AXIS_MODES.includes(raw.axes as AxisMode) ? (raw.axes as AxisMode) : null;
    if (dir === null || ent === null || tile === null || snap === null || axes === null) return null;

    let pos: WorkspaceLink["pos"];
    if (raw.pos !== undefined) {
      if (!raw.pos || typeof raw.pos !== "object" || Array.isArray(raw.pos)) return null;
      const entries = Object.entries(raw.pos);
      if (!entries.every(
        ([id, offset]) =>
          !!id && Array.isArray(offset) && offset.length === 2 &&
          offset.every((value) =>
            typeof value === "number" && Number.isFinite(value) &&
            Math.abs(value) <= MAX_NODE_OFFSET
          )
      )) return null;
      pos = Object.fromEntries(
        entries.map(([id, offset]) => [readNodeKey(id) ? id : nodeKey("tensor", id), offset])
      ) as Record<string, [number, number]>;
    }

    let views: WorkspaceLink["views"];
    if (raw.views !== undefined) {
      if (!raw.views || typeof raw.views !== "object" || Array.isArray(raw.views)) return null;
      if (!Object.entries(raw.views).every(([id, cfg]) => !!id && isViewCfg(cfg))) return null;
      views = raw.views;
    }

    let sel: WorkspaceLink["sel"] = null;
    const candidate: unknown = raw.sel;
    if (Array.isArray(candidate)) {
      if (!candidate.every(
        (p): p is { t: string; box: [number, number][] } =>
          !!p && typeof p === "object" && typeof (p as { t?: unknown }).t === "string" &&
          isBox((p as { box?: unknown }).box)
      )) return null;
      if (candidate.length) sel = candidate;
    } else if (candidate && typeof candidate === "object") {
      // Legacy single-tensor form: every box belonged to one tensor.
      const legacy = candidate as Partial<LegacySel>;
      if (typeof legacy.t === "string" && Array.isArray(legacy.boxes)) {
        if (!legacy.boxes.every(isBox)) return null;
        const parts = legacy.boxes.map((box) => ({ t: legacy.t as string, box }));
        if (parts.length) sel = parts;
      } else return null;
    } else if (candidate !== undefined && candidate !== null) return null;

    if (raw.plan !== undefined && !isTiling(raw.plan)) return null;
    if (raw.task !== undefined) {
      const task = raw.task as Partial<{ t: unknown; at: unknown }> | null;
      if (
        !task || typeof task !== "object" || typeof task.t !== "string" ||
        !Array.isArray(task.at) || !task.at.every((v) => Number.isSafeInteger(v) && v >= 0)
      ) return null;
      // A task is a tile of the plan, so there is no task without one.
      if (raw.plan === undefined) return null;
    }
    if (raw.kept !== undefined && (
      !Array.isArray(raw.kept) || raw.kept.length > MAX_KEPT_PLANS || !raw.kept.every(isTiling)
    )) return null;
    if (raw.tab !== undefined && raw.tab !== "plan") return null;
    return {
      dsl: raw.dsl,
      dir,
      ...(ent === undefined ? {} : { ent }),
      tile,
      snap,
      axes,
      ...(pos ? { pos } : {}),
      ...(views ? { views } : {}),
      sel,
      ...(raw.plan ? { plan: raw.plan } : {}),
      ...(raw.task ? { task: raw.task } : {}),
      ...(raw.kept?.length ? { kept: raw.kept } : {}),
      ...(raw.tab ? { tab: raw.tab } : {}),
    };
  } catch {
    return null;
  }
}

/** The ordered selection parts, in link form. */
export function selectionToLink(
  selection: { parts: { tensorId: string; box: Box }[] } | null
): WorkspaceLink["sel"] {
  if (!selection || !selection.parts.length) return null;
  return selection.parts.map((p) => ({
    t: p.tensorId,
    box: p.box.map((interval) => [interval.lo, interval.hi]) as [number, number][],
  }));
}

/** The plan, the inspected task, the kept tilings and the Plan view, in link
 *  form. Each is left out when there is none, so a link without a plan reads
 *  exactly as one written before plans could be shared. */
export function planToLink(state: {
  planTiles: Record<string, number[]>;
  planTask: TaskRef | null;
  keptPlans: readonly { tiles: Record<string, number[]> }[];
  inspectorTab: InspectorTab;
}): Pick<WorkspaceLink, "plan" | "task" | "kept" | "tab"> {
  const tiled = Object.keys(state.planTiles).length > 0;
  return {
    ...(tiled ? { plan: { ...state.planTiles } } : {}),
    ...(tiled && state.planTask
      ? { task: { t: state.planTask.tensorId, at: [...state.planTask.coord] } }
      : {}),
    ...(state.keptPlans.length ? { kept: state.keptPlans.map((k) => ({ ...k.tiles })) } : {}),
    ...(state.inspectorTab === "plan" ? { tab: "plan" as const } : {}),
  };
}

/** Only semantic inputs travel; results and worker identities are rebuilt. */
export function workspaceToLink(state: State): WorkspaceLink {
  return {
    dsl: state.dslText, dir: state.direction, ent: state.showEntangled,
    tile: state.tileScale, snap: state.snapToGrid, axes: state.axisMode,
    views: Object.fromEntries(Object.entries(state.viewCfgs).filter(([, cfg]) =>
      !cfg.projection || !!cfg.tile || !!cfg.axes || cfg.scale !== undefined || cfg.sliders.some((v) => v !== 0))),
    pos: Object.fromEntries(Object.entries(state.nodeOffsets).map(([id, { dx, dy }]) => [id, [dx, dy]])),
    sel: selectionToLink(state.selection),
    ...planToLink(state),
  };
}

export function linkToRestore(link: WorkspaceLink): WorkspaceRestore {
  return {
    dsl: link.dsl, direction: link.dir, showEntangled: link.ent === true,
    tileScale: link.tile, snapToGrid: link.snap !== false, axisMode: link.axes ?? "numeric",
    viewCfgs: link.views,
    nodeOffsets: Object.fromEntries(Object.entries(link.pos ?? {}).map(([id, [dx, dy]]) => [id, { dx, dy }])),
    parts: link.sel?.map((p) => ({ tensorId: p.t, box: p.box.map(([lo, hi]) => ({ lo, hi })) })) ?? null,
    planTiles: link.plan,
    planTask: link.task ? { tensorId: link.task.t, coord: link.task.at } : null,
    keptTilings: link.kept,
    planView: link.tab === "plan",
  };
}

/** A shareable URL, or the raw payload when it is too long to live in one. */
export function shareTarget(origin: string, pathname: string, state: WorkspaceLink): string {
  const hash = encodeWorkspace(state);
  return hash.length > MAX_HASH_LENGTH
    ? JSON.stringify(state)
    : `${origin}${pathname}#s=${hash}`;
}
