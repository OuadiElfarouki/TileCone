import { Box } from "../core/region";
import { Direction } from "./store";
import { AxisMode } from "./shape-label";

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
  tile: number;
  /** Optional: links written before snapping existed simply default to on. */
  snap?: boolean;
  /** Optional: links written before the shape view existed keep numeric cards. */
  axes?: AxisMode;
  /**
   * One entry per drawn part, in order, each naming its own tensor. The order
   * is load-bearing: it is what assigns hues and footprint rows, so parts are
   * listed flat rather than grouped by tensor.
   *
   * Links written before tiles could span tensors carry the older
   * `{ t, boxes }` object; `decodeWorkspace` accepts both.
   */
  sel: { t: string; box: [number, number][] }[] | null;
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
  try {
    const raw = JSON.parse(decodeURIComponent(escape(atob(m[1])))) as Partial<WorkspaceLink>;
    if (typeof raw.dsl !== "string" || !raw.dsl) return null;
    const dir = raw.dir === undefined
      ? "backward"
      : DIRECTIONS.includes(raw.dir as Direction) ? (raw.dir as Direction) : null;
    const tile = raw.tile === undefined
      ? 0
      : typeof raw.tile === "number" && Number.isFinite(raw.tile) ? raw.tile : null;
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
    if (dir === null || tile === null || snap === null || axes === null) return null;

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
    return { dsl: raw.dsl, dir, tile, snap, axes, sel };
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

/** A shareable URL, or the raw payload when it is too long to live in one. */
export function shareTarget(origin: string, pathname: string, state: WorkspaceLink): string {
  const hash = encodeWorkspace(state);
  return hash.length > MAX_HASH_LENGTH
    ? JSON.stringify(state)
    : `${origin}${pathname}#s=${hash}`;
}
