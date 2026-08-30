import { Region } from "../core/region";
import { Direction } from "./store";

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
  sel: { t: string; boxes: [number, number][][] } | null;
};

const DIRECTIONS: Direction[] = ["none", "backward", "forward", "both"];

/** Hash payloads above this are impractical as URLs; callers fall back to JSON. */
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
    const dir = DIRECTIONS.includes(raw.dir as Direction) ? (raw.dir as Direction) : "backward";
    const tile = typeof raw.tile === "number" && Number.isFinite(raw.tile) ? raw.tile : 0;
    const snap = typeof raw.snap === "boolean" ? raw.snap : true;

    let sel: WorkspaceLink["sel"] = null;
    const candidate = raw.sel;
    if (candidate && typeof candidate.t === "string" && Array.isArray(candidate.boxes)) {
      const boxes = candidate.boxes.filter(
        (box) =>
          Array.isArray(box) &&
          box.length > 0 &&
          box.every(
            (interval) =>
              Array.isArray(interval) &&
              interval.length === 2 &&
              interval.every((v) => Number.isSafeInteger(v)) &&
              interval[0] < interval[1]
          )
      );
      if (boxes.length) sel = { t: candidate.t, boxes: boxes as [number, number][][] };
    }
    return { dsl: raw.dsl, dir, tile, snap, sel };
  } catch {
    return null;
  }
}

/** The ordered selection parts, in link form. */
export function selectionToLink(
  selection: { tensorId: string; region: Region } | null
): WorkspaceLink["sel"] {
  if (!selection) return null;
  return {
    t: selection.tensorId,
    boxes: selection.region.boxes.map(
      (box) => box.map((interval) => [interval.lo, interval.hi]) as [number, number][]
    ),
  };
}

/** A shareable URL, or the raw payload when it is too long to live in one. */
export function shareTarget(origin: string, pathname: string, state: WorkspaceLink): string {
  const hash = encodeWorkspace(state);
  return hash.length > MAX_HASH_LENGTH
    ? JSON.stringify(state)
    : `${origin}${pathname}#s=${hash}`;
}
