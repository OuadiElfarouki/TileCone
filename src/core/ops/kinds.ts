/**
 * Attribute kinds: the closed set of value types an operation's attributes are
 * written in.
 *
 * An operation declares each attribute as one of these, and the kind is the one
 * place that says which forms a value may take and what it means. `stride=2`
 * on a 2-D convolution is `[2, 2]` because `spatial` says so, for every
 * operation with a spatial attribute; `axis=seq` is an index because `axis`
 * says so, for every operation that takes an axis. No operation reads a written
 * form, and no front end converts one: both see the same kinds.
 *
 * A kind works in two steps, because the second needs the operands:
 *
 * - `check` is context-free. It runs while the graph is validated, before any
 *   shape is known, and refuses a value that is none of the kind's forms.
 * - `canonical` runs during shape inference, with the inputs' shapes and axis
 *   names, and produces the one form operation code reads: non-negative axis
 *   indices, one entry per spatial axis, `[lo, hi]` pairs.
 *
 * The source graph keeps what was written, so a printer can write it back; only
 * the resolved graph holds canonical values.
 */

import { DTYPES } from "../dtypes";
import { resolveInteger, resolveShape, type Sym } from "../shapes";
import type { Attrs, AxisNames } from "./types";

/** What `canonical` may consult. */
export type KindCtx = {
  inShapes: number[][];
  /** Per input, one name per axis; holes are axes with no name. */
  inAxisNames: AxisNames[];
  /** The node's attributes as written, for a default that refers to another one. */
  written: Attrs;
  /** Bound parameters and derived symbols, for a kind that sizes things. */
  params: Record<string, number>;
  /** Declared outputs, to bound attributes that expand into one entry per output. */
  outputCount: number;
};

/** A written value the operands rule out, with the reason in the kind's terms. */
export class KindIssue extends Error {}

/**
 * What sort of value a kind holds, for a front end that has to spell it: free
 * text is quoted, a dtype or an enum member is a bare word, everything else is
 * numbers, words and lists. A kind is identified by this, never by the name of
 * the attribute it happens to be used for.
 */
export type KindType =
  | "int" | "bool" | "enum" | "text" | "dtype" | "dtypes" | "ints"
  | "shape" | "shapes" | "axis" | "axes" | "permutation"
  | "spatial" | "padding" | "padWidth" | "spatialRank" | "splitSizes" | "sliceBounds";

export type Kind = {
  type: KindType;
  /** The written forms, as a noun phrase: "an int ≥ 1, or one per spatial axis". */
  describe: string;
  /** Null when `value` is one of the written forms, otherwise why it is not. */
  check(value: unknown): string | null;
  /** The canonical value. Only called on a value `check` accepted; throws `KindIssue`. */
  canonical(value: unknown, ctx: KindCtx): unknown;
  /**
   * The written value to use when the attribute is absent, computed from the
   * operands or another attribute. Without one the attribute is required,
   * unless it is `optional`.
   */
  default?: (ctx: KindCtx) => unknown;
  /** Absent stays absent: operation code tests for it. */
  optional?: true;
  /** The shortest written form meaning the same as `value`, for a printer. */
  shorten?(value: unknown): unknown;
};

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v);

/** A plain array of `n` items, or of any length when `n` is undefined. */
const isList = (v: unknown, n?: number): v is unknown[] =>
  Array.isArray(v) && (n === undefined || v.length === n);

const bound = (min?: number, max?: number) =>
  min !== undefined && max !== undefined
    ? ` from ${min} to ${max}`
    : min !== undefined
      ? ` ≥ ${min}`
      : max !== undefined
        ? ` ≤ ${max}`
        : "";

const within = (v: number, min?: number, max?: number) =>
  (min === undefined || v >= min) && (max === undefined || v <= max);

/** A kind whose canonical value is the written one. */
function plain(type: KindType, describe: string, check: (v: unknown) => boolean): Kind {
  return {
    type,
    describe,
    check: (v) => (check(v) ? null : `expected ${describe}`),
    canonical: (v) => (Array.isArray(v) ? structuredClone(v) : v),
  };
}

// ------------------------------------------------------------------ scalars

export function int(opts: { min?: number; max?: number; default?: number } = {}): Kind {
  const kind = plain("int", `an int${bound(opts.min, opts.max)}`, (v) => isInt(v) && within(v, opts.min, opts.max));
  return opts.default === undefined ? kind : { ...kind, default: () => opts.default };
}

export function bool(opts: { default?: boolean } = {}): Kind {
  const kind = plain("bool", "true or false", (v) => typeof v === "boolean");
  return opts.default === undefined ? kind : { ...kind, default: () => opts.default };
}

export function oneOf(values: readonly string[], opts: { default?: string } = {}): Kind {
  const kind = plain("enum", `one of ${values.join(", ")}`, (v) => typeof v === "string" && values.includes(v));
  return opts.default === undefined ? kind : { ...kind, default: () => opts.default };
}

export function text(opts: { nonEmpty?: boolean } = {}): Kind {
  return plain("text", opts.nonEmpty ? "a non-empty string" : "a string", (v) =>
    typeof v === "string" && (!opts.nonEmpty || v.length > 0)
  );
}

export const dtype = (): Kind => ({ ...oneOf(DTYPES), type: "dtype" });

/** One dtype per output; `null` leaves that output's dtype to inference. */
export const dtypeList = (): Kind =>
  plain("dtypes", "a list of dtypes or null", (v) =>
    isList(v) && v.every((d) => d === null || (typeof d === "string" && (DTYPES as readonly string[]).includes(d)))
  );

export const intList = (opts: { min?: number } = {}): Kind =>
  plain("ints", `a list of ints${bound(opts.min)}`, (v) => isList(v) && v.every((x) => isInt(x) && within(x, opts.min)));

// --------------------------------------------------------------- extents

/** An int ≥ 1 or a dimension expression over the graph's symbols. */
const isExtent = (v: unknown) => (isInt(v) && v >= 1) || (typeof v === "string" && v.length > 0);

/** A list of extents: a target shape, or the sizes a split cuts. */
export const shape = (): Kind =>
  plain("shape", "a list of extents (ints ≥ 1 or dimension expressions)", (v) => isList(v) && v.every(isExtent));

/** Several shapes, one per output. */
export const shapeList = (): Kind =>
  plain("shapes", "a non-empty list of shapes", (v) => isList(v) && v.length > 0 && v.every((s) => isList(s) && s.every(isExtent)));

// ------------------------------------------------------------------ axes

const isAxis = (v: unknown) => isInt(v) || (typeof v === "string" && v.length > 0);

/** One written axis, as an index into an operand of this rank. */
function resolveAxis(v: unknown, rank: number, names: AxisNames): number {
  if (typeof v === "string") {
    const at = names.indexOf(v);
    if (at < 0) {
      const named = names.filter((n): n is string => n !== undefined);
      throw new KindIssue(
        named.length
          ? `no axis named "${v}"; the axes are ${names.map((n, i) => n ?? String(i)).join(", ")}`
          : `no axis named "${v}"; this operand's axes have no names`
      );
    }
    // A contraction can carry one name onto two axes - query and key positions
    // are both `seq` after `Q @ K^T` - and picking either would be a guess.
    const also = names.indexOf(v, at + 1);
    if (also >= 0)
      throw new KindIssue(`"${v}" names more than one axis (${at} and ${also}); write the index`);
    return at;
  }
  const axis = (v as number) < 0 ? (v as number) + rank : (v as number);
  if (axis < 0 || axis >= rank) throw new KindIssue(`axis ${v} is out of range for rank ${rank}`);
  return axis;
}

const operand = (ctx: KindCtx, of: number) => ({
  rank: ctx.inShapes[of]?.length ?? 0,
  names: ctx.inAxisNames[of] ?? [],
});

const AXIS = "an axis (an int, negative from the end, or an axis name)";

/** One axis of input `of`. Canonical: its non-negative index. */
export function axis(opts: { of?: number; default?: number | string } = {}): Kind {
  const of = opts.of ?? 0;
  return {
    type: "axis",
    describe: AXIS,
    check: (v) => (isAxis(v) ? null : `expected ${AXIS}`),
    canonical: (v, ctx) => {
      const { rank, names } = operand(ctx, of);
      return resolveAxis(v, rank, names);
    },
    ...(opts.default !== undefined ? { default: () => opts.default } : {}),
  };
}

/**
 * Some axes of input `of`, written as one axis or a list. Canonical: distinct
 * indices in ascending order. With `all`, leaving it out means every axis.
 */
export function axes(opts: { of?: number; all?: boolean; default?: number | string | (number | string)[] } = {}): Kind {
  const of = opts.of ?? 0;
  const describe = opts.all
    ? "an axis or a list of axes (empty only for a scalar)"
    : "an axis or a non-empty list of axes";
  const fallback = opts.all
    ? (ctx: KindCtx) => Array.from({ length: operand(ctx, of).rank }, (_, i) => i)
    : opts.default !== undefined
      ? () => opts.default
      : undefined;
  return {
    type: "axes",
    describe,
    check: (v) => (isAxis(v) || (isList(v) && (v.length > 0 || opts.all) && v.every(isAxis)) ? null : `expected ${describe}`),
    canonical: (v, ctx) => {
      const { rank, names } = operand(ctx, of);
      const list = isList(v) ? v : [v];
      if (list.length === 0 && rank !== 0)
        throw new KindIssue("an empty list of axes is only valid for a scalar");
      return [...new Set(list.map((a) => resolveAxis(a, rank, names)))].sort((x, y) => x - y);
    },
    ...(fallback ? { default: fallback } : {}),
  };
}

/** Every axis of input `of`, each once, in the order the output takes them. */
export function permutation(opts: { of?: number } = {}): Kind {
  const of = opts.of ?? 0;
  const describe = "a list naming every axis once";
  return {
    type: "permutation",
    describe,
    check: (v) => (isList(v) && v.every(isAxis) ? null : `expected ${describe}`),
    canonical: (v, ctx) => {
      const { rank, names } = operand(ctx, of);
      const list = (v as unknown[]).map((a) => resolveAxis(a, rank, names));
      if (list.length !== rank || new Set(list).size !== rank)
        throw new KindIssue(`[${(v as unknown[]).join(", ")}] does not name each of the ${rank} axes once`);
      return list;
    },
  };
}

// --------------------------------------------------------------- spatial

/** The axes after batch and channels, for an `N, C, ...` operand. */
const spatialRank = (ctx: KindCtx, of: number) => operand(ctx, of).rank - 2;

/**
 * One int per spatial axis of input `of`, or one int for all of them: a
 * stride, a dilation, a kernel size.
 */
export function spatial(opts: { of?: number; min?: number; default?: number | ((ctx: KindCtx) => unknown) } = {}): Kind {
  const of = opts.of ?? 0;
  const item = `an int${bound(opts.min)}`;
  const describe = `${item}, or one per spatial axis`;
  const ok = (x: unknown) => isInt(x) && within(x, opts.min);
  const fallback =
    typeof opts.default === "function" ? opts.default : opts.default !== undefined ? () => opts.default : undefined;
  return {
    type: "spatial",
    describe,
    check: (v) => (ok(v) || (isList(v) && v.every(ok)) ? null : `expected ${describe}`),
    canonical: (v, ctx) => {
      const n = spatialRank(ctx, of);
      // An operand with no spatial axes is the operation's own error to report,
      // in terms of the layout it expects.
      if (n < 1) return isList(v) ? v.slice() : [];
      if (!isList(v)) return Array.from({ length: n }, () => v);
      if (v.length !== n)
        throw new KindIssue(`has ${v.length} entries for ${n} spatial ${n === 1 ? "axis" : "axes"}`);
      return v.slice();
    },
    shorten: (v) => (isList(v) && v.length > 0 && v.every((x) => x === v[0]) ? v[0] : v),
    ...(fallback ? { default: fallback } : {}),
  };
}

const isPair = (v: unknown): v is [number, number] =>
  isList(v, 2) && v.every((x) => isInt(x) && x >= 0);
const isCount = (v: unknown) => isInt(v) && v >= 0;

/** `[lo, hi]` pairs for `n` axes from an int, per-axis ints, or per-axis pairs. */
function pairs(v: unknown, n: number): [number, number][] {
  if (isInt(v)) return Array.from({ length: n }, () => [v, v] as [number, number]);
  const list = v as unknown[];
  if (list.length !== n)
    throw new KindIssue(`has ${list.length} entries for ${n} ${n === 1 ? "axis" : "axes"}`);
  return list.map((x) => (isInt(x) ? [x, x] : [(x as number[])[0], (x as number[])[1]]));
}

/**
 * Padding on each spatial axis of input `of`: one int for every side, one int
 * per axis for both of its sides, or one `[lo, hi]` pair per axis.
 */
export function padding(opts: { of?: number; default?: number } = {}): Kind {
  const of = opts.of ?? 0;
  const describe = "an int ≥ 0, one per spatial axis, or one [lo, hi] pair per spatial axis";
  return {
    type: "padding",
    describe,
    check: (v) =>
      isCount(v) || (isList(v) && v.every((x) => isCount(x) || isPair(x))) ? null : `expected ${describe}`,
    canonical: (v, ctx) => {
      const n = spatialRank(ctx, of);
      if (n < 1) return isList(v) ? v.slice() : [];
      return pairs(v, n);
    },
    shorten: (v) => {
      if (!isList(v) || !v.length) return v;
      // Each pair to one int where both sides agree, then the list to one int
      // where every axis agrees.
      const each = v.map((x) => (isPair(x) && x[0] === x[1] ? x[0] : x));
      return each.every((x) => isInt(x) && x === each[0]) ? each[0] : each;
    },
    ...(opts.default !== undefined ? { default: () => opts.default } : {}),
  };
}

/**
 * Padding on every axis of input `of`, in NumPy's `pad_width` forms: one int
 * for every side, one `[lo, hi]` pair for every axis, or one pair per axis.
 */
export function padWidth(opts: { of?: number } = {}): Kind {
  const of = opts.of ?? 0;
  const describe = "an int ≥ 0, a [lo, hi] pair, or one [lo, hi] pair per axis";
  return {
    type: "padWidth",
    describe,
    check: (v) => (isCount(v) || isPair(v) || (isList(v) && v.every(isPair)) ? null : `expected ${describe}`),
    canonical: (v, ctx) => {
      const n = operand(ctx, of).rank;
      if (isInt(v)) return pairs(v, n);
      // A lone pair is every axis's padding. A list of pairs is per axis, and
      // is told apart from a lone pair by its items being lists.
      if (isPair(v)) return Array.from({ length: n }, () => [v[0], v[1]] as [number, number]);
      return pairs(v, n);
    },
    shorten: (v) => {
      if (!isList(v) || !v.length || !v.every(isPair)) return v;
      const [lo, hi] = v[0] as [number, number];
      if (!v.every((x) => (x as number[])[0] === lo && (x as number[])[1] === hi)) return v;
      return lo === hi ? lo : [lo, hi];
    },
  };
}

/**
 * How many spatial axes input `of` has, for an operation written for one
 * count (`conv2d`). Canonical: that count. Left out, it is read off the operand.
 */
export function spatialRankOf(opts: { of?: number } = {}): Kind {
  const of = opts.of ?? 0;
  return {
    type: "spatialRank",
    describe: "1, 2 or 3",
    check: (v) => (isInt(v) && v >= 1 && v <= 3 ? null : "expected 1, 2 or 3"),
    canonical: (v, ctx) => {
      const n = spatialRank(ctx, of);
      if (n !== v)
        throw new KindIssue(
          `expects ${v} spatial ${v === 1 ? "axis" : "axes"} (a rank-${(v as number) + 2} input), ` +
            `got a rank-${n + 2} input`
        );
      return v;
    },
    default: (ctx) => spatialRank(ctx, of),
  };
}

// ----------------------------------------------------------------- splits

/**
 * Where a split cuts the axis named by attribute `axisAttr` of input `of`: one
 * extent per piece, or one extent for equal pieces with a shorter last one.
 * Canonical: one extent per piece. Equal pieces keep the extent as written,
 * symbol and all; only a shorter last piece is a plain number.
 */
export function splitSizes(opts: { of?: number; axisAttr: string }): Kind {
  const of = opts.of ?? 0;
  const describe = "an extent for equal pieces, or a non-empty list of extents, one per piece";
  return {
    type: "splitSizes",
    describe,
    check: (v) => (isExtent(v) || (isList(v) && v.length > 0 && v.every(isExtent)) ? null : `expected ${describe}`),
    canonical: (v, ctx) => {
      if (isList(v)) return v.slice();
      const { rank, names } = operand(ctx, of);
      const written = ctx.written[opts.axisAttr];
      // The axis attribute reports its own problems; without a usable one there
      // is no extent to divide, and nothing more to say here.
      let along: number;
      try {
        along = resolveAxis(written ?? 0, rank, names);
      } catch {
        return [v];
      }
      const extent = ctx.inShapes[of][along];
      let piece: number;
      try {
        piece = resolveShape([v as Sym], ctx.params)[0];
      } catch (e) {
        throw new KindIssue((e as Error).message);
      }
      const whole = Math.floor(extent / piece);
      const rest = extent - whole * piece;
      const count = whole + Number(rest !== 0);
      if (count !== ctx.outputCount)
        throw new KindIssue(`declares ${count} outputs, got ${ctx.outputCount}`);
      return [...Array.from({ length: whole }, () => v), ...(rest ? [rest] : [])];
    },
  };
}

// ----------------------------------------------------------------- slices

/** An int (negative counts from the end), a dimension expression, or null. */
const isBound = (v: unknown) => v === null || isInt(v) || (typeof v === "string" && v.length > 0);

/**
 * One part of a slice of input `of`, per axis, as the subscript `X[a:b:c, ...]`
 * writes it: starts, stops or steps. A list may be shorter than the rank, and
 * an entry may be null; both mean the part's default (0, the extent, 1), as in
 * NumPy. A negative start or stop counts from the end, and both are clipped to
 * the axis. Canonical: one number per axis.
 */
export function sliceBounds(opts: { of?: number; part: "start" | "stop" | "step" }): Kind {
  const of = opts.of ?? 0;
  const describe =
    opts.part === "step"
      ? "a list of steps (ints ≥ 1, expressions or null), at most one per axis"
      : `a list of ${opts.part}s (ints, expressions or null), at most one per axis`;
  return {
    type: "sliceBounds",
    describe,
    check: (v) =>
      isList(v) && v.every((x) => isBound(x) && (opts.part !== "step" || x === null || !isInt(x) || x >= 1))
        ? null
        : `expected ${describe}`,
    canonical: (v, ctx) => {
      const shapeOf = ctx.inShapes[of] ?? [];
      const list = v as unknown[];
      if (list.length > shapeOf.length)
        throw new KindIssue(`has ${list.length} entries for ${shapeOf.length} axes`);
      return shapeOf.map((extent, i) => {
        const written = list[i] ?? null;
        if (written === null) return opts.part === "start" ? 0 : opts.part === "stop" ? extent : 1;
        let n: number;
        try {
          n = resolveInteger(written as Sym, ctx.params);
        } catch (e) {
          throw new KindIssue((e as Error).message);
        }
        if (opts.part === "step") {
          if (n < 1) throw new KindIssue(`step ${written} on axis ${i} is not positive`);
          return n;
        }
        return Math.min(extent, Math.max(0, n < 0 ? n + extent : n));
      });
    },
  };
}

// ------------------------------------------------------------- modifiers

/** Absent means absent: operation code tests for it. */
export const optional = (kind: Kind): Kind => {
  const { default: _default, ...rest } = kind;
  return { ...rest, optional: true };
};

// ------------------------------------------------------- whole attribute sets

function editDistance(a: string, b: string): number {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const above = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] === b[j - 1] ? 0 : 1));
      diagonal = above;
    }
  }
  return row[b.length];
}

/** The nearest of `known`, when `unknown` is close enough to be a misspelling
 * of it rather than a different idea. */
export function nearestName(unknown: string, known: readonly string[]): string | null {
  let best: string | null = null;
  let bestDistance = Infinity;
  for (const candidate of known) {
    const d = editDistance(unknown.toLowerCase(), candidate.toLowerCase());
    if (d < bestDistance) {
      best = candidate;
      bestDistance = d;
    }
  }
  return best !== null && bestDistance <= 2 && bestDistance < unknown.length ? best : null;
}

/** What is wrong with a set of attributes, and which attribute to point at. */
export type AttrProblem = { message: string; attribute?: string };

const has = (o: object, key: string) => Object.prototype.hasOwnProperty.call(o, key);

/**
 * The context-free half: every written attribute is declared and in one of its
 * kind's forms, and every required one is written. Null when all is well.
 */
export function checkAttrs(kinds: Record<string, Kind>, written: Attrs): AttrProblem | null {
  const known = Object.keys(kinds);
  const problems: { message: string; attribute: string }[] = [];
  for (const key of Object.keys(written)) {
    if (has(kinds, key)) continue;
    // An unrecognized attribute is the failure worth explaining well: a
    // misspelled one would otherwise be a different graph without a word.
    const suggestion = nearestName(key, known);
    problems.push({
      attribute: key,
      message: suggestion
        ? `unknown attribute "${key}" (did you mean "${suggestion}"?)`
        : known.length
          ? `unknown attribute "${key}"; this op takes ${known.join(", ")}`
          : `unknown attribute "${key}"; this op takes no attributes`,
    });
  }
  for (const [name, kind] of Object.entries(kinds)) {
    const value = has(written, name) ? written[name] : undefined;
    if (value === undefined) {
      if (!kind.default && !kind.optional)
        problems.push({ attribute: name, message: `${name}: missing; expected ${kind.describe}` });
      continue;
    }
    const issue = kind.check(value);
    if (issue) problems.push({ attribute: name, message: `${name}: ${issue}` });
  }
  if (!problems.length) return null;
  return { message: problems.map((p) => p.message).join("; "), attribute: problems[0].attribute };
}

/**
 * The half that needs the operands: every declared attribute in canonical form,
 * defaults filled in, optional ones left out when absent. Assumes `checkAttrs`
 * passed.
 */
export function canonicalAttrs(
  kinds: Record<string, Kind>,
  ctx: KindCtx
): { attrs: Attrs } | { problem: AttrProblem } {
  // Only declared names are written, so an ordinary object is safe here.
  const attrs: Attrs = {};
  for (const [name, kind] of Object.entries(kinds)) {
    let value = has(ctx.written, name) ? ctx.written[name] : undefined;
    if (value === undefined && kind.default) value = kind.default(ctx);
    if (value === undefined) continue;
    try {
      attrs[name] = kind.canonical(value, ctx);
    } catch (e) {
      if (!(e instanceof KindIssue)) throw e;
      return { problem: { attribute: name, message: `${name}: ${e.message}` } };
    }
  }
  return { attrs };
}
