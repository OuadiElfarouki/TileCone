/**
 * The DSL's call-name sugar, described once for both directions.
 *
 * `relu(X)` is an `elementwise` node and `layernorm(X, W)` a `normalize` node,
 * and something has to know that in each direction: lowering to build the node,
 * the printer to write it back as the call the author used. Those were two
 * separate pieces of hardcoded knowledge, and they had already drifted -
 * lowering accepted `amax`/`amin` while the printer only reversed `sum`, `mean`
 * and `prod`, so `amax(X, axis=1)` came back as `reduce(X, fn=max, ...)`. The
 * asymmetry was structural rather than an oversight, and a table is the fix.
 *
 * `implied` is what makes the printer's half work: those attributes are carried
 * by the call name itself, so writing them again would be redundant at best and
 * contradictory at worst. Every *other* attribute must still be printed, or
 * expanding a composite and recompiling silently drops it.
 */

import { ELEMENTWISE_FNS } from "../core/ops/elementwise";

export type SugarForm = {
  /** The name written in the source. */
  call: string;
  /** The operation it lowers to. */
  op: string;
  /** Attributes the call name determines, given the operand count written. */
  attrsFor(inputCount: number): Record<string, unknown>;
  /** Attribute names the call form expresses; the printer omits these. */
  implied: string[];
  /** Whether a node with these attributes is this call written back. */
  matches(attrs: Record<string, unknown>): boolean;
  /** Attributes that default when the author omits them. */
  defaults?: Record<string, unknown>;
  /** Accept the singular `axis=` spelling for an `axes=` attribute. */
  axisAlias?: boolean;
  /** Attributes with no default that the call is meaningless without. */
  requires?: string[];
};

const elementwise: SugarForm[] = Object.keys(ELEMENTWISE_FNS).map((fn) => ({
  call: fn,
  op: "elementwise",
  attrsFor: (inputCount) => ({ fn, nary: inputCount }),
  implied: ["fn", "nary"],
  matches: (attrs) => attrs.fn === fn,
}));

/** `amax`/`amin` are the NumPy spellings; the op's own names are `max`/`min`. */
const reduce: SugarForm[] = [
  { call: "sum", fn: "sum" },
  { call: "mean", fn: "mean" },
  { call: "prod", fn: "prod" },
  { call: "amax", fn: "max" },
  { call: "amin", fn: "min" },
].map(({ call, fn }) => ({
  call,
  op: "reduce",
  attrsFor: () => ({ fn }),
  implied: ["fn"],
  matches: (attrs) => attrs.fn === fn,
  defaults: { keepdim: false },
  axisAlias: true,
  requires: ["axes"],
}));

const normalize: SugarForm[] = ["layernorm", "rmsnorm"].map((kind) => ({
  call: kind,
  op: "normalize",
  // Weight and bias are positional: a second operand is the weight, a third the
  // bias. The flags exist because the op needs to know its own arity.
  attrsFor: (inputCount) => ({
    kind,
    hasWeight: inputCount >= 2,
    hasBias: inputCount >= 3,
  }),
  implied: ["kind", "hasWeight", "hasBias"],
  matches: (attrs) => attrs.kind === kind,
  defaults: { axes: [-1] },
}));

export const SUGAR_FORMS: SugarForm[] = [...elementwise, ...reduce, ...normalize];

const byCall = new Map(SUGAR_FORMS.map((form) => [form.call, form]));

/** The sugar a written call name refers to, if any. */
export function sugarForCall(call: string): SugarForm | undefined {
  return byCall.get(call);
}

/**
 * How to write this node as a call, if it can be written as one.
 *
 * An operation with no sugar, or one whose attributes no sugared form claims
 * (`reduce(fn=logsumexp)`), prints in the general `op(inputs, attrs...)` form.
 */
export function sugarForNode(op: string, attrs: Record<string, unknown>): SugarForm | undefined {
  return SUGAR_FORMS.find((form) => form.op === op && form.matches(attrs));
}
