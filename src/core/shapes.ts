/** Symbolic dimension binding, shared by graph resolution and shape-manipulating ops. */

export type Sym = string | number; // "M" | "K" | 128
export type Shape = Sym[];

export type GraphErrorCode =
  | "GRAPH_INVALID"
  | "GRAPH_UNKNOWN_OP"
  | "GRAPH_INVALID_ATTRIBUTES"
  | "GRAPH_ARITY"
  | "GRAPH_CYCLE"
  | "GRAPH_SHAPE"
  | "GRAPH_DTYPE"
  | "GRAPH_UNBOUND_SYMBOL"
  | "GRAPH_DEFINITION";

export type GraphErrorSubject = {
  kind: "node" | "tensor" | "parameter";
  id: string;
};

export class GraphError extends Error {
  constructor(
    message: string,
    public code: GraphErrorCode = "GRAPH_INVALID",
    public subject?: GraphErrorSubject
  ) {
    super(message);
    this.name = "GraphError";
  }
}

/* ------------------------------------------------- dimension expressions */

/**
 * A dimension may be written as arithmetic over parameters — `H*D`, `E/H`,
 * `S+1` — so a relationship the author relies on is stated where it is used
 * instead of being precomputed into a literal that silently stops agreeing.
 *
 * `Sym` stays `string | number`: the string simply holds the expression, a bare
 * parameter being the one-symbol case. Nothing downstream had to learn a new
 * shape type, and the text a reader wrote is what `notes.ts` and `toDSL` see.
 */
type DimNode =
  | { kind: "num"; value: number }
  | { kind: "sym"; name: string }
  | { kind: "neg"; arg: DimNode }
  | { kind: "bin"; op: "+" | "-" | "*" | "/"; left: DimNode; right: DimNode };

/** The DSL's numeric literal, defined once so a scalar attribute and a
 * dimension cannot disagree about what a number looks like. */
export const NUMBER_RE = /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/;

const skipWs = (src: string, from: number): number => {
  let i = from;
  while (i < src.length && /\s/.test(src[i])) i++;
  return i;
};

/**
 * Recognize one dimension expression starting at `start` and report where it
 * ends. The DSL parser needs the end so it can consume exactly the expression
 * and keep its written form; `resolveDim` needs the tree so it can evaluate it.
 * Both come from here, so the reader and the evaluator cannot drift apart.
 */
export function readDimExpr(src: string, start = 0): { node: DimNode; end: number } | null {
  let pos = start;

  function atom(): DimNode | null {
    pos = skipWs(src, pos);
    if (src[pos] === "(") {
      pos++;
      const inner = expr();
      if (!inner) return null;
      pos = skipWs(src, pos);
      if (src[pos] !== ")") return null;
      pos++;
      return inner;
    }
    // Checked before the numeric literal so `-A` negates a symbol and `1e-5`
    // is still read as a single number rather than a subtraction.
    if (src[pos] === "-") {
      pos++;
      const arg = atom();
      return arg && { kind: "neg", arg };
    }
    const num = NUMBER_RE.exec(src.slice(pos));
    if (num) {
      pos += num[0].length;
      return { kind: "num", value: Number(num[0]) };
    }
    const ident = /^[A-Za-z_][A-Za-z0-9_.$]*/.exec(src.slice(pos));
    if (ident) {
      pos += ident[0].length;
      return { kind: "sym", name: ident[0] };
    }
    return null;
  }

  function binary(
    next: () => DimNode | null,
    ops: string[]
  ): () => DimNode | null {
    return () => {
      let left = next();
      if (!left) return null;
      for (;;) {
        pos = skipWs(src, pos);
        const op = src[pos];
        if (!ops.includes(op)) return left;
        pos++;
        const right = next();
        if (!right) return null;
        left = { kind: "bin", op: op as "+" | "-" | "*" | "/", left, right };
      }
    };
  }

  const term = binary(atom, ["*", "/"]);
  const expr = binary(term, ["+", "-"]);

  const node = expr();
  return node ? { node, end: pos } : null;
}

function evalDimExpr(node: DimNode, params: Record<string, number>): number {
  switch (node.kind) {
    case "num":
      if (!Number.isSafeInteger(node.value))
        throw new GraphError(
          `dimension literal ${node.value} is not a safe integer`,
          "GRAPH_SHAPE"
        );
      return node.value;
    case "sym": {
      const bound = params[node.name];
      if (bound === undefined)
        throw new GraphError(`unbound symbolic dim "${node.name}"`, "GRAPH_UNBOUND_SYMBOL");
      if (!Number.isSafeInteger(bound) || bound <= 0)
        throw new GraphError(`bad binding ${node.name}=${bound}`, "GRAPH_SHAPE");
      return bound;
    }
    case "neg":
      return -evalDimExpr(node.arg, params);
    case "bin": {
      const left = evalDimExpr(node.left, params);
      const right = evalDimExpr(node.right, params);
      let value: number;
      if (node.op === "+") value = left + right;
      else if (node.op === "-") value = left - right;
      else if (node.op === "*") value = left * right;
      else {
        if (right === 0)
          throw new GraphError("division by zero in a dimension", "GRAPH_SHAPE");
        // A tensor axis is a whole number of elements; an inexact division means
        // the relationship the author wrote does not actually hold.
        if (left % right !== 0)
          throw new GraphError(
            `${left}/${right} is not a whole number of elements`,
            "GRAPH_SHAPE"
          );
        value = left / right;
      }
      // Check every step, not only the final result: unsafe arithmetic can
      // round and later return to a plausible safe integer with the wrong value.
      if (!Number.isSafeInteger(value))
        throw new GraphError(
          `dimension arithmetic produced ${value}, outside the safe integer range`,
          "GRAPH_SHAPE"
        );
      return value;
    }
  }
}

function resolveDim(s: Sym, params: Record<string, number>): number {
  if (typeof s === "number") {
    if (!Number.isSafeInteger(s) || s < 0)
      throw new GraphError(`bad dimension ${s}`, "GRAPH_SHAPE");
    return s;
  }
  const parsed = readDimExpr(s);
  if (!parsed || skipWs(s, parsed.end) !== s.length)
    throw new GraphError(`bad dimension expression "${s}"`, "GRAPH_SHAPE");
  const value = evalDimExpr(parsed.node, params);
  if (!Number.isSafeInteger(value) || value < 0)
    throw new GraphError(`dimension "${s}" resolves to ${value}`, "GRAPH_SHAPE");
  return value;
}

export function resolveShape(shape: Shape, params: Record<string, number>): number[] {
  return shape.map((s) => resolveDim(s, params));
}
