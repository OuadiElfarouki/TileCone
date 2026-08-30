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

export function resolveDim(s: Sym, params: Record<string, number>): number {
  if (typeof s === "number") {
    if (!Number.isSafeInteger(s) || s < 0)
      throw new GraphError(`bad dimension ${s}`, "GRAPH_SHAPE");
    return s;
  }
  const v = params[s];
  if (v === undefined)
    throw new GraphError(`unbound symbolic dim "${s}"`, "GRAPH_UNBOUND_SYMBOL");
  if (!Number.isSafeInteger(v) || v <= 0)
    throw new GraphError(`bad binding ${s}=${v}`, "GRAPH_SHAPE");
  return v;
}

export function resolveShape(shape: Shape, params: Record<string, number>): number[] {
  return shape.map((s) => resolveDim(s, params));
}
