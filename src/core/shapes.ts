/** Symbolic dimension binding, shared by graph resolution and shape-manipulating ops. */

export type Sym = string | number; // "M" | "K" | 128
export type Shape = Sym[];

export class GraphError extends Error {}

export function resolveDim(s: Sym, params: Record<string, number>): number {
  if (typeof s === "number") {
    if (!Number.isInteger(s) || s < 0) throw new GraphError(`bad dimension ${s}`);
    return s;
  }
  const v = params[s];
  if (v === undefined) throw new GraphError(`unbound symbolic dim "${s}"`);
  if (!Number.isInteger(v) || v <= 0) throw new GraphError(`bad binding ${s}=${v}`);
  return v;
}

export function resolveShape(shape: Shape, params: Record<string, number>): number[] {
  return shape.map((s) => resolveDim(s, params));
}
