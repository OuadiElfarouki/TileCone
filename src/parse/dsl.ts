/** Text DSL (IDEA.md §6.2):
 *   M = 512
 *   K = 2048
 *   A = Tensor(M, K, dtype=fp16)
 *   C = einsum("mk,kn->mn", A, B)
 *   Y0, Y1 = split(X, axis=0, sizes=[2, 2])
 * One statement per line, `#` comments.
 *
 * This module is now a facade. Text becomes an AST in `parser.ts` and a graph
 * in `lower.ts`; both collect errors rather than stopping at the first, and
 * `compiler.ts` is the entry point that surfaces all of them. The functions
 * here keep their old throwing contract for callers that want one graph or one
 * error, and `toDSL` still lives here because printing is the inverse of the
 * whole pipeline rather than of either half.
 */

import { Graph } from "../core/graph";
import { DType } from "../core/dtypes";
import { lowerProgram, DTYPE_TO_DSL } from "./lower";
import { parseProgram } from "./parser";
import { sugarForNode } from "./sugar";
import { DSLSourceMap } from "./source";

export { DSLError } from "./parser";
export { DSL_DTYPES } from "./lower";

export function parseDSL(text: string): Graph {
  return parseDSLWithSource(text).graph;
}

export type ParsedDSL = { graph: Graph; sourceMap: DSLSourceMap };

/** Parse and lower, throwing the first error. `compileDSL` reports them all. */
export function parseDSLWithSource(text: string): ParsedDSL {
  const { program, errors: parseErrors } = parseProgram(text);
  if (parseErrors.length) throw parseErrors[0];
  const { graph, sourceMap, errors } = lowerProgram(program);
  if (errors.length) throw errors[0];
  return { graph, sourceMap };
}

// ------------------------------------------------------------------- toDSL

function attrValueToDSL(v: unknown, name?: string): string {
  if (Array.isArray(v)) return `[${v.map((item) => attrValueToDSL(item)).join(", ")}]`;
  if (name === "dtype" && typeof v === "string" && v in DTYPE_TO_DSL)
    return DTYPE_TO_DSL[v as DType];
  if (typeof v === "string") return /^[A-Za-z_][A-Za-z0-9_]*$/.test(v) ? v : JSON.stringify(v);
  return String(v);
}

export function toDSL(g: Graph): string {
  const lines: string[] = [];
  for (const [name, value] of Object.entries(g.params)) lines.push(`${name} = ${value}`);
  for (const t of Object.values(g.tensors)) {
    if (t.producer || g.nodes.some((n) => n.outputs.includes(t.id))) continue;
    const constructor = t.role === "weight" ? "Parameter" : "Tensor";
    const dims = t.shape.map((dim, axis) => {
      const axisName = t.axisNames?.[axis];
      return axisName ? `${axisName}=${String(dim)}` : String(dim);
    });
    lines.push(
      `${t.name} = ${constructor}(${[...dims, `dtype=${DTYPE_TO_DSL[t.dtype]}`].join(", ")})`
    );
  }
  for (const n of g.nodes) {
    const outNames = n.outputs.map((o) => g.tensors[o].name);
    const inNames = n.inputs.map((i) => g.tensors[i].name);
    const attrs = { ...n.attrs };
    // einsum's equation is positional rather than named: it leads the call, in
    // front of the operands, the way the author wrote it.
    const leading: string[] = [];
    if (n.op === "einsum" && typeof attrs.equation === "string") {
      leading.push(`"${attrs.equation}"`);
      delete attrs.equation;
    }
    // The sugared call name carries some attributes itself; every *other* one
    // must still be written, or expanding a composite and recompiling drops it.
    const sugar = sugarForNode(n.op, attrs);
    const hidden = new Set(sugar?.implied ?? []);
    const named = Object.entries(attrs)
      .filter(([k]) => !hidden.has(k))
      .map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
    const call = `${sugar?.call ?? n.op}(${[...leading, ...inNames, ...named].join(", ")})`;
    lines.push(`${outNames.join(", ")} = ${call}`);
  }
  return lines.join("\n") + "\n";
}
