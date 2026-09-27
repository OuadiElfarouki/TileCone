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
import { readDimExpr } from "../core/shapes";
import { DType } from "../core/dtypes";
import { lowerProgram, DTYPE_TO_DSL } from "./lower";
import { parseProgram } from "./parser";
import { sugarForNode } from "./sugar";
import { DSLSourceMap } from "./source";
import { isDSLBooleanLiteral } from "./lexical";

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

/** Whether this attribute string is a dimension expression in full: a bare
 *  symbol, or arithmetic over symbols and literals. An einsum equation and any
 *  other free text are not, and stay quoted. */
function isDimExpr(v: string): boolean {
  if (!v) return false;
  const read = readDimExpr(v);
  return read !== null && read.end === v.length;
}

function attrValueToDSL(v: unknown, name?: string): string {
  // The attribute's name travels into its items: a per-output `dtypes` list
  // holds canonical dtypes that have to come back out in the DSL's spellings,
  // and dropping the name on the way in printed `f16` where only `fp16` parses.
  if (Array.isArray(v)) return `[${v.map((item) => attrValueToDSL(item, name)).join(", ")}]`;
  if (
    (name === "dtype" || name === "dtypes") &&
    typeof v === "string" &&
    Object.prototype.hasOwnProperty.call(DTYPE_TO_DSL, v)
  )
    return DTYPE_TO_DSL[v as DType];
  // A symbolic dimension is source the author wrote - `H*D`, `P+T` - and has to
  // come back out that way. Quoting it printed `shape=[B, "KVH*G", D]`, which
  // parses back to the same graph but is not what anyone typed, and expanding a
  // composite writes this text into the editor.
  if (typeof v === "string")
    return isDimExpr(v) && !isDSLBooleanLiteral(v) ? v : JSON.stringify(v);
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
