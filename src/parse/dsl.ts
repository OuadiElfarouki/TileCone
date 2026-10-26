/** Text DSL:
 *   M = 512
 *   K = 2048
 *   A = Tensor(M, K, dtype=fp16)
 *   C = einsum("mk,kn->mn", A, B)
 *   Y0, Y1 = split(X, [2, 2], axis=0)
 *   Z = Y0[:, 0:K/2]
 * One statement per line, `#` comments. The functions and their parameters are
 * the signatures in `signatures.ts`.
 *
 * This module is a facade. Text becomes an AST in `parser.ts` and a graph in
 * `lower.ts`; both collect errors rather than stopping at the first, and
 * `compiler.ts` is the entry point that surfaces all of them. The functions
 * here keep a throwing contract for callers that want one graph or one error,
 * and `toDSL` lives here because printing is the inverse of the whole pipeline
 * rather than of either half.
 */

import { Graph, Node, resolveGraphCollecting } from "../core/graph";
import { DType } from "../core/dtypes";
import type { Kind } from "../core/ops/kinds";
import type { Attrs } from "../core/ops/types";
import { DTYPE_TO_DSL } from "./bind";
import { lowerProgram } from "./lower";
import { parseProgram } from "./parser";
import { kindOf, signatureForNode, SUBSCRIPT_OP, type Signature } from "./signatures";
import { DSLSourceMap } from "./source";

export { DSLError } from "./parser";
export { DSL_DTYPES } from "./bind";

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

/** A value as source text, spelled for its kind as the binder reads it. */
function valueText(v: unknown, kind: Kind): string {
  switch (kind.type) {
    case "text":
      return JSON.stringify(v);
    case "dtype":
      return DTYPE_TO_DSL[v as DType];
    case "dtypes":
      return `[${(v as (DType | null)[]).map((d) => (d === null ? "null" : DTYPE_TO_DSL[d])).join(", ")}]`;
    default:
      return plainText(v);
  }
}

/** Numbers, booleans, lists, and words or dimension expressions, which are bare. */
function plainText(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(plainText).join(", ")}]`;
  return String(v);
}

/** `X[a:b, :, ::2]` from a slice's starts, stops and steps as written. */
function subscriptText(target: string, attrs: Attrs): string {
  const starts = (attrs.starts as unknown[]) ?? [];
  const stops = (attrs.stops as unknown[]) ?? [];
  const steps = (attrs.steps as unknown[]) ?? [];
  const n = Math.max(starts.length, stops.length, steps.length, 1);
  const part = (v: unknown) => (v === null || v === undefined ? "" : String(v));
  const items = Array.from({ length: n }, (_, i) => {
    const step = steps[i];
    return `${part(starts[i])}:${part(stops[i])}${step === null || step === undefined || step === 1 ? "" : `:${step}`}`;
  });
  return `${target}[${items.join(", ")}]`;
}

/**
 * A node as a call to its function. Tensors are positional; an attribute is
 * positional while every parameter before it was, and by name after the first
 * that was left out or is keyword-only. Values are shortened by their kind.
 */
function callText(sig: Signature, node: Node, inNames: string[]): string {
  const hidden = new Set([...Object.keys(sig.implied), ...(sig.inputAttrs ?? [])]);
  const covered = new Set(hidden);
  const parts: string[] = [];
  let positional = true;
  let next = 0;
  const later = (i: number) => sig.params.slice(i + 1).filter((p) => "role" in p && p.role !== "variadic").length;

  sig.params.forEach((param, i) => {
    const inOrder = positional && i < sig.keywordOnly;
    if ("role" in param) {
      if (param.role === "optional-input" && param.present && node.attrs[param.present] !== true) {
        positional = false;
        return;
      }
      if (param.role === "inputs" || param.role === "variadic") {
        const taken = inNames.slice(next, inNames.length - later(i));
        next += taken.length;
        if (param.role === "inputs") parts.push(`[${taken.join(", ")}]`);
        else parts.push(...taken);
        return;
      }
      const name = inNames[next++];
      parts.push(inOrder ? name : `${param.name}=${name}`);
      return;
    }
    covered.add(param.attr);
    if (!(param.attr in node.attrs)) {
      positional = false;
      return;
    }
    const kind = kindOf(sig, param);
    const value = node.attrs[param.attr];
    const text = valueText(kind.shorten ? kind.shorten(value) : value, kind);
    parts.push(inOrder ? text : `${param.name}=${text}`);
  });

  // Every attribute is written by some parameter, or implied by the name.
  // Dropping one would print a different graph.
  for (const key of Object.keys(node.attrs))
    if (!covered.has(key))
      throw new Error(`${sig.name}() has no parameter for the ${node.op} attribute "${key}"`);
  return `${sig.name}(${parts.join(", ")})`;
}

/**
 * Canonical attributes for every node resolution reaches. Which function a
 * node is written as can depend on a value the source left to a default - a
 * convolution's spatial rank - and that value is only known once resolved.
 */
function canonicalAttrsByNode(g: Graph): Map<string, Attrs> {
  const resolved = resolveGraphCollecting(g).resolved;
  return new Map((resolved?.topo ?? []).map((node) => [node.id, node.attrs]));
}

export function toDSL(g: Graph): string {
  const lines: string[] = [];
  for (const [name, value] of Object.entries(g.params)) lines.push(`${name} = ${value}`);
  // After every parameter: a definition only reads names declared before it.
  for (const [name, definition] of Object.entries(g.derived ?? {}))
    lines.push(`${name} = ${definition}`);
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
  const canonical = canonicalAttrsByNode(g);
  for (const n of g.nodes) {
    const outNames = n.outputs.map((o) => g.tensors[o].name);
    const inNames = n.inputs.map((i) => g.tensors[i].name);
    if (n.op === SUBSCRIPT_OP) {
      lines.push(`${outNames[0]} = ${subscriptText(inNames[0], n.attrs)}`);
      continue;
    }
    const sig = signatureForNode(n.op, { ...n.attrs, ...(canonical.get(n.id) ?? {}) });
    if (!sig) throw new Error(`no DSL function writes node "${n.id}" (${n.op})`);
    lines.push(`${outNames.join(", ")} = ${callText(sig, n, inNames)}`);
  }
  return lines.join("\n") + "\n";
}
