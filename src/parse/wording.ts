/**
 * A resolver message, said the way the source says it.
 *
 * The resolver names a node by its id and registry operation, its attributes by
 * their registry names, and its operands as "input 0", "input 1": the terms of
 * the graph, which any front end can build. The author wrote `conv2d`,
 * `padding=` and `matmul(A, B)`, so at the DSL boundary those are put back:
 * the function as called, the keyword as written, and each operand as the
 * parameter it was passed for, with the tensor it was. The rule is the same for
 * every operation, because it is read off the signature rather than written per
 * message.
 */

import type { Graph, Node } from "../core/graph";
import { DTYPES } from "../core/dtypes";
import { DTYPE_TO_DSL } from "./bind";
import type { GraphErrorSubject } from "../core/shapes";
import { signatureFor, type Signature } from "./signatures";
import type { DSLSourceMap } from "./source";

/** The registry's dtype names, which the DSL spells `float16` and so on. */
const IR_DTYPE = new RegExp(`\\b(${DTYPES.filter((d) => d !== "bool").join("|")})\\b`, "g");

/** `x2 (B)`, `arrays[1] (B)`: how each input of `node` was passed. */
function inputLabels(sig: Signature, node: Node, graph: Graph): string[] {
  const names = node.inputs.map((id) => graph.tensors[id]?.name ?? id);
  const labels: string[] = [];
  let next = 0;
  sig.params.forEach((param, i) => {
    if (!("role" in param)) return;
    if (param.role === "optional-input" && param.present && node.attrs[param.present] !== true) return;
    if (param.role === "inputs" || param.role === "variadic") {
      const after = sig.params
        .slice(i + 1)
        .filter((p) => "role" in p && (p.role === "input" || p.role === "optional-input")).length;
      const taken = names.slice(next, names.length - after);
      taken.forEach((name, k) => labels.push(`${param.name}[${k}] (${name})`));
      next += taken.length;
      return;
    }
    labels.push(`${param.name} (${names[next++]})`);
  });
  return labels;
}

/**
 * `detail` in the source's terms, or null when the subject is not a node the
 * source wrote.
 */
export function inSourceTerms(
  detail: string,
  subject: GraphErrorSubject | undefined,
  sourceMap: DSLSourceMap,
  graph: Graph
): string | null {
  if (subject?.kind !== "node") return null;
  const args = sourceMap.nodeArgs[subject.id];
  const node = graph.nodes.find((candidate) => candidate.id === subject.id);
  if (!args || !node) return null;

  let text = detail;
  // An operation's own message may lead with its registry name; the call
  // replaces it.
  if (text.startsWith(`${node.op}: `)) text = text.slice(node.op.length + 2);
  const attribute = subject.attribute;
  const leading = attribute !== undefined && text.startsWith(`${attribute}: `);

  if ("subscript" in args.written) {
    // Starts, stops and steps are one subscript to the author.
    if (leading) text = text.slice(attribute!.length + 2);
    return `${args.written.subscript}[…]: ${text}`;
  }

  const sig = signatureFor(args.written.function);
  if (!sig) return `${args.written.function}: ${text}`;
  if (leading) {
    // An attribute the name implies (`spatialRank` for `conv2d`) is the call
    // itself to the author, so it is not named again.
    const param = sig.params.find((p) => "attr" in p && p.attr === attribute);
    text = `${param ? `${param.name}: ` : ""}${text.slice(attribute!.length + 2)}`;
  }
  const labels = inputLabels(sig, node, graph);
  text = text.replace(/\binput (\d+)\b/g, (match, slot: string) => labels[Number(slot)] ?? match);
  text = text.replace(IR_DTYPE, (d) => DTYPE_TO_DSL[d as keyof typeof DTYPE_TO_DSL]);
  return `${sig.name}: ${text}`;
}
