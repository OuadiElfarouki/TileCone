/** Text DSL (IDEA.md §6.2):
 *   params M=512 K=2048
 *   input A [M, K] f16
 *   C = einsum("mk,kn->mn", A, B)
 *   Y0, Y1 = split(X, axis=0, sizes=[2, 2])
 * One statement per line, `#` comments. Round-trips losslessly via toDSL.
 */

import { Graph, Node, Tensor } from "../core/graph";
import { DTYPES, DType } from "../core/dtypes";
import { Sym } from "../core/shapes";
import { documentSpan, DSLSourceMap, lineSpan, SourceSpan } from "./source";

export class DSLError extends Error {
  constructor(
    public detail: string,
    public span: SourceSpan,
    public code = "DSL_SYNTAX"
  ) {
    super(`line ${span.start.line}: ${detail}`);
    this.name = "DSLError";
  }

  get line(): number {
    return this.span.start.line;
  }

  get column(): number {
    return this.span.start.column;
  }
}

const DTYPE_SET = new Set<string>(DTYPES);

/**
 * Keywords that declare a graph input. `weight` and `param` are the same thing
 * to the analysis — a tensor with no producer — but are tagged so the UI can
 * tell a learned parameter apart from an activation.
 */
const DECLARATIONS: Record<string, "activation" | "weight"> = {
  input: "activation",
  weight: "weight",
  param: "weight",
};

/** fn-name sugar -> op + fixed attrs */
const ELEMENTWISE_FNS = new Set([
  "add", "sub", "mul", "div", "pow", "maximum", "minimum",
  "relu", "gelu", "silu", "exp", "log", "sqrt", "rsqrt", "neg", "abs", "sigmoid", "tanh",
]);
const REDUCE_FNS = new Set(["sum", "mean", "prod", "amax", "amin"]);

type Value = number | string | boolean | Value[] | { ident: string };

class LineParser {
  pos = 0;
  constructor(
    public src: string,
    public line: number,
    public lineOffset: number,
    public columnOffset: number
  ) {}
  span(pos = this.pos, length = 1): SourceSpan {
    return lineSpan(this.line, this.lineOffset, this.columnOffset + pos + 1, length);
  }
  error(msg: string): never {
    throw new DSLError(`${msg} (at "${this.src.slice(this.pos, this.pos + 12)}...")`, this.span());
  }
  ws() {
    while (this.pos < this.src.length && /\s/.test(this.src[this.pos])) this.pos++;
  }
  eat(tok: string): boolean {
    this.ws();
    if (this.src.startsWith(tok, this.pos)) {
      this.pos += tok.length;
      return true;
    }
    return false;
  }
  expect(tok: string) {
    if (!this.eat(tok)) this.error(`expected "${tok}"`);
  }
  ident(): string | null {
    this.ws();
    // `$` is reserved by composite expansion for generated tensor names. It is
    // accepted after the first character so an expanded graph can be serialized
    // back to executable DSL without renaming the tensors shown in the UI.
    const m = /^[A-Za-z_][A-Za-z0-9_.$]*/.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return m[0];
  }
  identReq(what: string): string {
    const v = this.ident();
    if (v === null) throw new DSLError(`expected ${what}`, this.span());
    return v;
  }
  numberReq(what: string): number {
    const v = this.number();
    if (v === null) throw new DSLError(`expected ${what}`, this.span());
    return v;
  }
  number(): number | null {
    this.ws();
    // Accepts scientific notation, so an attribute like eps=1e-5 is expressible
    // and `toDSL` cannot emit a literal (String(1e-21) === "1e-21") that this
    // parser then rejects.
    const m = /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return Number(m[0]);
  }
  string(): string | null {
    this.ws();
    if (this.src[this.pos] !== '"') return null;
    const start = this.pos;
    this.pos++;
    let escaped = false;
    while (this.pos < this.src.length) {
      const ch = this.src[this.pos++];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === "\\") {
        escaped = true;
        continue;
      }
      if (ch === '"') {
        const literal = this.src.slice(start, this.pos);
        try {
          return JSON.parse(literal) as string;
        } catch {
          throw new DSLError("invalid string escape", this.span(start, this.pos - start));
        }
      }
    }
    throw new DSLError("unterminated string", this.span(start, this.pos - start));
  }
  value(): Value {
    this.ws();
    const s = this.string();
    if (s !== null) return s;
    if (this.eat("[")) {
      const arr: Value[] = [];
      if (!this.eat("]")) {
        do arr.push(this.value());
        while (this.eat(","));
        this.expect("]");
      }
      return arr;
    }
    const n = this.number();
    if (n !== null) return n;
    const id = this.ident();
    if (id === "true") return true;
    if (id === "false") return false;
    if (id !== null) return { ident: id };
    this.error("expected value");
  }
  atEnd(): boolean {
    this.ws();
    return this.pos >= this.src.length;
  }
}

function valueToAttr(v: Value): unknown {
  if (Array.isArray(v)) return v.map(valueToAttr);
  if (typeof v === "object" && v !== null && "ident" in v) return v.ident; // symbolic dim
  return v;
}

/** Remove a line comment without treating a # inside a string as a comment. */
function stripComment(line: string): string {
  let quoted = false;
  let escaped = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"') quoted = !quoted;
    else if (ch === "#" && !quoted) return line.slice(0, i);
  }
  return line;
}

export function parseDSL(text: string): Graph {
  return parseDSLWithSource(text).graph;
}

export type ParsedDSL = { graph: Graph; sourceMap: DSLSourceMap };

/** Parse DSL text while retaining statement spans for later semantic diagnostics. */
export function parseDSLWithSource(text: string): ParsedDSL {
  const params: Record<string, number> = {};
  const tensors: Record<string, Tensor> = {};
  const nodes: Node[] = [];
  const sourceMap: DSLSourceMap = {
    document: documentSpan(text),
    params: {},
    tensors: {},
    nodes: {},
  };
  const lines = text.split("\n");
  let lineOffset = 0;

  for (let ln = 0; ln < lines.length; ln++) {
    const physicalLine = lines[ln];
    const code = stripComment(physicalLine);
    const raw = code.trim();
    const columnOffset = raw ? code.indexOf(raw) : 0;
    const statementSpan = lineSpan(ln + 1, lineOffset, columnOffset + 1, raw.length);
    if (!raw) {
      lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
      continue;
    }
    const p = new LineParser(raw, ln + 1, lineOffset, columnOffset);

    if (/^params(?:\s|$)/.test(raw)) {
      p.expect("params");
      while (!p.atEnd()) {
        const name = p.identReq("param name");
        p.expect("=");
        if (params[name] !== undefined)
          throw new DSLError(`parameter "${name}" redefined`, p.span(), "DSL_DUPLICATE_PARAM");
        params[name] = p.numberReq("number");
        sourceMap.params[name] = statementSpan;
      }
      lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
      continue;
    }

    const leading = /^([A-Za-z_]\w*)\s/.exec(raw)?.[1];
    if (leading && leading in DECLARATIONS) {
      p.expect(leading);
      const role = DECLARATIONS[leading];
      const name = p.identReq("tensor name");
      p.expect("[");
      const shape: Sym[] = [];
      if (!p.eat("]")) {
        do {
          const n = p.number();
          if (n !== null) shape.push(n);
          else shape.push(p.identReq("dim"));
        } while (p.eat(","));
        p.expect("]");
      }
      let dtype: DType = "f32";
      const dt = p.ident();
      if (dt) {
        if (!DTYPE_SET.has(dt)) p.error(`unknown dtype "${dt}"`);
        dtype = dt as DType;
      }
      // Assignments already refuse trailing input; declarations used to accept
      // it silently, so `input X [4, 8] 32` declared an f32 tensor and the wrong
      // dtype went on to size every byte estimate.
      if (!p.atEnd())
        p.error(`unexpected input after declaration (dtype must be one of ${DTYPES.join(", ")})`);
      if (tensors[name])
        throw new DSLError(`tensor "${name}" redefined`, statementSpan, "DSL_DUPLICATE_TENSOR");
      tensors[name] = { id: name, name, shape, dtype, ...(role === "weight" ? { role } : {}) };
      sourceMap.tensors[name] = statementSpan;
      lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
      continue;
    }

    // assignment: NAME[, NAME...] = op(args)
    const outs: string[] = [];
    outs.push(p.identReq("statement"));
    while (p.eat(",")) outs.push(p.identReq("output name"));
    if (!p.eat("=")) {
      // The most common cause is a declaration keyword we do not know, e.g.
      // "tensor A [M, K]" — say so instead of demanding an "=".
      throw new DSLError(
        `unknown statement starting with "${outs[0]}". Expected either an assignment ` +
          `\`NAME = op(...)\`, or a declaration \`${Object.keys(DECLARATIONS).join("|")} NAME [dims] dtype\``,
        statementSpan,
        "DSL_UNKNOWN_STATEMENT"
      );
    }
    const callee = p.identReq("op name");
    p.expect("(");
    const positional: Value[] = [];
    const named: Record<string, unknown> = {};
    if (!p.eat(")")) {
      do {
        const save = p.pos;
        const id = p.ident();
        if (id && p.eat("=")) {
          if (Object.prototype.hasOwnProperty.call(named, id))
            throw new DSLError(
              `attribute "${id}" specified more than once`,
              statementSpan,
              "DSL_DUPLICATE_ATTRIBUTE"
            );
          named[id] = valueToAttr(p.value());
        }
        else {
          p.pos = save;
          positional.push(p.value());
        }
      } while (p.eat(","));
      p.expect(")");
    }
    if (!p.atEnd()) p.error("trailing input");

    let op = callee;
    const attrs: Record<string, unknown> = { ...named };
    const inputs: string[] = [];
    for (const v of positional) {
      if (typeof v === "string") {
        if (op === "einsum" && attrs.equation === undefined) attrs.equation = v;
        else p.error("unexpected string argument");
      } else if (typeof v === "object" && v !== null && "ident" in v) {
        inputs.push(v.ident);
      } else p.error("positional args must be tensors or an einsum equation");
    }

    if (ELEMENTWISE_FNS.has(callee)) {
      op = "elementwise";
      attrs.fn = callee;
      attrs.nary = inputs.length;
    } else if (REDUCE_FNS.has(callee)) {
      op = "reduce";
      attrs.fn = callee === "amax" ? "max" : callee === "amin" ? "min" : callee;
      if (attrs.keepdim === undefined) attrs.keepdim = false;
      // accept the singular spelling, which is what softmax/cumsum take
      if (attrs.axes === undefined && attrs.axis !== undefined) {
        attrs.axes = [attrs.axis];
        delete attrs.axis;
      }
      if (attrs.axes === undefined)
        throw new DSLError(
          `${callee}() needs an axis, e.g. ${callee}(X, axis=-1)`,
          statementSpan,
          "DSL_MISSING_ATTRIBUTE"
        );
    } else if (callee === "layernorm" || callee === "rmsnorm") {
      op = "normalize";
      attrs.kind = callee;
      attrs.hasWeight = inputs.length >= 2;
      attrs.hasBias = inputs.length >= 3;
      if (attrs.axes === undefined) attrs.axes = [-1];
    }

    for (const t of inputs)
      if (!tensors[t])
        throw new DSLError(`unknown tensor "${t}"`, statementSpan, "DSL_UNKNOWN_TENSOR");
    for (const o of outs) {
      if (tensors[o])
        throw new DSLError(`tensor "${o}" redefined`, statementSpan, "DSL_DUPLICATE_TENSOR");
      tensors[o] = { id: o, name: o, shape: [], dtype: tensors[inputs[0]]?.dtype ?? "f32" };
      sourceMap.tensors[o] = statementSpan;
    }
    const node = { id: `${op}_${outs[0]}`, op, inputs, outputs: outs, attrs };
    nodes.push(node);
    sourceMap.nodes[node.id] = statementSpan;
    lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
  }
  return { graph: { nodes, tensors, params }, sourceMap };
}

// ------------------------------------------------------------------- toDSL

function attrValueToDSL(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(attrValueToDSL).join(", ")}]`;
  if (typeof v === "string") return /^[A-Za-z_][A-Za-z0-9_]*$/.test(v) ? v : JSON.stringify(v);
  return String(v);
}

const REVERSE_ELEMENTWISE = ELEMENTWISE_FNS;

export function toDSL(g: Graph): string {
  const lines: string[] = [];
  const paramKeys = Object.keys(g.params);
  if (paramKeys.length)
    lines.push("params " + paramKeys.map((k) => `${k}=${g.params[k]}`).join(" "));
  for (const t of Object.values(g.tensors)) {
    if (t.producer || g.nodes.some((n) => n.outputs.includes(t.id))) continue;
    const kw = t.role === "weight" ? "weight" : "input";
    lines.push(`${kw} ${t.name} [${t.shape.map(String).join(", ")}] ${t.dtype}`);
  }
  for (const n of g.nodes) {
    const outNames = n.outputs.map((o) => g.tensors[o].name);
    const inNames = n.inputs.map((i) => g.tensors[i].name);
    let call: string;
    const attrs = { ...n.attrs };
    if (n.op === "elementwise" && REVERSE_ELEMENTWISE.has(attrs.fn as string)) {
      const fn = attrs.fn as string;
      // The call itself carries `fn` and `nary`; every other attribute has to be
      // written out, or expanding a composite and recompiling drops it silently.
      const named = Object.entries(attrs)
        .filter(([k]) => k !== "fn" && k !== "nary")
        .map(([k, v]) => `${k}=${attrValueToDSL(v)}`);
      call = `${fn}(${[...inNames, ...named].join(", ")})`;
    } else if (n.op === "einsum") {
      const eq = attrs.equation as string;
      delete attrs.equation;
      const rest = Object.entries(attrs).map(([k, v]) => `${k}=${attrValueToDSL(v)}`);
      call = `einsum(${[`"${eq}"`, ...inNames, ...rest].join(", ")})`;
    } else if (n.op === "normalize") {
      const kind = attrs.kind as string;
      const named = Object.entries(attrs)
        .filter(([k]) => !["kind", "hasWeight", "hasBias"].includes(k))
        .map(([k, v]) => `${k}=${attrValueToDSL(v)}`);
      call = `${kind}(${[...inNames, ...named].join(", ")})`;
    } else if (n.op === "reduce" && ["sum", "mean", "prod"].includes(attrs.fn as string)) {
      const fn = attrs.fn as string;
      const named = Object.entries(attrs)
        .filter(([k]) => k !== "fn")
        .map(([k, v]) => `${k}=${attrValueToDSL(v)}`);
      call = `${fn}(${[...inNames, ...named].join(", ")})`;
    } else {
      const named = Object.entries(attrs).map(([k, v]) => `${k}=${attrValueToDSL(v)}`);
      call = `${n.op}(${[...inNames, ...named].join(", ")})`;
    }
    lines.push(`${outNames.join(", ")} = ${call}`);
  }
  return lines.join("\n") + "\n";
}
