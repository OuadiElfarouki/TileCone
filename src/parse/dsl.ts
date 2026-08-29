/** Text DSL (IDEA.md §6.2):
 *   params M=512 K=2048
 *   input A [M, K] f16
 *   C = einsum("mk,kn->mn", A, B)
 *   Y0, Y1 = split(X, axis=0, sizes=[2, 2])
 * One statement per line, `#` comments. Round-trips losslessly via toDSL.
 */

import { DType, Graph, Node, Tensor } from "../core/graph";
import { Sym } from "../core/shapes";

export class DSLError extends Error {
  constructor(message: string, public line: number) {
    super(`line ${line}: ${message}`);
  }
}

const DTYPES = new Set(["f32", "f16", "bf16", "f8", "i32", "i8", "bool"]);

/** fn-name sugar -> op + fixed attrs */
const ELEMENTWISE_FNS = new Set([
  "add", "sub", "mul", "div", "pow", "maximum", "minimum",
  "relu", "gelu", "silu", "exp", "log", "sqrt", "rsqrt", "neg", "abs", "sigmoid", "tanh",
]);
const REDUCE_FNS = new Set(["sum", "mean", "prod", "amax", "amin"]);

type Value = number | string | boolean | Value[] | { ident: string };

class LineParser {
  pos = 0;
  constructor(public src: string, public line: number) {}
  error(msg: string): never {
    throw new DSLError(`${msg} (at "${this.src.slice(this.pos, this.pos + 12)}...")`, this.line);
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
    const m = /^[A-Za-z_][A-Za-z0-9_.]*/.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return m[0];
  }
  identReq(what: string): string {
    const v = this.ident();
    if (v === null) throw new DSLError(`expected ${what}`, this.line);
    return v;
  }
  numberReq(what: string): number {
    const v = this.number();
    if (v === null) throw new DSLError(`expected ${what}`, this.line);
    return v;
  }
  number(): number | null {
    this.ws();
    const m = /^-?\d+(\.\d+)?/.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return Number(m[0]);
  }
  string(): string | null {
    this.ws();
    if (this.src[this.pos] !== '"') return null;
    const end = this.src.indexOf('"', this.pos + 1);
    if (end < 0) this.error("unterminated string");
    const s = this.src.slice(this.pos + 1, end);
    this.pos = end + 1;
    return s;
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

export function parseDSL(text: string): Graph {
  const params: Record<string, number> = {};
  const tensors: Record<string, Tensor> = {};
  const nodes: Node[] = [];
  const lines = text.split("\n");

  for (let ln = 0; ln < lines.length; ln++) {
    const raw = lines[ln].replace(/#.*$/, "").trim();
    if (!raw) continue;
    const p = new LineParser(raw, ln + 1);

    if (raw.startsWith("params")) {
      p.expect("params");
      while (!p.atEnd()) {
        const name = p.identReq("param name");
        p.expect("=");
        params[name] = p.numberReq("number");
      }
      continue;
    }

    if (raw.startsWith("input ")) {
      p.expect("input");
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
        if (!DTYPES.has(dt)) p.error(`unknown dtype "${dt}"`);
        dtype = dt as DType;
      }
      if (tensors[name]) p.error(`tensor "${name}" redefined`);
      tensors[name] = { id: name, name, shape, dtype };
      continue;
    }

    // assignment: NAME[, NAME...] = op(args)
    const outs: string[] = [];
    outs.push(p.identReq("statement"));
    while (p.eat(",")) outs.push(p.identReq("output name"));
    p.expect("=");
    const callee = p.identReq("op name");
    p.expect("(");
    const positional: Value[] = [];
    const named: Record<string, unknown> = {};
    if (!p.eat(")")) {
      do {
        const save = p.pos;
        const id = p.ident();
        if (id && p.eat("=")) named[id] = valueToAttr(p.value());
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
    } else if (callee === "layernorm" || callee === "rmsnorm") {
      op = "normalize";
      attrs.kind = callee;
      attrs.hasWeight = inputs.length >= 2;
      attrs.hasBias = inputs.length >= 3;
      if (attrs.axes === undefined) attrs.axes = [-1];
    }

    for (const t of inputs)
      if (!tensors[t]) throw new DSLError(`unknown tensor "${t}"`, ln + 1);
    for (const o of outs) {
      if (tensors[o]) throw new DSLError(`tensor "${o}" redefined`, ln + 1);
      tensors[o] = { id: o, name: o, shape: [], dtype: tensors[inputs[0]]?.dtype ?? "f32" };
    }
    nodes.push({ id: `${op}_${outs[0]}`, op, inputs, outputs: outs, attrs });
  }
  return { nodes, tensors, params };
}

// ------------------------------------------------------------------- toDSL

function attrValueToDSL(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(attrValueToDSL).join(", ")}]`;
  if (typeof v === "string") return /^[A-Za-z_][A-Za-z0-9_]*$/.test(v) ? v : `"${v}"`;
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
    lines.push(`input ${t.name} [${t.shape.map(String).join(", ")}] ${t.dtype}`);
  }
  for (const n of g.nodes) {
    const outNames = n.outputs.map((o) => g.tensors[o].name);
    const inNames = n.inputs.map((i) => g.tensors[i].name);
    let call: string;
    const attrs = { ...n.attrs };
    if (n.op === "elementwise" && REVERSE_ELEMENTWISE.has(attrs.fn as string)) {
      const fn = attrs.fn as string;
      call = `${fn}(${inNames.join(", ")})`;
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
