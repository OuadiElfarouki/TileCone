/** Text DSL (IDEA.md §6.2):
 *   M = 512
 *   K = 2048
 *   A = Tensor(M, K, dtype=fp16)
 *   C = einsum("mk,kn->mn", A, B)
 *   Y0, Y1 = split(X, axis=0, sizes=[2, 2])
 * One statement per line, `#` comments. Round-trips losslessly via toDSL.
 */

import { Graph, Node, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";
import { NUMBER_RE, readDimExpr, Sym } from "../core/shapes";
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

export const DSL_DTYPES = ["fp32", "fp16", "bf16", "fp8", "int32", "int8", "bool"] as const;

const DTYPE_FROM_DSL: Record<(typeof DSL_DTYPES)[number], DType> = {
  fp32: "f32",
  fp16: "f16",
  bf16: "bf16",
  fp8: "f8",
  int32: "i32",
  int8: "i8",
  bool: "bool",
};

const DTYPE_TO_DSL: Record<DType, (typeof DSL_DTYPES)[number]> = {
  f32: "fp32",
  f16: "fp16",
  bf16: "bf16",
  f8: "fp8",
  i32: "int32",
  i8: "int8",
  bool: "bool",
};

/** Anchored `NUMBER_RE`: does this expression consist of one literal? */
const NUMBER_ONLY = new RegExp(`${NUMBER_RE.source}$`);

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
  number(): number | null {
    this.ws();
    // Scientific notation included, so an attribute like eps=1e-5 is expressible
    // and `toDSL` cannot emit a literal (String(1e-21) === "1e-21") that this
    // parser then rejects. Shared with dimension expressions.
    const m = NUMBER_RE.exec(this.src.slice(this.pos));
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
    // Booleans are identifiers but are values, not one-symbol dimensions.
    const save = this.pos;
    const word = this.ident();
    if (word === "true") return true;
    if (word === "false") return false;
    this.pos = save;
    // Everything else is read as a dimension expression, so a shape attribute
    // can say `shape=[B, S, H, E/H]` in the same language a declaration uses.
    // A lone literal is still a number; anything else keeps its written form.
    const parsed = readDimExpr(this.src, this.pos);
    if (!parsed) this.error("expected value");
    const text = this.src.slice(this.pos, parsed.end).trim();
    this.pos = parsed.end;
    return NUMBER_ONLY.test(text) ? Number(text) : { ident: text };
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

function dimensionValue(v: Value, p: LineParser): Sym {
  if (typeof v === "number") return v;
  if (typeof v === "object" && v !== null && !Array.isArray(v) && "ident" in v) {
    // `Tensor(4, fp16)` is the shape of the removed grammar, where the dtype
    // was a trailing positional word. Read as a dimension it is a symbol that
    // happens to be spelled like a dtype, and the failure surfaces much later
    // as an unbound symbol - naming a real mistake something it is not.
    if (v.ident in DTYPE_FROM_DSL)
      p.error(`dtype is an attribute here: write dtype=${v.ident}`);
    return v.ident;
  }
  p.error("tensor dimensions must be numbers or symbolic expressions");
}

function dtypeValue(v: Value, p: LineParser): DType {
  if (typeof v === "object" && v !== null && !Array.isArray(v) && "ident" in v) {
    const dtype = DTYPE_FROM_DSL[v.ident as keyof typeof DTYPE_FROM_DSL];
    if (dtype) return dtype;
  }
  p.error(`dtype must be one of ${DSL_DTYPES.join(", ")}`);
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

    // Every statement is an assignment: a scalar dimension, an input
    // constructor, or an operation with one or more outputs.
    const outs: string[] = [];
    outs.push(p.identReq("statement"));
    while (p.eat(",")) outs.push(p.identReq("output name"));
    // `Tensor = Tensor(4)` parses, but the constructor always wins in call
    // position, so the name can never be read back as the tensor it bound.
    for (const out of outs)
      if (out === "Tensor" || out === "Parameter")
        throw new DSLError(
          `"${out}" is a constructor and cannot name a tensor`,
          statementSpan,
          "DSL_RESERVED_NAME"
        );
    if (!p.eat("=")) {
      throw new DSLError(
        `expected an assignment: NAME = number, Tensor(...), Parameter(...), or op(...)`,
        statementSpan,
        "DSL_UNKNOWN_STATEMENT"
      );
    }

    const rhsStart = p.pos;
    if (outs.length === 1) {
      const scalar = p.number();
      if (scalar !== null && p.atEnd()) {
        const name = outs[0];
        if (Object.prototype.hasOwnProperty.call(params, name) || tensors[name])
          throw new DSLError(`symbol "${name}" redefined`, statementSpan, "DSL_DUPLICATE_PARAM");
        params[name] = scalar;
        sourceMap.params[name] = statementSpan;
        lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
        continue;
      }
      p.pos = rhsStart;
    }

    const callee = p.identReq("op name");
    p.expect("(");

    if (callee === "Tensor" || callee === "Parameter") {
      if (outs.length !== 1)
        throw new DSLError(`${callee} declares exactly one tensor`, statementSpan);
      const shape: Sym[] = [];
      const axisNames: (string | undefined)[] = [];
      let dtype: DType = "f32";
      let sawDType = false;
      if (!p.eat(")")) {
        do {
          const save = p.pos;
          const label = p.ident();
          if (label && p.eat("=")) {
            if (label === "dtype") {
              if (sawDType)
                throw new DSLError(
                  `attribute "dtype" specified more than once`,
                  statementSpan,
                  "DSL_DUPLICATE_ATTRIBUTE"
                );
              dtype = dtypeValue(p.value(), p);
              sawDType = true;
            } else {
              axisNames.push(label);
              shape.push(dimensionValue(p.value(), p));
            }
          } else {
            p.pos = save;
            axisNames.push(undefined);
            shape.push(dimensionValue(p.value(), p));
          }
        } while (p.eat(","));
        p.expect(")");
      }
      if (!p.atEnd()) p.error("trailing input");

      const namedAxes = axisNames.filter((axis): axis is string => axis !== undefined);
      if (namedAxes.length && namedAxes.length !== axisNames.length)
        throw new DSLError(
          `tensor "${outs[0]}": name every axis or none`,
          statementSpan,
          "DSL_PARTIAL_AXIS_NAMES"
        );
      if (new Set(namedAxes).size !== namedAxes.length)
        throw new DSLError(
          `tensor "${outs[0]}": duplicate axis name`,
          statementSpan,
          "DSL_DUPLICATE_AXIS_NAME"
        );

      const name = outs[0];
      if (tensors[name] || Object.prototype.hasOwnProperty.call(params, name))
        throw new DSLError(`symbol "${name}" redefined`, statementSpan, "DSL_DUPLICATE_TENSOR");
      tensors[name] = {
        id: name,
        name,
        shape,
        dtype,
        ...(namedAxes.length ? { axisNames: namedAxes } : {}),
        ...(callee === "Parameter" ? { role: "weight" as const } : {}),
      };
      sourceMap.tensors[name] = statementSpan;
      lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
      continue;
    }

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
          const value = p.value();
          named[id] = id === "dtype" ? dtypeValue(value, p) : valueToAttr(value);
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
      if (tensors[o] || Object.prototype.hasOwnProperty.call(params, o))
        throw new DSLError(`symbol "${o}" redefined`, statementSpan, "DSL_DUPLICATE_TENSOR");
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

function attrValueToDSL(v: unknown, name?: string): string {
  if (Array.isArray(v)) return `[${v.map((item) => attrValueToDSL(item)).join(", ")}]`;
  if (name === "dtype" && typeof v === "string" && v in DTYPE_TO_DSL)
    return DTYPE_TO_DSL[v as DType];
  if (typeof v === "string") return /^[A-Za-z_][A-Za-z0-9_]*$/.test(v) ? v : JSON.stringify(v);
  return String(v);
}

const REVERSE_ELEMENTWISE = ELEMENTWISE_FNS;

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
    let call: string;
    const attrs = { ...n.attrs };
    if (n.op === "elementwise" && REVERSE_ELEMENTWISE.has(attrs.fn as string)) {
      const fn = attrs.fn as string;
      // The call itself carries `fn` and `nary`; every other attribute has to be
      // written out, or expanding a composite and recompiling drops it silently.
      const named = Object.entries(attrs)
        .filter(([k]) => k !== "fn" && k !== "nary")
        .map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
      call = `${fn}(${[...inNames, ...named].join(", ")})`;
    } else if (n.op === "einsum") {
      const eq = attrs.equation as string;
      delete attrs.equation;
      const rest = Object.entries(attrs).map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
      call = `einsum(${[`"${eq}"`, ...inNames, ...rest].join(", ")})`;
    } else if (n.op === "normalize") {
      const kind = attrs.kind as string;
      const named = Object.entries(attrs)
        .filter(([k]) => !["kind", "hasWeight", "hasBias"].includes(k))
        .map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
      call = `${kind}(${[...inNames, ...named].join(", ")})`;
    } else if (n.op === "reduce" && ["sum", "mean", "prod"].includes(attrs.fn as string)) {
      const fn = attrs.fn as string;
      const named = Object.entries(attrs)
        .filter(([k]) => k !== "fn")
        .map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
      call = `${fn}(${[...inNames, ...named].join(", ")})`;
    } else {
      const named = Object.entries(attrs).map(([k, v]) => `${k}=${attrValueToDSL(v, k)}`);
      call = `${n.op}(${[...inNames, ...named].join(", ")})`;
    }
    lines.push(`${outNames.join(", ")} = ${call}`);
  }
  return lines.join("\n") + "\n";
}
