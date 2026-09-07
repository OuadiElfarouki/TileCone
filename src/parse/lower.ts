/**
 * AST -> Graph.
 *
 * This is where the DSL's call-name sugar becomes operations and attributes,
 * and where names are resolved. It collects errors instead of throwing at the
 * first one, and it *poisons* rather than cascades: a statement that fails
 * still registers the names it was going to bind, and a later statement reading
 * a poisoned name is skipped in silence. Reporting `unknown tensor "Y"` on
 * every subsequent line because line 3 failed would bury the one error the
 * author actually has to fix.
 */

import { Graph, Node, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";
import { Sym } from "../core/shapes";
import { Arg, Expr, Program, Spanned, Stmt } from "./ast";
import { DSLError } from "./parser";
import { sugarForCall } from "./sugar";
import { DSLSourceMap, SourceSpan } from "./source";

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

export const DTYPE_TO_DSL: Record<DType, (typeof DSL_DTYPES)[number]> = {
  f32: "fp32",
  f16: "fp16",
  bf16: "bf16",
  f8: "fp8",
  i32: "int32",
  i8: "int8",
  bool: "bool",
};

export type LowerResult = {
  graph: Graph;
  sourceMap: DSLSourceMap;
  errors: DSLError[];
};

/** An attribute value, with symbolic dimensions kept as their written text. */
function exprToAttr(e: Expr): unknown {
  switch (e.kind) {
    case "list":
      return e.items.map(exprToAttr);
    case "dim":
      return e.text;
    default:
      return e.value;
  }
}

function dtypeFromExpr(e: Expr): DType | null {
  if (e.kind !== "dim") return null;
  return DTYPE_FROM_DSL[e.text as keyof typeof DTYPE_FROM_DSL] ?? null;
}

export function lowerProgram(program: Program): LowerResult {
  const params: Record<string, number> = {};
  const tensors: Record<string, Tensor> = {};
  const nodes: Node[] = [];
  const sourceMap: DSLSourceMap = {
    document: program.span,
    params: {},
    tensors: {},
    nodes: {},
    nodeArgs: {},
  };
  const errors: DSLError[] = [];
  /** Names bound by a statement that failed; reading one is not a new error. */
  const poisoned = new Set<string>();

  const fail = (detail: string, span: SourceSpan, code: string) => {
    errors.push(new DSLError(detail, span, code));
  };
  const defined = (name: string) =>
    Object.prototype.hasOwnProperty.call(params, name) || !!tensors[name] || poisoned.has(name);

  /** Report a redefinition, or claim the name. Returns false when taken. */
  const claim = (name: Spanned<string>, code: string): boolean => {
    if (!defined(name.value)) return true;
    fail(`symbol "${name.value}" redefined`, name.span, code);
    return false;
  };

  for (const stmt of program.stmts) {
    if (stmt.kind === "error") {
      // The line already produced a parse error. Claim its names as poisoned so
      // that reading one below is not reported as a second, separate mistake.
      for (const out of stmt.outs) poisoned.add(out.value);
      continue;
    }

    if (stmt.kind === "param") {
      if (!claim(stmt.name, "DSL_DUPLICATE_PARAM")) continue;
      params[stmt.name.value] = stmt.value.value;
      sourceMap.params[stmt.name.value] = stmt.span;
      continue;
    }

    if (stmt.kind === "declare") {
      lowerDeclare(stmt);
      continue;
    }

    lowerCall(stmt);
  }

  function lowerDeclare(stmt: Stmt & { kind: "declare" }): void {
    const shape: Sym[] = [];
    const axisNames: (string | undefined)[] = [];
    let dtype: DType = "f32";
    let sawDType = false;
    let bad = false;

    for (const arg of stmt.args) {
      if (arg.kind === "named" && arg.name.value === "dtype") {
        if (sawDType) {
          fail(`attribute "dtype" specified more than once`, arg.span, "DSL_DUPLICATE_ATTRIBUTE");
          bad = true;
          continue;
        }
        const resolvedDType = dtypeFromExpr(arg.value);
        if (!resolvedDType) {
          fail(`dtype must be one of ${DSL_DTYPES.join(", ")}`, arg.value.span, "DSL_SYNTAX");
          bad = true;
          continue;
        }
        dtype = resolvedDType;
        sawDType = true;
        continue;
      }
      const value = arg.kind === "named" ? arg.value : arg.value;
      const dim = dimensionOf(value);
      if (dim === null) {
        bad = true;
        continue;
      }
      axisNames.push(arg.kind === "named" ? arg.name.value : undefined);
      shape.push(dim);
    }

    const named = axisNames.filter((a): a is string => a !== undefined);
    if (named.length && named.length !== axisNames.length) {
      fail(`tensor "${stmt.name.value}": name every axis or none`, stmt.span, "DSL_PARTIAL_AXIS_NAMES");
      bad = true;
    }
    if (new Set(named).size !== named.length) {
      fail(`tensor "${stmt.name.value}": duplicate axis name`, stmt.span, "DSL_DUPLICATE_AXIS_NAME");
      bad = true;
    }
    if (!claim(stmt.name, "DSL_DUPLICATE_TENSOR")) return;
    if (bad) {
      poisoned.add(stmt.name.value);
      return;
    }
    tensors[stmt.name.value] = {
      id: stmt.name.value,
      name: stmt.name.value,
      shape,
      dtype,
      ...(named.length ? { axisNames: named } : {}),
      ...(stmt.ctor.value === "Parameter" ? { role: "weight" as const } : {}),
    };
    sourceMap.tensors[stmt.name.value] = stmt.span;
  }

  /** A dimension expression, or null after reporting why it is not one. */
  function dimensionOf(e: Expr): Sym | null {
    if (e.kind === "number") return e.value;
    if (e.kind === "dim") {
      // `Tensor(4, fp16)` is the shape of a removed grammar, where the dtype
      // was a trailing positional word. Read as a dimension it is a symbol that
      // happens to be spelled like a dtype, and the failure would surface much
      // later as an unbound symbol - naming a real mistake something it is not.
      if (e.text in DTYPE_FROM_DSL) {
        fail(`dtype is an attribute here: write dtype=${e.text}`, e.span, "DSL_SYNTAX");
        return null;
      }
      return e.text;
    }
    fail("tensor dimensions must be numbers or symbolic expressions", e.span, "DSL_SYNTAX");
    return null;
  }

  function lowerCall(stmt: Stmt & { kind: "call" }): void {
    const callee = stmt.callee.value;
    const attrs: Record<string, unknown> = {};
    const inputs: string[] = [];
    const inputSpans: SourceSpan[] = [];
    let bad = false;
    const seenAttrs = new Set<string>();

    for (const arg of stmt.args) {
      if (arg.kind === "named") {
        if (seenAttrs.has(arg.name.value)) {
          fail(
            `attribute "${arg.name.value}" specified more than once`,
            arg.span,
            "DSL_DUPLICATE_ATTRIBUTE"
          );
          bad = true;
          continue;
        }
        seenAttrs.add(arg.name.value);
        if (arg.name.value === "dtype") {
          const d = dtypeFromExpr(arg.value);
          if (!d) {
            fail(`dtype must be one of ${DSL_DTYPES.join(", ")}`, arg.value.span, "DSL_SYNTAX");
            bad = true;
            continue;
          }
          attrs.dtype = d;
          continue;
        }
        attrs[arg.name.value] = exprToAttr(arg.value);
        continue;
      }
      // positional: a tensor reference, or einsum's equation
      const e = arg.value;
      if (e.kind === "string") {
        if (callee === "einsum" && attrs.equation === undefined) attrs.equation = e.value;
        else {
          fail("unexpected string argument", e.span, "DSL_SYNTAX");
          bad = true;
        }
        continue;
      }
      if (e.kind === "dim") {
        inputs.push(e.text);
        inputSpans.push(e.span);
        continue;
      }
      fail("positional args must be tensors or an einsum equation", e.span, "DSL_SYNTAX");
      bad = true;
    }

    // Desugaring is table-driven, and the printer reads the same table, so a
    // call the DSL accepts is a call the DSL can write back.
    const sugar = sugarForCall(callee);
    let op = callee;
    if (sugar) {
      op = sugar.op;
      // The singular spelling, which is what softmax and cumsum take.
      if (sugar.axisAlias && attrs.axes === undefined && attrs.axis !== undefined) {
        attrs.axes = [attrs.axis];
        delete attrs.axis;
      }
      for (const [key, value] of Object.entries(sugar.defaults ?? {}))
        if (attrs[key] === undefined) attrs[key] = value;
      Object.assign(attrs, sugar.attrsFor(inputs.length));
      for (const key of sugar.requires ?? [])
        if (attrs[key] === undefined) {
          fail(
            `${callee}() needs an axis, e.g. ${callee}(X, axis=-1)`,
            stmt.span,
            "DSL_MISSING_ATTRIBUTE"
          );
          bad = true;
        }
    }

    // Unknown inputs are reported per reference, on the reference's own span.
    // A reference to a name that a *failed* statement would have bound is not
    // reported at all: the author has one error there, not two.
    let missing = false;
    inputs.forEach((name, i) => {
      if (tensors[name]) return;
      if (poisoned.has(name)) {
        missing = true;
        return;
      }
      fail(`unknown tensor "${name}"`, inputSpans[i], "DSL_UNKNOWN_TENSOR");
      missing = true;
    });

    const claimed = stmt.outs.filter((out) => claim(out, "DSL_DUPLICATE_TENSOR"));
    if (claimed.length !== stmt.outs.length) return;
    if (bad || missing) {
      for (const out of stmt.outs) poisoned.add(out.value);
      return;
    }

    for (const out of stmt.outs) {
      tensors[out.value] = {
        id: out.value,
        name: out.value,
        shape: [],
        dtype: tensors[inputs[0]]?.dtype ?? "f32",
      };
      sourceMap.tensors[out.value] = stmt.span;
    }
    const node: Node = {
      id: `${op}_${stmt.outs[0].value}`,
      op,
      inputs,
      outputs: stmt.outs.map((o) => o.value),
      attrs,
    };
    nodes.push(node);
    sourceMap.nodes[node.id] = stmt.span;
    // Argument-level spans, so a diagnostic about one operand or one attribute
    // can underline it rather than the whole statement.
    sourceMap.nodeArgs[node.id] = {
      inputs: inputSpans,
      attrs: Object.fromEntries(
        stmt.args
          .filter((a): a is Arg & { kind: "named" } => a.kind === "named")
          .map((a) => [a.name.value, a.span])
      ),
      callee: stmt.callee.span,
    };
  }

  return { graph: { nodes, tensors, params }, sourceMap, errors };
}
