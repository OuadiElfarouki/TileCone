/**
 * AST -> Graph.
 *
 * This is where a call is bound to its signature and becomes an operation with
 * attributes, and where names are resolved. It collects errors instead of
 * throwing at the first one, and it *poisons* rather than cascades: a statement that fails
 * still registers the names it was going to bind, and a later statement reading
 * a poisoned name is skipped in silence. Reporting `unknown tensor "Y"` on
 * every subsequent line because line 3 failed would bury the one error the
 * author actually has to fix.
 */

import { Graph, Node, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";
import { dimSymbols, Sym } from "../core/shapes";
import { nearestName } from "../core/ops/kinds";
import { Expr, Program, Spanned, Stmt } from "./ast";
import { bindCall, dtypeFromWord, DSL_DTYPES } from "./bind";
import { DSLError } from "./parser";
import { foreignName, SIGNATURES, signatureFor, SUBSCRIPT_OP } from "./signatures";
import { DSLSourceMap, NodeArgSpans, SourceSpan } from "./source";

export { DSL_DTYPES, DTYPE_TO_DSL } from "./bind";

export type LowerResult = {
  graph: Graph;
  sourceMap: DSLSourceMap;
  errors: DSLError[];
};

function dtypeFromExpr(e: Expr): DType | null {
  return e.kind === "dim" ? dtypeFromWord(e.text) : null;
}

export function lowerProgram(program: Program): LowerResult {
  const dictionary = <T>(): Record<string, T> => Object.create(null) as Record<string, T>;
  const params = dictionary<number>();
  const derived = dictionary<Sym>();
  const tensors = dictionary<Tensor>();
  const nodes: Node[] = [];
  const sourceMap: DSLSourceMap = {
    document: program.span,
    params: dictionary<SourceSpan>(),
    tensors: dictionary<SourceSpan>(),
    nodes: dictionary<SourceSpan>(),
    nodeArgs: dictionary<DSLSourceMap["nodeArgs"][string]>(),
  };
  const errors: DSLError[] = [];
  /** Names bound by a statement that failed; reading one is not a new error. */
  const poisoned = new Set<string>();

  const fail = (detail: string, span: SourceSpan, code: string) => {
    errors.push(new DSLError(detail, span, code));
  };
  const defined = (name: string) =>
    Object.prototype.hasOwnProperty.call(params, name) ||
    Object.prototype.hasOwnProperty.call(derived, name) ||
    Object.prototype.hasOwnProperty.call(tensors, name) ||
    poisoned.has(name);

  /** Whether a written value reads a symbol a failed statement would have
   * bound. Such a value is skipped in silence, like a poisoned tensor. */
  const readsPoisoned = (e: Expr): boolean =>
    e.kind === "list"
      ? e.items.some(readsPoisoned)
      : e.kind === "dim" && dimSymbols(e.text).some((name) => poisoned.has(name));

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

    if (stmt.kind === "derive") {
      lowerDerive(stmt);
      continue;
    }

    if (stmt.kind === "declare") {
      lowerDeclare(stmt);
      continue;
    }

    if (stmt.kind === "subscript") {
      lowerSubscript(stmt);
      continue;
    }

    lowerCall(stmt);
  }

  /**
   * A derived symbol may only be written over the parameters and derived
   * symbols above it. That keeps definitions acyclic by construction, and an
   * unknown name is reported here, on the definition, rather than at every
   * dimension that later reads it.
   */
  function lowerDerive(stmt: Stmt & { kind: "derive" }): void {
    if (!claim(stmt.name, "DSL_DUPLICATE_PARAM")) return;
    let bad = false;
    for (const name of dimSymbols(stmt.value.text)) {
      if (
        Object.prototype.hasOwnProperty.call(params, name) ||
        Object.prototype.hasOwnProperty.call(derived, name)
      )
        continue;
      bad = true;
      if (poisoned.has(name)) continue;
      fail(
        Object.prototype.hasOwnProperty.call(tensors, name)
          ? `"${name}" is a tensor; a derived symbol is written over size parameters`
          : `unknown parameter "${name}"; a derived symbol can use only parameters defined above it`,
        stmt.value.span,
        "DSL_UNKNOWN_PARAM"
      );
    }
    if (bad) {
      poisoned.add(stmt.name.value);
      return;
    }
    derived[stmt.name.value] = stmt.value.text;
    sourceMap.params[stmt.name.value] = stmt.span;
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
      if (readsPoisoned(value)) {
        bad = true;
        continue;
      }
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
      if (dtypeFromWord(e.text)) {
        fail(`dtype is an attribute here: write dtype=${e.text}`, e.span, "DSL_SYNTAX");
        return null;
      }
      return e.text;
    }
    fail("tensor dimensions must be numbers or symbolic expressions", e.span, "DSL_SYNTAX");
    return null;
  }

  /**
   * The node a statement builds, once its inputs and attributes are known.
   * Unknown inputs are reported per reference, on the reference's own span. A
   * reference to a name that a *failed* statement would have bound is not
   * reported at all: the author has one error there, not two.
   */
  function emit(
    outs: Spanned<string>[],
    op: string,
    inputs: Spanned<string>[],
    attrs: Record<string, unknown>,
    spans: {
      statement: SourceSpan;
      callee: SourceSpan;
      attrs: Record<string, SourceSpan>;
      written: NodeArgSpans["written"];
    },
    bad: boolean
  ): void {
    let missing = false;
    for (const ref of inputs) {
      if (tensors[ref.value]) continue;
      missing = true;
      if (poisoned.has(ref.value)) continue;
      fail(`unknown tensor "${ref.value}"`, ref.span, "DSL_UNKNOWN_TENSOR");
    }

    const claimed = outs.filter((out) => claim(out, "DSL_DUPLICATE_TENSOR"));
    if (claimed.length !== outs.length) return;
    if (bad || missing) {
      for (const out of outs) poisoned.add(out.value);
      return;
    }

    for (const out of outs) {
      tensors[out.value] = {
        id: out.value,
        name: out.value,
        shape: [],
        dtype: tensors[inputs[0]?.value]?.dtype ?? "f32",
      };
      sourceMap.tensors[out.value] = spans.statement;
    }
    const node: Node = {
      id: `${op}_${outs[0].value}`,
      op,
      inputs: inputs.map((ref) => ref.value),
      outputs: outs.map((o) => o.value),
      attrs,
    };
    nodes.push(node);
    sourceMap.nodes[node.id] = spans.statement;
    // Argument-level spans, so a diagnostic about one operand or one attribute
    // can underline it rather than the whole statement.
    sourceMap.nodeArgs[node.id] = {
      inputs: inputs.map((ref) => ref.span),
      attrs: spans.attrs,
      callee: spans.callee,
      written: spans.written,
    };
  }

  function lowerCall(stmt: Stmt & { kind: "call" }): void {
    const callee = stmt.callee.value;
    const sig = signatureFor(callee);
    if (!sig) {
      const near = foreignName(callee) ?? nearestName(callee, SIGNATURES.map((s) => s.name));
      fail(
        `unknown function "${callee}"${near ? ` (did you mean ${near}?)` : ""}`,
        stmt.callee.span,
        "DSL_UNKNOWN_FUNCTION"
      );
      for (const out of stmt.outs) if (claim(out, "DSL_DUPLICATE_TENSOR")) poisoned.add(out.value);
      return;
    }
    // A value reading a symbol that a failed statement would have bound is
    // skipped in silence, like a poisoned tensor.
    const silent = stmt.args.some((arg) => arg.kind === "named" && readsPoisoned(arg.value));
    const { bound, errors: bindErrors } = bindCall(sig, stmt.args, stmt.span);
    errors.push(...bindErrors);
    emit(
      stmt.outs,
      sig.op,
      bound?.inputs ?? [],
      bound?.attrs ?? {},
      {
        statement: stmt.span,
        callee: stmt.callee.span,
        attrs: bound?.attrSpans ?? {},
        written: { function: callee },
      },
      !bound || silent
    );
  }

  /**
   * `X[a:b, :, ::2]` as a slice: one start, stop and step per written axis,
   * null where a part is left out. An integer index would drop its axis, which
   * the slice operation does not do, so it is refused with the range to write.
   */
  function lowerSubscript(stmt: Stmt & { kind: "subscript" }): void {
    const starts: unknown[] = [];
    const stops: unknown[] = [];
    const steps: unknown[] = [];
    let bad = false;
    const bound = (e: Expr | undefined): unknown => {
      if (e === undefined) return null;
      if (readsPoisoned(e)) {
        bad = true;
        return null;
      }
      if (e.kind === "number") return e.value;
      if (e.kind === "dim") return e.text;
      fail("a slice bound is an int or a dimension expression", e.span, "DSL_SYNTAX");
      bad = true;
      return null;
    };
    for (const item of stmt.items) {
      if (item.kind === "index") {
        const text = item.value.kind === "dim" ? item.value.text : String((item.value as { value?: unknown }).value);
        fail(
          `an index would drop the axis, which slicing here does not do; write ${text}:${text}+1 to keep it`,
          item.span,
          "DSL_UNSUPPORTED_INDEX"
        );
        bad = true;
        continue;
      }
      starts.push(bound(item.start));
      stops.push(bound(item.stop));
      steps.push(bound(item.step));
    }
    const at = { starts: stmt.brackets, stops: stmt.brackets, steps: stmt.brackets };
    emit(
      stmt.outs,
      SUBSCRIPT_OP,
      [stmt.target],
      { starts, stops, steps },
      { statement: stmt.span, callee: stmt.brackets, attrs: at, written: { subscript: stmt.target.value } },
      bad
    );
  }

  return {
    graph: { nodes, tensors, params, ...(Object.keys(derived).length ? { derived } : {}) },
    sourceMap,
    errors,
  };
}
