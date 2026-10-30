/**
 * Binding a call's arguments to a signature's parameters.
 *
 * One procedure for every function, with Python's rules: positional arguments
 * fill parameters in order up to the keyword-only marker, a variadic parameter
 * takes the rest, keywords bind by name, and a parameter bound twice, a
 * keyword nobody declared, or a required parameter left out is an error that
 * names it. The value written for an attribute is checked against that
 * attribute's kind here, so the error underlines the value.
 *
 * What a value means is the kind's business, not this module's. Here it is
 * only spelled: free text is quoted, a dtype or an enum member is a bare word,
 * and everything else is numbers, words and lists.
 */

import { DSLError } from "./parser";
import type { Arg, Expr, Spanned } from "./ast";
import { IDENT_RE } from "./lexical";
import type { SourceSpan } from "./source";
import { nearestName, type Kind } from "../core/ops/kinds";
import type { Attrs } from "../core/ops/types";
import type { DType } from "../core/dtypes";
import {
  foreignKeyword,
  kindOf,
  signatureLine,
  type Param,
  type Signature,
} from "./signatures";

/** The DSL's dtype names: those of the Array API, NumPy and PyTorch. */
export const DSL_DTYPES = [
  "float32", "float16", "bfloat16", "float8", "int64", "int32", "uint8", "int8", "bool",
] as const;

export type DSLDType = (typeof DSL_DTYPES)[number];

export const DTYPE_FROM_DSL: Record<DSLDType, DType> = {
  float32: "f32",
  float16: "f16",
  bfloat16: "bf16",
  float8: "f8",
  int64: "i64",
  int32: "i32",
  uint8: "u8",
  int8: "i8",
  bool: "bool",
};

export const DTYPE_TO_DSL: Record<DType, DSLDType> = {
  f32: "float32",
  f16: "float16",
  bf16: "bfloat16",
  f8: "float8",
  i64: "int64",
  i32: "int32",
  u8: "uint8",
  i8: "int8",
  bool: "bool",
};

/** The IR dtype a written word names, or null. */
export function dtypeFromWord(word: string): DType | null {
  return Object.prototype.hasOwnProperty.call(DTYPE_FROM_DSL, word)
    ? DTYPE_FROM_DSL[word as DSLDType]
    : null;
}

export type BoundCall = {
  /** Tensor references, in the order the operation takes its inputs. */
  inputs: Spanned<string>[];
  /** Attributes by registry name, as written: canonical forms come later. */
  attrs: Attrs;
  /** Where each attribute was written, by registry name. */
  attrSpans: Record<string, SourceSpan>;
};

const isWord = (e: Expr): e is Expr & { kind: "dim" } => {
  if (e.kind !== "dim") return false;
  const m = IDENT_RE.exec(e.text);
  return m !== null && m[0].length === e.text.length;
};

const isVariadic = (p: Param) => "role" in p && p.role === "variadic";

/** A value with no kind-specific spelling: numbers, words, lists. */
function plainValue(e: Expr, name: string): unknown {
  switch (e.kind) {
    case "number":
    case "bool":
      return e.value;
    case "dim":
      return e.text;
    case "list":
      return e.items.map((item) => plainValue(item, name));
    case "string":
      throw new DSLError(
        `${name}: quotes are for free text; write ${e.value} without them`,
        e.span,
        "DSL_INVALID_ARGUMENT"
      );
  }
}

const DTYPE_LIST = DSL_DTYPES.join(", ");

/** The written value of an attribute parameter, spelled for its kind. */
function writtenValue(e: Expr, kind: Kind, name: string): unknown {
  switch (kind.type) {
    case "text":
      if (e.kind !== "string")
        throw new DSLError(`${name}: expected ${kind.describe}, in quotes`, e.span, "DSL_INVALID_ARGUMENT");
      return e.value;
    case "dtype": {
      const d = isWord(e) ? dtypeFromWord(e.text) : null;
      if (!d) throw new DSLError(`${name}: expected one of ${DTYPE_LIST}`, e.span, "DSL_INVALID_ARGUMENT");
      return d;
    }
    case "dtypes": {
      if (e.kind !== "list")
        throw new DSLError(`${name}: expected a list of dtypes or null`, e.span, "DSL_INVALID_ARGUMENT");
      return e.items.map((item) => {
        if (isWord(item) && item.text === "null") return null;
        const d = isWord(item) ? dtypeFromWord(item.text) : null;
        if (!d)
          throw new DSLError(`${name}: expected one of ${DTYPE_LIST}, or null`, item.span, "DSL_INVALID_ARGUMENT");
        return d;
      });
    }
    case "enum":
      if (e.kind === "string")
        throw new DSLError(`${name}: write ${e.value} without quotes`, e.span, "DSL_INVALID_ARGUMENT");
      return plainValue(e, name);
    default:
      return plainValue(e, name);
  }
}

/** A tensor named by a bare word. */
function tensorRef(e: Expr, name: string): Spanned<string> {
  if (!isWord(e)) throw new DSLError(`${name}: expected a tensor name`, e.span, "DSL_EXPECTED_TENSOR");
  return { value: e.text, span: e.span };
}

/** The parameter an unknown keyword most likely meant, if any. */
function meantParameter(sig: Signature, keyword: string): string | null {
  const names = sig.params.filter((p) => !isVariadic(p)).map((p) => p.name);
  const foreign = foreignKeyword(keyword).find((candidate) => names.includes(candidate));
  return foreign ?? nearestName(keyword, names);
}

/**
 * Bind `args` to `sig`, collecting every problem rather than stopping at the
 * first. `bound` is absent when there was any.
 */
export function bindCall(
  sig: Signature,
  args: Arg[],
  statement: SourceSpan
): { bound?: BoundCall; errors: DSLError[] } {
  const errors: DSLError[] = [];
  const fail = (message: string, span: SourceSpan, code: string) =>
    errors.push(new DSLError(message, span, code));

  /** What each parameter was given: one value, or a list for a variadic one. */
  const given = new Map<Param, { values: Expr[]; span: SourceSpan }>();
  const positional = sig.params.slice(0, sig.keywordOnly);
  let next = 0;
  let sawKeyword = false;
  /** Parameters an unknown keyword was taken to mean. Reporting them missing
   * as well would count one mistake twice. */
  const meant = new Set<string>();
  let tooMany = false;

  for (const arg of args) {
    if (arg.kind === "positional") {
      if (sawKeyword) {
        fail("a positional argument cannot follow a keyword argument", arg.span, "DSL_POSITIONAL_AFTER_KEYWORD");
        continue;
      }
      const param = positional[next];
      if (!param) {
        // Reported once, at the first argument too many.
        if (!tooMany) {
          tooMany = true;
          const names = positional.filter((p) => !isVariadic(p)).map((p) => p.name);
          const byName = sig.params.slice(sig.keywordOnly).filter((p) => !isVariadic(p)).map((p) => p.name);
          const given = args.filter((a) => a.kind === "positional").length;
          fail(
            `${sig.name}() takes ${names.length} positional argument${names.length === 1 ? "" : "s"}` +
              `${names.length ? ` (${names.join(", ")})` : ""}, got ${given}` +
              (byName.length ? `; ${byName.join(", ")} ${byName.length === 1 ? "is" : "are"} written by name` : ""),
            arg.span,
            "DSL_TOO_MANY_POSITIONAL"
          );
        }
        continue;
      }
      if (isVariadic(param)) {
        const slot = given.get(param) ?? { values: [], span: arg.span };
        slot.values.push(arg.value);
        given.set(param, slot);
        continue;
      }
      given.set(param, { values: [arg.value], span: arg.span });
      next++;
      continue;
    }

    sawKeyword = true;
    const keyword = arg.name.value;
    const param = sig.params.find((p) => p.name === keyword && !isVariadic(p));
    if (!param) {
      const near = meantParameter(sig, keyword);
      if (near) meant.add(near);
      fail(
        `${sig.name}() has no parameter "${keyword}"${near ? ` (did you mean ${near}?)` : ""}; ${signatureLine(sig)}`,
        arg.name.span,
        "DSL_UNKNOWN_KEYWORD"
      );
      continue;
    }
    if (given.has(param)) {
      fail(`${sig.name}() got two values for ${keyword}`, arg.span, "DSL_DUPLICATE_ARGUMENT");
      continue;
    }
    given.set(param, { values: [arg.value], span: arg.span });
  }

  const inputs: Spanned<string>[] = [];
  const present = new Set<string>();
  const attrs: Attrs = {};
  const attrSpans: Record<string, SourceSpan> = {};

  for (const param of sig.params) {
    const slot = given.get(param);
    if ("role" in param) {
      if (!slot) {
        if ((param.role === "input" || param.role === "inputs") && !meant.has(param.name))
          fail(`${sig.name}() is missing ${param.name}: ${signatureLine(sig)}`, statement, "DSL_MISSING_ARGUMENT");
        continue;
      }
      try {
        if (param.role === "inputs") {
          const list = slot.values[0];
          if (list.kind !== "list" || !list.items.length)
            throw new DSLError(`${param.name}: expected a list of tensors, as in [A, B]`, list.span, "DSL_EXPECTED_TENSOR");
          inputs.push(...list.items.map((item) => tensorRef(item, param.name)));
        } else {
          inputs.push(...slot.values.map((value) => tensorRef(value, param.name)));
        }
        present.add(param.name);
      } catch (e) {
        if (!(e instanceof DSLError)) throw e;
        errors.push(e);
      }
      continue;
    }

    const kind = kindOf(sig, param);
    if (!slot) {
      if (!kind.default && !kind.optional && !(param.attr in sig.implied) && !meant.has(param.name))
        fail(`${sig.name}() is missing ${param.name}: ${signatureLine(sig)}`, statement, "DSL_MISSING_ARGUMENT");
      continue;
    }
    try {
      const value = writtenValue(slot.values[0], kind, param.name);
      const issue = kind.check(value);
      if (issue) throw new DSLError(`${param.name}: ${issue}`, slot.values[0].span, "DSL_INVALID_ARGUMENT");
      attrs[param.attr] = value;
      attrSpans[param.attr] = slot.span;
    } catch (e) {
      if (!(e instanceof DSLError)) throw e;
      errors.push(e);
    }
  }

  if (errors.length) return { errors };
  return {
    bound: {
      inputs,
      attrs: {
        ...sig.implied,
        ...(sig.fromInputs?.({ count: inputs.length, present }) ?? {}),
        ...attrs,
      },
      attrSpans,
    },
    errors,
  };
}
