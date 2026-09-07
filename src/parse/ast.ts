/**
 * The DSL's syntax tree.
 *
 * The parser used to build graph nodes directly from text, which cost three
 * things this layer exists to give back:
 *
 * - **Recovery.** A tree can hold an `error` statement, so one bad line no
 *   longer ends the compile. Every phase after this one collects rather than
 *   throws, and the editor gets the whole list.
 * - **Spans on the parts.** A diagnostic about the third argument can underline
 *   the third argument. Previously every semantic error underlined the whole
 *   statement, because a statement span was the only span there was.
 * - **One description of the syntax.** Desugaring (`relu` -> `elementwise`) now
 *   happens in lowering, over a tree, instead of inline in the parser where the
 *   printer had to independently mirror it.
 *
 * The tree deliberately stops at syntax. It does not know that `relu` is an
 * elementwise function or that `axes` takes a list: those are questions about
 * the operation registry, and answering them here would put op knowledge in the
 * parser again.
 */

import { SourceSpan } from "./source";

export type Spanned<T> = { value: T; span: SourceSpan };

/**
 * A value written in argument position.
 *
 * `dim` is the general case and carries the author's own text: a bare symbol
 * (`H`), arithmetic over symbols (`E/H`), or a word that later turns out to be
 * a dtype or an enum member (`reflect`). The parser cannot tell those apart
 * without knowing the operation, so it keeps the text and lets lowering decide.
 */
export type Expr =
  | { kind: "number"; value: number; span: SourceSpan }
  | { kind: "string"; value: string; span: SourceSpan }
  | { kind: "bool"; value: boolean; span: SourceSpan }
  | { kind: "dim"; text: string; span: SourceSpan }
  | { kind: "list"; items: Expr[]; span: SourceSpan };

export type Arg =
  | { kind: "positional"; value: Expr; span: SourceSpan }
  | { kind: "named"; name: Spanned<string>; value: Expr; span: SourceSpan };

export type Stmt =
  /** `M = 512` */
  | { kind: "param"; name: Spanned<string>; value: Spanned<number>; span: SourceSpan }
  /** `A = Tensor(M, K, dtype=fp16)` / `W = Parameter(...)` */
  | {
      kind: "declare";
      name: Spanned<string>;
      ctor: Spanned<"Tensor" | "Parameter">;
      args: Arg[];
      span: SourceSpan;
    }
  /** `C = matmul(A, B)` / `Y0, Y1 = split(X, ...)` */
  | {
      kind: "call";
      outs: Spanned<string>[];
      callee: Spanned<string>;
      args: Arg[];
      span: SourceSpan;
    }
  /**
   * A line that could not be parsed.
   *
   * `outs` holds whatever names the parser had already read before it failed.
   * They matter: `X = Tensor(4, 4` binds `X` as far as the author is concerned,
   * and without recording that, every later line reading `X` reports a second,
   * invented error about a name the author did write.
   */
  | { kind: "error"; outs: Spanned<string>[]; span: SourceSpan };

export type Program = { stmts: Stmt[]; span: SourceSpan };
