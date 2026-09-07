/**
 * Text -> AST, with recovery.
 *
 * The DSL is line-oriented, and that is what makes recovery cheap and honest:
 * a statement is exactly one line, so a line that fails to parse can be
 * reported, replaced by an `error` statement, and left behind. There is no
 * question of resynchronising mid-expression, and no risk of a cascade of
 * invented errors from a parser guessing its way forward.
 */

import { NUMBER_RE, readDimExpr } from "../core/shapes";
import { IDENT_RE, scanStringLiteral, stripComment } from "./lexical";
import { Arg, Expr, Program, Spanned, Stmt } from "./ast";
import { documentSpan, lineSpan, SourceSpan } from "./source";

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

/** Anchored `NUMBER_RE`: does this expression consist of one literal? */
const NUMBER_ONLY = new RegExp(`${NUMBER_RE.source}$`);

class LineParser {
  pos = 0;
  constructor(
    public src: string,
    public line: number,
    public lineOffset: number,
    public columnOffset: number
  ) {}

  /** A span covering `[from, to)` of this line. */
  spanFrom(from: number, to = this.pos): SourceSpan {
    return lineSpan(this.line, this.lineOffset, this.columnOffset + from + 1, Math.max(0, to - from));
  }
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
    const m = IDENT_RE.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return m[0];
  }
  /** An identifier plus the span it occupied, for diagnostics that name it. */
  spannedIdent(): Spanned<string> | null {
    this.ws();
    const from = this.pos;
    const value = this.ident();
    return value === null ? null : { value, span: this.spanFrom(from) };
  }
  identReq(what: string): Spanned<string> {
    const v = this.spannedIdent();
    if (v === null) throw new DSLError(`expected ${what}`, this.span());
    return v;
  }
  number(): number | null {
    this.ws();
    // Scientific notation included, so an attribute like eps=1e-5 is expressible
    // and the printer cannot emit a literal (String(1e-21) === "1e-21") that
    // this parser then rejects. Shared with dimension expressions.
    const m = NUMBER_RE.exec(this.src.slice(this.pos));
    if (!m) return null;
    this.pos += m[0].length;
    return Number(m[0]);
  }
  string(): { value: string; span: SourceSpan } | null {
    this.ws();
    if (this.src[this.pos] !== '"') return null;
    const start = this.pos;
    const { end, terminated } = scanStringLiteral(this.src, start);
    this.pos = end;
    if (!terminated) throw new DSLError("unterminated string", this.spanFrom(start));
    try {
      return { value: JSON.parse(this.src.slice(start, end)) as string, span: this.spanFrom(start) };
    } catch {
      throw new DSLError("invalid string escape", this.spanFrom(start));
    }
  }

  expr(): Expr {
    this.ws();
    const from = this.pos;
    const s = this.string();
    if (s !== null) return { kind: "string", value: s.value, span: s.span };
    if (this.eat("[")) {
      const items: Expr[] = [];
      if (!this.eat("]")) {
        do items.push(this.expr());
        while (this.eat(","));
        this.expect("]");
      }
      return { kind: "list", items, span: this.spanFrom(from) };
    }
    // Booleans are identifiers but are values, not one-symbol dimensions.
    const save = this.pos;
    const word = this.ident();
    if (word === "true") return { kind: "bool", value: true, span: this.spanFrom(save) };
    if (word === "false") return { kind: "bool", value: false, span: this.spanFrom(save) };
    this.pos = save;
    // Everything else is read as a dimension expression, so a shape attribute
    // can say `shape=[B, S, H, E/H]` in the same language a declaration uses.
    // A lone literal is still a number; anything else keeps its written form.
    const parsed = readDimExpr(this.src, this.pos);
    if (!parsed) this.error("expected value");
    const text = this.src.slice(this.pos, parsed.end).trim();
    this.pos = parsed.end;
    const span = this.spanFrom(save);
    return NUMBER_ONLY.test(text)
      ? { kind: "number", value: Number(text), span }
      : { kind: "dim", text, span };
  }

  /** `name=value` when an `=` follows the identifier, otherwise a bare value. */
  arg(): Arg {
    this.ws();
    const from = this.pos;
    const save = this.pos;
    const name = this.spannedIdent();
    if (name && this.eat("=")) {
      const value = this.expr();
      return { kind: "named", name, value, span: this.spanFrom(from) };
    }
    this.pos = save;
    return { kind: "positional", value: this.expr(), span: this.spanFrom(from) };
  }

  /** Arguments up to and including the closing paren. */
  argList(): Arg[] {
    const args: Arg[] = [];
    if (this.eat(")")) return args;
    do args.push(this.arg());
    while (this.eat(","));
    this.expect(")");
    return args;
  }

  atEnd(): boolean {
    this.ws();
    return this.pos >= this.src.length;
  }
}

export { stripComment } from "./lexical";

export type ParseResult = { program: Program; errors: DSLError[] };

/**
 * Parse one already-comment-stripped statement. Throws `DSLError` on failure.
 *
 * Output names are appended to `outs` as they are read rather than returned, so
 * a caller recovering from a failure later in the line still knows which names
 * this statement was binding.
 */
function parseStatement(p: LineParser, statementSpan: SourceSpan, outs: Spanned<string>[]): Stmt {
  outs.push(p.identReq("statement"));
  while (p.eat(",")) outs.push(p.identReq("output name"));
  for (const out of outs)
    if (out.value === "Tensor" || out.value === "Parameter")
      throw new DSLError(
        `"${out.value}" is a constructor and cannot name a tensor`,
        out.span,
        "DSL_RESERVED_NAME"
      );
  if (!p.eat("="))
    throw new DSLError(
      "expected an assignment: NAME = number, Tensor(...), Parameter(...), or op(...)",
      statementSpan,
      "DSL_UNKNOWN_STATEMENT"
    );

  const rhsStart = p.pos;
  if (outs.length === 1) {
    const from = p.pos;
    const scalar = p.number();
    if (scalar !== null && p.atEnd())
      return {
        kind: "param",
        name: outs[0],
        value: { value: scalar, span: p.spanFrom(from) },
        span: statementSpan,
      };
    p.pos = rhsStart;
  }

  const callee = p.identReq("op name");
  p.expect("(");
  const args = p.argList();
  if (!p.atEnd()) p.error("trailing input");

  if (callee.value === "Tensor" || callee.value === "Parameter") {
    if (outs.length !== 1)
      throw new DSLError(`${callee.value} declares exactly one tensor`, statementSpan);
    return {
      kind: "declare",
      name: outs[0],
      ctor: { value: callee.value, span: callee.span },
      args,
      span: statementSpan,
    };
  }
  return { kind: "call", outs, callee, args, span: statementSpan };
}

/**
 * Parse a whole document, collecting one error per unparseable line.
 *
 * A failing line yields an `error` statement rather than being dropped, so
 * later phases can see that something was written there and stay quiet about
 * names it would have bound.
 */
export function parseProgram(text: string): ParseResult {
  const stmts: Stmt[] = [];
  const errors: DSLError[] = [];
  const lines = text.split("\n");
  let lineOffset = 0;

  for (let ln = 0; ln < lines.length; ln++) {
    const physicalLine = lines[ln];
    const code = stripComment(physicalLine);
    const raw = code.trim();
    const columnOffset = raw ? code.indexOf(raw) : 0;
    const statementSpan = lineSpan(ln + 1, lineOffset, columnOffset + 1, raw.length);
    const advance = () => {
      lineOffset += physicalLine.length + (ln < lines.length - 1 ? 1 : 0);
    };
    if (!raw) {
      advance();
      continue;
    }
    const p = new LineParser(raw, ln + 1, lineOffset, columnOffset);
    const outs: Spanned<string>[] = [];
    try {
      stmts.push(parseStatement(p, statementSpan, outs));
    } catch (e) {
      if (!(e instanceof DSLError)) throw e;
      errors.push(e);
      stmts.push({ kind: "error", outs, span: statementSpan });
    }
    advance();
  }
  return { program: { stmts, span: documentSpan(text) }, errors };
}
