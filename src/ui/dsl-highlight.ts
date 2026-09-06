import { DTYPES } from "../core/dtypes";

export type DSLHighlightKind = "plain" | "comment" | "keyword";
export type DSLHighlightToken = { kind: DSLHighlightKind; text: string };

const RESERVED = new Set<string>([
  "params",
  "input",
  "weight",
  "param",
  "true",
  "false",
  ...DTYPES,
]);

const IDENT_START = /[A-Za-z_]/;
const IDENT_CONTINUE = /[A-Za-z0-9_.$]/;

/**
 * Tokenize only the syntax whose meaning is stable across operation plugins:
 * declarations, scalar literals, dtypes, and any identifier used as a call.
 * Strings remain opaque, including `#` in an einsum equation; outside a
 * string, `#` owns the rest of its physical line exactly as the DSL parser does.
 */
export function highlightDSL(source: string): DSLHighlightToken[] {
  const tokens: DSLHighlightToken[] = [];
  const push = (kind: DSLHighlightKind, text: string) => {
    if (!text) return;
    const previous = tokens[tokens.length - 1];
    if (previous?.kind === kind) previous.text += text;
    else tokens.push({ kind, text });
  };

  let plainStart = 0;
  let i = 0;
  while (i < source.length) {
    if (source[i] === '"') {
      i++;
      let escaped = false;
      while (i < source.length) {
        const ch = source[i++];
        // The parser handles physical lines independently. An unterminated
        // string is an error on this line, but it must not swallow highlighting
        // on every line below it while the author fixes the draft.
        if (ch === "\n") break;
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') break;
      }
      continue;
    }

    if (source[i] === "#") {
      push("plain", source.slice(plainStart, i));
      const end = source.indexOf("\n", i);
      const commentEnd = end < 0 ? source.length : end;
      push("comment", source.slice(i, commentEnd));
      i = commentEnd;
      plainStart = i;
      continue;
    }

    if (IDENT_START.test(source[i])) {
      const start = i++;
      while (i < source.length && IDENT_CONTINUE.test(source[i])) i++;
      const word = source.slice(start, i);
      let next = i;
      while (next < source.length && /\s/.test(source[next])) next++;
      if (RESERVED.has(word) || source[next] === "(") {
        push("plain", source.slice(plainStart, start));
        push("keyword", word);
        plainStart = i;
      }
      continue;
    }

    i++;
  }
  push("plain", source.slice(plainStart));
  return tokens;
}
