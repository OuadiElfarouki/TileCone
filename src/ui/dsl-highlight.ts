import { DSL_DTYPES } from "../parse/dsl";
import { IDENT_CONTINUE, IDENT_START, scanStringLiteral } from "../parse/lexical";

export type DSLHighlightKind = "plain" | "comment" | "keyword";
export type DSLHighlightToken = { kind: DSLHighlightKind; text: string };

const RESERVED = new Set<string>([
  "true",
  "false",
  ...DSL_DTYPES,
]);

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
      // The parser handles physical lines independently. An unterminated
      // string is an error on this line, but it must not swallow highlighting
      // on every line below it while the author fixes the draft, so the scan
      // stops at the newline.
      i = scanStringLiteral(source, i, true).end;
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

/**
 * The same tokens, padded for the highlight layer that sits under the textarea.
 *
 * A textarea keeps an empty last line after a trailing newline; a `pre` drops
 * it, so the two layers disagree about how tall the source is and the overlay
 * ends one line short of the caret as soon as the box scrolls. One appended
 * newline restores that line. It is unconditional: when the source does not end
 * in a newline the extra one is itself the dropped last line, so the padded
 * text always renders exactly the lines the textarea does.
 */
export function overlayTokens(source: string): DSLHighlightToken[] {
  const tokens = highlightDSL(source);
  const last = tokens[tokens.length - 1];
  if (last?.kind === "plain") return [...tokens.slice(0, -1), { ...last, text: last.text + "\n" }];
  return [...tokens, { kind: "plain", text: "\n" }];
}
