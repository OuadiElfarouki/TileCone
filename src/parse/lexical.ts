/**
 * The DSL's lexical rules, in one place.
 *
 * Two consumers need them and used to carry separate copies: the parser, which
 * turns text into a tree, and the editor highlighter, which colours text that
 * may not parse at all. The highlighter's copy was written to match the
 * parser's "exactly", which is a comment rather than a guarantee — a change to
 * what may appear in an identifier, or to how a string escape ends, would have
 * silently desynchronised the colours from the grammar.
 *
 * Both forms of every rule are derived here from one definition, so they cannot
 * disagree. The highlighter needs character-wise predicates because it scans
 * partial text; the parser needs anchored patterns because it consumes tokens.
 */

/** Characters an identifier may start with. */
const IDENT_START_CLASS = "A-Za-z_";
/**
 * Characters an identifier may continue with. `$` is reserved by composite
 * expansion for generated tensor names, and is accepted after the first
 * character so an expanded graph can be printed back as executable DSL without
 * renaming the tensors shown in the UI.
 */
const IDENT_CONTINUE_CLASS = "A-Za-z0-9_.$";

/** Anchored identifier match, for a parser consuming the next token. */
export const IDENT_RE = new RegExp(`^[${IDENT_START_CLASS}][${IDENT_CONTINUE_CLASS}]*`);

/** Character predicates, for a scanner walking text one position at a time. */
export const IDENT_START = new RegExp(`[${IDENT_START_CLASS}]`);
export const IDENT_CONTINUE = new RegExp(`[${IDENT_CONTINUE_CLASS}]`);

/**
 * Index just past the string literal starting at `start` (which must be `"`),
 * or the index at which it was found to be unterminated.
 *
 * `terminated` distinguishes the two, because the callers want different
 * things: the parser raises an error, while the highlighter is looking at a
 * draft someone is still typing and simply stops at the end of the line.
 */
export function scanStringLiteral(
  source: string,
  start: number,
  stopAtNewline = false
): { end: number; terminated: boolean } {
  let i = start + 1;
  let escaped = false;
  while (i < source.length) {
    const ch = source[i];
    if (stopAtNewline && ch === "\n") return { end: i, terminated: false };
    i++;
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"') return { end: i, terminated: true };
  }
  return { end: i, terminated: false };
}

/**
 * Remove a line comment without treating a `#` inside a string as one.
 *
 * Operates on a single physical line, which is what both callers have: the
 * parser strips per line before parsing a statement, and the highlighter needs
 * the same rule to decide where a comment run begins.
 */
export function stripComment(line: string): string {
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
