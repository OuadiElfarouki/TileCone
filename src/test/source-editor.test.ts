/// <reference types="vite/client" />
import { describe, expect, it } from "vitest";
import styles from "../styles.css?raw";
import { EXAMPLES } from "../examples";
import { highlightDSL, overlayTokens } from "../ui/dsl-highlight";

const overlayText = (source: string) => overlayTokens(source).map((t) => t.text).join("");

/** What the `pre` actually paints: a forced break at the end leaves no line. */
const renderedLines = (preContent: string) => {
  const lines = preContent.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
};

const SOURCES: [string, string][] = [
  ["empty", ""],
  ["one line, no newline", "A = 1"],
  ["one line, trailing newline", "A = 1\n"],
  ["blank line in the middle", "A = Tensor(M,K, dtype=fp16)\n\nB = relu(A)\n"],
  ["blank lines at the end", "A = 1\n\n\n"],
  ["only newlines", "\n\n\n"],
  ["trailing spaces", "A = 1   \n   \nB = 2\n"],
  ["ends in a comment", "A = 1 # note"],
  ["ends in a comment and a newline", "A = 1 # note\n"],
  ["ends inside a string", 'Y = einsum("unfinished'],
  ["carriage returns", "A = 1\r\nB = 2\r\n"],
  ...EXAMPLES.map((ex) => [`example ${ex.name}`, ex.dsl] as [string, string]),
];

describe("source editor overlay", () => {
  /* The caret lives in the textarea and the ink lives in the `pre` beneath it.
     Every test here is one way those two can stop describing the same text. */
  it.each(SOURCES)("mirrors %s byte for byte", (_name, source) => {
    expect(overlayText(source)).toBe(`${source}\n`);
  });

  it.each(SOURCES)("paints one line per textarea line for %s", (_name, source) => {
    expect(renderedLines(overlayText(source))).toEqual(source.split("\n"));
  });

  it.each(SOURCES)("keeps the tokens %s was highlighted with", (_name, source) => {
    const kinds = (tokens: { kind: string; text: string }[]) =>
      tokens.filter((t) => t.text.trim()).map((t) => `${t.kind}:${t.text.trim()}`);
    expect(kinds(overlayTokens(source))).toEqual(kinds(highlightDSL(source)));
  });

  it("pads with plain text rather than extending a comment", () => {
    const tokens = overlayTokens("A = 1 # note");
    expect(tokens[tokens.length - 1]).toEqual({ kind: "plain", text: "\n" });
  });

  it("adds exactly one line, never two", () => {
    expect(overlayText("A = 1\n")).toBe("A = 1\n\n");
    expect(renderedLines(overlayText("A = 1\n"))).toEqual(["A = 1", ""]);
  });
});

/* The two layers only stay aligned while they agree on every property that can
   move a line break or a line box. CSS cannot express "these two elements are
   the same box", so the contract is that such properties are declared once, in
   the rule that selects both, and nowhere else. */
describe("source editor layer geometry", () => {
  const css = styles.replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, selector, body]) => ({
    selector: selector.trim(),
    props: new Set(
      body
        .split(";")
        .map((d: string) => d.split(":")[0].trim())
        .filter(Boolean)
    ),
  }));

  const shared = rules.find(
    (r) => /(^|,)\s*\.source-highlight\s*(,|$)/.test(r.selector) && /textarea\.source\s*$/.test(r.selector)
  );

  it("declares the alignment-critical properties for both layers at once", () => {
    expect(shared).toBeDefined();
    for (const prop of [
      "width",
      "height",
      "margin",
      "font",
      "tab-size",
      "white-space",
      "overflow-wrap",
      "border",
      "padding",
      // Without a reserved gutter the textarea's own scrollbar narrows its wrap
      // width the moment the source outgrows the box, and every wrapped line
      // below that point lands a row away from its caret.
      "scrollbar-gutter",
    ])
      expect(shared!.props).toContain(prop);
  });

  it("lets neither layer set that geometry alone", () => {
    const forbidden = [
      "width", "height", "margin", "font", "font-size", "font-family", "line-height",
      "letter-spacing", "word-spacing", "tab-size", "white-space", "overflow-wrap",
      "word-break", "text-indent", "box-sizing", "border", "border-width", "border-style",
      "padding", "padding-top", "padding-right", "padding-bottom", "padding-left",
      "scrollbar-gutter", "scrollbar-width",
    ];
    const offenders = rules
      .filter((rule) => rule !== shared && /^(\.source-highlight|textarea\.source)[.:\w-]*$/.test(rule.selector))
      .flatMap((rule) => forbidden.filter((p) => rule.props.has(p)).map((p) => `${rule.selector} { ${p} }`));
    expect(offenders).toEqual([]);
  });
});
