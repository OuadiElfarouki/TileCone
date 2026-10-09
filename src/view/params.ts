/**
 * Size parameters as the source panel's controls see them.
 *
 * The controls show the built program's values and change only on a run. An
 * edit in a control rewrites the number in that parameter's `M = 256` line of
 * the draft, found by name, which is exactly what typing the new value would
 * do; Run builds it like any other edit.
 */

import { parseProgram } from "../parse/parser";

export type SourceParam = {
  name: string;
  value: number;
  /** Offsets of the number literal in the draft, half-open. */
  start: number;
  end: number;
  /** Offset of the end of the parameter's physical line. */
  lineEnd: number;
};

/** The draft's size parameters that a slider can set: positive integers. */
export function sourceParams(text: string): SourceParam[] {
  const params: SourceParam[] = [];
  for (const stmt of parseProgram(text).program.stmts) {
    if (stmt.kind !== "param") continue;
    const { value, span } = stmt.value;
    if (!Number.isSafeInteger(value) || value < 1) continue;
    // The parser's span for the value starts after `=` and so may hold the
    // whitespace before the number; the rewrite has to replace the number only.
    const written = text.slice(span.start.offset, span.end.offset);
    const start = span.start.offset + (written.length - written.trimStart().length);
    const end = span.end.offset - (written.length - written.trimEnd().length);
    const newline = text.indexOf("\n", end);
    params.push({
      name: stmt.name.value,
      value,
      start,
      end,
      lineEnd: newline < 0 ? text.length : newline,
    });
  }
  return params;
}

/** The draft with one parameter's number replaced. */
export function withParamValue(text: string, param: SourceParam, value: number): string {
  return text.slice(0, param.start) + String(value) + text.slice(param.end);
}

/** The draft with the parameter called `name` set to `value`, or `null` when
 * the draft no longer has a parameter of that name to rewrite. */
export function withParamNamed(text: string, name: string, value: number): string | null {
  const param = sourceParams(text).find((p) => p.name === name);
  return param ? withParamValue(text, param, value) : null;
}

/**
 * How a parameter's stepper moves: a power of two doubles and halves, any other
 * even value moves by `STRIDE`, and an odd value by one.
 *
 * The mode is read from the value when a run of steps begins and held until the
 * run ends, so stepping an odd 3 upward passes through 4 and 8 one at a time
 * instead of switching to doubling there. Stopping and starting again reads it
 * afresh from wherever the value was left.
 */
export type StepMode = "double" | "stride" | "unit";

export const STRIDE = 32;

const isPowerOfTwo = (value: number): boolean => {
  if (!Number.isSafeInteger(value) || value < 1) return false;
  // log2 rounds nearby large integers to the same result as a power of two.
  const integer = BigInt(value);
  return (integer & (integer - 1n)) === 0n;
};

export function stepModeOf(value: number): StepMode {
  if (isPowerOfTwo(value)) return "double";
  return value % 2 === 0 ? "stride" : "unit";
}

/** One step from `value` in `direction`, never below 1. A step that would go
 * below 1 or outside the safe integer range leaves the value where it is. */
export function stepValue(value: number, mode: StepMode, direction: 1 | -1): number {
  let next: number;
  if (mode === "double") {
    // The nearest power of two strictly beyond the value, so a value that is
    // not one (typed mid-run) still lands on the lattice.
    next = 1;
    if (direction > 0) while (next <= value) next *= 2;
    else {
      while (next * 2 < value) next *= 2;
      if (next >= value) return value;
    }
  } else {
    next = value + direction * (mode === "stride" ? STRIDE : 1);
  }
  return Number.isSafeInteger(next) && next >= 1 ? next : value;
}

/** How a step mode reads beside its stepper. */
export const STEP_LABEL: Record<StepMode, string> = {
  double: "×2",
  stride: `±${STRIDE}`,
  unit: "±1",
};
