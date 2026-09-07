/**
 * The DSL's sugar table, checked in both directions off the table itself.
 *
 * Lowering and printing used to hold separate, hardcoded descriptions of which
 * call name means which operation, and they had drifted: `amax`/`amin` lowered
 * but printed back as `reduce(fn=max, ...)`. Driving both from `SUGAR_FORMS`
 * makes that class of asymmetry impossible, and driving the test from the same
 * table means a form added later is covered without anyone remembering.
 */

import { describe, expect, it } from "vitest";
import { parseDSL, toDSL } from "../parse/dsl";
import { SUGAR_FORMS, sugarForCall, sugarForNode } from "../parse/sugar";
import { tryCompileDSL } from "../parse/compiler";
import { ELEMENTWISE_FNS } from "../core/ops/elementwise";

/** A minimal, valid program exercising one sugared call. */
function programFor(form: (typeof SUGAR_FORMS)[number]): string {
  if (form.op === "reduce") return `X = Tensor(4, 5)\nZ = ${form.call}(X, axis=1)\n`;
  if (form.op === "normalize") return `X = Tensor(4, 5)\nW = Tensor(5)\nZ = ${form.call}(X, W)\n`;
  // The function's own arity lives with the operation, not with the sugar:
  // the sugar table says which op a call means, the op table says how many
  // operands that call takes. Asking the right one of the two is the point.
  const binary = ELEMENTWISE_FNS[form.call].minInputs >= 2;
  return binary
    ? `X = Tensor(4, 5)\nY = Tensor(4, 5)\nZ = ${form.call}(X, Y)\n`
    : `X = Tensor(4, 5)\nZ = ${form.call}(X)\n`;
}

describe("sugar table drives both directions", () => {
  it.each(SUGAR_FORMS.map((f) => [f.call, f] as const))(
    "%s survives a round trip as itself",
    (call, form) => {
      const source = programFor(form);
      const printed = toDSL(parseDSL(source));
      // The call name is preserved, not replaced by the underlying op.
      expect(printed).toContain(`${call}(`);
      expect(printed).not.toContain(`${form.op}(`);
      // And printing is a fixpoint: re-reading it yields the same text.
      expect(toDSL(parseDSL(printed))).toBe(printed);
    }
  );

  it("amax and amin survive, which is what the split tables got wrong", () => {
    const printed = toDSL(parseDSL("X = Tensor(4, 5)\nZ = amax(X, axis=1)\n"));
    expect(printed).toContain("amax(");
    expect(printed).not.toContain("fn=max");
  });

  it("an op with no matching sugar prints in the general form", () => {
    // `logsumexp` is a real reduce with no call-name spelling of its own.
    const printed = toDSL(
      parseDSL("X = Tensor(4, 5)\nZ = reduce(X, fn=logsumexp, axes=[1], keepdim=false)\n")
    );
    expect(printed).toContain("reduce(");
    expect(printed).toContain("fn=logsumexp");
  });

  it("keeps attributes the call name does not carry", () => {
    const printed = toDSL(parseDSL("X = Tensor(4, 5)\nZ = sum(X, axes=[1], keepdim=true)\n"));
    // `fn` is implied by writing `sum`; `keepdim` is not, and dropping it would
    // silently change the shape on recompile.
    expect(printed).toContain("keepdim=true");
    expect(printed).not.toContain("fn=");
  });

  it("einsum keeps its equation leading and unnamed", () => {
    const printed = toDSL(
      parseDSL('A = Tensor(2, 3)\nB = Tensor(3, 4)\nC = einsum("ij,jk->ik", A, B)\n')
    );
    expect(printed).toContain('einsum("ij,jk->ik", A, B)');
  });

  it("every sugared form actually compiles", () => {
    for (const form of SUGAR_FORMS) {
      const result = tryCompileDSL(programFor(form));
      expect(result.ok, `${form.call}: ${result.ok ? "" : result.diagnostics[0].message}`).toBe(
        true
      );
    }
  });
});

describe("sugar lookup", () => {
  it("resolves a call name to its form and back from a node", () => {
    const form = sugarForCall("layernorm");
    expect(form?.op).toBe("normalize");
    expect(sugarForNode("normalize", { kind: "layernorm" })?.call).toBe("layernorm");
  });

  it("returns nothing for an unsugared op", () => {
    expect(sugarForCall("conv")).toBeUndefined();
    expect(sugarForNode("conv", {})).toBeUndefined();
  });

  it("has no two forms claiming the same call name", () => {
    const calls = SUGAR_FORMS.map((f) => f.call);
    expect(new Set(calls).size).toBe(calls.length);
  });

  it("has no two forms of one op matching the same attributes", () => {
    // Otherwise `sugarForNode` would pick whichever came first in the table and
    // the printer's choice would depend on declaration order.
    for (const form of SUGAR_FORMS) {
      const claimants = SUGAR_FORMS.filter(
        (other) => other.op === form.op && other.matches(form.attrsFor(2))
      );
      expect(claimants.map((c) => c.call)).toEqual([form.call]);
    }
  });
});
