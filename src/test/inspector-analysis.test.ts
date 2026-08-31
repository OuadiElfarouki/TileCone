import { describe, expect, it } from "vitest";
import { contributionAnalysisEnabled } from "../ui/inspector-analysis";

describe("inspector analysis gating", () => {
  it("runs contribution probes only when downstream rows are visible", () => {
    expect(contributionAnalysisEnabled("both")).toBe(true);
    expect(contributionAnalysisEnabled("forward")).toBe(true);
    expect(contributionAnalysisEnabled("backward")).toBe(false);
    expect(contributionAnalysisEnabled("none")).toBe(false);
  });
});
