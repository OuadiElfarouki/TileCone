import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { useStore } from "../../../src/state/store";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const CHAIN = "A = Tensor(32, 32, dtype=fp16)\nB = Tensor(32, 32, dtype=fp16)\n" +
  "W = Tensor(32, 16, dtype=fp16)\nC = matmul(A, B)\nY = matmul(C, W)\n";
const S = () => useStore.getState();
const render = () => renderToStaticMarkup(createElement(Inspector));

describe("operand reuse in the Plan view", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTileAt("Y", [8, 4], [8, 4]);
    S().setPlanTile("C", [8, 8]);
  });

  it("counts what the inspected task computes, under its own heading", () => {
    const html = render();
    expect(html).toContain("Operand reuse in this task");
    // An 8 x 4 tile of Y reads a row of C with its 4 columns and a column of W with its 8 rows.
    expect(html).toContain("via matmul → Y");
    expect(html).toContain("each read 4×");
    expect(html).toContain("each read 8×");
    expect(html).not.toContain("via matmul → C");
  });

  it("shows the fused operand's own reuse once the tasks compute it", () => {
    S().setPlanTile("C", null);
    const html = render();
    expect(html).toContain("via matmul → C");
    // The task computes C's 8-row band: each of those rows of A is read by all 32 columns.
    expect(html).toContain("each read 32×");
  });
});

describe("producer paging in the Plan view", () => {
  it("shows a bounded first page with the full producer count", () => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTile("C", [1, 1]);
    S().setPlanTileAt("Y", [1, 1], [0, 0]);
    const html = render();
    expect(html).toContain("needs 32 of 1024 C tasks");
    expect(html).toContain("1–8 of 32");
    expect(html).toContain("previous");
    expect(html).toContain("next");
    expect(html).not.toContain("show all");
  });
});
