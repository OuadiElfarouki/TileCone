import { describe, expect, it } from "vitest";
import { EXAMPLES } from "../../src/examples/index";
import { useStore } from "../../src/state/store";
import { parseWorkspaceDocument, serializeWorkspace } from "../../src/state/workspace-document";

const S = () => useStore.getState();
const named = (name: string) => EXAMPLES.find((example) => example.name === name)!;

describe("which view a built or restored workspace opens on", () => {
  it("opens an ordinary example on Dependencies after a Plan experiment", () => {
    S().applyDSL(named("Experiment: fusion and recomputation").dsl);
    expect(S().inspectorTab).toBe("plan");
    S().applyDSL(named("Reshape trap").dsl);
    expect(S().inspectorTab).toBe("dependencies");
    expect(S().selection?.parts.map((part) => part.tensorId)).toEqual(["F"]);
  });

  it("selects the operation that computes an example's starter tile", () => {
    S().applyDSL(named("Multi-head attention").dsl);
    const out = S().resolved!.tensors.Out;
    expect(S().selectedOp).toBe(out.producer!.nodeId);
  });

  it("keeps the reader's view when they build their own source", () => {
    S().applyDSL("X = Tensor(8, 8, dtype=float32)\nY = relu(X)\n");
    S().setInspectorTab("plan");
    S().applyDSL("X = Tensor(16, 8, dtype=float32)\nY = relu(X)\n");
    expect(S().inspectorTab).toBe("plan");
  });

  it("reopens a workspace saved on Dependencies on Dependencies, from the Plan view", () => {
    S().applyDSL(EXAMPLES[0].dsl);
    expect(S().inspectorTab).toBe("dependencies");
    const saved = serializeWorkspace(S());
    S().setInspectorTab("plan");
    expect(S().restoreWorkspace(parseWorkspaceDocument(saved))).toBe(true);
    expect(S().inspectorTab).toBe("dependencies");
  });

  it("reopens a workspace saved on Plan on Plan, from the Dependencies view", () => {
    S().applyDSL(EXAMPLES[0].dsl);
    S().setInspectorTab("plan");
    const saved = serializeWorkspace(S());
    S().setInspectorTab("dependencies");
    expect(S().restoreWorkspace(parseWorkspaceDocument(saved))).toBe(true);
    expect(S().inspectorTab).toBe("plan");
  });
});
