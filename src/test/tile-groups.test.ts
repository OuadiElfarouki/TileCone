import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Inspector } from "../ui/Inspector";
import { compileDSL } from "../parse/compiler";
import {
  groupPropResult,
  analysisTarget,
  MAX_PER_BOX_PROPS,
  partsOn,
  useStore,
  type BoxProp,
  type SelPart,
} from "../ui/store";
import { executeQuery } from "../core/executor";
import { box, count, fromBox } from "../core/region";
import type { ResolvedGraph } from "../core/graph";
import { analysisTensorId, groupAttribution, groupFocus, measuredParts, measuredElements } from "../ui/inspector-analysis";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../ui/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ui/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

/** Two tensors where one feeds the other, so their cones genuinely overlap. */
const chain = () =>
  compileDSL(`A = Tensor(64, 64, dtype=fp32)
B = Tensor(64, 64, dtype=fp32)
C = matmul(A, B)
D = relu(C)
`).resolved;

const propsFor = (resolved: ResolvedGraph, parts: SelPart[]): BoxProp[] =>
  parts.map((part) => {
    const r = executeQuery(resolved, {
      tensorId: part.tensorId,
      region: fromBox(part.box),
      direction: "both",
    });
    return { backward: r.backward, forward: r.forward };
  });

const byTensorOf = (resolved: ResolvedGraph, parts: SelPart[]) => {
  const out: Record<string, { backward: any; forward: any }> = {};
  for (const part of parts) {
    const r = executeQuery(resolved, {
      tensorId: part.tensorId,
      region: fromBox(part.box),
      direction: "both",
    });
    out[part.tensorId] = { backward: r.backward, forward: r.forward };
  }
  return out;
};

describe("tile groups", () => {
  it("counts the enabled union in the header, including the cap fallback", () => {
    const tiles: SelPart[] = [
      { tensorId: "C", box: box([0, 4]) },
      { tensorId: "C", box: box([2, 6]) },
      { tensorId: "D", box: box([0, 20]) },
    ];
    expect(measuredElements(measuredParts(tiles, "C", new Set(), null, true))).toBe(6);
    expect(measuredElements(measuredParts(tiles, "C", new Set([1]), null, true))).toBe(4);
    expect(measuredElements(measuredParts(tiles, "C", new Set([0, 1]), null, true))).toBe(0);
    expect(measuredElements(measuredParts(tiles, "C", new Set([1]), null, false))).toBe(6);
  });
  const parts: SelPart[] = [
    { tensorId: "C", box: box([0, 16], [0, 16]) },
    { tensorId: "D", box: box([32, 48], [32, 48]) },
  ];

  it("scopes to the selected group, falling back to the anchor tensor", () => {
    const tiles = [parts[0], parts[0], parts[1]];
    expect(analysisTensorId(tiles, null)).toBe("D");
    expect(analysisTensorId(tiles, "C")).toBe("C");
    // A group nothing is drawn on cannot be analysed, so the anchor stands in.
    expect(analysisTensorId([parts[1]], "C")).toBe("D");
  });

  it("lets focus narrow within the group and never re-scope it", () => {
    const tiles = [parts[0], parts[0], parts[1]];
    // Hovering D's row while C is the group leaves the group alone: a preview
    // must not change what the panel below is about.
    expect(analysisTensorId(tiles, "C")).toBe("C");
    expect(groupFocus(tiles, 2, "C")).toBeNull();
    // Within the group it narrows to that one tile.
    expect(groupFocus(tiles, 1, "C")).toBe(1);
    expect(groupFocus(tiles, null, "C")).toBeNull();
  });

  it("excludes other groups from bars without renumbering their colors", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);
    const scoped = groupAttribution(perBox, parts, "D")!;
    expect(scoped).toHaveLength(2);
    expect(scoped[0]).toEqual({ backward: null, forward: null });
    expect(scoped[1]).toBe(perBox[1]);
    expect(groupAttribution(null, parts, "D")).toBeNull();
  });

  it("analyses one tensor's tiles and never the other's", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);

    const onC = groupPropResult(null, perBox, parts, new Set(), null, "C", "backward")!;
    const onD = groupPropResult(null, perBox, parts, new Set(), null, "D", "backward")!;

    // C's cone stops at A and B. D's reaches through C to A and B as well, but
    // over the region *its* tile needs - a different one.
    expect(count(onC.tensors.get("A")!.region)).toBe(16 * 64);
    expect(onC.tensors.has("D")).toBe(false);
    expect(onD.tensors.has("C")).toBe(true);
    // Same size band, different rows of A: C's tile needs rows 0:16, D's needs
    // 32:48. Merging them would claim a job that reads both, which is nobody's
    // kernel - and would charge the shared work through C only once.
    const rowsOf = (region: { boxes: { lo: number; hi: number }[][] }) =>
      region.boxes.map((b) => `${b[0].lo}:${b[0].hi}`);
    expect(rowsOf(onC.tensors.get("A")!.region)).toEqual(["0:16"]);
    expect(rowsOf(onD.tensors.get("A")!.region)).toEqual(["32:48"]);
  });

  it("honours hidden tiles within a group", () => {
    const resolved = chain();
    const twoOnC: SelPart[] = [
      { tensorId: "C", box: box([0, 16], [0, 16]) },
      { tensorId: "C", box: box([32, 48], [0, 16]) },
      { tensorId: "D", box: box([0, 8], [0, 8]) },
    ];
    const perBox = propsFor(resolved, twoOnC);

    const both = groupPropResult(null, perBox, twoOnC, new Set(), null, "C", "backward")!;
    const one = groupPropResult(null, perBox, twoOnC, new Set([1]), null, "C", "backward")!;

    expect(count(both.tensors.get("A")!.region)).toBe(2 * 16 * 64);
    expect(count(one.tensors.get("A")!.region)).toBe(16 * 64);
  });

  it("ignores a focused tile that belongs to another group", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);

    // Focus is on D (index 1) while the C group is being read: the C answer
    // must be C's tiles, not D's.
    const onC = groupPropResult(null, perBox, parts, new Set(), 1, "C", "backward")!;
    expect(onC.tensors.has("D")).toBe(false);
    expect(count(onC.tensors.get("A")!.region)).toBe(16 * 64);
  });

  it("stays grouped past the attribution cap, where per-tile cones are gone", () => {
    const resolved = chain();
    const byTensor = byTensorOf(resolved, parts);

    // `perBox` is null above MAX_PER_BOX_PROPS. The per-tensor query stands in,
    // so the analysis is coarser but still never mixes two tensors.
    expect(MAX_PER_BOX_PROPS).toBeGreaterThan(0);
    const onC = groupPropResult(byTensor, null, parts, new Set(), null, "C", "backward")!;
    expect(onC.tensors.has("D")).toBe(false);
    expect(groupPropResult(byTensor, null, parts, new Set(), null, "D", "backward")!.tensors.has("C"))
      .toBe(true);
  });

  it("has no answer without an active tensor", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);
    expect(groupPropResult(null, perBox, parts, new Set(), null, null, "backward")).toBeNull();
  });
});

/**
 * Scope is named by a draw, a pin, or the group header, and by nothing else.
 * Each of these was a way to strand the reader in a group they had left.
 */
describe("moving between tile groups", () => {
  const SRC = `A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
CC = matmul(A, B)
W = Tensor(256, 256, dtype=fp16)
D = matmul(CC, W)
`;
  const S = () => useStore.getState();
  /** What the inspector computes each render. */
  const active = () => analysisTensorId(S().selection?.parts ?? [], S().analysisGroup);
  const render = () => renderToStaticMarkup(createElement(Inspector));

  beforeEach(() => {
    S().setInspectorTab("dependencies");
    S().applyDSL(SRC);
    S().setSelection("D", fromBox(box([160, 224], [48, 80])));
    S().setSelection("D", fromBox(box([16, 80], [128, 160])), "union");
  });

  it("follows a tile drawn on another tensor, dropping a pin left behind", () => {
    S().togglePinBox(1);
    expect(active()).toBe("D");

    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    // The pin was on D. Keeping it would leave the panel describing the tile
    // the reader just left, with the one they drew dimmed in another group.
    expect(S().pinnedBox).toBeNull();
    expect(active()).toBe("CC");
  });

  it("releases the pin when another tile is drawn on the same tensor", () => {
    S().togglePinBox(1);
    S().setSelection("D", fromBox(box([200, 240], [8, 40])), "union");
    // The reader just added a tile to this group; staying pinned to the old one
    // would hide what they did. The group is unchanged, and now has three tiles.
    expect(S().pinnedBox).toBeNull();
    expect(active()).toBe("D");
    expect(partsOn(S().selection, "D")).toHaveLength(3);
  });

  it("returns to the group the last deliberate act named, not to a stale one", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");
    expect(active()).toBe("D");

    // Drawing on CC again renames the group, so Escape lands on CC.
    S().setSelection("CC", fromBox(box([0, 32], [0, 32])), "union");
    S().clearFocus();
    expect(active()).toBe("CC");
  });

  it("does not let a hover over another group's row re-scope the panel", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");

    // The header sits inside the hovered list, so the pointer crosses other
    // rows to reach it. Those crossings must not undo the click that got there.
    S().hoverBox(2);
    expect(active()).toBe("D");
    expect(groupFocus(S().selection!.parts, S().focusedBox, active())).toBeNull();
    S().hoverBox(null);
    expect(active()).toBe("D");
  });

  it("releases the pin when a group is chosen, so the new group can be hovered", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().togglePinBox(2);
    expect(S().pinnedBox).toBe(2);
    S().selectAnalysisGroup("D");
    expect(S().pinnedBox).toBeNull();

    // A pin outranks hovering, so leaving it set would freeze the group just chosen.
    S().hoverBox(0);
    expect(S().focusedBox).toBe(0);
  });

  it("switches group when a tile in another one is clicked", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");
    S().togglePinBox(2);
    expect(active()).toBe("CC");
    expect(groupFocus(S().selection!.parts, S().focusedBox, active())).toBe(2);
  });

  it("retires a group once nothing is drawn on it", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    expect(active()).toBe("CC");
    S().deleteBox(2);
    expect(S().analysisGroup).toBeNull();
    expect(active()).toBe("D");
  });

  it("moves the selected group, ignoring hover on another tensor", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");
    S().hoverBox(2);
    const before = S().selection!.parts;
    expect(analysisTarget(before, S().analysisGroup, S().focusedBox))
      .toEqual({ tensorId: "D", focusedBox: null, index: 1 });
    S().moveSelection(0, 1);
    expect(S().selection!.parts[0].box[0].lo).toBe(before[0].box[0].lo + 1);
    expect(S().selection!.parts[1].box[0].lo).toBe(before[1].box[0].lo + 1);
    expect(S().selection!.parts[2]).toBe(before[2]);
  });

  it("renders ranges only in tile rows, including a single-tile workspace", () => {
    S().setSelection("D", fromBox(box([0, 16], [0, 16])), "replace");
    const html = render();
    expect(html.match(/aria-label="selection range"/g)).toHaveLength(1);
    const header = html.match(/<header class="tile-identity">[\s\S]*?<\/header>/)![0];
    expect(header).not.toContain("selection range");
    expect(html).not.toContain("hover an enabled tile");
    expect(html).not.toContain("Select a tensor header");
  });

  /* The split the panel promises in its tab labels: a figure is either a
     function of the graph and the drawn region, or it assumes an execution, and
     the second kind never renders beside the first. */
  it("keeps modelled figures behind the execution tab", () => {
    S().setSelection("D", fromBox(box([0, 16], [0, 16])), "replace");

    const dependencies = render();
    expect(dependencies).toContain("Cost to compute");
    expect(dependencies).toContain("Backward Cone");
    expect(dependencies).toContain("Shared across tiles");
    expect(dependencies).not.toContain("Arithmetic intensity");
    expect(dependencies).not.toContain("Reuse sweep");
    expect(dependencies).not.toContain("materialized views, and no cross-op cache reuse");

    S().setInspectorTab("execution");
    const execution = render();
    expect(execution).toContain("Arithmetic intensity");
    expect(execution).toContain("Reuse sweep");
    // The assumptions each scenario rests on stay on the figure itself.
    expect(execution).toContain("materialized views, and no cross-op cache reuse");
    expect(execution).toContain("models, not bounds");
    expect(execution).not.toContain("Cost to compute");
    expect(execution).not.toContain("Backward Cone");
    expect(execution).not.toContain('<p class="hint">Idealized scenarios');
  });

  it("names the tab strip for assistive technology and the pointer alike", () => {
    const html = render();
    expect(html).toContain('role="tablist"');
    expect(html).toContain('id="ins-tab-dependencies" role="tab"');
    expect(html).toContain('aria-selected="true" aria-controls="ins-panel-dependencies"');
    expect(html).toContain('aria-selected="false" aria-controls="ins-panel-execution"');
    /* No roving tabindex, and no arrow handler behind it: the arrows are the
       tile's, and a focused tab that answered them shadowed that binding. */
    expect(html).not.toContain("tabindex");
  });

  it("does not render another group's entanglement in the inspector", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");
    if (!S().showEntangled) S().toggleEntangled();
    expect(S().entangled![2].length).toBeGreaterThan(0);
    expect(render()).not.toContain('class="ent-list"');
    S().selectAnalysisGroup("CC");
    expect(render()).toContain('class="ent-list"');
  });
});
