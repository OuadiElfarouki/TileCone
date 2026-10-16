import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { compileDSL } from "../../../src/parse/compiler";
import { useStore } from "../../../src/state/store";
import { executeQuery } from "../../../src/core/executor";
import { removalTarget } from "../../../src/components/hooks/useKeyboard";
import { box, count, fromBox } from "../../../src/core/region";
import type { ResolvedGraph } from "../../../src/core/graph";
import { analysisTensorId, groupAttribution, groupFocus, measuredParts, measuredElements } from "../../../src/components/inspector/inspector-analysis";
import { groupPropResult, analysisTarget, MAX_PER_BOX_PROPS, partsOn, BoxProp, SelPart } from "../../../src/view/workspace";
import { formatBytes } from "../../../src/view/format";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
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
    const rowsOf = (region: { boxes: readonly (readonly { lo: number; hi: number }[])[] }) =>
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

  it("states how many outputs read each operand element, per operation", () => {
    const html = render();
    expect(html).toContain("Operand reuse");
    // Both tiles are 64 rows by 32 columns of D and share no row or column:
    // a row of CC is read by the 32 outputs across its tile, a column of W by
    // the 64 down it.
    expect(html).toContain("32 uses per element");
    expect(html).toContain("64 uses per element");
  });

  /* Operand reuse follows the same scope as every figure below the tiles list,
     and its heading says which scope that is, so counts that change under a
     hover never read as counts about something else. */
  describe("operand reuse scope", () => {
    const B_BYTES = formatBytes(256 * 256 * 2);

    it("names the merged tiles, and each operation by what it computes", () => {
      const html = render();
      expect(html).toContain("Operand reuse in 2 tiles · merged");
      expect(html).toContain("via matmul → D");
      expect(html).toContain("via matmul → CC");
      // B is read in full by the 128 rows of CC the two tiles need.
      expect(html).toContain(`128 uses per element · ${B_BYTES}`);
    });

    it("narrows to a hovered tile, and says so", () => {
      S().hoverBox(0);
      const html = render();
      expect(html).toContain("Operand reuse in tile 1");
      expect(html).toContain(`64 uses per element · ${B_BYTES}`);
      expect(html).not.toContain(`128 uses per element · ${B_BYTES}`);
      S().hoverBox(null);
      expect(render()).toContain(`128 uses per element · ${B_BYTES}`);
    });

    it("leaves a hidden tile out", () => {
      S().toggleBoxHidden(1);
      const html = render();
      expect(html).toContain(`64 uses per element · ${B_BYTES}`);
      S().toggleBoxHidden(1);
      expect(render()).toContain(`128 uses per element · ${B_BYTES}`);
    });

    it("says when the tile is on a graph input rather than claiming single use", () => {
      S().setSelection("A", fromBox(box([0, 8], [0, 8])), "union");
      const html = render();
      expect(html).toContain("No operation computes this tile: it is on a graph input.");
      expect(html).not.toContain("is used by one output");
    });

    it("drops the old counts when the graph is rebuilt at other sizes", () => {
      S().applyDSL(SRC.split("256").join("128"));
      S().setSelection("D", fromBox(box([0, 64], [0, 32])));
      const html = render();
      expect(html).not.toContain(B_BYTES);
      expect(html).toContain(`64 uses per element · ${formatBytes(128 * 128 * 2)}`);
    });
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

  /* What the Del binding removes: only a tile that is actually pointed at. */
  it("removes the focused tile, dropping the pin with it", () => {
    const drawn = S().selection!.parts.length;
    expect(removalTarget(S().focusedBox, S().inspectorTab)).toBeNull();

    S().togglePinBox(0);
    const victim = removalTarget(S().focusedBox, S().inspectorTab);
    expect(victim).toBe(0);
    S().deleteBox(victim!);

    expect(S().selection!.parts).toHaveLength(drawn - 1);
    // The removed tile was the pinned one, and a surviving index would now name
    // a different tile.
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
    S().undoWorkspace();
    expect(S().selection!.parts).toHaveLength(drawn);

    // Escape means "I am done pointing at it", so the key stops acting.
    S().togglePinBox(0);
    S().clearFocus();
    expect(removalTarget(S().focusedBox, S().inspectorTab)).toBeNull();
  });

  it("retires a group once nothing is drawn on it", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    expect(active()).toBe("CC");
    S().deleteBox(2);
    expect(S().analysisGroup).toBeNull();
    expect(active()).toBe("D");
  });

  /* The list is the selector: a hovered row is the tile the arrows move,
     whichever group is analysed, and no group moves as a whole. */
  it("moves the tile hovered in the list, whatever group is analysed", () => {
    S().setSelection("CC", fromBox(box([112, 144], [192, 224])), "union");
    S().selectAnalysisGroup("D");
    S().hoverBox(2);
    const before = S().selection!.parts;
    expect(analysisTarget(before, S().analysisGroup, S().focusedBox))
      .toEqual({ tensorId: "D", focusedBox: null, index: 1 });
    S().moveSelection(0, 1);
    expect(S().selection!.parts[0]).toBe(before[0]);
    expect(S().selection!.parts[1]).toBe(before[1]);
    expect(S().selection!.parts[2].box[0].lo).toBe(before[2].box[0].lo + 1);
  });

  it("renders ranges only in tile rows, including a single-tile workspace", () => {
    S().setSelection("D", fromBox(box([0, 16], [0, 16])), "replace");
    const html = render();
    expect(html.match(/aria-label="selection range for tile/g)).toHaveLength(1);
    const header = html.match(/<header class="tile-identity">[\s\S]*?<\/header>/)![0];
    expect(header).not.toContain("selection range");
    expect(html).toContain('aria-label="pin tile 1 on D"');
    expect(html).toContain('aria-label="remove tile 1 from D"');
    expect(html).not.toContain("hover an enabled tile");
    expect(html).not.toContain("Select a tensor header");
  });

  /* The split the panel promises in its view labels: a figure is either a
     function of the graph and the drawn region, or it rests on a declared
     tiling. Nothing renders that assumes an execution the reader did not state. */
  it("keeps drawn-tile figures and plan figures in their own views, and models none", () => {
    S().setSelection("D", fromBox(box([0, 16], [0, 16])), "replace");

    const dependencies = render();
    expect(dependencies).toContain("Cost to compute");
    expect(dependencies).toContain("Backward Cone");
    expect(dependencies).toContain("Shared graph-input demand");
    expect(dependencies).not.toContain("Arithmetic intensity");
    expect(dependencies).not.toContain("Reuse sweep");
    expect(dependencies).not.toContain("Cost model");

    S().setInspectorTab("plan");
    const plan = render();
    expect(plan).toContain("Tiled tensors");
    expect(plan).not.toContain("Cost to compute");
    expect(plan).not.toContain("Backward Cone");
    S().setInspectorTab("dependencies");
  });

  it("exposes the view switch as buttons without claiming tab arrow navigation", () => {
    const html = render();
    expect(html).toContain('role="group" aria-label="analysis class"');
    expect(html).toContain('id="ins-tab-dependencies" class="ins-tab" aria-pressed="true"');
    expect(html).toContain('id="ins-tab-plan" class="ins-tab" aria-pressed="false"');
    expect(html).toContain('role="region" id="ins-panel-dependencies"');
    expect(html).not.toContain('role="tab"');
    expect(html).not.toContain('role="tablist"');
  });

  it("reports when per-tile sharing is unavailable past the attribution cap", () => {
    S().setSelection("D", fromBox(box([0, 1], [0, 1])), "replace");
    for (let index = 1; index <= MAX_PER_BOX_PROPS; index++) {
      S().setSelection(
        "D",
        fromBox(box([index * 2, index * 2 + 1], [0, 1])),
        "union"
      );
    }

    expect(S().selection!.parts).toHaveLength(MAX_PER_BOX_PROPS + 1);
    expect(S().perBox).toBeNull();
    const html = render();
    expect(html).toContain(`Per-tile sharing is unavailable above ${MAX_PER_BOX_PROPS} total tiles.`);
    expect(html).not.toContain("Enable and analyse at least two tiles together");
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
