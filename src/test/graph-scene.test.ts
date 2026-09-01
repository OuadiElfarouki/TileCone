import { describe, expect, it } from "vitest";
import { resolveGraph } from "../core/graph";
import {
  buildBaseGraphLayout,
  buildGraphScene,
  type BaseGraphLayout,
} from "../ui/graph-scene";
import { WORLD_MARGIN } from "../ui/graph-geometry";
import { G } from "./harness";

const measure = () => ({ w: 100, h: 60 });

describe("structural graph layout", () => {
  it("preserves repeated operand links that Dagre collapses for placement", () => {
    const resolved = resolveGraph(
      G({ X: [4, 4] }, [["mul", "matmul", ["X", "X"], ["Y"]]])
    );
    const layout = buildBaseGraphLayout(resolved, measure);

    expect(layout.links.filter((link) => link.from === "t:X" && link.to === "n:mul"))
      .toHaveLength(2);
    expect(layout.links).toHaveLength(3);
    expect(layout.nodes).toHaveLength(3);
  });

  it("uses the supplied tensor measurement policy verbatim", () => {
    const resolved = resolveGraph(G({ X: [4] }, []));
    const layout = buildBaseGraphLayout(resolved, () => ({ w: 173, h: 91 }));
    const tensor = layout.nodes.find((node) => node.id === "X")!;

    expect({ w: tensor.w, h: tensor.h }).toEqual({ w: 173, h: 91 });
  });
});

describe("live graph scene projection", () => {
  const base: BaseGraphLayout = {
    nodes: [
      { id: "X", kind: "tensor", x: 20, y: 30, w: 100, h: 60 },
      { id: "op", kind: "op", x: 200, y: 45, w: 64, h: 30 },
    ],
    links: [{ from: "t:X", to: "n:op", tensorId: "X", opId: "op" }],
    width: 284,
    height: 110,
  };

  it("moves tensors, leaves operators fixed, and reroutes their connectors", () => {
    const before = buildGraphScene(base, {});
    const after = buildGraphScene(base, { X: { dx: 15, dy: 25 } });

    expect(after.nodes.find((node) => node.id === "X")).toMatchObject({ x: 35, y: 55 });
    expect(after.nodes.find((node) => node.id === "op")).toEqual(base.nodes[1]);
    expect(after.edges[0].path).not.toBe(before.edges[0].path);
    expect(base.nodes[0]).toMatchObject({ x: 20, y: 30 });
  });

  it("expands scene bounds to contain moved cards and the world margin", () => {
    const scene = buildGraphScene(base, { X: { dx: 300, dy: 200 } });
    const tensor = scene.nodes.find((node) => node.id === "X")!;

    expect(scene.width).toBe(tensor.x + tensor.w + WORLD_MARGIN);
    expect(scene.height).toBe(tensor.y + tensor.h + WORLD_MARGIN);
  });

  it("includes cards moved left or above the original scene origin", () => {
    const scene = buildGraphScene(base, { X: { dx: -150, dy: -100 } });
    const tensor = scene.nodes.find((node) => node.id === "X")!;

    expect(scene.left).toBe(tensor.x - WORLD_MARGIN);
    expect(scene.top).toBe(tensor.y - WORLD_MARGIN);
    expect(scene.left + scene.width).toBe(base.width);
    expect(scene.top + scene.height).toBe(base.height);
  });

  it("fans repeated links into independently keyed routes", () => {
    const repeated: BaseGraphLayout = {
      ...base,
      links: [base.links[0], base.links[0]],
    };
    const scene = buildGraphScene(repeated, {});

    expect(scene.edges.map((edge) => edge.key)).toEqual(["t:X|n:op#0", "t:X|n:op#1"]);
    expect(scene.edges[0].path).not.toBe(scene.edges[1].path);
    expect(scene.edges[0].mark).not.toBe(scene.edges[1].mark);
  });
});
