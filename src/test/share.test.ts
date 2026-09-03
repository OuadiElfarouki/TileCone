import { describe, expect, it } from "vitest";
import {
  decodeWorkspace,
  encodeWorkspace,
  MAX_HASH_LENGTH,
  selectionToLink,
  shareTarget,
  WorkspaceLink,
} from "../ui/share";
import { box } from "../core/region";

const LINK: WorkspaceLink = {
  dsl: "input A [4, 4] f32\nB = relu(A)\n",
  dir: "both",
  tile: -2,
  snap: false,
  axes: "numeric",
  pos: { B: [35, -20] },
  sel: [{ t: "B", box: [[0, 2], [1, 3]] }],
};

describe("workspace links round-trip", () => {
  it("survives encode then decode unchanged", () => {
    expect(decodeWorkspace(`#s=${encodeWorkspace(LINK)}`)).toEqual(LINK);
  });

  it("accepts the hash with or without its leading #", () => {
    const hash = encodeWorkspace(LINK);
    expect(decodeWorkspace(`s=${hash}`)).toEqual(decodeWorkspace(`#s=${hash}`));
  });

  it("carries non-ASCII source through base64 intact", () => {
    const link = { ...LINK, dsl: "# λ — tensor ≈ région\ninput A [2] f32\n" };
    expect(decodeWorkspace(`#s=${encodeWorkspace(link)}`)?.dsl).toBe(link.dsl);
  });

  it("preserves ordered selection parts, including their order", () => {
    const selection = {
      parts: [
        { tensorId: "C", box: box([4, 8], [0, 4]) },
        { tensorId: "C", box: box([0, 2], [2, 6]) },
      ],
    };
    const link = { ...LINK, sel: selectionToLink(selection) };
    expect(decodeWorkspace(`#s=${encodeWorkspace(link)}`)?.sel).toEqual([
      { t: "C", box: [[4, 8], [0, 4]] },
      { t: "C", box: [[0, 2], [2, 6]] },
    ]);
  });

  it("keeps each part's tensor, so a link can span several", () => {
    const selection = {
      parts: [
        { tensorId: "A", box: box([0, 2], [0, 2]) },
        { tensorId: "B", box: box([2, 4], [1, 3]) },
        { tensorId: "A", box: box([2, 4], [2, 4]) },
      ],
    };
    const link = { ...LINK, sel: selectionToLink(selection) };
    expect(decodeWorkspace(`#s=${encodeWorkspace(link)}`)?.sel?.map((p) => p.t)).toEqual([
      "A",
      "B",
      "A",
    ]);
  });

  it("reads a pre-multi-tensor link as parts on that one tensor", () => {
    const legacy = { ...LINK, sel: { t: "B", boxes: [[[0, 2], [1, 3]], [[2, 4], [0, 1]]] } };
    expect(decodeWorkspace(`#s=${encodeWorkspace(legacy as never)}`)?.sel).toEqual([
      { t: "B", box: [[0, 2], [1, 3]] },
      { t: "B", box: [[2, 4], [0, 1]] },
    ]);
  });

  it("round-trips every direction, including none", () => {
    for (const dir of ["none", "backward", "forward", "both"] as const)
      expect(decodeWorkspace(`#s=${encodeWorkspace({ ...LINK, dir })}`)?.dir).toBe(dir);
  });
});

describe("a link that cannot be trusted is refused, not repaired", () => {
  it("rejects junk rather than throwing", () => {
    for (const hash of ["", "#", "#s=", "#s=not-base64!!", "#x=abc", "#s=" + btoa("{")])
      expect(decodeWorkspace(hash)).toBeNull();
  });

  it("rejects a payload with no source", () => {
    expect(decodeWorkspace(`#s=${btoa(JSON.stringify({ dir: "both" }))}`)).toBeNull();
  });

  it("rejects an unknown direction", () => {
    const bad = encodeWorkspace({ ...LINK, dir: "sideways" as never });
    expect(decodeWorkspace(`#s=${bad}`)).toBeNull();
  });

  it("rejects malformed boxes rather than restoring half a selection", () => {
    const bad = encodeWorkspace({
      ...LINK,
      sel: [{ t: "B", box: [[3, 1]] as [number, number][] }], // hi <= lo
    });
    expect(decodeWorkspace(`#s=${bad}`)).toBeNull();
  });

  it("rejects a mixed valid and invalid selection rather than keeping a prefix", () => {
    const bad = encodeWorkspace({
      ...LINK,
      sel: [
        { t: "B", box: [[0, 1], [0, 1]] },
        { t: "B", box: [[3, 1]] },
      ],
    });
    expect(decodeWorkspace(`#s=${bad}`)).toBeNull();
  });

  it("rejects explicitly malformed settings while defaulting absent legacy ones", () => {
    for (const patch of [{ tile: "large" }, { tile: Infinity }, { snap: "yes" }]) {
      const bad = btoa(JSON.stringify({ ...LINK, ...patch }));
      expect(decodeWorkspace(`#s=${bad}`)).toBeNull();
    }
  });

  it("rejects malformed tensor layout offsets", () => {
    for (const pos of [{ B: [1] }, { B: [1, Infinity] }, { B: [1e20, 0] }, { B: "near" }]) {
      const bad = btoa(JSON.stringify({ ...LINK, pos }));
      expect(decodeWorkspace(`#s=${bad}`)).toBeNull();
    }
  });

  it("defaults snapping on for links written before it existed", () => {
    const legacy = btoa(JSON.stringify({ dsl: "input A [2] f32\n", dir: "both", tile: 0, sel: null }));
    expect(decodeWorkspace(`#s=${legacy}`)?.snap).toBe(true);
  });

  it("keeps a null selection null", () => {
    expect(decodeWorkspace(`#s=${encodeWorkspace({ ...LINK, sel: null })}`)?.sel).toBeNull();
  });
});

describe("share target", () => {
  it("is a URL when the payload fits", () => {
    const target = shareTarget("https://x.dev", "/app", LINK);
    expect(target.startsWith("https://x.dev/app#s=")).toBe(true);
    expect(decodeWorkspace(target.slice(target.indexOf("#")))).toEqual(LINK);
  });

  it("falls back to raw JSON when the hash would be unusable", () => {
    const huge = { ...LINK, dsl: "# pad\n".repeat(4000) };
    const target = shareTarget("https://x.dev", "/app", huge);
    expect(target.startsWith("https://x.dev")).toBe(false);
    expect(JSON.parse(target).dsl).toBe(huge.dsl);
    expect(encodeWorkspace(huge).length).toBeGreaterThan(MAX_HASH_LENGTH);
  });
});

describe("selectionToLink", () => {
  it("is null without a selection", () => {
    expect(selectionToLink(null)).toBeNull();
  });

  it("flattens intervals to [lo, hi] pairs", () => {
    expect(selectionToLink({ parts: [{ tensorId: "A", box: box([1, 5]) }] })).toEqual([
      { t: "A", box: [[1, 5]] },
    ]);
  });

  it("is null for a selection with no parts", () => {
    expect(selectionToLink({ parts: [] })).toBeNull();
  });
});

describe("the shape view travels with the link", () => {
  it("round-trips the chosen reading", () => {
    const link = {
      dsl: "input X [2] f32\n",
      dir: "both" as const,
      tile: 0,
      snap: true,
      axes: "numeric" as const,
      sel: null,
    };
    expect(decodeWorkspace(`#s=${encodeWorkspace(link)}`)).toEqual(link);
  });

  it("keeps a link written before the setting existed on numeric shapes", () => {
    // Absent is not malformed: a legacy field takes its documented default.
    const legacy = { dsl: "input X [2] f32\n", dir: "both" as const, tile: 0, sel: null };
    const decoded = decodeWorkspace(`#s=${encodeWorkspace(legacy)}`);
    expect(decoded).not.toBeNull();
    expect(decoded!.axes).toBe("numeric");
  });

  it("refuses a link whose shape view is not a known reading", () => {
    const bad = { dsl: "input X [2] f32\n", dir: "both", tile: 0, axes: "hieroglyphs", sel: null };
    expect(decodeWorkspace(`#s=${encodeWorkspace(bad as never)}`)).toBeNull();
  });
});
