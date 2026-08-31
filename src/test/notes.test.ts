import { describe, expect, it } from "vitest";
import {
  coneFindings,
  coneIsFullyElementwise,
  dependencyNotes,
  MAX_NOTES,
} from "../core/notes";
import { propagateBackward } from "../core/propagate";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

const notesFor = (dsl: string, tensor: string, sel: [number, number][]) => {
  const { resolved } = compileDSL(dsl);
  const back = propagateBackward(resolved, {
    tensorId: tensor,
    region: fromBox(sel.map(([lo, hi]) => ({ lo, hi }))),
  });
  return { resolved, back, notes: dependencyNotes(resolved, back) };
};

const GEMM = `params M=64 N=64 K=128
input A [M, K] f16
input B [K, N] f16
C = matmul(A, B)
`;

describe("contraction notes", () => {
  it("names the contracted axis and its extent", () => {
    const { notes } = notesFor(GEMM, "C", [[0, 16], [0, 16]]);
    expect(notes).toHaveLength(1);
    expect(notes[0].op).toBe("matmul");
    // the DSL declared this axis as K, so the note must say K — not the
    // internal einsum label
    expect(notes[0].text).toContain("contracts K=128 in full");
    expect(notes[0].text).toContain("A and B");
  });

  it("attributes the note to the node that causes it", () => {
    const { notes } = notesFor(GEMM, "C", [[0, 16], [0, 16]]);
    const { resolved } = notesFor(GEMM, "C", [[0, 16], [0, 16]]);
    expect(resolved.topo.some((n) => n.id === notes[0].nodeId)).toBe(true);
  });

  it("says nothing about a tensor the cone never reached", () => {
    // selecting an input means no operation is upstream of it at all
    const { notes } = notesFor(GEMM, "A", [[0, 8], [0, 8]]);
    expect(notes).toEqual([]);
  });
});

describe("reduction and normalization notes", () => {
  it("softmax names the axis it normalises over", () => {
    const { notes } = notesFor(
      `input X [8, 32] f32
Y = softmax(X, axis=-1)
`,
      "Y",
      [[0, 2], [0, 4]]
    );
    expect(notes).toHaveLength(1);
    expect(notes[0].text).toContain("softmax normalises X along axis 1");
    expect(notes[0].text).toContain("32 wide");
  });

  it("normalization also reports the axes that stay independent", () => {
    // the fusable half matters: reporting only the constraint would make
    // layernorm look less tileable than it is
    const { notes } = notesFor(
      `input X [16, 32] f16
input W [32] f16
input Bb [32] f16
H = layernorm(X, W, Bb, axes=[-1])
`,
      "H",
      [[0, 4], [0, 32]]
    );
    const text = notes.map((n) => n.text).join(" ");
    expect(text).toContain("statistics");
    // one free axis: singular agreement
    expect(text).toContain("axis 0 stays independent");
  });

  it("agrees in number when several axes stay independent", () => {
    const { notes } = notesFor(
      `input X [4, 8, 32] f16
input W [32] f16
input Bb [32] f16
H = layernorm(X, W, Bb, axes=[-1])
`,
      "H",
      [[0, 2], [0, 2], [0, 32]]
    );
    expect(notes.map((n) => n.text).join(" ")).toContain("axes 0, 1 stay independent");
  });

  it("a reduction names its collapsed axes", () => {
    const { notes } = notesFor(
      `input X [8, 16] f32
Y = sum(X, axes=[1])
`,
      "Y",
      [[0, 4]]
    );
    expect(notes.map((n) => n.text).join(" ")).toContain("reduces X over axis 1");
  });
});

describe("notes never over-claim", () => {
  it("a chain of elementwise ops produces no notes but is not 'no cone'", () => {
    const dsl = `input X [8, 8] f32
Y = relu(X)
Z = add(Y, X)
`;
    const { resolved, back, notes } = notesFor(dsl, "Z", [[0, 4], [0, 4]]);
    expect(notes).toEqual([]);
    expect(coneIsFullyElementwise(resolved, back)).toBe(true);
  });

  it("a cone that reached no operation is not elementwise either", () => {
    const { resolved, back } = notesFor(GEMM, "A", [[0, 8], [0, 8]]);
    expect(coneIsFullyElementwise(resolved, back)).toBe(false);
  });

  it("a constrained cone is not reported as elementwise", () => {
    const { resolved, back } = notesFor(GEMM, "C", [[0, 16], [0, 16]]);
    expect(coneIsFullyElementwise(resolved, back)).toBe(false);
  });

  it("is capped so the panel stays a summary", () => {
    const dsl = `params S=32
input X [S, S] f16
A1 = softmax(X, axis=-1)
A2 = softmax(A1, axis=-1)
A3 = softmax(A2, axis=-1)
A4 = softmax(A3, axis=-1)
A5 = softmax(A4, axis=-1)
A6 = softmax(A5, axis=-1)
`;
    const { notes } = notesFor(dsl, "A6", [[0, 4], [0, 32]]);
    expect(notes.length).toBeLessThanOrEqual(MAX_NOTES);
  });

  it("identical statements are not repeated", () => {
    const { notes } = notesFor(
      `params S=32
input X [S, S] f16
A1 = softmax(X, axis=-1)
A2 = softmax(A1, axis=-1)
`,
      "A2",
      [[0, 4], [0, 32]]
    );
    const texts = notes.map((n) => n.text);
    expect(new Set(texts).size).toBe(texts.length);
  });
});

describe("shapes the prototype never covered", () => {
  it("convolution reports the halo neighbouring tiles re-read", () => {
    const { notes } = notesFor(
      `params N=1 C=3 F=8 H=32 W=32
input X [N, C, H, W] f16
input W1 [F, C, 3, 3] f16
Y = conv(X, W1, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`,
      "Y",
      [[0, 1], [0, 1], [0, 4], [0, 4]]
    );
    const text = notes.map((n) => n.text).join(" ");
    expect(text).toContain("3×3 window");
    expect(text).toContain("overlap by 1×1");
  });

  it("stays silent when the stride is wide enough to tile cleanly", () => {
    // stride 3 with a 3x3 kernel leaves no halo, so there is nothing to warn about
    const { notes } = notesFor(
      `params N=1 C=3 F=8 H=33 W=33
input X [N, C, H, W] f16
input W1 [F, C, 3, 3] f16
Y = conv(X, W1, stride=[3, 3], pads=[[0, 0], [0, 0]], dilation=[1, 1], groups=1)
`,
      "Y",
      [[0, 1], [0, 1], [0, 4], [0, 4]]
    );
    expect(notes.filter((n) => n.op === "conv")).toEqual([]);
  });

  it("a scan is described as triangular, and knows its direction", () => {
    const forward = notesFor(
      `input X [32] f32
Y = cumsum(X, axis=0, reverse=false)
`,
      "Y",
      [[8, 16]]
    );
    expect(forward.notes[0].text).toContain("every earlier element");
    expect(forward.notes[0].text).toContain("triangular");

    const backward = notesFor(
      `input X [32] f32
Y = cumsum(X, axis=0, reverse=true)
`,
      "Y",
      [[8, 16]]
    );
    expect(backward.notes[0].text).toContain("every later element");
  });
});

describe("reshape, the op that looks free and is not", () => {
  const TRAP = `input X [4, 4] f32
F = reshape(X, shape=[16])
`;

  it("warns when a contiguous tile lands on several runs of the input", () => {
    // F[2:6] straddles two rows of X
    const { notes } = notesFor(TRAP, "F", [[2, 6]]);
    const reshape = notes.find((n) => n.op === "reshape");
    expect(reshape).toBeDefined();
    expect(reshape!.text).toContain("2 disjoint runs");
    expect(reshape!.text).toContain("strided in memory");
  });

  it("stays silent when the tile happens to split cleanly", () => {
    // F[0:4] is exactly row 0 of X — one run in, one run out
    const { notes } = notesFor(TRAP, "F", [[0, 4]]);
    expect(notes.filter((n) => n.op === "reshape")).toEqual([]);
  });

  it("counts the runs it actually found", () => {
    // spanning three rows yields three runs, not a generic "several"
    const { notes } = notesFor(TRAP, "F", [[2, 10]]);
    const reshape = notes.find((n) => n.op === "reshape");
    expect(reshape!.text).toContain("3 disjoint runs");
  });
});

describe("look-alike notes are merged, not repeated", () => {
  const QKV = `params S=16 E=32
input X [S, E] f16
input Wq [E, E] f16
input Wk [E, E] f16
input Wv [E, E] f16
Q = matmul(X, Wq)
K = matmul(X, Wk)
V = matmul(X, Wv)
Kt = transpose(K, perm=[1, 0])
S1 = matmul(Q, Kt)
Y = matmul(S1, V)
`;

  it("states one constraint once and names the tensors it also covers", () => {
    const { notes } = notesFor(QKV, "Y", [[0, 4], [0, 8]]);
    const contractionNotes = notes.filter((n) => n.text.includes("contracts E=32"));
    expect(contractionNotes).toHaveLength(1);
    expect(contractionNotes[0].alsoApplies?.length).toBeGreaterThan(0);
    expect(contractionNotes[0].text).toContain("The same holds for");
  });

  it("merging frees cap slots for genuinely different constraints", () => {
    // Without merging, three identical projection notes would fill the cap and
    // the distinct downstream contraction would never be shown.
    const { notes } = notesFor(QKV, "Y", [[0, 4], [0, 8]]);
    const keys = new Set(notes.map((n) => n.text.replace(/The same holds.*/, "")));
    expect(keys.size).toBe(notes.length);
    expect(notes.length).toBeGreaterThan(1);
  });
});

describe("notes describe the cone that exists", () => {
  it("stays silent when the contraction axis was not pulled in full", () => {
    // A degenerate selection with an empty extent pulls nothing, so the
    // contraction claim would be false.
    const { resolved } = compileDSL(GEMM);
    const back = propagateBackward(resolved, {
      tensorId: "C",
      region: { boxes: [box([0, 0], [0, 16])], exact: true, reasons: [] },
    });
    expect(dependencyNotes(resolved, back)).toEqual([]);
  });
});

describe("cone findings: flags, and what survives the note cap", () => {
  const findingsFor = (dsl: string, tensor: string, sel: [number, number][]) => {
    const { resolved } = compileDSL(dsl);
    const back = propagateBackward(resolved, {
      tensorId: tensor,
      region: fromBox(sel.map(([lo, hi]) => ({ lo, hi }))),
    });
    return { resolved, back, findings: coneFindings(resolved, back) };
  };

  // Five distinct constraints, with the hardest one produced by the *last* node
  // in graph order, so a verdict taken from the capped list would miss it.
  const STACK = `params S=16 E=16
input X [S, E] f16
input W [E] f16
input Bb [E] f16
input V [E, E] f16
C1 = cumsum(X, axis=0, reverse=false)
C2 = cumsum(C1, axis=0, reverse=true)
P = softmax(C2, axis=-1)
H = layernorm(P, W, Bb, axes=[-1])
Y = matmul(H, V)
`;

  it("keeps the hardest constraint when the cap has to drop one", () => {
    const { findings } = findingsFor(STACK, "Y", [[0, 4], [0, 4]]);
    expect(findings.notes).toHaveLength(MAX_NOTES);
    // the contraction is the fifth and last in graph order: dropping by graph
    // order would lose the one constraint that forces staging or accumulation
    expect(findings.notes.some((note) => note.severity === 3)).toBe(true);
    // the weakest constraints are what gave way
    expect(findings.notes.filter((note) => note.severity === 1)).toHaveLength(1);
  });

  it("reads the surviving notes in graph order, not in ranked order", () => {
    const { resolved, findings } = findingsFor(STACK, "Y", [[0, 4], [0, 4]]);
    const position = (nodeId: string) => resolved.topo.findIndex((n) => n.id === nodeId);
    const order = findings.notes.map((note) => position(note.nodeId));
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  it("flags a tensor even when its note did not fit the list", () => {
    const { findings } = findingsFor(STACK, "Y", [[0, 4], [0, 4]]);
    expect(findings.flags.get("V")?.[0]).toContain("contracted");
    expect(findings.flags.get("H")?.[0]).toContain("contracted");
  });

  it("addresses flags to the tensors the constraint is visible on", () => {
    const { resolved, findings } = findingsFor(STACK, "Y", [[0, 4], [0, 4]]);
    for (const tensorId of findings.flags.keys())
      expect(resolved.tensors[tensorId]).toBeDefined();
    // softmax closes an axis of its own input, not of its output
    expect(findings.flags.get("C2")?.some((f) => f.includes("softmax"))).toBe(true);
    expect(findings.flags.get("P")?.some((f) => f.includes("softmax"))).toBe(false);
  });

  it("breaks a severity tie on the constraint nearest the tile", () => {
    // three equally hard contractions: the projections are four steps upstream,
    // the attention product one. The reader hits the near one first.
    const QKV = `params S=16 E=32
input X [S, E] f16
input Wq [E, E] f16
input Wk [E, E] f16
Q = matmul(X, Wq)
K = matmul(X, Wk)
Kt = transpose(K, perm=[1, 0])
Y = matmul(Q, Kt)
`;
    const { resolved, back, findings } = findingsFor(QKV, "Y", [[0, 4], [0, 8]]);
    // Q and K contract E four steps away; Y contracts its own axis one step
    // away, so a cap of one has to keep Y
    expect(dependencyNotes(resolved, back, 1)[0].subject).toBe("Y");
    expect(
      findings.notes.some(
        (note) => note.subject === "Q" || note.alsoApplies?.includes("Q")
      )
    ).toBe(true);
  });

  it("states the constraint an operation puts on a tensor it takes twice", () => {
    // matmul(D, D) contracts D against itself: one slot pulls whole rows, the
    // other whole columns. The union covers the axis; the disjoint boxes that
    // represent it do not, and the note used to vanish because of that.
    const SELF = `params M=256 N=256 K=512
input A [M, K] f16
input B [K, N] f16
input F [N, M] f16
C = matmul(A, B)
D = matmul(F, C)
DD = matmul(D, D)
`;
    const { resolved, back, findings } = findingsFor(SELF, "DD", [[0, 32], [0, 32]]);
    // the tile's own operation is stated, and is the nearest of the three
    expect(findings.notes.some((note) => note.subject === "DD")).toBe(true);
    expect(dependencyNotes(resolved, back, 1)[0].subject).toBe("DD");
    // both operand slots are reported against the one tensor they share
    expect(findings.flags.get("D")).toHaveLength(2);
  });

  it("a merged note flags every tensor it was merged from", () => {
    const QKV = `params S=16 E=32
input X [S, E] f16
input Wq [E, E] f16
input Wk [E, E] f16
input Wv [E, E] f16
Q = matmul(X, Wq)
K = matmul(X, Wk)
V = matmul(X, Wv)
Kt = transpose(K, perm=[1, 0])
S1 = matmul(Q, Kt)
Y = matmul(S1, V)
`;
    const { findings } = findingsFor(QKV, "Y", [[0, 4], [0, 8]]);
    for (const weight of ["Wq", "Wk", "Wv"])
      expect(findings.flags.get(weight)?.length).toBeGreaterThan(0);
  });

  it("states the elementwise finding as its own verdict, with no author", () => {
    const { findings } = findingsFor(
      `params S=8
input X [S] f32
Y = add(X, X)
`,
      "Y",
      [[0, 4]]
    );
    expect(findings.elementwise).toBe(true);
    expect(findings.notes).toEqual([]);
  });

  it("says nothing at all when the cone reached no operation", () => {
    const { findings } = findingsFor(GEMM, "A", [[0, 4], [0, 4]]);
    expect(findings.elementwise).toBe(false);
    expect(findings.notes).toEqual([]);
    expect(findings.flags.size).toBe(0);
  });

});
