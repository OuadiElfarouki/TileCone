import { describe, expect, it } from "vitest";
import { box } from "../core/region";
import { formatSelectionBox, parseSelectionBox } from "../ui/selection-range";

describe("inspector selection ranges", () => {
  it("formats singleton and interval axes compactly", () => {
    expect(formatSelectionBox(box([2, 3], [4, 9]))).toBe("[2, 4:9]");
  });

  it("parses bracketed rank-preserving ranges", () => {
    expect(parseSelectionBox("[2, 4:9]", [8, 16])).toEqual(box([2, 3], [4, 9]));
    expect(parseSelectionBox("2, 4:9", [8, 16])).toEqual(box([2, 3], [4, 9]));
  });

  it("supports scalar selections", () => {
    expect(parseSelectionBox("[]", [])).toEqual([]);
  });

  it("rejects rank mismatches, empty intervals, and out-of-bounds axes", () => {
    expect(parseSelectionBox("[2]", [8, 16])).toBeNull();
    expect(parseSelectionBox("[2:2, 0:1]", [8, 16])).toBeNull();
    expect(parseSelectionBox("[0:9, 0:1]", [8, 16])).toBeNull();
  });
});
