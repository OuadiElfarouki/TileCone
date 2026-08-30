import { describe, expect, it } from "vitest";
import { cardSize } from "../ui/TensorCard";
import { graphScale } from "../ui/tiling";
import type { ViewCfg } from "../ui/store";

const cfg: ViewCfg = { sliders: [0, 0], projection: true };

describe("frameless tensor footprint", () => {
  it("reserves collision width for the visible name and numeric shape", () => {
    const shape = [4, 4];
    const px = graphScale([{ rows: 4, cols: 4 }]);
    const short = cardSize(shape, cfg, px, "X");
    const long = cardSize(shape, cfg, px, "attention_output_projection");
    expect(long.w).toBeGreaterThan(short.w);
    expect(long.h).toBe(short.h);
  });

  it("keeps the label and grid inside the solid collision height", () => {
    const shape = [16, 16];
    const px = graphScale([{ rows: 16, cols: 16 }]);
    const size = cardSize(shape, cfg, px, "X");
    expect(size.h).toBeGreaterThan(16 * px); // numeric label sits above the grid
  });
});
