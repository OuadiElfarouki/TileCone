import { describe, expect, it } from "vitest";
import type { MenuChoice } from "../../../src/view/menu";
import { cardViewMenu, type CardViewChoice } from "../../../src/view/tensor/menus";

const NAMES = ["B", "S", "H", "D"];
const name = (axis: number) => NAMES[axis];
const SHAPE = [1, 128, 4, 32];

const choice = (spec: ReturnType<typeof cardViewMenu>, id: CardViewChoice) =>
  spec.find((entry): entry is MenuChoice<CardViewChoice> => entry.kind === "choice" && entry.id === id)!;
const action = (spec: ReturnType<typeof cardViewMenu>, id: string) =>
  spec.find((entry) => entry.kind === "action" && entry.id === id)!;

describe("a card's view menu", () => {
  it("offers every axis for the rows and for the columns, with the drawn pair checked", () => {
    const spec = cardViewMenu(SHAPE, undefined, name);
    expect(choice(spec, "rows").options.map((o) => [o.label, o.checked])).toEqual([
      ["B", false], ["S", false], ["H", true], ["D", false],
    ]);
    expect(choice(spec, "cols").options.filter((o) => o.checked).map((o) => o.value)).toEqual([3]);
    expect(choice(spec, "rows").options[1].title).toContain("display only, the graph is unchanged");
  });

  it("offers the default plane only when the card draws another", () => {
    expect(action(cardViewMenu(SHAPE, undefined, name), "default")).toMatchObject({
      disabled: true,
      title: "the card already draws the last two axes",
    });
    const chosen = cardViewMenu(SHAPE, { axes: [1, 3] }, name);
    expect(action(chosen, "default")).toMatchObject({ disabled: false });
    expect(choice(chosen, "rows").options.find((o) => o.checked)!.label).toBe("S");
    expect(action(chosen, "swap")).toMatchObject({ disabled: false, title: "draw D down and S across" });
  });

  it("is empty below rank two, where there is no pair to choose", () => {
    expect(cardViewMenu([16], undefined, name)).toEqual([]);
    expect(cardViewMenu([], undefined, name)).toEqual([]);
    expect(cardViewMenu([8, 16], undefined, name).length).toBeGreaterThan(0);
  });
});
