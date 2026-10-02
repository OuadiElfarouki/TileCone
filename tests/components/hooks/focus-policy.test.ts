import { describe, expect, it } from "vitest";
import { pressReleases, releasedAfterPointer, type FocusNode } from "../../../src/components/hooks/useFocusPolicy";

/** A fake element: a pointer control or not, inside a focus-managing container or not. */
const node = ({ control = false, managed = false, children = [] as FocusNode[] } = {}): FocusNode => {
  const self: FocusNode = {
    matches: () => control,
    closest: () => (managed ? {} : null),
    contains: (other) => other === self || children.includes(other as FocusNode),
  };
  return self;
};

describe("a press releases what it lands outside of", () => {
  it("releases a field or control when the workspace is pressed", () => {
    const canvas = node();
    expect(pressReleases(node(), canvas)).toBe(true); // the source editor, then a card
    expect(pressReleases(node({ control: true }), canvas)).toBe(true); // a slider, then a card
    expect(pressReleases(node(), null)).toBe(true);
  });

  it("keeps it for a press inside it", () => {
    const thumb = node();
    expect(pressReleases(node({ children: [thumb] }), thumb)).toBe(false);
  });

  it("leaves focus to a menu, a list box or a dialog", () => {
    expect(pressReleases(node(), node({ managed: true }))).toBe(false); // pressing into a menu
    expect(pressReleases(node({ managed: true }), node())).toBe(false); // the dialog's trap
  });
});

describe("a pointer leaves no control holding the keys", () => {
  it("releases a button or slider once the press ends", () => {
    expect(releasedAfterPointer(node({ control: true }))).toBe(true);
  });

  it("keeps a text field, and anything in a menu", () => {
    expect(releasedAfterPointer(node())).toBe(false);
    expect(releasedAfterPointer(node({ control: true, managed: true }))).toBe(false);
  });
});
