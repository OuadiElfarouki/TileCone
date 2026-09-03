import { describe, expect, it } from "vitest";
import { matchesShortcut, SHORTCUT_GROUPS, SHORTCUTS } from "../ui/shortcuts";

const key = (
  value: string,
  modifiers: Partial<Pick<KeyboardEvent, "altKey" | "ctrlKey" | "metaKey" | "shiftKey">> = {}
) => ({
  key: value,
  altKey: false,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  ...modifiers,
});

describe("shortcut manifest", () => {
  it("is the grouped inventory rendered by the help sheet", () => {
    const grouped = SHORTCUT_GROUPS.flatMap((group) => group.items);
    expect(grouped).toContain(SHORTCUTS.fit);
    expect(grouped).toContain(SHORTCUTS.run);
    expect(new Set(grouped.map((item) => item.id)).size).toBe(grouped.length);
  });

  it("matches the modifiers advertised by local and global bindings", () => {
    expect(matchesShortcut(key("f"), SHORTCUTS.fit)).toBe(true);
    expect(matchesShortcut(key("1", { altKey: true }), SHORTCUTS.leftPanel)).toBe(true);
    expect(matchesShortcut(key("1"), SHORTCUTS.leftPanel)).toBe(false);
    expect(matchesShortcut(key("Enter", { metaKey: true }), SHORTCUTS.run)).toBe(true);
    expect(matchesShortcut(key("Enter"), SHORTCUTS.run)).toBe(false);
    expect(matchesShortcut(key("ArrowLeft", { shiftKey: true }), SHORTCUTS.moveFast)).toBe(true);
    expect(matchesShortcut(key("ArrowLeft"), SHORTCUTS.moveFast)).toBe(false);
    // The base arrow binding stays permissive: the fast variant refines it
    // rather than replacing it, so both have to match a shifted arrow.
    expect(matchesShortcut(key("ArrowLeft", { shiftKey: true }), SHORTCUTS.move)).toBe(true);
  });

  it("does not read a shifted printable key as its unshifted binding", () => {
    // Ctrl+Shift+Z is redo by convention and unimplemented here, so it must not
    // fall through to undo. The browser reports the shifted character directly.
    expect(matchesShortcut(key("Z", { ctrlKey: true, shiftKey: true }), SHORTCUTS.undo)).toBe(false);
    expect(matchesShortcut(key("z", { ctrlKey: true }), SHORTCUTS.undo)).toBe(true);
    expect(matchesShortcut(key("D", { shiftKey: true }), SHORTCUTS.feeds)).toBe(false);
    expect(matchesShortcut(key("?", { shiftKey: true }), SHORTCUTS.help)).toBe(true);
  });

  it("does not pretend a pointer gesture is a keyboard match", () => {
    expect(matchesShortcut(key("Scroll"), SHORTCUTS.zoom)).toBe(false);
  });
});
