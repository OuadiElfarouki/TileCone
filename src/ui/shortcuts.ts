/** One manifest drives both key matching and the visible shortcut sheet. Local
 * controls may own an action (the source editor owns Run), but neither they nor
 * the dialog restate its key or label. An entry without `keys` is a pointer
 * gesture that the sheet still has to teach; it never matches a key event. */
export type Shortcut = {
  id: string;
  label: string;
  action: string;
  keys?: readonly string[];
  alt?: boolean;
  primary?: boolean;
  shift?: boolean;
};

const shortcut = <T extends Shortcut>(value: T): T => value;

export const SHORTCUTS = {
  move: shortcut({ id: "move", label: "Arrow keys", action: "move the focused tile", keys: ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"] }),
  moveFast: shortcut({ id: "move-fast", label: "Shift + arrow", action: "move it eight steps", keys: ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"], shift: true }),
  toggleTile: shortcut({ id: "toggle-tile", label: "H", action: "show or hide the focused tile's needs and feeds", keys: ["h"] }),
  escape: shortcut({ id: "escape", label: "Esc", action: "cancel a gesture, unpin a tile, or leave a field", keys: ["Escape"] }),
  undo: shortcut({ id: "undo", label: "Ctrl/Cmd + Z", action: "undo the last tile or tensor move", keys: ["z"], primary: true }),
  needs: shortcut({ id: "needs", label: "U", action: "toggle What it needs", keys: ["u"] }),
  feeds: shortcut({ id: "feeds", label: "D", action: "toggle What it feeds", keys: ["d"] }),
  fit: shortcut({ id: "fit", label: "F", action: "fit the graph to the viewport", keys: ["f"] }),
  scrub: shortcut({ id: "scrub", label: "[ / ]", action: "scrub the first hidden tensor axis", keys: ["[", "]"] }),
  zoom: shortcut({ id: "zoom", label: "Scroll", action: "zoom around the pointer" }),
  help: shortcut({ id: "help", label: "?", action: "open this shortcut sheet", keys: ["?"] }),
  leftPanel: shortcut({ id: "left-panel", label: "Alt + 1", action: "toggle the source panel", keys: ["1"], alt: true }),
  rightPanel: shortcut({ id: "right-panel", label: "Alt + 2", action: "toggle the tile inspector", keys: ["2"], alt: true }),
  run: shortcut({ id: "run", label: "Ctrl/Cmd + Enter", action: "run edited graph source", keys: ["Enter"], primary: true }),
} as const;

export const SHORTCUT_GROUPS = [
  { title: "Selection", items: [SHORTCUTS.move, SHORTCUTS.moveFast, SHORTCUTS.toggleTile, SHORTCUTS.escape, SHORTCUTS.undo] },
  { title: "View", items: [SHORTCUTS.needs, SHORTCUTS.feeds, SHORTCUTS.fit, SHORTCUTS.scrub, SHORTCUTS.zoom, SHORTCUTS.help] },
  { title: "Panels", items: [SHORTCUTS.leftPanel, SHORTCUTS.rightPanel, SHORTCUTS.run] },
] as const;

type KeyboardLike = Pick<KeyboardEvent, "key" | "altKey" | "ctrlKey" | "metaKey" | "shiftKey">;

export function matchesShortcut(event: KeyboardLike, binding: Shortcut): boolean {
  if (!binding.keys) return false;
  // `key` already folds Shift into a printable character, so an exact match is
  // what keeps Ctrl+Shift+Z — the redo chord this app does not implement — from
  // silently undoing, and what lets `?` bind without claiming which layout
  // produces it. Named keys do not fold, so a binding that wants Shift with one
  // says so and its unshifted base binding still matches.
  if (!binding.keys.includes(event.key)) return false;
  if (binding.alt && !event.altKey) return false;
  if (binding.primary && !(event.ctrlKey || event.metaKey)) return false;
  if (binding.shift && !event.shiftKey) return false;
  return true;
}
