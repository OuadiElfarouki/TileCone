/**
 * Colors for selection boxes and their dependency cones.
 *
 * Hue encodes *which selected box* a highlighted region came from. The three
 * categorical slots below are the only ones used, because these regions are
 * compared all-pairs (any two can land side by side in one grid) and three is
 * the largest set that clears the all-pairs CVD and normal-vision floors on
 * both canvas surfaces — verified with the dataviz validator:
 *
 *   light (surface #f3f4f6): CVD ΔE 9.2, normal ΔE 24.0 — PASS
 *   dark  (surface #16181d): CVD ΔE 9.4, normal ΔE 20.9 — PASS
 *
 * A 4th hue fails in dark mode (violet↔blue ΔE 1.9), so boxes past the third
 * render in a neutral instead of an invented hue. Identity is never carried by
 * color alone: the inspector lists every box with its index, and hovering a row
 * isolates that box's cone regardless of how many boxes there are.
 */

export type RGB = [number, number, number];

const CATEGORICAL_LIGHT: RGB[] = [
  [42, 120, 214], // blue   #2a78d6
  [235, 104, 52], // orange #eb6834
  [27, 175, 122], // aqua   #1baf7a
];

const CATEGORICAL_DARK: RGB[] = [
  [57, 135, 229], // blue   #3987e5
  [217, 89, 38], // orange #d95926
  [25, 158, 112], // aqua   #199e70
];

const NEUTRAL_LIGHT: RGB = [107, 114, 128];
const NEUTRAL_DARK: RGB = [139, 147, 163];

/** Fallback hues when there are too many boxes to attribute individually. */
const AGGREGATE_LIGHT = { upstream: [42, 120, 214] as RGB, downstream: [124, 58, 237] as RGB };
const AGGREGATE_DARK = { upstream: [57, 135, 229] as RGB, downstream: [167, 139, 250] as RGB };

export function isDarkTheme(): boolean {
  const t = document.documentElement.dataset.theme;
  if (t === "light") return false;
  if (t === "dark") return true;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function boxColor(index: number, dark: boolean): RGB {
  const pal = dark ? CATEGORICAL_DARK : CATEGORICAL_LIGHT;
  if (index < pal.length) return pal[index];
  return dark ? NEUTRAL_DARK : NEUTRAL_LIGHT;
}

export const MAX_DISTINCT_HUES = CATEGORICAL_LIGHT.length;

export function aggregateColors(dark: boolean) {
  return dark ? AGGREGATE_DARK : AGGREGATE_LIGHT;
}

export function rgbCss(c: RGB, alpha = 1): string {
  return alpha >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}
