/**
 * Colors for selection boxes and their dependency cones.
 *
 * Hue encodes *which selected box* a highlighted region came from. The three
 * categorical slots below are the only ones used, because these regions are
 * compared all-pairs (any two can land side by side in one grid) and three is
 * the largest set that clears the all-pairs CVD and normal-vision floors on
 * both canvas surfaces — verified with the dataviz validator, all-pairs:
 *
 *   light (surface #f3f4f6): CVD ΔE 9.4, normal ΔE 22.0, contrast >= 3:1 — PASS
 *   dark  (surface #1c1f24): CVD ΔE 9.4, normal ΔE 20.9, contrast >= 3:1 — PASS
 *
 * Separation is measured between the marks, so both ΔE figures are independent
 * of the surface. A surface move needs only the contrast check re-run; a *hue*
 * move needs the whole validator.
 *
 * Aqua is deeper in light than in dark rather than mirroring it. At the
 * original values orange and aqua sat at 2.91:1 and 2.56:1 against the light
 * surface — under the 3:1 floor. That was a relief condition, not a
 * pass, and it could not be discharged the way relief usually is: a cone drawn
 * on a card carries no label, so out there hue *is* the identifier, and the
 * inspector's swatch-plus-index sits on a different surface entirely. Deepening
 * them (scaling linear RGB, which holds chromaticity exactly, so the hue is
 * unchanged) clears the floor and costs nothing elsewhere — the two palettes are
 * independent, and light's CVD separation improved 9.2 -> 9.4 in the process.
 *
 * Orange needs no such split: #e2603a clears the floor on both surfaces
 * (light 3.20:1, dark 4.70:1) and so is the same value in each theme. It
 * replaced a light #e56532 / dark #d95926 pair. The swap does not touch the
 * binding constraint — the all-pairs minimum is blue-vs-aqua in both themes,
 * and orange sits far above it — but it also moves orange further from the
 * dark amber chrome (normal ΔE 22.4 -> 22.8) while raising both surface
 * contrasts. The CVD figures recorded above predate this change and were not
 * regenerated: they come from the dataviz validator, and re-running them needs
 * that script. Nothing measured without it regressed.
 *
 * A 4th hue fails in dark mode (violet↔blue ΔE 1.9), so boxes past the third
 * render in a neutral instead of an invented hue. Identity is never carried by
 * color alone: the inspector lists every box with its index, and hovering a row
 * emphasises that box's cone when per-box attribution is available.
 */

export type RGB = [number, number, number];

const CATEGORICAL_LIGHT: RGB[] = [
  [42, 120, 214], // blue   #2a78d6  4.01:1
  [226, 96, 58], // orange #e2603a  3.20:1
  [24, 159, 111], // aqua   #189f6f  3.06:1
];

const CATEGORICAL_DARK: RGB[] = [
  [57, 135, 229], // blue   #3987e5
  [226, 96, 58], // orange #e2603a
  [25, 158, 112], // aqua   #199e70
];

const NEUTRAL_LIGHT: RGB = [107, 114, 128];
const NEUTRAL_DARK: RGB = [139, 147, 163];

/** Fallback hues when there are too many boxes to attribute individually. */
const AGGREGATE_LIGHT = { upstream: [42, 120, 214] as RGB, downstream: [124, 58, 237] as RGB };
const AGGREGATE_DARK = { upstream: [57, 135, 229] as RGB, downstream: [167, 139, 250] as RGB };

/**
 * The surface a tensor card's canvas is painted on. Exported because the
 * validation figures above are recorded against these exact values, and because
 * `styles.css` must keep `--card` in step: the canvas is drawn by `drawGrid`
 * while the plate around it is drawn by CSS, and a mismatch shows as a seam.
 */
export const CARD_SURFACE = { light: "#f3f4f6", dark: "#1c1f24" } as const;

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
