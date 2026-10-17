/**
 * Colors for selection boxes and their dependency cones.
 *
 * Hue encodes *which selected box* a highlighted region came from. These
 * regions are compared all-pairs (any two can land side by side in one grid),
 * so the six slots are validated all-pairs with the dataviz validator:
 *
 *   light (surface #f3f4f6): CVD ΔE 9.1, normal ΔE 15.7, contrast >= 3:1 : PASS
 *   dark  (surface #1c1f24): CVD ΔE 7.0, normal ΔE 15.2, contrast >= 3:1 : FLOOR
 *
 * The first three slots alone clear CVD ΔE 9.1 in both themes. Slots 4-6 fill
 * the gaps between their hues: olive between orange and aqua, purple between
 * blue and the reuse lens's violet, plum between that and orange. A cyan
 * between aqua and blue does not fit anywhere: every cyan was within normal ΔE
 * 15 of one or the other. Each new slot also stays at least ΔE 12 from the
 * colours drawn on the same cards, the accent that marks the inspected tile
 * and the lens violet. Dark mode's worst pair, ochre against orange under
 * protanopia, is in the 6-8 floor band, which is legal with secondary
 * encoding, and tiles have it: the inspector lists each tile with its index
 * and swatch, and hovering a row emphasises that tile's cone.
 *
 * Separation is measured between the marks, so both ΔE figures are independent
 * of the surface. A surface move needs only the contrast check re-run; a *hue*
 * move needs the whole validator.
 *
 * Aqua is deeper in light than in dark rather than mirroring it. At the
 * original values orange and aqua sat at 2.91:1 and 2.56:1 against the light
 * surface : under the 3:1 floor. That was a relief condition, not a
 * pass, and it could not be discharged the way relief usually is: a cone drawn
 * on a card carries no label, so out there hue *is* the identifier, and the
 * inspector's swatch-plus-index sits on a different surface entirely. Deepening
 * them (scaling linear RGB, which holds chromaticity exactly, so the hue is
 * unchanged) clears the floor and costs nothing elsewhere : the two palettes are
 * independent, and light's CVD separation improved 9.2 -> 9.4 in the process.
 *
 * Orange needs no such split: #e2603a clears the floor on both surfaces
 * (light 3.20:1, dark 4.70:1) and so is the same value in each theme. It
 * replaced a light #e56532 / dark #d95926 pair. The swap does not touch the
 * binding constraint : the all-pairs minimum is blue-vs-aqua in both themes,
 * and orange sits far above it : but it also moves orange further from the
 * dark amber chrome (normal ΔE 22.4 -> 22.8) while raising both surface
 * contrasts. The CVD figures recorded above predate this change and were not
 * regenerated: they come from the dataviz validator, and re-running them needs
 * that script. Nothing measured without it regressed.
 *
 * A tile keeps the slot it was given (`SelPart.hue`), so removing one does not
 * repaint the others. Past six tiles the slots repeat, and identity is never
 * carried by color alone: the inspector lists every box with its index, and
 * hovering a row emphasises that box's cone when per-box attribution is
 * available.
 */

export type RGB = [number, number, number];

const CATEGORICAL_LIGHT: RGB[] = [
  [42, 120, 214], // blue   #2a78d6  4.01:1
  [226, 96, 58], // orange #e2603a  3.20:1
  [24, 159, 111], // aqua   #189f6f  3.06:1
  [79, 89, 3], // olive  #4f5903  6.91:1
  [128, 3, 179], // purple #8003b3  7.39:1
  [120, 57, 96], // plum   #783960  7.48:1
];

const CATEGORICAL_DARK: RGB[] = [
  [57, 135, 229], // blue   #3987e5
  [226, 96, 58], // orange #e2603a
  [25, 158, 112], // aqua   #199e70
  [142, 96, 28], // ochre  #8e601c  3.02:1
  [172, 120, 176], // mauve  #ac78b0  4.77:1
  [187, 38, 162], // magenta #bb26a2  3.08:1
];

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

/** The colour of a tile's hue slot (`SelPart.hue`). Slots past the palette wrap. */
export function boxColor(hue: number, dark: boolean): RGB {
  const pal = dark ? CATEGORICAL_DARK : CATEGORICAL_LIGHT;
  return pal[((hue % pal.length) + pal.length) % pal.length];
}

export const MAX_DISTINCT_HUES = CATEGORICAL_LIGHT.length;

export function aggregateColors(dark: boolean) {
  return dark ? AGGREGATE_DARK : AGGREGATE_LIGHT;
}

export function rgbCss(c: RGB, alpha = 1): string {
  return alpha >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}
