/**
 * The paint stack a tensor card draws, as plain data: cone fills and rulings,
 * entanglement stipples, seeds, previews, and the Plan view's task paint.
 * `components/card/draw-grid.ts` rasterises it.
 */

import { Supply } from "../../core/plan/interfaces";
import { TilePlan } from "../../core/plan/plan";
import { tileBox } from "../../core/plan/tile-family";
import { Box, fromBox, intersect, isEmpty, Region, subtract } from "../../core/region";
import { aggregateColors, boxColor } from "../palette";
import { Layer, PlanPaint } from "./grid";
import { BoxProp, Direction } from "../workspace";

/**
 * Ruling angle per selection box, in degrees counter-clockwise from horizontal.
 *
 * Two boxes whose downstream cones overlap used to draw the same ruling at the
 * same phase, so the later one landed exactly on the earlier and the shared
 * area read as a single region. Hue cannot resolve that : the two hues are
 * painted over each other : so the angle has to. Given its own slope, an
 * overlap crosses itself and says "both of these reach here", which is a fact
 * the panel otherwise only states as two separate rows.
 *
 * The twelve slots match the per-box attribution cap. Every angle stays clear
 * of 0/90 and the 45-degree approximation hatch. The order maximises separation
 * for the common first few boxes; later neutral boxes use the remaining safe
 * slopes, so no two attributable boxes can erase each other exactly.
 */
const STRIPE_ANGLES_DEG = [
  135, 165, 15, 75, 105, 111, 117, 123, 129, 147, 153, 159,
];

/** @internal Pure encoding rule for renderer tests. */
export function stripeAngleDeg(boxIndex: number): number {
  const n = STRIPE_ANGLES_DEG.length;
  return STRIPE_ANGLES_DEG[((Math.trunc(boxIndex) % n) + n) % n];
}

const CONE_ALPHA = 0.72;
const previewAlpha = (depth: number) =>
  (0.22 * CONE_ALPHA) / (1 + 0.35 * Math.max(0, depth - 1));

/** How far a non-focused box's cone fades. Fading, never hiding: the peers have
 * to stay legible or there is nothing to compare the focused one against. */
const PEER_FADE = 0.4;
/** Outline weight on the focused box's cone. */
const EMPHASIS_LINE_PX = 2.7;
/** Placeholder until supplied-share is quantitative. Direction owns the ruling's
 * slope; a later analysis may vary this full 0–1 channel, which the renderer
 * spends on the spacing between lines rather than on their weight. */
export const DEFAULT_DOWNSTREAM_DENSITY = 0.46;

export const downstreamPattern = (boxIndex: number): NonNullable<Layer["pattern"]> => ({
  kind: "stripe",
  density: DEFAULT_DOWNSTREAM_DENSITY,
  angle: stripeAngleDeg(boxIndex),
});

/** Everything the canvas needs to decide what to paint, as plain data. */
/** @internal Input contract for the directly tested layer builder. */
export type LayerInputs = {
  tensorId: string;
  dark: boolean;
  direction: Direction;
  isSelected: boolean;
  /**
   * The selection parts drawn on *this* tensor, each carrying the index it has
   * in the whole workspace. Parts on other tensors are not this card's business
   * to draw, but the index is: hue, focus and visibility are all keyed by it,
   * so a tile must keep the same colour whichever card it lives on.
   */
  parts: { index: number; box: Box }[];
  /** Parts in the whole workspace, so the rubber band previews the next hue. */
  partCount: number;
  perBox: BoxProp[] | null;
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  back?: { region: Region; depth: number };
  fwd?: { region: Region; depth: number };
  prev?: { region: Region; depth: number };
  prevForward?: { region: Region; depth: number };
  /** The in-progress rubber band, already in element space. */
  dragRegion: Region | null;
  /** Regions on this tensor that the selection is combined with, by part index. */
  entangled?: { index: number; region: Region }[];
  showEntangled?: boolean;
};

/**
 * Decide the paint stack for one tensor. Pure so the encoding rules - which
 * cone is filled, which is outlined, what fades, what is hidden - can be
 * asserted directly instead of inferred from pixels.
 */
/** @internal Pure rendering seam exported for deterministic canvas tests. */
export function buildLayers({
  tensorId,
  dark,
  direction,
  isSelected,
  parts,
  partCount,
  perBox,
  hiddenBoxes,
  focusedBox,
  back,
  fwd,
  prev,
  prevForward,
  dragRegion,
  entangled,
  showEntangled,
}: LayerInputs): Layer[] {
  const layers: Layer[] = [];
  const agg = aggregateColors(dark);
  // Both cones are always analysed; `direction` decides which are painted. A
  // Direction filtering happens at paint time; per-tile visibility is shared
  // with the inspector so the picture and merged numbers describe the same set.
  const showBack = direction === "backward" || direction === "both";
  const showFwd = direction === "forward" || direction === "both";

  /**
   * The solid same-hue fill this tile has already put on this card, which a
   * stipple in that hue would disappear into.
   *
   * Only the needs cone qualifies. The feeds cone is ruled, so dots read fine
   * over it, and the selection rectangle paints later and is opaque whatever
   * colour the dots are. The condition is the cone's own paint condition, not
   * an approximation of it: if the fill is not there, there is nothing to knock
   * out of and the dots should stay in the tile's hue.
   */
  const solidGroundFor = (index: number): Region | null => {
    const tr = perBox?.[index]?.backward?.tensors.get(tensorId);
    if (!showBack || !tr || (isSelected && tr.depth === 0)) return null;
    return tr.region;
  };

  // Transient hover preview, drawn faintly under everything else.
  if (prev && !isSelected && showBack)
    layers.push({ region: prev.region, color: agg.upstream, alpha: previewAlpha(prev.depth), hatch: !prev.region.exact });
  if (prevForward && !isSelected && showFwd)
    layers.push({ region: prevForward.region, color: agg.downstream, alpha: previewAlpha(prevForward.depth), hatch: !prevForward.region.exact, pattern: downstreamPattern(0) });

  if (perBox) {
    // Hue identifies which selected box produced this region.
    // Focusing a box fades its peers rather than hiding them: this feature
    // exists to compare cones, and blanking every other cone destroys the
    // comparison being made. Only the explicit visibility toggle removes paint.
    const plain: Layer[] = [];
    const emphasised: Layer[] = [];
    perBox.forEach((bp, i) => {
      if (hiddenBoxes.has(i)) return;
      const emph = focusedBox === i;
      const alphaScale = focusedBox !== null && !emph ? PEER_FADE : 1;
      const color = boxColor(i, dark);
      const bTr = bp.backward?.tensors.get(tensorId);
      const fTr = bp.forward?.tensors.get(tensorId);
      const into = emph ? emphasised : plain;
      // Hue means "which box", so direction lives in fill geometry in every
      // view: required input is uniform; downstream reach is ruled.
      if (showBack && bTr && !(isSelected && bTr.depth === 0))
        into.push({
          region: bTr.region,
          color,
          // Persistent cone alpha means hidden-axis coverage in both directions;
          // graph distance is stated exactly by the inspector's dN badge.
          alpha: CONE_ALPHA * alphaScale,
          hatch: !bTr.region.exact,
          outline: emph,
          lineWidth: emph ? EMPHASIS_LINE_PX : undefined,
        });
      if (showFwd && fTr && !(isSelected && fTr.depth === 0))
        into.push({
          region: fTr.region,
          color,
          // Distance is already stated exactly as dN in the inspector. Keeping
          // it out of downstream alpha leaves alpha to hidden-axis coverage and
          // density to the future supplied-share measurement.
          alpha: CONE_ALPHA * alphaScale,
          hatch: !fTr.region.exact,
          // Angle follows the box index, like the hue does. Where two boxes
          // reach the same elements the rulings cross instead of the later one
          // hiding the earlier, which is the only place that overlap is visible.
          pattern: downstreamPattern(i),
        });
    });
    // The emphasised cone draws last so it sits over its faded peers. Scoped
    // to this group on purpose: sorting all layers would also lift it over the
    // selection outline, which is meant to stay on top.
    layers.push(...plain, ...emphasised);
  } else {
    // Too many boxes to attribute: fall back to one hue per direction.
    if (showBack && back && !(isSelected && back.depth === 0))
      layers.push({ region: back.region, color: agg.upstream, alpha: CONE_ALPHA, hatch: !back.region.exact });
    if (showFwd && fwd && !(isSelected && fwd.depth === 0))
      layers.push({
        region: fwd.region,
        color: agg.downstream,
        alpha: CONE_ALPHA,
        hatch: !fwd.region.exact,
        // One merged cone, so there is nothing to tell apart: the base slope.
        pattern: downstreamPattern(0),
      });
  }

  /* Entanglement is orthogonal to the cone, so it paints whichever direction is
     shown, and when neither is: what a tile reads and what it is multiplied
     against are separate questions, and a reader may want either alone. Hue
     still follows the part index, and a hidden part is hidden here too - one
     visibility control, not two. */
  if (showEntangled && entangled)
    for (const { index, region } of entangled) {
      if (hiddenBoxes.has(index)) continue;
      const alphaScale = focusedBox !== null && focusedBox !== index ? PEER_FADE : 1;
      const color = boxColor(index, dark);
      const alpha = CONE_ALPHA * alphaScale;
      /* Dots in the tile's hue vanish on a solid fill of that same hue, and
         that overlap is the headline case rather than an edge one: in
         `matmul(A, A)` both operands are the one tensor, so what the tile is
         combined with lands inside what it reads. Split the region against the
         ground it will be drawn on and knock the covered part out in the card
         surface, exactly as the approximation hatch already does on a solid
         fill. Hue where there is bare surface under it, surface where there is
         not — the mark stays legible either way, and "third relation, third
         texture" stays true where it matters most. */
      const solid = solidGroundFor(index);
      const over = solid ? intersect(region, solid) : null;
      const bare = solid ? subtract(region, solid) : region;
      if (over && !isEmpty(over))
        layers.push({
          region: over,
          color,
          knockout: true,
          alpha,
          hatch: !region.exact,
          pattern: { kind: "stipple", density: 0.5 },
        });
      if (!isEmpty(bare))
        layers.push({
          region: bare,
          color,
          alpha,
          hatch: !region.exact,
          pattern: { kind: "stipple", density: 0.5 },
        });
    }

  // The selection itself: each box in its own hue, dimmed when another is
  // focused. A hidden box keeps its rectangle - hiding removes the *cone*, and
  // a probe you cannot see is a probe you cannot move back.
  parts.forEach(({ index, box: b }) => {
    const isFocused = focusedBox === null || focusedBox === index;
    const hidden = hiddenBoxes.has(index);
    layers.push({
      region: fromBox(b),
      color: boxColor(index, dark),
      alpha: hidden ? 0.18 : isFocused ? 0.9 : 0.35,
      hatch: false,
      outline: isFocused && !hidden,
      seed: true,
    });
  });

  // The in-progress rubber band, drawn on top of everything it will replace.
  if (dragRegion)
    layers.push({
      region: dragRegion,
      color: boxColor(partCount, dark),
      alpha: 0.4,
      hatch: false,
      outline: true,
    });

  return layers;
}

/**
 * The Plan view's paint for one card.
 *
 * The meanings are the ones the cards already use. The inspected task's tile
 * is drawn like a placed tile: outline and corner marks in the first hue. What
 * it reads on this tensor is the solid needs fill in that hue, one layer per
 * tensor so two slots reading one element do not paint it twice, with the
 * approximation hatch when the demand is widened. The lattice is the plan's
 * tiling, and the producer tiles the task needs are outlined in neutral ink.
 */
/** @internal Pure rendering seam exported for deterministic plan-view tests. */
export function buildPlanPaint({
  tensorId,
  rowAxis,
  colAxis,
  dark,
  plan,
  supply,
  proposed,
  pointer,
}: {
  tensorId: string;
  rowAxis: number;
  colAxis: number;
  dark: boolean;
  plan: TilePlan | null;
  supply: Supply | null;
  /** Extents a click would divide this tensor at, while the plan does not tile it. */
  proposed?: number[] | null;
  /** The tile under the pointer, or the band being dragged out. */
  pointer?: Region | null;
}): { layers: Layer[]; paint: PlanPaint } {
  const family = plan?.families.get(tensorId);
  const hue = boxColor(0, dark);
  const layers: Layer[] = [];

  const demand = supply?.demand.filter((d) => d.tensorId === tensorId) ?? [];
  if (demand.length) {
    const region: Region = {
      boxes: demand.flatMap((d) => d.region.boxes),
      exact: demand.every((d) => d.region.exact),
      reasons: [...new Set(demand.flatMap((d) => d.region.reasons))],
    };
    layers.push({ region, color: hue, alpha: CONE_ALPHA, hatch: !region.exact });
  }
  if (family && supply?.task.tensorId === tensorId)
    layers.push({
      region: fromBox(tileBox(family, supply.task.coord)),
      color: hue,
      alpha: 0.35,
      hatch: false,
      outline: true,
      seed: true,
    });

  const tiles = family
    ? (supply?.producers ?? [])
        .filter((p) => p.task.tensorId === tensorId)
        .map((p) => ({ box: tileBox(family, p.task.coord), definite: p.definite }))
    : [];
  // What the gesture would take, outlined in the task hue: it is about to
  // become the inspected tile, not a producer of one.
  if (pointer) layers.push({ region: pointer, color: hue, alpha: 0.2, hatch: false, outline: true });

  const extents = family ? family.tile : proposed;
  const lattice = extents
    ? {
        rows: rowAxis >= 0 ? extents[rowAxis] : 1,
        cols: colAxis >= 0 ? extents[colAxis] : 1,
        ...(family ? {} : { proposed: true }),
      }
    : null;
  return { layers, paint: { lattice, tiles } };
}
