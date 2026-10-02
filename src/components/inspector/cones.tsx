/** The Backward and Forward Cone sections: one row per tensor, with its footprint. */

import React from "react";
import { Contribution } from "../../core/contribution";
import { sumFigures, TensorReadout } from "../../core/metrics";
import { count, Region, subtract, union } from "../../core/region";
import { fmt, formatBytes, formatFigure } from "../../view/format";
import { boxColor, rgbCss } from "../../view/palette";
import { SHORTCUTS } from "../../view/shortcuts";
import { ConeDirection, MAX_PER_BOX_PROPS } from "../../view/workspace";
import { useStore } from "../../state/store";
import { CopyButton } from "../chrome/CopyButton";

/** Past the cap there is no per-tile propagation left to read, and three places
 *  used to say so in three different ways. */
export const MERGED_AT_CAP = `over ${MAX_PER_BOX_PROPS} tiles: traced as one merged region, not per tile`;

/** Both cones report an empty selection the same way. */
export const NO_ENABLED_TILES = "No tiles are enabled, include one above to analyse it.";

export const EMPTY_REGION: Region = { boxes: [], exact: true, reasons: [] };

/**
 * How a readout is attributed to individual tiles: the per-tile propagation
 * when there is one, which tiles are switched off, which one is focused, and
 * the theme the tile hues are mixed for. The footprint bar, the row around it
 * and the section around that all need the same four, so they travel together
 * rather than being restated at every level.
 */
export type TileAttribution = {
  perBox: ReturnType<typeof useStore.getState>["perBox"];
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  dark: boolean;
};

export function FootprintBar({
  tensorId,
  direction,
  elements,
  totalElements,
  attr: { perBox, hiddenBoxes, focusedBox, dark },
}: {
  tensorId: string;
  direction: ConeDirection;
  elements: number;
  totalElements: number;
  attr: TileAttribution;
}): React.ReactElement {
  if (totalElements <= 0) return <span className="footprint-bar" />;

  const regions = perBox?.map(
    (prop, index) => hiddenBoxes.has(index)
      ? EMPTY_REGION
      : prop[direction]?.tensors.get(tensorId)?.region ?? EMPTY_REGION
  );
  const segments: { elements: number; color: string; title: string; shared?: boolean }[] = [];

  if (regions && focusedBox !== null && regions[focusedBox]) {
    segments.push({
      elements,
      color: rgbCss(boxColor(focusedBox, dark)),
      title: `tile ${focusedBox + 1}: ${fmt(elements)} elements`,
    });
  } else if (regions) {
    let exclusiveTotal = 0;
    regions.forEach((region, index) => {
      let others = EMPTY_REGION;
      regions.forEach((other, otherIndex) => {
        if (otherIndex !== index) others = union(others, other);
      });
      const exclusive = count(subtract(region, others));
      exclusiveTotal += exclusive;
      if (exclusive > 0)
        segments.push({
          elements: exclusive,
          color: rgbCss(boxColor(index, dark)),
          title: `tile ${index + 1} only: ${fmt(exclusive)} elements`,
        });
    });
    const shared = Math.max(0, elements - exclusiveTotal);
    if (shared > 0)
      segments.push({
        elements: shared,
        color: "",
        title: `shared by multiple tiles: ${fmt(shared)} elements`,
        shared: true,
      });
  } else if (elements > 0) {
    segments.push({ elements, color: "var(--muted)", title: `${fmt(elements)} elements touched` });
  }

  return (
    <span
      className="footprint-bar"
      title={`whole bar = ${fmt(totalElements)} tensor elements`}
      aria-label={`${fmt(elements)} of ${fmt(totalElements)} elements touched`}
    >
      {segments.map((segment, index) => (
        <i
          key={index}
          className={segment.shared ? "shared" : ""}
          title={segment.title}
          style={{
            width: `${Math.min(100, (segment.elements / totalElements) * 100)}%`,
            background: segment.color || undefined,
          }}
        />
      ))}
    </span>
  );
}

/** Slice expressions minus the over-approximation comment the readout appends. */
export const sliceLines = (row: TensorReadout) =>
  row.sliceExprs.filter((line) => !line.startsWith("#"));

/**
 * One tensor in one direction: how much of it the tile touches, why it has that
 * shape, and -downstream- whether the tile finishes it or only feeds it.
 */
export function ConeRow({
  row,
  direction,
  hue,
  flags,
  contribution,
  attr,
}: {
  row: TensorReadout;
  direction: ConeDirection;
  hue: string;
  flags: string[];
  contribution?: Contribution;
  attr: TileAttribution;
}): React.ReactElement {
  const exprs = sliceLines(row);
  const share = row.totalElements > 0 ? (row.elements / row.totalElements) * 100 : 0;

  return (
    <div className="cone-row">
      <div className="cone-row-head">
        <b>{row.name}</b>
        {/* One box is the common case and reads best inline. More than one is
            exactly when a single truncated expression would hide the answer, so
            those move to their own block below. */}
        {exprs.length === 1 && (
          <code className="cone-expr">{exprs[0].slice(row.name.length)}</code>
        )}
        {!row.exact && <span className="badge approx" title={row.reasons.join("; ")}>≈</span>}
        {row.isInput && <span className="badge input">in</span>}
        <span className="badge depth" title={`${row.depth} step${row.depth === 1 ? "" : "s"} along this direction`}>
          d{row.depth}
        </span>
        <span className="row-stats">
          {share.toFixed(1)}% · {formatFigure(row.byteFigure, formatBytes)} · {row.boxCount} box{row.boxCount === 1 ? "" : "es"}
          {/* Boxes may overlap - two operand slots reading one tensor give two
              bands sharing a corner. Without this the listed boxes visibly sum
              past the element total and the row looks like it is miscounting,
              when in fact the total is the union and the shared elements are
              read twice. Stated only when there are some. */}
          {row.overlap > 0 && (
            <span
              className="row-shared"
              title={`${row.overlap.toLocaleString()} element${row.overlap === 1 ? "" : "s"} lie in more than one box, counted once in the total`}
            >
              {" "}· {row.overlap.toLocaleString()} shared
            </span>
          )}
        </span>
      </div>
      {attr.perBox ? (
        <FootprintBar
          tensorId={row.tensorId}
          direction={direction}
          elements={row.elements}
          totalElements={row.totalElements}
          attr={attr}
        />
      ) : (
        <span
          className="cone-bar"
          title={`${fmt(row.elements)} of ${fmt(row.totalElements)} elements`}
          aria-label={`${share.toFixed(1)} percent of ${row.name}`}
        >
          <i style={{ width: `${Math.min(100, share)}%`, background: hue }} />
        </span>
      )}
      {flags.map((flag) => (
        <span className="cone-flag" key={flag}>
          {flag}
        </span>
      ))}
      {contribution?.partial && (
        <span className="cone-flag">
          partial : {contribution.detail}
          {!contribution.exact && " (from an over-approximated region)"}
        </span>
      )}
      {exprs.length > 1 && (
        <div className="slice-exprs">
          {exprs.slice(0, 4).map((line, index) => (
            <code key={index}>{line}</code>
          ))}
          {exprs.length > 4 && (
            <code className="muted"># … {exprs.length - 4} more boxes</code>
          )}
        </div>
      )}
      <CopyButton
        className="mini copy-exprs"
        title="copy this tensor's slice expressions"
        text={() => exprs.join("\n")}
        label="copy"
      />
    </div>
  );
}

/* One direction, one identity: the heading, the glyph, the paint and the key
   that toggles it all follow from `direction`, so a call site names only the
   direction. The shortcut letters come from the manifest that binds them. */

export const CONE = {
  backward: { title: "Backward Cone", arrow: "↑", paint: "needs", key: SHORTCUTS.needs },
  forward: { title: "Forward Cone", arrow: "↓", paint: "feeds", key: SHORTCUTS.feeds },
} as const satisfies Record<ConeDirection, unknown>;

/* The two classes of question the panel answers, and the promise each one
   makes. A figure under Dependencies is a function of the graph and the drawn
   region - exact, or a bound that names why. A figure under Cost model exists
   only once an execution is assumed, so it is modelled and could be wrong in
   either direction; mixing the two in one column would lend the models the
   others' credibility. */

export function ConeSection({
  direction,
  rows,
  hue,
  flags,
  contributions: contrib,
  enabled,
  onToggle,
  empty,
  attr,
}: {
  direction: ConeDirection;
  rows: TensorReadout[];
  hue: string;
  flags: Map<string, string[]>;
  contributions?: Map<string, Contribution>;
  enabled: boolean;
  onToggle: () => void;
  empty: string;
  attr: TileAttribution;
}): React.ReactElement {
  const { title, arrow, paint, key } = CONE[direction];
  const bytes = sumFigures(rows.map((row) => row.byteFigure));
  const verdicts = contrib ? [...contrib.values()] : [];
  const partial = verdicts.filter((item) => item.partial).length;
  const completed = verdicts.length - partial;

  return (
    <section className="cone-section">
      <div className="cone-head">
        <button
          className={`cone-toggle${enabled ? " on" : ""}`}
          onClick={onToggle}
          aria-pressed={enabled}
          aria-expanded={enabled}
          title={`${enabled ? "hide" : "show"} ${title.toLowerCase()} (${key.keys[0]})`}
        >
          <span className={`cone-key ${paint}`} aria-hidden />
          <span className="cone-arrow" aria-hidden>{arrow}</span>
          <span>{title}</span>
        </button>
        <span className="rollup">
          {rows.length} tensor{rows.length === 1 ? "" : "s"}
          {verdicts.length > 0 && ` · ${completed} completed, ${partial} partial`}
          {` · ${formatFigure(bytes, formatBytes)}`}
        </span>
      </div>
      {enabled &&
        (rows.length ? (
          rows.map((row) => (
            <ConeRow
              key={row.tensorId}
              row={row}
              direction={direction}
              hue={hue}
              flags={flags.get(row.tensorId) ?? []}
              contribution={contrib?.get(row.tensorId)}
              attr={attr}
            />
          ))
        ) : (
          <p className="hint">{empty}</p>
        ))}
    </section>
  );
}
