import React, { useMemo } from "react";
import { axisName } from "../../view/tensor/shape-label";
import { MAX_CONTRIBUTION_PROBES } from "../../core/contribution";
import { ratioFigure, sumFigures } from "../../core/metrics";
import { formatBoxIndices } from "../../core/region";
import { fmt, formatBytes, formatFigure, formatIntensity } from "../../view/format";
import { demandDetail, demandSummary } from "../../view/demand";
import { aggregateColors, boxColor, rgbCss } from "../../view/palette";
import { neighbourShares, reuseFigures } from "../../view/reuse-rows";
import { SHORTCUTS } from "../../view/shortcuts";
import { MAX_PER_BOX_PROPS, partsOn } from "../../view/workspace";
import { useDark, useStore } from "../../state/store";
import { AxisEditor } from "./AxisEditor";
import { PlanPanel } from "./PlanPanel";
import { ConeSection, MERGED_AT_CAP, NO_ENABLED_TILES, sliceLines, TileAttribution } from "./cones";
import {
  analysisTensorId,
  type ConeCost,
  groupAttribution,
  groupFocus,
  useInspectorAnalysis,
} from "./inspector-analysis";
import { DependencyNotes } from "./notes";
import { EmptyPanel, InspectorTabs } from "./tabs";
import { RegionEditor, TileIdentity } from "./tiles";
import { useReuseSweep } from "./useReuseSweep";

export function Inspector(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const dark = useDark();
  const selection = useStore((s) => s.selection);
  const byTensorRes = useStore((s) => s.byTensorRes);
  const direction = useStore((s) => s.direction);
  const toggleDirection = useStore((s) => s.toggleDirection);
  const showEntangled = useStore((s) => s.showEntangled);
  const toggleEntangled = useStore((s) => s.toggleEntangled);
  const entangled = useStore((s) => s.entangled);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const tab = useStore((s) => s.inspectorTab);
  const tileFocus = useStore((s) => s.focusedBox);
  /**
   * The tile group everything below the tiles list describes, and the focus
   * that is allowed to narrow it.
   *
   * The group is store state named by a draw, a pin, or the group header, so a
   * tile drawn on another tensor moves the readout with it. Focus narrows
   * within that group and stops there: `groupFocus` is null while the pointer
   * is over another group's row, which leaves that row lit on the canvas
   * without re-scoping the panel under it.
   */
  const selectedGroup = useStore((s) => s.analysisGroup);
  // Above the attribution cap, tile hover must not masquerade as a single-tile readout.
  const rawFocus = perBox ? tileFocus : null;
  // Stable across renders, so the memos below key off the selection itself
  // rather than off a fresh empty array.
  const parts = useMemo(() => selection?.parts ?? [], [selection]);
  const activeTensorId = analysisTensorId(parts, selectedGroup);
  /** Everything below the tiles list reads the narrowed focus, never the raw
   *  one, so a hover outside the group cannot reach any of it. */
  const focusedBox = groupFocus(parts, rawFocus, activeTensorId);
  const selectGroup = useStore((s) => s.selectAnalysisGroup);
  /** The tile the axis table describes: the focused one, else the last drawn in
   *  the group, which is also the one the arrow keys move. */
  const axisIndex = focusedBox ?? (activeTensorId
    ? parts.reduce<number | null>((last, part, i) => part.tensorId === activeTensorId ? i : last, null)
    : null);

  const { reuseProbe, reuse, reusePending, reuseError, visiblePlayback, computeReuse } =
    useReuseSweep({ resolved, selection, selectedGroup, rawFocus, hiddenBoxes, direction, showEntangled, tab });

  const {
    metrics,
    bounds,
    sharing,
    findings,
    seeds,
    contribution: contrib,
    upstream,
    downstream,
  } = useInspectorAnalysis({
    resolved,
    selection,
    perBox,
    byTensorRes,
    hiddenBoxes,
    focusedBox,
    activeTensorId,
    direction,
  });

  if (!resolved) return <aside className="inspector" aria-label="Tile inspector" />;

  /**
   * Everything below the tiles list is counted over the active group, not over
   * the whole selection: a tile on another tensor is not part of this answer
   * and must not appear in its totals or its "N tiles" wording.
   */
  const groupParts = activeTensorId ? partsOn(selection, activeTensorId) : [];
  const selBoxes = groupParts.length;
  const enabledBoxes = groupParts.filter((part) => !hiddenBoxes.has(part.index)).length;
  /** What the cone sections need to paint a readout per tile, in one piece. */
  const attribution: TileAttribution = {
    perBox: groupAttribution(perBox, parts, activeTensorId),
    hiddenBoxes,
    focusedBox,
    dark,
  };
  /** Past the cap the cones are still correct, but merged rather than attributed.
   *  The cap counts the whole selection, because that is what `perBox` tracks. */
  const merged = !perBox;
  const aggregate = aggregateColors(dark);
  /** A bar carries the hue of the tile it belongs to, or the aggregate when the
   *  rows describe several tiles at once. */
  const coneHue = (cone: "upstream" | "downstream") => {
    if (focusedBox !== null) return rgbCss(boxColor(focusedBox, dark));
    if (selBoxes === 1 && !merged) return rgbCss(boxColor(groupParts[0].index, dark));
    return rgbCss(cone === "upstream" ? aggregate.upstream : aggregate.downstream);
  };

  // Both questions keep their heading in the inspector; the filter collapses
  // its rows and canvas paint. The store refuses to collapse the final one.
  const showUpstream = direction === "backward" || direction === "both";
  const showDownstream = direction === "forward" || direction === "both";

  // An empty section has two causes, and they are different findings: the
  // selection sits at the edge of the graph, or what it reaches is selected too
  // and is therefore not something it reads or feeds.
  const seedIds = [...seeds.keys()];
  const upstreamEmpty = enabledBoxes === 0
    ? NO_ENABLED_TILES
    : seedIds.some((id) => resolved.tensors[id].producer)
      ? "Everything the selection needs is itself selected."
      : "Every selected tensor is a graph input, it needs nothing earlier.";
  const downstreamEmpty = enabledBoxes === 0
    ? NO_ENABLED_TILES
    : seedIds.some((id) => resolved.consumers[id]?.length)
      ? "Everything the selection feeds is itself selected."
      : "Nothing consumes this selection, it feeds no later tensor.";

  const visibleRows = [...(showUpstream ? upstream : []), ...(showDownstream ? downstream : [])];
  const approxRows = visibleRows.filter((row) => !row.exact).length;
  const approxReasons = [
    ...new Set(visibleRows.filter((row) => !row.exact).flatMap((row) => row.reasons)),
  ];
  /* One row per entangled region, skipping tiles the user hid so the list and
     the paint describe the same set. */
  const entangledRows = (entangled ?? []).flatMap((forPart, index) =>
    parts[index]?.tensorId !== activeTensorId || hiddenBoxes.has(index) || (focusedBox !== null && focusedBox !== index)
      ? []
      : forPart.map((e) => ({
          name: resolved?.tensors[e.tensorId]?.name ?? e.tensorId,
          op: e.op,
          region: e.region,
        }))
  );

  /**
   * What the Cost figures actually cover, named in the heading.
   *
   * Merged is the common case and it was the one the heading got wrong: with
   * several tiles and no focus every figure describes their *union*, and the
   * heading still read "this tile". Past the attribution cap `hiddenBoxes` is
   * not honoured either (`enabledPropResult` returns the whole aggregate), so
   * the count has to come from what was measured rather than from what is
   * switched on.
   */
  const measuredBoxes = perBox ? enabledBoxes : selBoxes;
  const costScope =
    focusedBox !== null && perBox
      ? `tile ${focusedBox + 1}`
      : measuredBoxes > 1
        ? `${measuredBoxes === selBoxes ? measuredBoxes : `${measuredBoxes} of ${selBoxes}`} tiles · merged`
        : "this tile";

  /* Each figure now carries its own status, so its mark comes from the figure
     rather than from whether anything in the readout was approximate. One
     widened weight used to put `\u2264` on an output-byte count that was exact, and
     - worse - the same `\u2264` on a FLOP total spanning a barrier, which is not a
     bound in that direction or any other. `formatFigure` writes what each one
     has earned, including the word `unknown` where there is no number. */

  /**
   * A scenario's intensity, or the reason there isn't one.
   *
   * Built through `ratioFigure` so the quotient inherits what its parts know:
   * `~` where both moved over a widened region, and `unknown` where the
   * numerator crossed an operation nobody described - a ratio of an unknown
   * quantity is not a looser ratio, it is not a ratio.
   */
  const intensityOf = (cost: ConeCost | null | undefined): string => {
    if (!cost) return "—";
    const ratio = ratioFigure(cost.flops, cost.bytes);
    if (ratio.value === null) return "unknown";
    if (cost.bytes.value === 0) return "—";
    return formatIntensity(ratio);
  };

  /* Summed through the figure algebra rather than by adding three numbers, so
     the total inherits the weakest of the three claims instead of presenting
     itself as whatever the last one was. */
  const workingSet = metrics
    ? sumFigures([metrics.inputBytes, metrics.intermediateBytes, metrics.outputBytes])
    : null;

  /** Reuse neighbours step along the anchor tensor's axes, so it names them. */
  const activeTensor = activeTensorId ? resolved.tensors[activeTensorId] : undefined;
  const axisLabelOf = (axis: number) => axisName(activeTensor ?? {}, axis);

  return (
    <aside className="inspector" aria-label="Tile inspector">
      <div className="inspector-scroll">
        {/* The Plan view describes tasks rather than drawn tiles, so it is
            reachable with nothing drawn, and the switch sits above the empty
            panel because it is the only route to it. */}
        {tab === "plan" ? (
          <>
            <InspectorTabs />
            <PlanPanel />
          </>
        ) : !selection ? (
          <>
            <InspectorTabs />
            <EmptyPanel />
          </>
        ) : (
          <>
            <TileIdentity
              activeTensorId={activeTensorId}
              focusedBox={focusedBox}
              coneExprs={() =>
                [...upstream, ...downstream].flatMap((row) => sliceLines(row)).join("\n")
              }
            />
            {/* The tile header sits above the strip: both classes of question
                are about the same tile, and it is the tiles list below that
                picks which one. */}
            <InspectorTabs />
            {tab === "dependencies" ? (
              <div
                className="ins-tabpanel"
                role="region"
                id="ins-panel-dependencies"
                aria-labelledby="ins-tab-dependencies"
              >
            {/* The tiles list is the selector for everything below it: it picks
                which cone the two sections describe, so it sits above them. */}
            <RegionEditor activeTensorId={activeTensorId} onSelectGroup={selectGroup} />
            {axisIndex !== null && <AxisEditor index={axisIndex} />}

            <ConeSection
              direction="backward"
              rows={upstream}
              hue={coneHue("upstream")}
              flags={findings?.flags ?? new Map()}
              enabled={showUpstream}
              onToggle={() => toggleDirection("backward")}
              empty={upstreamEmpty}
              attr={attribution}
            />
            <ConeSection
              direction="forward"
              rows={downstream}
              hue={coneHue("downstream")}
              flags={new Map()}
              contributions={contrib?.byTensor}
              enabled={showDownstream}
              onToggle={() => toggleDirection("forward")}
              empty={downstreamEmpty}
              attr={attribution}
            />
            {direction === "none" && (
              <p className="view-mode-note" role="status">
                Figures only · Backward & Forward cones are hidden.
              </p>
            )}
            {showDownstream && contrib?.capped && (
              <p className="hint">
                more than {MAX_CONTRIBUTION_PROBES} tensors are fed: the rows above do not
                say whether this tile completes them or only feeds them
              </p>
            )}

            {/* A third relation, so a third section rather than rows folded into
                one of the cones: "what this tile is multiplied against" is not
                a hop along the graph and does not belong under a direction. */}
            <section className="cone-section">
              <div className="cone-head">
                <button
                  className={`cone-toggle${showEntangled ? " on" : ""}`}
                  onClick={toggleEntangled}
                  aria-pressed={showEntangled}
                  aria-expanded={showEntangled}
                  title={`${showEntangled ? "hide" : "show"} the co-access surface (${SHORTCUTS.entangled.keys[0]})`}
                >
                  <span className="cone-key combined" aria-hidden="true" />
                  Co-access Surface
                </button>
              </div>
              {showEntangled &&
                (merged ? (
                  <p className="hint">{MERGED_AT_CAP}</p>
                ) : entangledRows.length ? (
                  <ul className="ent-list">
                    {entangledRows.map((row, i) => (
                      <li key={i}>
                        <code>
                          {row.name}[{formatBoxIndices(row.region.boxes[0])}
                          {row.region.boxes.length > 1 ? ", …" : ""}]
                        </code>
                        <span className="muted"> via {row.op}</span>
                        {!row.region.exact && (
                          <span className="badge approx" title={row.region.reasons.join("; ")}>
                            ≈
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">Nothing meets this tile in a shared term.</p>
                ))}
            </section>

            {approxReasons.length > 0 && (
              /* Contract A1 asks this section to name a *reason*. Counting the
                 badges restated something already on screen and left the
                 reasons in a title, which is the one place a mouse-less reader
                 never reaches. The count stays because it says how far to
                 look; the reasons are now the content. */
              <div className="ins-section warn">
                ≈ {approxRows} row{approxRows === 1 ? "" : "s"} over-approximated:{" "}
                {approxReasons.join(", ")}
              </div>
            )}

            {metrics && (
              <>
                {/* Each figure says for itself what may be read into it: a
                    bare number is a count, `<=` is "no more than", and a figure
                    whose arithmetic crosses an operation nobody described says
                    `unknown` instead of offering a number that is neither a
                    ceiling nor a floor. */}
                <div className="ins-section">
                  <div className="ins-title">
                    {`Cost to compute ${costScope} `}
                    {!metrics.exact && (
                      <span className="badge approx" title={metrics.reasons.join("; ")}>≈</span>
                    )}
                  </div>
                  <div className="kv">
                    <span>FLOPs</span>
                    <span title={metrics.flops.reasons.join("; ")}>
                      {formatFigure(metrics.flops, fmt)}
                    </span>
                    <span>input bytes</span>
                    <span>{formatFigure(metrics.inputBytes, formatBytes)}</span>
                    <span>intermediate</span>
                    <span>{formatFigure(metrics.intermediateBytes, formatBytes)}</span>
                    <span>output bytes</span>
                    <span>{formatFigure(metrics.outputBytes, formatBytes)}</span>
                    <span>working set</span>
                    <span>{workingSet && formatFigure(workingSet, formatBytes)}</span>
                  </div>
                  {metrics.flops.status === "unknown" && (
                    /* The one qualification a reader cannot act on without
                       being told: every other figure here is still a bound,
                       and only the arithmetic is missing. Without this the
                       word `unknown` beside a full set of byte counts reads
                       as a glitch rather than as the answer. */
                    <p className="hint overlap">
                      No FLOP total through {metrics.unknownOperations} unmodelled
                      {metrics.unknownOperations === 1 ? " operation" : " operations"}:
                      its arithmetic is not described, so a sum across it would be
                      neither an upper nor a lower bound. The byte figures still hold.
                    </p>
                  )}
                  {metrics.exact || metrics.flops.status === "unknown" ? null : (
                    <p className="hint overlap">
                      Bounds, not counts: read ≤ as “no more than”, never understated.
                    </p>
                  )}
                </div>
                {/* A function of the tiles on screen: duplicate element demand
                    at graph inputs. It is a shareable footprint, not a claim
                    about cache hits, transactions, or bytes actually loaded.
                    The sweep that asks the same question of tiles nobody drew
                    is modelled, and lives under Cost model. */}
                <div className="ins-section">
                  <div className="ins-title">Shared graph-input demand</div>
                  {sharing && sharing.length > 0 ? (
                    <ul className="reuse-list">
                      {sharing.map((row) => (
                        <li key={row.tensorId}>
                          <code>{resolved.tensors[row.tensorId].name}</code>
                          <span
                            className="muted"
                            title={demandDetail(row, `${row.readers} of ${row.selectedTiles} tiles`)}
                          >
                            {demandSummary(row)}
                          </span>
                          {!row.exact && (
                            <span className="badge approx" title={row.reasons.join("; ")}>≈</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : !perBox && parts.length > MAX_PER_BOX_PROPS ? (
                    <p className="hint">
                      Per-tile sharing is unavailable above {MAX_PER_BOX_PROPS} total tiles.
                    </p>
                  ) : enabledBoxes < 2 || focusedBox !== null ? (
                    <p className="hint">
                      Enable and analyse at least two tiles together to compare their demand.
                    </p>
                  ) : (
                    <p className="hint">
                      These tiles have no graph-input demand.
                    </p>
                  )}
                </div>
              </>
            )}
            <DependencyNotes
              findings={findings}
              hasSelection={enabledBoxes > 0}
              focusedBox={focusedBox}
              attributed={!!perBox}
            />
              </div>
            ) : (
              <div
                className="ins-tabpanel"
                role="region"
                id="ins-panel-execution"
                aria-labelledby="ins-tab-execution"
              >
                <p className="tab-note">
                  Everything here assumes an execution the graph does not fix: a tiling of the
                  whole tensor, an order to run it in, what fuses with what. These are models,
                  not bounds - a wrong assumption moves them in either direction.
                </p>
                {/* Both scenarios assume a fusion decision the graph does not
                    make, which is what moved them off the cost table: the
                    counts there are measured, these are argued. */}
                <div className="ins-section">
                  <div className="ins-title">Arithmetic intensity</div>
                  <div className="kv">
                    <span title={
                      bounds
                        ? `one kernel over every op and every tile: intermediates never reach memory and a shared operand band is fetched once - ${formatFigure(bounds.fused.flops, fmt)} FLOP / ${formatFigure(bounds.fused.bytes, formatBytes)}`
                        : undefined
                    }>
                      fused
                    </span>
                    <span>{intensityOf(bounds?.fused)}</span>
                    <span title={
                      bounds?.unfused
                        ? `every op of every tile as its own job: separate input reads and output writes, and no cross-op cache reuse; reshape, transpose, slice and expand are views that move nothing where the layout allows - ${formatFigure(bounds.unfused.flops, fmt)} FLOP / ${formatFigure(bounds.unfused.bytes, formatBytes)}`
                        : `per-tile cones are not traced past ${MAX_PER_BOX_PROPS} tiles, and the merged result cannot be taken apart again`
                    }>
                      unfused
                    </span>
                    <span>{intensityOf(bounds?.unfused)}</span>
                  </div>
                  {metrics?.flops.status === "unknown" ? (
                    <p className="hint overlap">
                      Unavailable: the numerator is a FLOP total across an operation whose
                      arithmetic is not modelled. A ratio of an unknown quantity is not a
                      looser ratio, it is not one.
                    </p>
                  ) : metrics && !metrics.exact ? (
                    <p className="hint overlap">
                      Both are ratios of figures measured over a widened region, which moves
                      the numerator and the denominator at once: ~ says disturbed in an
                      unknown direction, where ≤ would claim a side.
                    </p>
                  ) : null}
                </div>
                <div className="ins-section">
                  <div className="ins-title with-action">
                    Reuse sweep
                    <button
                      className="mini"
                      onClick={computeReuse}
                      disabled={!reuseProbe || reusePending}
                      title="sample tiles of the selection's size across the anchor tensor and estimate how many demand part of each graph-input footprint"
                    >
                      {reusePending ? "estimating…" : reuse ? "replay" : "estimate"}
                    </button>
                  </div>
                  <p className="hint">
                    Tiles of this tile's size, laid over the whole tensor: how many of them demand
                    part of the same graph-input footprint.
                  </p>
                  {visiblePlayback?.phase === "playing" && (
                    <p className="hint">
                      Probe {visiblePlayback.visited} of {visiblePlayback.frames.length}
                    </p>
                  )}
                  {reuseError ? (
                    <p className="hint overlap">Reuse estimate failed: {reuseError}</p>
                  ) : reusePending ? (
                    <p className="hint">Evaluating sampled tiles off the UI thread…</p>
                  ) : !reuse ? (
                    <p className="hint">sampled sweep · run on demand</p>
                  ) : reuse.length === 0 ? (
                    <p className="hint">This tile has no graph-input demand.</p>
                  ) : (
                    <ul className="reuse-list">
                      {reuse.map((estimate) => {
                        const neighbours = neighbourShares(estimate.neighbors, axisLabelOf);
                        const figures = reuseFigures(estimate);
                        return (
                          <li key={estimate.tensorId}>
                            <code>{resolved.tensors[estimate.tensorId].name}</code>
                            <span
                              className="muted"
                              title={
                                estimate.exhaustive
                                  ? "every tile of this size was probed"
                                  : `${estimate.probes} of ${estimate.totalTiles} tiles probed, one per stratum`
                              }
                            >
                              {formatFigure(figures.tiles, String)} of {estimate.totalTiles} tiles
                              {figures.sharedFraction &&
                                ` · ${formatFigure(figures.sharedFraction, (v) => `${Math.round(v * 100)}%`)} of the footprint each`}
                            </span>
                            {!estimate.geometryExact && (
                              <span className="badge approx" title={estimate.reasons.join("; ")}>≈</span>
                            )}
                            {(!estimate.exhaustive || neighbours.text) && (
                              <span
                                className="reuse-detail"
                                title={neighbours.reasons.length
                                  ? `approximate neighbours: ${neighbours.reasons.join("; ")}`
                                  : undefined}
                              >
                                {!estimate.exhaustive && `${estimate.probes} sampled`}
                                {!estimate.exhaustive && neighbours.text && " · "}
                                {neighbours.text && `neighbouring tile shares ${neighbours.text}`}
                              </span>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
