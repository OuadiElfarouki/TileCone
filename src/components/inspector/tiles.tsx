/** The tiles list, and the range field each row edits. */

import React from "react";
import { Box, partsOverlap, sameBox } from "../../core/region";
import { fmt } from "../../view/format";
import { boxColor, MAX_DISTINCT_HUES, rgbCss } from "../../view/palette";
import { formatSelectionBox, parseSelectionBox } from "../../view/selection-range";
import { viewAxes } from "../../view/tensor/tensor-view";
import { graphIsInteractive, partHue, partsOn, selectedTensorIds } from "../../view/workspace";
import { SHORTCUTS } from "../../view/shortcuts";
import { useDark, useStore } from "../../state/store";
import { CopyButton } from "../chrome/CopyButton";
import { DraftField } from "../chrome/DraftField";
import { MERGED_AT_CAP } from "./cones";

export function SelectionRangeInput({
  box,
  shape,
  label,
  onCommit,
}: {
  box: Box;
  shape: number[];
  label: string;
  onCommit: (box: Box) => void;
}): React.ReactElement {
  return (
    <DraftField
      value={formatSelectionBox(box)}
      parse={(text) => parseSelectionBox(text, shape)}
      apply={(parsed) => {
        if (!sameBox(parsed, box)) onCommit(parsed);
      }}
      label={label}
      title="edit range; Enter or leaving the field applies, Escape abandons"
      invalidTitle={`expected ${shape.length} in-bounds index or lo:hi fields`}
      className="box-range"
    />
  );
}

/**
 * The selection's tiles, one row each, including a single tile. Hovering a row
 * emphasises that tile's cone across the whole graph; clicking pins it.
 */
/**
 * The visible way into tile growth, which Alt+arrow otherwise is: one step
 * out along the card's columns or rows. Shrinking back is offered with the
 * growth it undoes, in the Growth section.
 */
function GrowButtons({ shape, box, axes, label, onGrow }: {
  shape: number[];
  box: Box;
  axes: { rowAxis: number; colAxis: number };
  label: string;
  onGrow: (axis: number) => void;
}): React.ReactElement | null {
  const edges = [
    { axis: axes.colAxis, glyph: "→", side: "right" },
    { axis: axes.rowAxis, glyph: "↓", side: "bottom" },
  ].filter(({ axis }) => axis >= 0);
  if (!edges.length) return null;
  return (
    <span className="grow-buttons" role="group" aria-label={`grow ${label}`}>
      {edges.map(({ axis, glyph, side }) => (
        <button
          key={side}
          className="mini"
          disabled={box[axis].hi >= shape[axis]}
          aria-label={`grow ${label} at its ${side} edge`}
          title={`grow this tile one step at its ${side} edge and show what that adds to its needs (${SHORTCUTS.growTile.label.replace("arrow", glyph)})`}
          onClick={(e) => {
            e.stopPropagation();
            onGrow(axis);
          }}
        >
          +{glyph}
        </button>
      ))}
    </span>
  );
}

export function RegionEditor({ activeTensorId, onSelectGroup, coneExprs }: {
  activeTensorId: string | null;
  onSelectGroup: (tensorId: string) => void;
  /** Slice expressions for everything the analysed tiles need and feed. */
  coneExprs: () => string;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const selection = useStore((s) => s.selection);
  const replaceBox = useStore((s) => s.replaceBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const focusedBox = useStore((s) => s.focusedBox);
  const pinned = useStore((s) => s.pinnedBox);
  const hoverBox = useStore((s) => s.hoverBox);
  const togglePinBox = useStore((s) => s.togglePinBox);
  const clearSelection = useStore((s) => s.clearSelection);
  const toggleBoxHidden = useStore((s) => s.toggleBoxHidden);
  const growSelection = useStore((s) => s.growSelection);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const interactive = useStore(graphIsInteractive);
  const dark = useDark();

  if (!resolved || !selection || selection.parts.length === 0) return null;
  const parts = selection.parts;
  /**
   * Tiles in drawn order, split into one group per tensor.
   *
   * The split is the semantics made visible: everything below this list
   * analyses one group, so the list has to show where a group ends. Order
   * follows first appearance, which keeps a row from jumping to a new place
   * when a second tile lands on a tensor already in the list.
   */
  const groups = selectedTensorIds(selection).map((tensorId) => ({
    tensorId,
    rows: partsOn(selection, tensorId),
  }));

  // Overlap is a within-tensor question: two boxes on different tensors index
  // different things and cannot double-count each other.
  const overlap = { unique: 0, summed: 0 };
  for (const id of selectedTensorIds(selection)) {
    const one = partsOverlap(partsOn(selection, id).map((p) => p.box));
    overlap.unique += one.unique;
    overlap.summed += one.summed;
  }

  return (
    <div className="ins-section">
      {/* The tiles are the panel's subject, so this list is its heading: the
          tensor, the element count and the range are each stated once, here. */}
      <div className="ins-title with-action">
        Tiles
        <span className="ins-title-actions">
          <CopyButton
            className="mini"
            title="copy slice expressions for everything these tiles need and feed"
            text={coneExprs}
            label="copy slices"
          />
          <button className="mini" onClick={clearSelection} title="remove every tile from the selection">
            clear all
          </button>
        </span>
      </div>
      {!perBox && <p className="hint">{MERGED_AT_CAP}</p>}
      {overlap.summed > overlap.unique && (
        <p className="hint overlap">
          tiles overlap: {fmt(overlap.summed)} counted across parts,{" "}
          <b>{fmt(overlap.unique)}</b> distinct elements; both analyses use the deduplicated set
        </p>
      )}

      <div className="box-list" onMouseLeave={() => hoverBox(null)}>
        {groups.map((group) => (
          <div
            className={`tile-group${group.tensorId === activeTensorId ? " active" : ""}`}
            key={group.tensorId}
          >
            {groups.length > 1 && (
              <div className="tile-group-head">
                <button className="mini" aria-pressed={group.tensorId === activeTensorId}
                  onClick={() => onSelectGroup(group.tensorId)}
                  title="analyse this tensor's enabled tiles together">
                  <b>{resolved.tensors[group.tensorId].name}</b>
                </button>
                <span className="muted">
                  {group.rows.length} tile{group.rows.length === 1 ? "" : "s"}
                </span>
                {group.tensorId === activeTensorId && (
                  <span className="tile-group-mark">analysed below</span>
                )}
              </div>
            )}
        {group.rows.map(({ index: i, box: b }) => {
          const tensorId = group.tensorId;
          const active = focusedBox === i;
          const hidden = hiddenBoxes.has(i);
          const t = resolved.tensors[tensorId];
          const shape = t.resolved;
          return (
            <div
              className={`box-row${active ? " active" : ""}${pinned === i ? " pinned" : ""}${hidden ? " hidden-cone" : ""}`}
              key={i}
              onMouseEnter={() => hidden ? hoverBox(null) : hoverBox(i)}
              onClick={() => perBox && !hidden && togglePinBox(i)}
              title={perBox && !hidden ? "hover to emphasise this tile's analysis; click to pin (Esc unpins)" : undefined}
            >
              {/* One control, not two: the swatch *is* the visibility toggle, so
                  the row states the tile's hue and its shown/hidden state once. */}
              <button
                className="vis-toggle"
                disabled={!perBox}
                style={{
                  borderColor: rgbCss(boxColor(partHue(parts, i), dark)),
                  background: hidden ? "transparent" : rgbCss(boxColor(partHue(parts, i), dark)),
                }}
                title={
                  perBox
                    ? hidden
                      ? "include this tile in the merged analysis and graph highlights"
                      : "exclude this tile from the merged analysis and graph highlights (h)"
                    : "per-tile needs and feeds are not traced at this many tiles"
                }
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBoxHidden(i);
                }}
                aria-label={`${hidden ? "include" : "exclude"} tile ${i + 1}`}
                aria-pressed={!hidden}
              />
              <span className="box-body">
                <button
                  className="box-label box-pin"
                  disabled={!perBox || hidden}
                  aria-label={`${pinned === i ? "unpin" : "pin"} tile ${i + 1} on ${t.name}`}
                  aria-pressed={pinned === i}
                  title={pinned === i
                    ? "unpin this tile so the group is analysed together"
                    : "pin this tile so the readout follows it"}
                  onClick={(event) => {
                    event.stopPropagation();
                    togglePinBox(i);
                  }}
                >
                  <b>{t.name}</b> · {fmt(b.reduce((a, I) => a * (I.hi - I.lo), 1))} elements
                </button>
                <SelectionRangeInput
                  box={b}
                  shape={shape}
                  label={`selection range for tile ${i + 1} on ${t.name}`}
                  onCommit={(next) => replaceBox(i, next)}
                />
              </span>
              {interactive && (
                <GrowButtons
                  shape={shape}
                  box={b}
                  axes={viewAxes(shape, viewCfgs[tensorId])}
                  label={`tile ${i + 1} on ${t.name}`}
                  onGrow={(axis) => growSelection(axis, true, true, i)}
                />
              )}
              <button
                className="mini danger"
                aria-label={`remove tile ${i + 1} from ${t.name}`}
                title="remove this tile from the selection"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteBox(i);
                }}
              >
                ×
              </button>
            </div>
          );
        })}
          </div>
        ))}
      </div>
      {parts.length > MAX_DISTINCT_HUES && (
        <p className="hint">
          {MAX_DISTINCT_HUES} hues are in use, so some tiles share one; hover a row to tell them apart.
        </p>
      )}
    </div>
  );
}
