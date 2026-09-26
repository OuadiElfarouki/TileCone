/** The tile header, the tiles list, and the range field each row edits. */

import React, { useEffect, useState } from "react";
import { Box, partsOverlap } from "../../core/region";
import { fmt } from "../../view/format";
import { boxColor, MAX_DISTINCT_HUES, rgbCss } from "../../view/palette";
import { formatSelectionBox, parseSelectionBox } from "../../view/selection-range";
import { viewAxes } from "../../view/tensor/tensor-view";
import { partsOn, selectedTensorIds } from "../../view/workspace";
import { useDark, useStore } from "../../state/store";
import { CopyButton } from "../chrome/CopyButton";
import { MERGED_AT_CAP } from "./cones";
import { measuredElements, measuredParts } from "./inspector-analysis";

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
  const formatted = formatSelectionBox(box);
  const [draft, setDraft] = useState(formatted);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    setDraft(formatted);
    setInvalid(false);
  }, [formatted]);

  const commit = () => {
    const parsed = parseSelectionBox(draft, shape);
    if (!parsed) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    const unchanged = parsed.every(
      (interval, axis) => interval.lo === box[axis].lo && interval.hi === box[axis].hi
    );
    if (!unchanged) onCommit(parsed);
  };

  return (
    <input
      className={`box-range${invalid ? " invalid" : ""}`}
      value={draft}
      aria-label={label}
      aria-invalid={invalid}
      title={invalid ? `expected ${shape.length} in-bounds index or lo:hi fields` : "edit range; Enter or blur applies"}
      spellCheck={false}
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") commit();
        if (event.key === "Escape") {
          setDraft(formatted);
          setInvalid(false);
        }
      }}
    />
  );
}

/**
 * The analysed tile set and its distinct element count. Ranges are edited only
 * in the tile list, including when the workspace contains a single tile.
 */
export function TileIdentity({
  coneExprs,
  activeTensorId,
  focusedBox,
}: {
  coneExprs: () => string;
  activeTensorId: string | null;
  focusedBox: number | null;
}): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved)!;
  const selection = useStore((s) => s.selection);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const perBox = useStore((s) => s.perBox);
  const viewCfg = useStore((s) => (activeTensorId ? s.viewCfgs[activeTensorId] : undefined));
  const dark = useDark();

  if (!selection) return null;
  const anchorId = activeTensorId;
  if (!anchorId) return null;
  /**
   * The header describes the same group as everything under it, so its counts
   * come from the active tensor's tiles rather than from the whole selection.
   * A tile on another tensor is in the list above, not in this answer.
   */
  const group = partsOn(selection, anchorId);
  const index = focusedBox ?? group[group.length - 1]?.index;
  const part = selection.parts[index];
  if (part === undefined || index === undefined) return null;
  const ordinal = group.findIndex((row) => row.index === index);

  // With several tiles in the group and none focused, everything below
  // describes their merged cone. Naming one of them here would put the wrong
  // tile at the top of a readout about all of them.
  const merged = group.length > 1 && focusedBox === null;
  const tensor = resolved.tensors[anchorId];
  const shape = tensor.resolved!;
  const { rowAxis, colAxis } = viewAxes(shape, viewCfg);
  const axisLabel = (axis: number) => tensor.axisNames?.[axis] ?? `ax${axis}`;
  const measured = measuredParts(selection.parts, anchorId, hiddenBoxes, focusedBox, !!perBox);
  const elements = measuredElements(measured);

  return (
    <header className="tile-identity">
      <div className="tile-ord">
        {/* The swatch is the tile's hue on the canvas: the header and the cone
            it describes have to be the same colour, so it follows the anchor's
            own index rather than the first one. A merged readout belongs to no
            single hue, so it carries none. */}
        {!merged && (
          <i className="swatch" style={{ background: rgbCss(boxColor(index, dark)) }} />
        )}
        <span>
          {measured.length === 0 ? "no enabled tiles" : merged
            ? `${measured.length} of ${group.length} tiles · merged`
            : `tile ${ordinal + 1} of ${group.length}`}
        </span>
        <CopyButton
          className="mini copy-cone"
          title="copy slice expressions for everything this tile needs and feeds"
          text={coneExprs}
          label="copy all"
        />
      </div>
      <div className="tile-name">
        <b
          title={`rows ${axisLabel(rowAxis >= 0 ? rowAxis : 0)} · cols ${axisLabel(colAxis >= 0 ? colAxis : 0)} · shape [${shape.join("×")}]`}
        >
          {tensor.name}
        </b>
        <span className="tile-count">{fmt(elements)} el</span>
      </div>
    </header>
  );
}

/**
 * The selection's tiles, one row each, including a single tile. Hovering a row
 * emphasises that tile's cone across the whole graph; clicking pins it.
 */
export function RegionEditor({ activeTensorId, onSelectGroup }: {
  activeTensorId: string | null;
  onSelectGroup: (tensorId: string) => void;
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
      <div className="ins-title with-action">
        Tiles
        <button className="mini" onClick={clearSelection} title="remove every tile from the selection">
          clear all
        </button>
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
          const shape = t.resolved!;
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
                  borderColor: rgbCss(boxColor(i, dark)),
                  background: hidden ? "transparent" : rgbCss(boxColor(i, dark)),
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
          tiles past the {MAX_DISTINCT_HUES}rd share a neutral color, only {MAX_DISTINCT_HUES} hues stay
          distinguishable side by side, so use hover to tell the rest apart.
        </p>
      )}
    </div>
  );
}
