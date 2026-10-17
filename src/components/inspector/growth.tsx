import React from "react";
import { formatBoxIndices } from "../../core/region";
import { DTYPE_BYTES } from "../../core/dtypes";
import { useStore } from "../../state/store";
import { FIGURE_MARK, fmt, formatBytes } from "../../view/format";
import { activeGrowth } from "../../view/growth";
import { keyboardTile } from "../../view/workspace";
import { viewAxes } from "../../view/tensor/tensor-view";

/**
 * What growing the keyboard tile has added to what it needs, since its first
 * Alt+arrow step: the starting and current tile, the outputs gained, and per
 * tensor the elements needed that the starting tile did not need. The cards
 * outline the same elements.
 *
 * "No new elements" is the finding as often as a count is - one more column
 * of a GEMM tile needs no more of A - and it is a statement about footprint
 * only, which the hint says.
 */
export function GrowthSummary(): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const growth = useStore((s) => activeGrowth(s.growth, s.selection?.parts, keyboardTile(s)));
  const viewCfgs = useStore((s) => s.viewCfgs);
  const growSelection = useStore((s) => s.growSelection);
  if (!resolved || !growth) return null;
  const shape = resolved.tensors[growth.tensorId].resolved!;
  const { rowAxis, colAxis } = viewAxes(shape, viewCfgs[growth.tensorId]);
  // One step back on each edge the growth moved; the arrows do the same.
  const shrinkable = [
    { axis: colAxis, glyph: "←", side: "right" },
    { axis: rowAxis, glyph: "↑", side: "bottom" },
  ].filter(({ axis }) => axis >= 0 && growth.box[axis].hi > growth.baseBox[axis].hi);
  const { delta } = growth;
  const name = resolved.tensors[growth.tensorId].name;
  const mark = delta.exact ? "" : FIGURE_MARK.approximate;
  const rangeOf = (box: typeof growth.box) => `${name}[${formatBoxIndices(box)}]`;
  return (
    <section className="ins-section" aria-label="tile growth">
      <div className="ins-title with-action">
        Growth
        {shrinkable.length > 0 && (
          <span className="grow-buttons">
            {shrinkable.map(({ axis, glyph, side }) => (
              <button
                key={side}
                className="mini"
                aria-label={`shrink the tile back at its ${side} edge`}
                title={`shrink the tile back one step at its ${side} edge (Alt + ${glyph})`}
                onClick={() => growSelection(axis, false, true, growth.index)}
              >
                −{glyph}
              </button>
            ))}
          </span>
        )}
      </div>
      <p className="plan-op">
        <code>{rangeOf(growth.baseBox)}</code> → <code>{rangeOf(growth.box)}</code>
        <span className="muted">
          {" "}· {delta.outputs > 0 ? `+${fmt(delta.outputs)}` : "no new"} output{delta.outputs === 1 ? "" : "s"}
        </span>
      </p>
      <ul className="reuse-list">
        {delta.tensors.map(({ tensorId, elements }) => {
          const tensor = resolved.tensors[tensorId];
          return (
            <li key={tensorId}>
              <code>{tensor.name}</code>
              <span className={`reads-figures${elements ? "" : " muted"}`}>
                {elements
                  ? `${mark}+${fmt(elements)} elements · ${mark}${formatBytes(elements * DTYPE_BYTES[tensor.dtype])}`
                  : `${mark}no new elements`}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="hint">
        Outlined on the cards: what the grown tile needs that the starting tile did not. No new elements
        means nothing further is read there, not that no more arithmetic is done. Alt+arrow grows and
        shrinks it from the keyboard; moving the tile starts over.
      </p>
      {!delta.exact && (
        <p className="hint overlap">Some cone is a bound, so these differences are approximate (~).</p>
      )}
    </section>
  );
}
