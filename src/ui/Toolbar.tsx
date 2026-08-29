import React, { useState } from "react";
import { EXAMPLES } from "../examples/index";
import { ComposeMode, Direction, SelectMode, useStore } from "./store";
import { boxColor, isDarkTheme, rgbCss } from "./palette";
import { TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";

const MODES: [SelectMode, string, string][] = [
  ["cell", "cell", "click a single tile cell"],
  ["box", "box", "drag a rectangle of tile cells"],
  ["row", "row", "click selects the whole tile row"],
  ["col", "col", "click selects the whole tile column"],
  ["all", "all", "click selects the whole tensor"],
];

const COMPOSE: [ComposeMode, string, string][] = [
  ["replace", "replace", "a new drag replaces the selection"],
  ["union", "∪ add", "a new drag adds to the selection — draw several disjoint regions (or hold Shift)"],
  ["subtract", "∖ sub", "a new drag cuts out of the selection (or hold Alt)"],
];

const DIRS: [Direction, string, string][] = [
  ["backward", "▲ upstream", "what this selection depends on (u)"],
  ["forward", "▼ downstream", "what this selection influences (d)"],
  ["both", "◆ both", "both directions (b)"],
];

export function Toolbar(): React.ReactElement {
  const exampleIndex = useStore((s) => s.exampleIndex);
  const loadExample = useStore((s) => s.loadExample);
  const direction = useStore((s) => s.direction);
  const setDirection = useStore((s) => s.setDirection);
  const selectMode = useStore((s) => s.selectMode);
  const setSelectMode = useStore((s) => s.setSelectMode);
  const composeMode = useStore((s) => s.composeMode);
  const setComposeMode = useStore((s) => s.setComposeMode);
  const hideInert = useStore((s) => s.hideInert);
  const setHideInert = useStore((s) => s.setHideInert);
  const clearSelection = useStore((s) => s.clearSelection);
  const setEditorOpen = useStore((s) => s.setEditorOpen);
  const editorOpen = useStore((s) => s.editorOpen);
  const dslText = useStore((s) => s.dslText);
  const selection = useStore((s) => s.selection);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const [linkCopied, setLinkCopied] = useState(false);
  const dark = isDarkTheme();

  const copyLink = () => {
    const state = {
      dsl: dslText,
      sel: selection
        ? { t: selection.tensorId, boxes: selection.region.boxes.map((b) => b.map((I) => [I.lo, I.hi])) }
        : null,
      dir: direction,
      tile: tileScale,
    };
    const hash = btoa(unescape(encodeURIComponent(JSON.stringify(state))));
    const url = `${location.origin}${location.pathname}#s=${hash}`;
    const write = (text: string) =>
      navigator.clipboard?.writeText(text).then(() => {
        setLinkCopied(true);
        setTimeout(() => setLinkCopied(false), 1200);
      });
    if (hash.length > 8000) write(JSON.stringify(state));
    else write(url);
  };

  return (
    <div className="toolbar">
      <span className="brand">TileCone</span>
      <select value={exampleIndex} onChange={(e) => loadExample(Number(e.target.value))} title="built-in examples">
        {EXAMPLES.map((ex, i) => (
          <option key={ex.name} value={i}>{ex.name}</option>
        ))}
      </select>
      <button className={editorOpen ? "on" : ""} onClick={() => setEditorOpen(!editorOpen)}>
        edit graph
      </button>
      <span className="sep" />
      <span className="group">
        {DIRS.map(([d, label, tip]) => (
          <button key={d} className={direction === d ? "on" : ""} title={tip} onClick={() => setDirection(d)}>
            {label}
          </button>
        ))}
      </span>
      <span className="sep" />
      <span className="group" title="selection mode">
        {MODES.map(([m, label, tip]) => (
          <button key={m} className={selectMode === m ? "on" : ""} title={tip} onClick={() => setSelectMode(m)}>
            {label}
          </button>
        ))}
      </span>
      <span className="sep" />
      <span className="group" title="how a new drag combines with the current selection">
        {COMPOSE.map(([m, label, tip]) => (
          <button key={m} className={composeMode === m ? "on" : ""} title={tip} onClick={() => setComposeMode(m)}>
            {label}
          </button>
        ))}
      </span>
      <span className="sep" />
      <label
        className="detail"
        title="tile size for every tensor: each drawn cell is one tile. Left = finer (down to one element), right = coarser (up to half the smallest axis). Defaults to ~5% of the smallest axis, snapped to a power of two."
      >
        detail
        <input
          type="range"
          min={-TILE_SCALE_MAX}
          max={-TILE_SCALE_MIN}
          step={1}
          value={-tileScale}
          onChange={(e) => setTileScale(-Number(e.target.value))}
        />
        <span className="detail-val">{tileScale === 0 ? "auto" : `${tileScale > 0 ? "×" : "÷"}${2 ** Math.abs(tileScale)}`}</span>
      </label>
      <span className="sep" />
      <label className="chk" title="collapse tensors/ops not involved in the current dependency cone">
        <input type="checkbox" checked={hideInert} onChange={(e) => setHideInert(e.target.checked)} />
        hide inert
      </label>
      <button onClick={clearSelection}>clear</button>
      <button onClick={copyLink}>{linkCopied ? "copied ✓" : "share link"}</button>
      <span className="legend" title="hue identifies which selected box a highlight came from">
        {[0, 1, 2].map((i) => (
          <i key={i} className="swatch" style={{ background: rgbCss(boxColor(i, dark)) }} />
        ))}
        box hue · <i className="swatch outlined" /> downstream · <i className="swatch hatched" /> approx
      </span>
    </div>
  );
}
