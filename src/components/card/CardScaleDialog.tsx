import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { MAX_CARD_SCALE, MIN_CARD_SCALE } from "../../view/tensor/tensor-view";

const contain = (event: React.SyntheticEvent) => event.stopPropagation();

/** Commit a numeric edit once, with native validation and modal focus. */
export function CardScaleDialog({ name, scale, onApply, onClose }: {
  name: string; scale: number; onApply: (scale: number) => void; onClose: () => void;
}): React.ReactElement {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  /** Whether the press began on the backdrop: a selection dragged out of the
   *  field ends there too, and must not dismiss the dialog. */
  const pressedBackdrop = useRef(false);
  useEffect(() => {
    const element = dialog.current!;
    element.showModal(); input.current?.select();
    return () => element.close();
  }, []);
  return createPortal(<dialog ref={dialog} className="card-scale-dialog" role="dialog" aria-modal="true" aria-label={`display size for ${name}`}
    // The portal is outside the card in the DOM but inside it in React, so
    // events are contained here before they reach the card and graph handlers.
    onKeyDown={contain} onPointerMove={contain} onPointerUp={contain} onWheel={contain}
    onContextMenu={contain}
    onPointerDown={(event) => { event.stopPropagation(); pressedBackdrop.current = event.target === event.currentTarget; }}
    onClick={(event) => { event.stopPropagation(); if (pressedBackdrop.current && event.target === event.currentTarget) onClose(); }}
    onCancel={(event) => { event.preventDefault(); onClose(); }}>
    <form onSubmit={(event) => {
      event.preventDefault();
      if (event.currentTarget.reportValidity()) onApply(Math.round(input.current!.valueAsNumber * 10) / 1000);
    }}>
      <label>Display size · {name}<span className="card-scale-field">
        <input ref={input} type="number" aria-label="display size percent" autoFocus required
          min={MIN_CARD_SCALE * 100} max={MAX_CARD_SCALE * 100} step="0.1" defaultValue={Number((scale * 100).toFixed(1))} /> %
      </span></label>
      <p className="hint">100% is the automatic fit. Neighbours may limit the displayed size; “make room for this size” rearranges them.</p>
      <div className="card-scale-actions"><button type="button" onClick={onClose}>cancel</button><button type="submit">apply</button></div>
    </form>
  </dialog>, document.body);
}
