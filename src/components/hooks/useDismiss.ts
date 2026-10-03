import { useEffect, useRef } from "react";

/**
 * Close a popover on a press outside it. With `onMove`, also on anything that
 * moves the page under it - for a popover placed at fixed coordinates, which
 * would otherwise be left pointing at where its owner used to be.
 *
 * `inside` lists what counts as the popover: the surface itself, and the
 * control that toggles it, whose own press would otherwise close the popover
 * only for its click to reopen it.
 */
export function useDismiss(
  open: boolean,
  inside: readonly React.RefObject<Element>[],
  dismiss: () => void,
  { onMove = false }: { onMove?: boolean } = {}
): void {
  // The latest callback, so the listeners are installed once per opening.
  const dismissRef = useRef(dismiss);
  dismissRef.current = dismiss;
  const insideRef = useRef(inside);
  insideRef.current = inside;

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (!insideRef.current.some((ref) => ref.current?.contains(target))) dismissRef.current();
    };
    const onPageMove = () => dismissRef.current();
    document.addEventListener("pointerdown", onPointerDown);
    if (onMove) {
      window.addEventListener("scroll", onPageMove, true);
      window.addEventListener("resize", onPageMove);
    }
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      if (onMove) {
        window.removeEventListener("scroll", onPageMove, true);
        window.removeEventListener("resize", onPageMove);
      }
    };
  }, [open, onMove]);
}
