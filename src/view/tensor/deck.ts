/**
 * A slice card drawn as a deck: the slice shown is the front card, and the
 * slices still to come in row-major order over the hidden axes lie behind it
 * as sheets peeking out up and to the right. Moving a slider deals the front
 * card away or deals one back, so the reader sees where in the stack they are
 * and which way they moved.
 *
 * The sheets carry no elements. They are a depth cue: their number says how
 * much of the tensor lies behind the slice, never which elements it holds.
 */

/** The most sheets drawn behind the front card. */
export const DECK_SHEETS = 5;
/** How far each sheet sits up and to the right of the one in front, in card px. */
export const DECK_STEP_PX = 3;
/** Room a slice card keeps above and beside its canvas for the deck, in card
 *  px. Constant, so stepping through slices never resizes the card. */
export const DECK_ROOM_PX = DECK_SHEETS * DECK_STEP_PX;

/**
 * How many sheets to draw behind slice `index` of `count`.
 *
 * One per remaining slice while they fit; past that the deck's thickness is
 * proportional to what remains, and never zero while anything does, so the
 * last slice is the only one with an empty deck.
 */
export function deckSheets(index: number, count: number): number {
  const remaining = Math.max(0, count - 1 - index);
  if (remaining === 0) return 0;
  if (count - 1 <= DECK_SHEETS) return remaining;
  return Math.max(1, Math.ceil((DECK_SHEETS * remaining) / (count - 1)));
}
