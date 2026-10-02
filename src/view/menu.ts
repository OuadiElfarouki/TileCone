/**
 * A menu as data: what it offers and whether each entry applies, without the
 * handlers that act on it. Builders in `view` describe a menu from the state
 * it is about; `OptionsMenu` renders it and calls handlers by entry id. The
 * split keeps the decision of what a menu offers testable without a DOM, and
 * keeps one rendering of every menu in the app.
 */

/** An entry that does one thing and closes the menu. */
export type MenuAction<A extends string> = {
  kind: "action";
  id: A;
  label: string;
  /** What the entry does, or why it does not apply. */
  title: string;
  /** Listed but inert, so the menu keeps its shape when an entry does not apply. */
  disabled: boolean;
};

/**
 * One value out of several, applied at once. The menu stays open, so a choice
 * that is made in two parts - a card's rows and its columns - is one visit.
 */
export type MenuChoice<C extends string> = {
  kind: "choice";
  id: C;
  label: string;
  options: { value: number; label: string; title: string; checked: boolean }[];
};

export type MenuSeparator = { kind: "separator" };

export type MenuSpec<A extends string, C extends string = never> = (
  | MenuAction<A>
  | MenuChoice<C>
  | MenuSeparator
)[];

/** What a menu's entries do, by id. */
export type MenuHandlers<A extends string, C extends string = never> = {
  action: Record<A, () => void>;
  choice: Record<C, (value: number) => void>;
};
