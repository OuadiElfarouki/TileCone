/**
 * Whether a press starts a gesture: the primary mouse button, or a pen or
 * touch contact, which report as button 0. A secondary press belongs to the
 * context menu, and a middle press to the browser, so neither may draw a
 * tile, pan, or drag a node on its way there.
 */
export const isPrimaryPress = (event: { button: number }): boolean => event.button === 0;
