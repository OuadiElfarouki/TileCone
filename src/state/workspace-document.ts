import { linkToRestore, parseWorkspaceLink, workspaceToLink, type WorkspaceLink } from "./share";
import type { State, WorkspaceRestore } from "./types";

/** Autosave and exported files share one versioned, portable document. */
export type WorkspaceDocument = {
  format: "tilecone-workspace";
  version: 1;
  workspace: WorkspaceLink;
  draftText: string;
  scope: { hidden: number[]; group: string | null; pinned: number | null };
};

export const MAX_WORKSPACE_BYTES = 4 * 1024 * 1024;

export function workspaceDocument(state: State): WorkspaceDocument {
  return {
    format: "tilecone-workspace", version: 1,
    workspace: workspaceToLink(state), draftText: state.draftText,
    scope: { hidden: [...state.hiddenBoxes].sort((a, b) => a - b), group: state.analysisGroup, pinned: state.pinnedBox },
  };
}

export function serializeWorkspace(state: State): string {
  const text = JSON.stringify(workspaceDocument(state), null, 2);
  if (new TextEncoder().encode(text).byteLength > MAX_WORKSPACE_BYTES)
    throw new Error("This workspace exceeds the 4 MB file limit.");
  return text;
}

/** Legacy raw Share JSON remains importable, including oversized URL payloads.
 * Unknown document versions are refused rather than guessed at. */
export function parseWorkspaceDocument(text: string): WorkspaceRestore {
  if (text.length > MAX_WORKSPACE_BYTES || new TextEncoder().encode(text).byteLength > MAX_WORKSPACE_BYTES)
    throw new Error("Workspace files must be smaller than 4 MB.");
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("This file is not valid workspace JSON."); }
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("This file is not a TileCone workspace.");
  const raw = value as Partial<WorkspaceDocument>;
  if ("format" in raw || "version" in raw || "workspace" in raw) {
    if (raw.format !== "tilecone-workspace") throw new Error("This file is not a TileCone workspace.");
    if (raw.version !== 1) throw new Error("This workspace version is not supported by this app.");
    const link = parseWorkspaceLink(raw.workspace);
    const scope = raw.scope;
    const count = link?.sel?.length ?? 0;
    const indexFits = (index: unknown): index is number => Number.isSafeInteger(index) && (index as number) >= 0 && (index as number) < count;
    if (!link || typeof raw.draftText !== "string" || !scope || typeof scope !== "object" ||
        !Array.isArray(scope.hidden) || !scope.hidden.every(indexFits) ||
        new Set(scope.hidden).size !== scope.hidden.length ||
        (scope.group !== null && (typeof scope.group !== "string" || !link.sel?.some((p) => p.t === scope.group))) ||
        (scope.pinned !== null && (!indexFits(scope.pinned) || scope.hidden.includes(scope.pinned) ||
          link.sel?.[scope.pinned].t !== scope.group)))
      throw new Error("This workspace contains invalid tiles or settings.");
    return { ...linkToRestore(link), draftText: raw.draftText,
      hiddenBoxes: scope.hidden, analysisGroup: scope.group, pinnedBox: scope.pinned };
  }
  const link = parseWorkspaceLink(value);
  if (!link) throw new Error("This file is not a valid TileCone workspace.");
  return linkToRestore(link);
}

/** Pointer previews, results, and playback never trigger a disk write. */
export function sameSavedWorkspace(a: State, b: State): boolean {
  return a.dslText === b.dslText && a.draftText === b.draftText &&
    a.selection === b.selection && a.direction === b.direction &&
    a.showEntangled === b.showEntangled && a.tileScale === b.tileScale &&
    a.snapToGrid === b.snapToGrid && a.axisMode === b.axisMode &&
    a.viewCfgs === b.viewCfgs && a.nodeOffsets === b.nodeOffsets &&
    a.planTiles === b.planTiles && a.planTask === b.planTask &&
    a.keptPlans === b.keptPlans && a.inspectorTab === b.inspectorTab &&
    a.hiddenBoxes === b.hiddenBoxes && a.analysisGroup === b.analysisGroup && a.pinnedBox === b.pinnedBox;
}
