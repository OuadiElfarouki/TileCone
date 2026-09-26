import { expandNode } from "../../core/expand";
import { Graph, hydrateResolvedGraph, ResolvedGraph } from "../../core/graph";
import { compileDSL, tryCompileDSL } from "../../parse/compiler";
import { toDSL } from "../../parse/dsl";
import { EXAMPLES } from "../../examples/index";
import { BaseGraphLayout } from "../../view/graph/graph-scene";
import { InspectorTab } from "../../view/workspace";
import { recompute } from "../analysis";
import { analysisWorkerAvailable, compileInWorker } from "../analysis-worker-client";
import { compile } from "../compile-epoch";
import { planEditOf, samePlanEdit, WorkspaceSnapshot } from "../history";
import {
  installedDSLState,
  loadResolvedGraph,
  restoredWorkspaceState,
  validWorkspaceFields,
} from "../load";
import { derivePlan } from "../plan";
import { GetState, SetState, State } from "../types";

/** Loading sources and examples, restoring shared links, undo, and composite expansion. */
export const workspaceActions = (set: SetState, get: GetState): Pick<
  State,
  "loadExample"
  | "loadExampleAsync"
  | "stageExample"
  | "setDraftText"
  | "applyDSL"
  | "applyDSLAsync"
  | "restoreWorkspace"
  | "restoreWorkspaceAsync"
  | "undoWorkspace"
  | "expandNodeInPlace"
> => ({
  loadExample: (i) => get().applyDSL(EXAMPLES[i].dsl),

  loadExampleAsync: (i) => get().applyDSLAsync(EXAMPLES[i].dsl),

  stageExample: (i) => {
    if (get().compiling) compile.epoch++;
    set({
      draftText: EXAMPLES[i].dsl,
      compiling: false,
      loadError: null,
      diagnostics: [],
    });
  },

  setDraftText: (text) => {
    // A response for the previous draft must never overwrite text entered
    // while that response was in flight.
    if (get().compiling) compile.epoch++;
    set({ draftText: text, compiling: false });
  },

  applyDSL: (text) => {
    compile.epoch++;
    // `tryCompileDSL` rather than the throwing form: a thrown CompilationError
    // flattens to its first diagnostic's message, and the editor wants all of
    // them. Everything the compiler found in one pass reaches the panel.
    const result = tryCompileDSL(text);
    if (!result.ok) {
      set({
        draftText: text,
        compiling: false,
        diagnostics: result.diagnostics,
        loadError: `line ${result.diagnostics[0].span.start.line}: ${result.diagnostics[0].message}`,
      });
      return;
    }
    try {
      const program = result.program;
      set(installedDSLState(text, program.graph, program.resolved));
    } catch (e) {
      // Compilation succeeded; anything failing here is a workspace-build
      // problem with no source span to attach it to.
      set({ draftText: text, compiling: false, loadError: (e as Error).message, diagnostics: [] });
    }
  },

  applyDSLAsync: async (text) => {
    if (!analysisWorkerAvailable()) {
      get().applyDSL(text);
      return get().dslText === text && get().loadError === null;
    }
    const epoch = ++compile.epoch;
    set({ compiling: true, draftText: text, loadError: null, diagnostics: [] });
    try {
      const result = await compileInWorker(text);
      if (epoch !== compile.epoch) return false;
      if (!result.ok) {
        set({
          compiling: false,
          diagnostics: result.diagnostics,
          loadError: `line ${result.diagnostics[0].span.start.line}: ${result.diagnostics[0].message}`,
        });
        return false;
      }
      const resolved = hydrateResolvedGraph(result.artifact.resolved);
      set(installedDSLState(text, result.artifact.graph, resolved, {
        graphId: result.graphId,
        layout: result.artifact.layout,
        graphPx: result.artifact.graphPx,
      }));
      return true;
    } catch (error) {
      if (epoch !== compile.epoch) return false;
      set({
        compiling: false,
        loadError: error instanceof Error ? error.message : String(error),
        diagnostics: [],
      });
      return false;
    }
  },

  restoreWorkspace: (workspace) => {
    compile.epoch++;
    try {
      if (!validWorkspaceFields(workspace)) return false;
      const program = compileDSL(workspace.dsl);
      set(restoredWorkspaceState(workspace, program.graph, program.resolved));
      return true;
    } catch {
      return false;
    }
  },

  restoreWorkspaceAsync: async (workspace) => {
    if (!analysisWorkerAvailable()) return get().restoreWorkspace(workspace);
    if (!validWorkspaceFields(workspace)) return false;
    const epoch = ++compile.epoch;
    set({ compiling: true });
    try {
      const result = await compileInWorker(workspace.dsl);
      if (epoch !== compile.epoch) return false;
      if (!result.ok) {
        set({ compiling: false });
        return false;
      }
      const resolved = hydrateResolvedGraph(result.artifact.resolved);
      set(restoredWorkspaceState(workspace, result.artifact.graph, resolved, {
        graphId: result.graphId,
        layout: result.artifact.layout,
        graphPx: result.artifact.graphPx,
      }));
      return true;
    } catch {
      if (epoch === compile.epoch) set({ compiling: false });
      return false;
    }
  },

  undoWorkspace: () => {
    const { workspaceHistory, resolved, selection, perBox, entangled } = get();
    if (!workspaceHistory.length) return;
    const prev = workspaceHistory[workspaceHistory.length - 1];
    if (prev.source) {
      // The complete pre-expansion graph is part of this one special history
      // entry. Restoring it directly avoids recompiling and relaying out a
      // potentially large graph on the UI thread during Undo.
      compile.epoch++;
      const source = prev.source;
      set({
        ...loadResolvedGraph(
          source.graph,
          source.resolved,
          source.baseLayout
            ? { graphId: null, layout: source.baseLayout, graphPx: source.graphPx }
            : undefined
        ),
        dslText: source.dslText,
        draftText: source.draftText,
        exampleIndex: source.exampleIndex,
        tensorOffsets: prev.tensorOffsets,
        selection: prev.selection,
        workspaceHistory: workspaceHistory.slice(0, -1),
        focusNode: null,
        compiling: false,
        ...recompute(source.resolved, prev.selection),
        ...derivePlan(source.resolved, prev.plan.tiles, prev.plan.task, get()),
      });
      return;
    }
    set({
      selection: prev.selection,
      tensorOffsets: prev.tensorOffsets,
      workspaceHistory: workspaceHistory.slice(0, -1),
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      // The restored selection may not contain the group at all, and undo is
      // not the place to guess which of its tensors the reader meant.
      analysisGroup: null,
      preview: null,
      ...recompute(resolved, prev.selection, { selection, perBox, entangled }),
      ...derivePlan(resolved, prev.plan.tiles, prev.plan.task, get()),
      /* One history holds tile edits and plan edits alike, so a step back can
         restore something the visible panel does not show. Undo then looks as
         if it did nothing and the change is found later by accident, so the
         view the restored edit belongs to is brought forward with it. */
      ...(samePlanEdit(planEditOf(get()), prev.plan)
        ? {}
        : { inspectorTab: "plan" as InspectorTab }),
    });
  },

  expandNodeInPlace: (nodeId) => {
    const state = get();
    const { graph, resolved } = state;
    if (!graph || !resolved) return;
    try {
      const g2 = expandNode(graph, nodeId);
      // Source and graph remain one transaction: rerunning or sharing the text
      // must restore the same primitive graph currently shown in the workspace.
      const source = toDSL(g2);
      // `loadResolvedGraph` clears the history, and rightly: its entries name
      // tensors and coordinates in the graph being replaced. The one entry that
      // survives is the one it cannot invalidate, because it is what to go back
      // *to* — recorded after the clear, for that reason.
      const restore: WorkspaceSnapshot = {
        selection: state.selection,
        tensorOffsets: state.tensorOffsets,
        plan: planEditOf(state),
        source: {
          dslText: state.dslText,
          draftText: state.draftText,
          graph,
          resolved,
          baseLayout: state.baseLayout,
          graphPx: state.graphPx,
          exampleIndex: state.exampleIndex,
        },
      };
      const install = (
        nextGraph: Graph,
        nextResolved: ResolvedGraph,
        worker?: { graphId: number; layout: BaseGraphLayout; graphPx: number }
      ) => set({
          ...loadResolvedGraph(nextGraph, nextResolved, worker),
          dslText: source,
          draftText: source,
          exampleIndex: -1,
          focusNode: null,
          compiling: false,
          workspaceHistory: [restore],
        });

      if (!analysisWorkerAvailable()) {
        compile.epoch++;
        const program = compileDSL(source);
        install(program.graph, program.resolved);
        return;
      }

      const epoch = ++compile.epoch;
      set({ compiling: true, loadError: null, diagnostics: [] });
      void compileInWorker(source).then((result) => {
        if (epoch !== compile.epoch) return;
        if (!result.ok) {
          set({
            compiling: false,
            diagnostics: result.diagnostics,
            loadError: result.diagnostics[0]?.message ?? "expanded graph did not compile",
          });
          return;
        }
        install(result.artifact.graph, hydrateResolvedGraph(result.artifact.resolved), {
          graphId: result.graphId,
          layout: result.artifact.layout,
          graphPx: result.artifact.graphPx,
        });
      }).catch((error) => {
        if (epoch === compile.epoch) set({
          compiling: false,
          loadError: error instanceof Error ? error.message : String(error),
          diagnostics: [],
        });
      });
    } catch (e) {
      set({ compiling: false, loadError: (e as Error).message });
    }
  },
});
