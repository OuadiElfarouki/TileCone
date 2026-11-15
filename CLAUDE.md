# TileCone

A tensor tile dependency explorer: pick a tile of a tensor in a small graph (one layer at
most) and see what it needs, what it feeds, and what a tiling plan costs. React + Zustand +
Vite, TypeScript throughout. `SYSTEM.md` is the canonical architecture; `README.md` is the
user-facing account; `docs/README.md` says which other documents are current.

## Commands
- Dev server: `npm run dev` (http://localhost:5173)
- Typecheck: `npx tsc --noEmit`
- Unit tests: `npx vitest run [paths]` - collects only `tests/**/*.test.ts`; tests mirror `src/` paths
- Browser journeys: `timeout 300 npx playwright test [e2e/file]` - `e2e/*.e2e.ts`, dev server in
  `e2e` mode on port 4179 (started automatically), store exposed as `window.__TILECONE__`
- Build: `npm run build`
- Screenshots: `node scripts/shoot.mjs <steps.mjs> --sizes 1280x800,760x800 --out <dir> [--theme dark]`
  (see the header of the script; use it instead of throwaway e2e files)

## Code rules
- Layers, imports only pointing down: `core`, `parse` <- `examples` <- `view` <- `worker`, `state`
  <- `components` <- `App`, `main`. `core` is pure and DOM-free.
- One correctness invariant outranks the rest: the picture is never a lie. An over-approximation
  stays marked; a figure says what it counts (docs/UI_CONTRACTS.md section A).
- DSL vocabulary: Array API names for array ops, `torch.nn.functional` names for NN ops, one
  spelling per thing, no aliases or compatibility shims.
- Behaviour or architecture changes update `SYSTEM.md` / `README.md` in the same change; a change
  to something listed in docs/UI_CONTRACTS.md updates that entry with the reason.
- Comments, docs and commit messages: plain technical prose, explaining why. No dramatic phrasing.
- Never `git add`, `git commit` or push unless explicitly asked; leave edits in the working tree.
  Do not report on the index, `.gitignore` or which files are tracked.
- `package.json` / `package-lock.json` must stay strict JSON (an editor formatter once added
  trailing commas and broke vite).

## Working agreement

### Threads
Keep the main thread for core work: the analysis engine, the DSL and symbolic shapes, plan
semantics, architecture. UI/UX side-topics go through `/ui-review <question>` (runs in the
`ui-polisher` agent, so its screenshots and visual back-and-forth stay out of the thread), or
in a separate session when the topic is a design project of its own.

### Delegation
The main thread plans, decides and integrates. Delegate execution:
- Mechanical, well-specified edits -> `implementer`. Give it the exact files and the intended
  change.
- Typecheck/tests/journeys/benchmarks -> `test-runner` (Haiku), or `/check`. Quick targeted runs
  (one test file, a typecheck) are fine inline.
- Codebase search / "where is X" -> built-in `Explore`.
- Review of non-trivial changes before accepting them -> `reviewer`.
- Visual checks, layout and interaction questions -> `ui-polisher`, or `/ui-review`.
Keep design decisions, debugging of non-obvious failures, and anything touching architecture in
the main thread. Run independent subagent tasks in parallel.

### Skills
- `/check [quick | paths]` - typecheck, unit tests and browser journeys, failures only.
- `/ui-review <question>` - screenshot-based UI/UX check or fix, in its own context.
- `/commit-msg` - commit message for the staged changes, in this repo's style.

### Command output
- When running `cat`, send the output to a file and show only the lines that matter (a `grep`
  or a line range of that file). Do not print a whole file into the thread.
