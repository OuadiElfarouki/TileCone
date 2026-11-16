# TileCone: Codex working agreement

TileCone explores tensor dependencies and tiling plans for small tensor programs.
Read `SYSTEM.md` for implemented architecture, `README.md` for product behavior,
and `docs/README.md` for the maintained documentation map. Read relevant sections,
not every document at the start of every task.

## Engineering rules

- Imports point down: core/parse <- examples <- view <- worker/state <- components
  <- App/main. Core is pure and DOM-free; operation semantics belong there.
- The picture must never lie. Conservative regions stay marked; exactness, bounds,
  unknown arithmetic and logical traffic must remain distinct through every layer.
- Check empty regions, scalars, degenerate axes, symbolic equalities, overlapping
  regions and shared operands. Plans must preserve demand coverage and clearly
  distinguish materialization, recomputation and execution-order assumptions.
- Worker results must belong to the current graph/request. Check cancellation,
  graph replacement, undo, workspace restoration and serialization together.
- UI changes must preserve keyboard ownership, pointer cancellation, focus and
  disclosure recovery. Read the relevant `docs/UI_CONTRACTS.md` entries.
- DSL vocabulary follows the Array API and torch.nn.functional, with one spelling
  per concept. Avoid aliases and compatibility shims unless explicitly requested.
- Update SYSTEM.md/README.md when behavior or architecture changes; update affected
  UI contracts with the reason. Use plain technical prose explaining why.
- Keep package.json and package-lock.json strict JSON. Follow existing patterns;
  introduce dependencies only when the task calls for them.
- Leave edits unstaged unless asked to stage or commit. Preserve unrelated work.
  Do not change .gitignore as part of another task. Respect the requested review
  scope (staged, unstaged, commit or complete project).

## Commands and evidence

- Typecheck: `npx tsc --noEmit`.
- Targeted tests: `npx vitest run <tests/...test.ts>`; unit tests live under tests/.
- Full unit suite: `npm test`.
- Browser journeys: `timeout 300 npx playwright test [e2e/file.e2e.ts]`.
  Playwright starts the e2e server on port 4179 and exposes window.__TILECONE__.
- Build: `npm run build` (includes typecheck).
- Visual checks: `node scripts/shoot.mjs <steps.mjs> --sizes 1280x800,760x800 --out /tmp/tilecone-shots`.
  Use the script's documented helpers; inspect the PNGs, not only page text.
- Write long logs to /tmp or scratchpad and report useful excerpts. Run checks
  proportional to the change; avoid concurrent build/full-suite/browser runs
  when measuring timings. The region.test.ts 3-second assertion has failed under
  load: if it is the only failure, rerun that file alone and report both results.

## Task routing and delegation

The primary agent owns scope, design decisions and integration. Delegate useful,
independent work with bounded ownership; do small searches and single-file checks
inline when spawning would cost more. Use at most three concurrent children.
Never assign overlapping edits. Give each child the objective, permitted files,
contracts to preserve, verification required and expected report. Children read
AGENTS.md, but need task-specific context and do not inherit unstated assumptions.

| Work | Preferred agent/model | Effort |
| --- | --- | --- |
| Routine searches, tracing one path, test execution | explorer/test_runner, gpt-6.1-sol | medium |
| Bounded implementation after behavior is decided | implementer, gpt-6.1-terra | high |
| Ordinary feature reasoning, state/UI changes, review | primary/reviewer/ui_polisher, gpt-6.1-sol | high |
| Region algebra, symbolic proofs, plan cost/demand semantics, sweep pruning, difficult races or disputed architecture | semantic_reviewer, gpt-6-astra | high |

Use Sol medium for a mechanical patch when cheaper than a delegation. Terra high
is the coding delegate for an explicit plan, not a mandatory extra design pass.
Use Astra when correctness depends on several interacting invariants or a failure
remains unexplained after a focused investigation. A file under core/ alone is not
reason to escalate: renaming an attribute or adding an established op pattern can
stay on Sol/Terra. For work spanning plan semantics, workers, state and persistence,
use Sol high to coordinate, Terra for separated implementation, and an Astra review
of the semantic boundaries when algorithms or dominance/pruning rules change.
Do not pay for duplicate broad reviews: give Astra the uncertain invariant and
minimal evidence, not an invitation to re-audit the entire repository.

Model names here are project preferences, not guarantees of account availability.
Use custom roles when the host exposes them. With explicit model spawning, use a
fresh child context and pass a self-contained brief. If the host cannot select a
role/model, follow the same workflow inline or report the available fallback;
never claim a model was used when it was not. Explicit user choices take priority.

## Reusable workflows

Repository skills live in .agents/skills: `$check`, `$ui-review`, `$commit-msg`.
Agent definitions and defaults live in .codex/. Configuration applies when the
host loads the trusted project; it does not change a running thread's model.
