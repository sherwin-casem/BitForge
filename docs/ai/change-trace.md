# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-07-15.

### 2026-07-15 — Phase 22.3: CLI/REPL polish (color, progress, live tok/s, markdown)
- What: Added `--color <auto|always|never>` (respects `NO_COLOR`), a coarse 4-stage model-load
  progress indicator (TTY in-place `\r` updates, plain one-line-per-stage otherwise), a live
  tok/s indicator updated per-token during REPL generation, and incremental markdown rendering
  (bold, inline code, fenced code blocks) for REPL output — handling constructs whose delimiters
  are split across separate streamed token pieces (e.g. `"**bo"` + `"ld**"`). Regrouped
  `--help` output into sections (Model & Generation / Hardware / Server / Multimodal / Memory &
  RAG / Output) with worked examples. Only the REPL path is affected — the one-shot `--prompt`
  path and the HTTP API's SSE callback are untouched, matching the plan's scoping.
- A manual TTY smoke test (via `script`) surfaced a real UX rough edge in the first cut of the
  markdown renderer: an unterminated code fence (common whenever `max_tokens` cuts generation off
  mid-block) buffered the ENTIRE rest of the response until the final flush, defeating live
  streaming. Fixed with a bounded safety valve (`MD_MAX_PENDING_UNCLOSED`, 4 KiB) — an opening
  marker that hasn't found its close within that many buffered bytes is flushed as plain text
  instead of waiting indefinitely, covered by a new test
  (`test_unclosed_fence_eventually_flushes_without_waiting_for_end`).
- Why: closes the CLI-polish gap vs. leading engines (colored output, progress bars, live stats,
  markdown rendering) identified in the original UI/UX audit.
- Areas: `src/cli/{color,progress,live_stats,md_render}.c` + matching headers (new),
  `src/cli/{args,main,repl}.c`/`include/cli/args.h` (new `--color` flag, progress-stage hooks,
  REPL composite callback), `CMakeLists.txt` (four new CLI sources registered),
  `tests/test_{color,progress,md_render}.c` (new).
- Result: `make release/test/debug` green on gcc and clang; new unit tests cover color
  resolution, progress-line formatting, and markdown rendering (including the split-delimiter
  and unclosed-construct edge cases); manual REPL smoke test under a real pty (`script`)
  confirmed live progress stages, live tok/s updates, and markdown styling all render correctly;
  one-shot `--prompt` golden output (Paris, 64 tok/s) unaffected.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.1: HTTP API hardening + concurrency rearchitecture
- What: Added CORS (`--cors`/`--cors-origin`), optional API-key auth (`--api-key`), `/metrics`
  (Prometheus text exposition, `--metrics`), `/docs` + `/openapi.json` (static OpenAPI 3.0 +
  hand-rolled docs page, no Swagger-UI dependency), and `POST /v1/chat/completions/cancel`
  (stop an in-flight generation). Rearchitected `http_server.c` from one serial listener thread
  to a detached-per-connection-thread model, with a `generation_mutex` serializing only the
  actual `generate_with_callback` calls (a second concurrent chat request gets `429` immediately
  rather than blocking). `generate_with_callback`'s `TokenCallback` now returns `int` (0 =
  continue, nonzero = stop early) so cancellation can actually halt the generation loop, not just
  the client's view of it. Raised `HTTP_MAX_BODY_BYTES` 512 KiB → 8 MiB ahead of Phase 22.2's
  image uploads.
- Real end-to-end testing (live server + `curl`/`curl --raw` against the SmolLM2-135M demo
  model) surfaced and fixed four pre-existing HTTP protocol bugs the "untested socket layer"
  label had been masking — see `docs/ai/mistakes.md` (2026-07-15 entry) for the full list
  (recv-loop hang on bodyless GETs, `Content-Length: 0` sent before the real non-streaming body,
  a false `Transfer-Encoding: chunked` claim over unframed bytes, and no `SIGPIPE` handling).
  Also fixed a bug introduced by the new cancel feature itself: the id registered internally was
  the raw id, but clients only ever see the `"chatcmpl-"`-prefixed id in the stream, so a client
  echoing it back to the cancel endpoint would never match — fixed by registering under the same
  public, prefixed id the client observes.
- Why: closes the biggest UI/UX gap vs. leading engines (CORS/auth/metrics/docs/cancel) and is a
  prerequisite for Phase 22.2's web UI (needs real CORS + a working stop button).
- Areas: `src/api/{server_config,cors,auth,metrics,openapi,cancel,http_server,sse_stream}.c` +
  matching headers, `src/transformer/generate.c`/`include/transformer/generate.h` (callback
  return type), `src/cli/{args.c,main.c}`/`include/cli/args.h` (new flags), `CMakeLists.txt`
  (six new API sources registered), `tests/test_{cors,auth,metrics,cancel,openapi}.c` (new).
- Result: `make release/test/debug` green on gcc and clang (ASan/UBSan); a ThreadSanitizer build
  of the full engine showed zero race warnings under concurrent traffic (metrics/models/health/
  docs requests firing while a generation held the mutex, correctly getting `429` for concurrent
  generation attempts); golden output ("Paris"/"Berlin") unchanged through both the CLI and the
  now-hardened API; manual verification of CORS allow/deny, auth 401/200, streaming, non-
  streaming, and mid-stream cancellation all confirmed working via live curl sessions.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.0: docs groundwork for Web UI & API/DX hardening
- What: Recorded the Phase 22 plan (web chat UI via Vite+Svelte embedded in the binary, HTTP API
  hardening with a concurrency rearchitecture, CLI/REPL polish, mandatory design-QA + regression
  screenshots) before any code changes. Justified and documented the `api` scope in
  `tool-sync-policy.md` (adapter files to be added once 22.1 lands).
- Why: `docs/ai/**` is canonical and updated before code per policy; this is a large multi-phase
  effort and needs the decision trail written down first.
- Areas: `docs/ai/decision-log.md`, `docs/ai/project-overview.md`, `docs/ai/tool-sync-policy.md`.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-06-19 — Portable `make dist` build + GitHub Release pipeline
- What: Added a portable distribution build and a release workflow that attaches a prebuilt
  x86-64 Linux binary to a GitHub Release. New `make dist` target compiles the bulk at
  `-march=x86-64-v2` with per-file SIMD ISA flags (AVX2/AVX-512/VNNI) so runtime `simd_dispatch`
  lights up the best tier on the host; `simd_dispatch.c` is compiled at the baseline with
  `-DTN_FORCE_DISPATCH_ALL` (new guard, no SIMD codegen there) so all branches are present;
  static `-static-libstdc++ -static-libgcc` leaves only libc/libm deps. Added a `--version`/`-v`
  flag (works without `--model`) and a `-DPZ_VERSION` build stamp (banner no longer hardcodes
  "Phase 16"). CMake gains an off-by-default `PZ_DIST` option mirroring the Makefile.
- Why: user asked for a prebuilt x86-64 binary on a GitHub Release, tested thoroughly; the
  existing `-march=native` release is not distributable on varied CPUs.
- Areas: `Makefile` (dist target, per-TU ISA rules, version stamp), `src/math/simd_dispatch.c`
  (`TN_FORCE_DISPATCH_ALL`), `src/cli/{args.c,main.c}` + `include/cli/args.h` (`--version`),
  `CMakeLists.txt` (`PZ_DIST`, `PZ_VERSION`), `.github/workflows/release.yml` (new),
  `.github/workflows/ci.yml` (dist build-check), `docs/RELEASING.md` (new).
- Result: gcc release/test(46)/debug/dist and clang release/debug/dist green; portable binary
  links only libc/libm; golden output (France→Paris, Germany→Berlin) correct across
  scalar/avx2/avx512f/vnni and T=1/2/8 on the SmolLM2-135M F16 model.
- Commit/PR: on branch `claude/x86-64-github-release-8xduj2`.

### 2026-06-14 — Docs reflect dense GGUF support (SmolLM2 + generic loader)
- What: README, ROADMAP, and project-overview said the engine runs only BitNet and
  DeepSeek-V2-Lite, but the benchmark docs (`.claude/BENCHMARK_SUMMARY.md`,
  `docs/PERFORMANCE_CEILING_REPORT.md`) already benchmark **SmolLM2-135M-Instruct F16**
  (dense GGUF) up to 83.79 tok/s, and `config_from_gguf()` in `src/core/gguf_loader.c` is
  architecture-agnostic. Added a third support tier: dense GGUF transformers (Llama-family)
  via the generic loader, with SmolLM2 as the verified model and other architectures flagged
  as loads-but-untested. MoE/MLA acceleration remains DeepSeek-V2-specific.
- Why: docs understated actual, already-tested capability; user asked for the correct picture.
- Areas: `README.md` (intro, new "Dense GGUF Models" section, footer), `.github/ROADMAP.md`
  (perf snapshot), `docs/ai/project-overview.md` (Purpose). Lean adapters (AGENTS/copilot/
  GEMINI/CLAUDE) left as-is per tool-sync-policy — they describe the targeted/special-cased
  architectures, not an exhaustive model list. Historical benchmark addenda left untouched.
- Branch: `claude/readme-llm-support-docs-3tg13v`.

### 2026-06-14 — README accuracy pass + repo best-practices + docs reorg
- What: (1) Corrected README intro to match canonical scope (BitNet + DeepSeek-V2-Lite
  GGUF + vision/agentic/RAG), kept "written in C", reframed Python as temporary
  dev/test tooling (zero-Python goal), added LLM-agnostic goal. (2) Reconciled the
  Phase 21 HTTP API claim to 🔄 partial/experimental across README, ROADMAP, and
  project-overview (it is real and wired but serial/loopback-only/untested-in-CI).
  (3) Added community-health files: `.github/CODEOWNERS`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`, `.editorconfig`, `CITATION.cff`. (4) Moved 27
  archival/design/report `.md` files out of the repo root into
  `docs/{architecture,phases,reports,weight-loading}/` and `docs/`, leaving 8 entry-point
  docs at root; rewrote all inbound markdown links path-aware and fixed 4 dangling links
  (verified 0 dangling repo-wide).
- Why: README/roadmap/overview contradicted each other and the tree; root had 35 `.md`
  files hurting discoverability; repo was missing standard GitHub best-practice files.
- Areas: `README.md`, `.github/ROADMAP.md`, `docs/ai/project-overview.md`, `.editorconfig`,
  `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/config.yml`,
  `CITATION.cff`, and `docs/{architecture,phases,reports,weight-loading}/**`.
- Branch: `claude/readme-accuracy-review-y9jk7u`.

### 2026-06-07 — Document branch-hygiene convention
- What: Added a "Version control & branch hygiene" section to `engineering-rules.md` (delete
  merged branches; enable auto-delete-head-branches; avoid flag/placeholder branches; don't
  commit artifacts/models/logs).
- Why: post-merge cleanup surfaced redundant branches; this sandbox's git proxy blocks ref
  deletion, so the convention + the repo auto-delete setting prevent future accumulation.
- Areas: `docs/ai/engineering-rules.md`. Canonical-only (adapters stay lean per tool-sync-policy).

### 2026-06-07 — Cross-tool AI development system
- What: Added `docs/ai/**` canonical docs + Claude/Copilot/Antigravity adapters.
- Why: one source of truth; continuity across Claude Code, GitHub Copilot, Antigravity.
- Areas: `docs/ai/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.github/`, `AGENTS.md`,
  `gemini/GEMINI.md`, `.agents/`.
- Commit/PR: (this change) — see commit checkpoints in PR to `master`.

### 2026-06-07 — Green CI + regression verification (PR #6, merged `cb9fa52`)
- What: Fixed the CI cascade and verified no regression across SmolLM2/BitNet/DeepSeek.
- Why: CI had never run to completion; ensure merge-safety and no regression.
- Areas: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`,
  `tests/test_vision_components.c`, `Makefile`, `src/core/run_state.c`,
  `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
- Result: all 7 CI checks green on PR #6 and on `master`; secrets scan clean (215 commits).
