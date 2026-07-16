# Decision Log — project-zero

> Timestamped architectural / tooling / workflow / process decisions. Newest first.
> Read at session start. Last updated: 2026-07-16.

### 2026-07-16 — Closed the Q2_0 perf gap and the remaining chat-template gaps on explicit request
- Decision: given a direct follow-up request to "fix all shortcomings" from the earlier Qwen 3.6
  benchmark (this same day), implemented the deferred Q2_0 VNNI kernel and the contained
  chat-template gaps (`|items` filter, tuple-unpacking for-loops, `loop.previtem`/`nextitem`,
  method calls on literals) rather than leaving them as documented-but-unaddressed. Left two
  items deliberately out of scope, each re-flagged explicitly rather than silently fixed or
  silently dropped:
  1. The Q2_0 *batch* matmul path (`parallel_matmul_q2_0_batch`, used for MoE per-expert
     routing) still uses the portable decode+FMA kernel. Ternary-Bonsai-27B is dense (MoE
     disabled), so this was never on the measured hot path this session, and extending the same
     VNNI trick to it wasn't verified against a real MoE+Q2_0 model.
  2. The byte-level-BPE detokenizer's incomplete GPT-2 byte-unicode reverse table (mojibake on
     non-ASCII output) — a real, pre-existing, unrelated-to-Qwen-3.6 bug found via a screenshot
     review, but a stateful rewrite of a function on every model's hot decode path, with zero
     existing non-ASCII test coverage to verify against; see mistakes.md for the full reasoning.
- Benchmark results (before -> after the VNNI kernel), same prompt, same host, 4 threads,
  greedy decoding: project-zero generation throughput went from ~0.11 tok/s to **~3.24 tok/s**
  (~29x) on the real model. Verified as a genuine speedup and not a correctness regression by
  comparing the exact sampled-token-ID sequence at matching generation steps before and after —
  identical for every step checked (both engines' greedy argmax agreed token-for-token), and a
  full 256-token completion was re-captured end-to-end producing the same coherent
  reasoning-then-answer response as before the optimization.
- Correctness verification method for the new VNNI kernel specifically: a dedicated unit test
  (`tests/test_q2_0_matmul.c`) comparing it against an independent reference decode path
  (`gguf_dequant_q2_0`, unrelated to either matmul kernel) across single-block/multi-block/
  real-model-dimension/non-block-aligned/all-zero-activation cases, run under full ASan+UBSan
  instrumentation of every object involved (not just the test file — see mistakes.md's note on
  why the first version of this test had false failures and how that was root-caused with a
  one-hot activation probe before concluding the kernel itself was correct).
- Status: ACCEPTED. gcc+clang release/test/debug all green (including the 2 new test binaries);
  real-model output re-verified coherent and consistent with the pre-optimization baseline.

### 2026-07-16 — Qwen 3.5/3.6 hybrid-attention support: architecture, benchmark engine, and scope
- Decision: implement full Qwen 3.5/3.6 (Gated DeltaNet + Gated-Attention hybrid) support,
  targeting `prism-ml/Ternary-Bonsai-27B-gguf` specifically, following the existing `has_mla`
  precedent (extend shared `MoEConfig`/`TransformerWeights`/`RunState` structs with
  `q35_*`/`has_linear_attn`-gated fields rather than new per-arch types).
- New tensor format: added `GGML_TYPE_Q2_0`/`gguf_dequant_q2_0`/`parallel_matmul_q2_0` for
  PrismML's ternary Q2_0 packing. Confirmed empirically (bytes-per-tensor computed from the real
  downloaded file, cross-checked against llama.cpp discussion #22019) that this specific file uses
  the **group-128** variant (34B/128 elems, 2.125 bpw), not mainline ggml's canonical **group-64**
  `block_q2_0` (18B/64 elems) — both share GGUF type ID 42, so the group size had to be derived
  from the file, not the type. Zero-copy: token embedding and LM head stay Q2_0-raw (dequantizing
  either fully to F32 would need ~5GB), dequanted per-row/per-token on demand.
- Benchmark engine choice: built **PrismML-Eng/llama.cpp** (the `prism` branch), not mainline
  ggml-org/llama.cpp, specifically because mainline's CPU/Metal Q2_0 kernel only covers the
  group-64 packing and cannot load this project's actual downloaded group-128 file without a
  second multi-GB download of a different GGUF variant. The fork's `QK2_0` constant (128) was
  checked in `ggml-common.h` before building to confirm compatibility with the exact file already
  on disk, letting both engines benchmark the identical `.gguf`.
- Full-model verification method: rather than trusting code review alone against the reference
  source (`src/models/qwen35.cpp`/`delta-net-base.cpp` in the same clone), correctness was proven
  by running the identical prompt through both engines and requiring matching, coherent output —
  this is what actually surfaced the final remaining bug (a transposed conv1d kernel read; see
  `mistakes.md` 2026-07-16 item 6) that line-by-line math comparison against the reference source
  had missed. Tokenization/chat-template rendering were separately confirmed byte-identical via
  `llama-tokenize -f <rendered_prompt.txt> --ids` against project-zero's own `--verbose` dump.
- Scope boundary: Q2_0 matmul performance (currently scalar-decode-to-stack-buffer +
  AVX2 FMA per 128-element block) was explicitly left unoptimized — the task was correctness +
  running the model + benchmarking existing behavior, not a performance pass. This shows up
  directly in the benchmark: project-zero's Q2_0 kernel is markedly slower than the reference
  engine's (see the benchmark artifact for numbers on this specific 4-core host). Flagged as a
  known, deliberately out-of-scope follow-up rather than silently accepted.
- Status: ACCEPTED. Full architecture verified end-to-end against the real 27B model: gcc+clang
  release/test/debug all green (incl. new `tests/test_chat_template.c`, 18 cases), coherent
  reasoning output produced and cross-checked token-for-token against the reference engine's
  continuation of the same prompt.

### 2026-07-16 — Process decision: bugs get fixed on discovery, not just logged
- Decision: any bug found during a task — including pre-existing ones unrelated to what's being
  worked on, surfaced incidentally via ASan/UBSan/ThreadSanitizer, manual/screenshot review, test
  failures, or code reading — gets fixed in the same pass. "Pre-existing" or "unrelated to this
  change" is no longer an acceptable reason to leave a confirmed real defect in place.
- Trigger: a ~1.2MB ASan leak (49k+ allocations) surfaced while verifying the Phase 22.5 CLI
  banner/spinner work was initially treated as a pre-existing, out-of-scope finding and merely
  documented. On follow-up, traced to two real, small, fixable bugs — (1) `tokenizer_free()` in
  `src/cli/main.c` was gated on `args.tokenizer_path`, which is unset for the common GGUF-auto-
  load tokenizer path, so cleanup silently never ran; (2) `GGUFHeader`'s heap-allocated
  string-metadata copies (`parse_meta_entry` in `src/core/gguf_reader.c`) had no corresponding
  free function at all. Both fixed same-session; see `docs/ai/mistakes.md` (2026-07-16) for the
  full writeup.
- Canonical rule added: `docs/ai/engineering-rules.md` § "Bug-fix policy" — only exception is a
  fix requiring a large architectural change, which must be flagged to the user explicitly
  rather than silently fixed or silently left. Synced to all adapters (`.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`).

### 2026-07-15 — Phase 22: Web UI & API/DX hardening — tech + scope decisions
- Decision: project-zero has no UI/UX beyond a raw JSON API and a plain-text CLI (confirmed:
  no `ui/`/`web/`/`static/`/`.html`/`package.json` anywhere pre-Phase-22; `src/api/http_server.c`
  exposed only `GET /v1/models`, `POST /v1/chat/completions`, `GET /health`). Closing the gap vs.
  leading engines (llama.cpp server, Ollama, LM Studio) requires a web chat UI, HTTP API
  hardening (CORS/auth/metrics/OpenAPI), and CLI/REPL polish (color/progress/markdown).
- Web UI tech: **Vite + Svelte** SPA (a lighter analogue of llama.cpp's SvelteKit server UI — no
  SSR needed since the C server only serves static files), built via an npm step used only by
  *contributors* touching `webui/src`. The built `webui/dist/` bundle is embedded into the C
  binary as a generated, **git-committed** byte-array TU (`src/api/webui_bundle_generated.c`),
  so ordinary `make release`/`cmake --build` and CI stay 100% Node-free by default — mirrors
  llama.cpp's own build-then-embed pattern. `make webui-bundle` (opt-in, never in `all`/`test`)
  regenerates it; a `webui.yml` CI job scoped to `webui/**` fails on drift.
- Scope decision (concurrency): the HTTP server's single-threaded/serial `handle_connection()`
  model is **rearchitected** to per-connection threads + a `generation_mutex` (only one
  `generate_with_callback` in flight at a time; second concurrent chat request gets `429`), so
  the web UI's static assets and Stop/Cancel button aren't blocked behind an in-flight
  generation. This is a real concurrency change to `http_server.c`, not pure presentation —
  verified with ThreadSanitizer in addition to the usual ASan/UBSan pass.
- Scope decision (image upload): included now. Requires extracting `main.c`'s inline
  vision-loading block into a reusable `multimodal/vision_pipeline.c` function callable from
  both the CLI and `--server` mode, plus raising `HTTP_MAX_BODY_BYTES`/`CHAT_MAX_CONTENT` to fit
  base64 images.
- Mandatory verification additions (user-specified, apply project-wide going forward for this
  kind of change): (1) any refactor of existing non-UI code done in service of a UI change (the
  two items above) must be verified with a before/after run of every currently-tested model,
  timestamped raw-output screenshots, and a text+tok/s diff against baseline — extends the
  existing golden-output/A/B practice, does not replace it; (2) every UI/UX screenshot (web +
  CLI) is graded against a written design-principles reference (`docs/design/ui-ux-principles.md`)
  via independent image review before being accepted, looping fix→recapture on failure.
- Capture tooling: investigated `docs/tty_bitnet.png`/`demo_bitnet.gif` — confirmed **manually
  captured**, no script/tool/CI step produces them (repo-wide search for vhs/asciinema/ttyrec/
  `.tape` files found zero hits). Phase 22 introduces the first real capture tooling: Playwright
  (headless Chromium, pre-installed in this environment) for the web UI, and a scripted terminal
  capture for the CLI/REPL (see `docs/design/ui-ux-principles.md`/Phase 22.4 notes for the exact
  tool, since `vhs` requires `ttyd`+`ffmpeg` which are not present in this environment).
- Full design: see the Phase 22 plan; sub-phases 22.0 (docs) → 22.1 (API hardening) → 22.2 (web
  UI) → 22.3 (CLI polish) → 22.4 (design QA/regression/README).

### 2026-06-21 — Claim corrected to same-SIMD/same-thread/same-precision; hero = beats bitnet.cpp
- Decision: The README headline claims only what holds apples-to-apples. Fresh three-engine
  measurement on one Xeon (same SIMD, per-thread, matched precision) shows: Project Zero
  beats Microsoft `bitnet.cpp` on BitNet at every thread (+19…+37%, BF16 head; 1.80× / 95%
  DRAM ceiling tuned), and on dense SmolLM2 beats `llama.cpp` at 1–3 threads but trails at the
  4-thread peak (−12%). So **drop any blanket "beats llama.cpp"**; hero = beats bitnet.cpp +
  only single no-dep binary running ternary AND dense. DeepSeek MoE 7× gap stays visible.
- Wording: marketing copy/PNGs must **not** use the word "honest" (reads as justifying);
  state facts confidently instead. (Internal docs may still discuss honesty.)
- Methodology: TG steady-state; PZ via `[gen]` over 128 tok; competitors via `llama-bench
  -n 128 -r 5`; warm cache; one test at a time; full results in BENCHMARK_REPORT.md Addendum AP.
- Gotcha (recorded): the container's host CPU migrated mid-session (Xeon 2.10→2.80 GHz; the
  2.80 lacks `avx_vnni`), which SIGILLs native-built competitor binaries and cripples
  bitnet.cpp's i2_s kernel (0.58 tok/s). PZ is unaffected (runtime SIMD dispatch). Comparison
  numbers are from the 2.10 GHz host where both engines had their VNNI kernels; live tty/video
  demos are PZ-only on the current host and labelled as such.
- Assets added: `docs/benchmark_bitnet.png`, `docs/benchmark_smollm2.png` (bar charts),
  `docs/tty_bitnet.png`, `docs/tty_smollm2.png` (live terminal captures), `docs/demo.webm`.

### 2026-06-20 — README repositioned as a "claim + proof" star-conversion page
- Decision: Lead the README with an honest performance claim ("beats `llama.cpp`/`bitnet.cpp`
  in some configs"), an above-the-fold benchmark table, visual proof, a one-command `make demo`,
  and exposed audit/QA links. Detailed technical content (CLI, architecture, DeepSeek, limits)
  is kept below, nothing removed.
- Honesty rule (binding): every headline win states its config; the losses stay visible —
  DeepSeek-V2 MoE ~7× behind llama.cpp (expert scatter) and SmolLM dense losing at T=1–2.
  Audience is systems devs who will scrutinize; an overclaim that doesn't survive `make demo`
  costs more credibility than it gains.
- Added: `make demo` target (downloads SmolLM2-135M GGUF, runs golden prompt; tokenizer is
  embedded so no `--tokenizer`), `docs/GROWTH_STRATEGY.md` (distribution playbook), and three
  committed proof images under `docs/` (fresh-Xeon terminal card + two OpenBenchmarking result
  screenshots, rendered via headless Chromium / Playwright).
- Reproduction note: the cloud host is itself an Intel Xeon 2.10 GHz / AVX-512 VNNI, so the
  fresh run (BitNet 40.42 tok/s, SmolLM2 142.39 tok/s, INT4) is legitimately comparable to the
  documented Xeon numbers. Project Zero reads BitNet only from native `.bin` (no ternary-GGUF
  support), so the BitNet model was converted from HF safetensors; bf16 tensors were re-encoded
  to f32 first because `convert_hf_bitnet.py` uses numpy (no bf16 slicing) — a one-off
  workaround, the committed tool was not modified.

### 2026-06-19 — Prebuilt x86-64 binary via a tagged GitHub Release
- Decision: Ship a portable prebuilt `adaptive_ai_engine` as a GitHub Release asset, built by a
  new `make dist` target and published by `.github/workflows/release.yml` on `v*` tag push.
  First release tagged `v0.1.0` (pre-1.0 → `prerelease`).
- Portability decision (key): `make release`'s `-march=native` is NOT distributable (the
  AVX-512-VNNI CI runner would bake in AVX-512 → SIGILL on older CPUs). Use **per-TU
  multiversioning**: bulk at `-march=x86-64-v2`, each SIMD kernel TU carries its own ISA flag,
  and `simd_dispatch.c` is compiled at the baseline with `-DTN_FORCE_DISPATCH_ALL` so AVX2/
  AVX-512/VNNI are selected at RUNTIME (the design `simd_dispatch` already implements). Static
  `-static-libstdc++ -static-libgcc` → only libc/libm at runtime.
- Min-CPU envelope: starts + runs BitNet ternary at x86-64-v2; quant/dense GGUF (Q4_K/F16) need
  AVX2 (those kernels are compile-time AVX2-or-scalar, called directly). Documented in
  `docs/RELEASING.md`.
- Supply chain: avoid third-party Actions (cut the release with the `gh` CLI), least-privilege
  `permissions` (only the publish job gets `contents: write`); SHA-pinning noted as a follow-up.
- Status: ACCEPTED; verified gcc release/test(46)/debug/dist + clang release/debug/dist green,
  golden output (France→Paris, Germany→Berlin) correct across scalar/avx2/avx512f/vnni and
  T=1/2/8. (clang `make test` is blocked only by a missing ASan runtime in the local container.)

### 2026-06-07 — Adopt a cross-tool AI development system with one source of truth
- Decision: `docs/ai/**` is canonical; Claude Code, GitHub Copilot, and Google Antigravity
  files are thin adapters that summarize and link here. All such files are dynamic and updated
  proactively every session.
- Rationale: prevent three drifting instruction systems; preserve context across tools.
- Sync rule: on a durable change, update `docs/ai/**` first, then adapters, then record in
  `change-trace.md`. See `tool-sync-policy.md`.

### 2026-06-07 — `master` is the canonical branch; `docs/...` branch archived, not merged
- Decision: Do not merge the unrelated-history `docs/readme-footer-openbenchmarking` branch.
  Verified `master` is a strict superset of its code; archived the branch as a git bundle and
  (pending) remove it from the public remote.
- Rationale: unrelated histories add no code; the branch holds stale tests and messy history.

### 2026-06-07 — Production OOM guard in `run_state_alloc`
- Decision: Add a deterministic size guard (reject buffers >32× available RAM) instead of
  relying on `calloc` returning NULL.
- Rationale: cross-platform determinism (macOS over-commits). 32× headroom never rejects a
  runnable config; all three models re-verified unchanged.
- Status: ACCEPTED, merged via PR #6.

### 2026-06-07 — CI fixes are test/build-only except the one guard above
- Decision: Keep production engine behavior unchanged; fix CI by fixing test harness + Makefile.
- Rationale: "no new development / no regression" constraint; the failures were test-side.

### 2026-06-07 — Regression measured by A/B on the same hardware
- Decision: Because the cloud CPU differs from documented i5-11300H/Xeon baselines, prove "no
  regression" by building HEAD vs a known-good commit on the same host and comparing tok/s +
  golden outputs, rather than comparing to the documented absolute numbers.
- Rationale: absolute tok/s isn't portable across CPUs; relative A/B is.
