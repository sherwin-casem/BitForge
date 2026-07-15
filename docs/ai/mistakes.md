# Mistake Log — project-zero

> Canonical, append-at-top (newest first). Read this at the start of every session.
> Add an entry **immediately** when a mistake, false assumption, regression, or avoidable
> rework is found. Propagate durable lessons into `engineering-rules.md` and the tool adapters.
> Last updated: 2026-07-15.

## Entry template (copy this)
```
### YYYY-MM-DD — <short title>
- Summary: <what went wrong>
- Root cause: <why>
- Affected files/modules: <paths>
- Detection: <test / CI job / ASan / review>
- Correction: <the fix>
- Prevention rule: <durable rule; also added to engineering-rules.md / adapters if durable>
```

---

### 2026-07-15 — Live tok/s indicator overwrote the streamed response text
- Summary: the REPL's live tok/s status line (`\r` + overwrite + `\x1b[K`) clobbered the actual
  response text every single token, because both were written to the same terminal row with no
  forced newline between them — a screenshot taken for design QA showed the response reduced to
  a couple of stray characters, with the stats line sitting where the reply should have been.
- Root cause: `\r` moves the cursor to the start of whatever line it currently occupies; since
  streamed response tokens don't force their own line, the live-stats update and the response
  text shared a cursor position, so each stats redraw erased the response printed so far on that
  row.
- Affected files/modules: `src/cli/live_stats.c` (`tn_live_stats_render`).
- Detection: a real terminal screenshot taken for the Phase 22.4 design-QA pass — not caught by
  `tests/test_progress.c`/`test_color.c`/`test_md_render.c`, which test formatting logic in
  isolation, not interaction between two features writing to the same terminal row. A second,
  smaller mistake compounded this while fixing it: the first attempt used `"\x1b7"` as a single
  string-literal escape, which C parses greedily as one (invalid, out-of-range) hex escape
  because `7` is a valid hex digit — had to split it into `"\x1b" "7"`.
- Correction: save the cursor (`\x1b7`), jump to the terminal's last row (`\x1b[999;1H` —
  terminals clamp an out-of-range row to their actual height, no need to query real size), print
  the stats there, restore the cursor (`\x1b8`) — the standard "status line" technique, and it
  never touches the response's own cursor position.
- Prevention rule: two features that both write control sequences to the same stream/terminal
  must be verified together with a real render (screenshot or manual terminal check), not just
  unit-tested independently — formatting-logic tests cannot catch cross-feature terminal-state
  interaction. Also: never write a bare `\xNN` immediately followed by another hex digit in a C
  string literal — split into separate literal segments.

### 2026-07-15 — "Untested socket layer" hid four real HTTP protocol bugs
- Summary: `docs/ai/project-overview.md` had long flagged the Phase 21 HTTP server's "socket
  layer" as untested/not-in-CI — taken at face value rather than verified. Doing the first real
  end-to-end `curl` testing (Phase 22 API hardening) surfaced four separate, real bugs, none of
  which any existing test caught because `tests/test_api_server.c` only exercises pure-logic
  units (JSON parse, chat compile, SSE format) via pipes — never a real socket:
  1. `handle_connection`'s receive loop never broke out when a request had no `Content-Length`
     (the normal case for `GET`/`HEAD`/`OPTIONS`) — it kept calling `recv()` forever waiting for
     a body that would never arrive, hanging indefinitely on every plain `GET /health`.
  2. The non-streaming chat-completion path sent `Content-Length: 0` (from a `NULL` body) and
     only *then* wrote the real JSON via a second `write()` — compliant HTTP clients stop reading
     at 0 bytes and never see the actual response.
  3. Streaming responses claimed `Transfer-Encoding: chunked` but the SSE writer emitted raw,
     unframed bytes (no hex chunk-size prefixes) — a real protocol violation that curl tolerated
     by accident but a strict client (browser `fetch`/`EventSource`) would not.
  4. The server never called `signal(SIGPIPE, SIG_IGN)`, so any client disconnecting mid-response
     delivered `SIGPIPE` to `write()`, whose default disposition kills the *entire* process, not
     just that connection — trivially triggered by any short `curl --max-time`.
- Root cause: all four are consequences of never having driven the server with a real HTTP
  client. Logic-level pipe tests validate formatting functions in isolation but cannot catch bugs
  in the receive-loop state machine, header/body sequencing, or process-wide signal disposition.
- Affected files/modules: `src/api/http_server.c` (recv loop, `send_response_ex`, `SIGPIPE`),
  `src/api/sse_stream.c`/`include/api/sse_stream.h` (new `sse_format_full_response()` so the
  caller can compute `Content-Length` before sending headers).
- Detection: manual `curl`/`curl --raw` end-to-end smoke testing against a real running server
  with a real model — not caught by `make test`, ASan, UBSan, or code review of the diff alone.
- Correction: stop reading once headers are complete and no `Content-Length` is present; build
  the JSON body before sending headers so `Content-Length` is correct; drop the false
  `Transfer-Encoding: chunked` claim (the response already closes the connection, so EOF
  delimits the body per RFC 7230 §3.3.3 case 7); ignore `SIGPIPE` at server start.
- Prevention rule: a component documented as "untested" is not verified — before extending or
  hardening it, exercise it for real (a live server + `curl`/`curl --raw`, not just unit tests)
  rather than trusting the label. Any new HTTP/socket work in this codebase must include a real
  end-to-end request/response check, not only logic-level tests.

### 2026-06-07 — `make debug` never linked the sanitizer runtime / lacked `-march=native`
- Summary: After tests passed, `make debug` failed for both compilers (undefined `__asan_*`,
  then undefined `ternary_matmul_packed_avx2/avx512`).
- Root cause: `debug` compiled objects with `-fsanitize` but the `$(TARGET)` link omitted it;
  and debug had no `-march=native`, so feature-gated fallback TUs compiled empty while VNNI
  dispatch TUs (built with explicit `-mavx512vnni`) referenced them.
- Affected files/modules: `Makefile` (`debug` target, `$(TARGET)` link, `CFLAGS/CXXFLAGS_DEBUG`).
- Detection: GitHub CI `make debug` step (newly reached after `make test` was fixed).
- Correction: debug `LDFLAGS += -fsanitize=address -fsanitize=undefined`; add `-march=native`
  to `CFLAGS_DEBUG`/`CXXFLAGS_DEBUG`; add `-mavx512vl` to the 256-bit VNNI rule.
- Prevention rule: a CI step only validates what it reaches; verify the **full** sequence
  (release+test+debug) for gcc **and** clang locally before declaring CI fixed.

### 2026-06-07 — Uninitialized struct field → nondeterministic ASan stack-overflow
- Summary: `test_vision_projector` crashed (read 64B past `patches`) only on the non-AVX512
  clang runner; passed elsewhere.
- Root cause: `VisionProjector proj;` left `scale_factor` uninitialized; garbage `>1` selected
  the pixel-shuffle path that over-reads.
- Affected files/modules: `tests/test_vision_components.c`, `src/multimodal/vision_projector.c`.
- Detection: GitHub CI ubuntu-22.04 clang ASan.
- Correction: `memset(&proj,0,sizeof(proj))` before use.
- Prevention rule: zero every struct before partial init (same class as the weights bug below).

### 2026-06-07 — Reliance on platform malloc behavior for OOM trapping
- Summary: `rb_mem_02_oom_resistance` (INT_MAX context) got OS `Killed:9` on macOS though it
  returned `TN_ERR_OOM` on Linux.
- Root cause: `run_state_alloc` depended on `calloc` returning NULL for absurd sizes; macOS
  over-commits then the OOM killer fires.
- Affected files/modules: `src/core/run_state.c`, `tests/test_redbox.c`.
- Detection: GitHub CI macOS job.
- Correction: deterministic guard `tn_alloc_too_large()` (overflow check + reject >32× free RAM
  via `tn_get_free_ram`) before the big allocations.
- Prevention rule: trap pathological sizes explicitly; never depend on allocator failure modes.

### 2026-06-07 — Missing `memset` before `weights_alloc_pointers` (caller contract)
- Summary: `test_blackbox` and `audit_sliding_window_crash` aborted under ASan (invalid free /
  wild-pointer memcpy).
- Root cause: tests declared `TransformerWeights` on the stack without zeroing; the documented
  contract requires the caller to `memset` first, else `weights_free_pointers` frees garbage and
  uninitialized `layers_are_ternary`/`layer_weight_type` misroute the forward pass.
- Affected files/modules: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`
  (contract in `src/core/weights.c`).
- Detection: GitHub CI ubuntu-22.04 + macOS ASan.
- Correction: add `memset(...)` (+ `#include <string.h>`) before `weights_alloc_pointers`.
- Prevention rule: honor documented zero-first caller contracts; codified in engineering-rules.md.

### 2026-06-07 — Initial parity confusion: filtered diff hid the real branch delta
- Summary: A `git diff -- src/ include/` suggested only 4 files differed between `master` and the
  unrelated `docs` branch; a full diff showed many more (tests, Makefile, docs, junk logs).
- Root cause: over-narrow diff path filter.
- Affected files/modules: branch-comparison process.
- Detection: cross-checking with `git diff --name-status` (no path filter).
- Correction: always run an unfiltered name-status diff first, then drill down.
- Prevention rule: verify branch parity with a full diff before concluding; recorded in decision-log.
