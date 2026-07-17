# RCA — Ternary-Bonsai-27B throughput drop: ~2.74 → ~1 tok/s (2026-07-16/17)

**Verdict: not a code regression.** The drop is a change in the underlying (virtualized) host's
performance profile, which happened to coincide in time with the classifier-compatibility work.
The exact commit that produced 2.74 tok/s, rebuilt and rerun unmodified on the current host,
measures ~1.0 tok/s — the same as current HEAD. A pre-VNNI control commit still measures ~13x
slower than both, proving the test methodology detects real code-driven gaps when they exist.

## 1. Symptom timeline

All runs: same `Ternary-Bonsai-27B-Q2_0.gguf` (7.16 GB), same prompt, `--max-tokens 60
--temperature 0 --threads 4`, greedy decoding. "Host profile" is the engine's own auto-detected
Hardware Profile printed at startup (visible in every screenshot).

| When | Build | Host profile (L2/core · L3 · DRAM bw · ceiling) | Result |
|---|---|---|---|
| 07-16 17:40 | `ce8e90d-dirty` (VNNI Q2_0 kernel) | 2048 KiB · **260 MiB** · **16.0 GB/s** · 17.1 tok/s | **2.74 tok/s** (t=4 peak of the 0.86/1.62/2.31/2.74 thread sweep) |
| 07-16 evening | host recycled mid-session (`/proc/uptime` ≈ 5 min) | 1024 KiB · **33 MiB** · 9.9–12.3 GB/s | classifier sweeps land at ~1.07–1.19 tok/s |
| 07-17 11:08 | HEAD `85d3b36` | 1024 KiB · 33 MiB · 12.2 GB/s · 10.4 tok/s | 0.59 tok/s (`classifier_auto.png`) |
| 07-17 11:51 | **`ce8e90d` rebuilt from that exact commit** | 1024 KiB · 33 MiB · 10.9 GB/s · 9.4 tok/s | **1.02 tok/s** (1.40 on an earlier same-day rerun) |
| 07-17 same session | `34d3ac9` (control: parent of `ce8e90d`, pre-VNNI kernel) | same host | **0.08 tok/s** (0.12 earlier) — the expected ~13x code gap still shows |
| 07-17 same session | HEAD, third leg | same host | 0.95 tok/s (1.08 earlier) |

Screenshots (primary evidence, hardware box + `[gen]` line embedded in each):
`benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/commit_bisect_{ce8e90d_1.02toks,34d3ac9_0.08toks,HEAD_0.95toks}.png`.
The original 2.74 capture with the host-A hardware box is recoverable from git history:
`git show 591333d:benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/pz_t4_peak.png`
(the in-tree copy was later recaptured on host B to fix display bugs and shows 1.07).

Note: the sweep between the two states was a **thread** sweep (t=1..4 = `--threads`), not a
temperature sweep — `--temperature 0` was constant throughout.

## 2. Suspects examined and ruled out

Every commit between the 2.74 measurement (`ce8e90d`) and HEAD was audited
(`git diff ce8e90d..HEAD -- src/ include/`: 16 files). The only ones touching engine code:

1. **`85d3b36` classifier materialization (the prime suspect — the drop was noticed right after
   this work).** `forward.c`'s LM-head dispatch gained a branch, but it is guarded by
   `hp->classifier_explicit`, which is set **only** by `tn_hardware_profile_set_classifier()`
   when `--classifier` is passed on the CLI. Without the flag, execution falls through to the
   byte-identical `parallel_matmul_q2_0(...)` call as before, plus one `tn_hardware_profile_get()`
   and two predictable branches **per token** (not per row) — nanoseconds against a ~1 s token.
   The loader-side materialization in `gguf_loader.c` is behind the same flag: default loads
   allocate nothing extra.
2. **`921e223` VBMI SIGILL fix (runtime dispatch instead of `#if`).** `bitunpack2_vnni.h`'s
   unpack now branches on a cached per-TU `static int` (one load+compare per 64 weights,
   fully predicted) and `matmul_i4_task` hoists the check once per task. Cannot cost 2.7x on a
   bandwidth-bound kernel.
3. **`dd295e5` detokenizer/batch-path fixes.** The Q2_0 batch path has no caller for dense
   models; the detokenizer is per-emitted-token string handling, off the matmul hot path.

Ruling out by diff was then made falsifiable by the commit bisection above: even if some audit
argument were wrong, **the pre-fix binary itself runs at ~1 tok/s today**. Conversely, the
`34d3ac9` control run proves the harness reliably resolves a genuine code-level performance gap
on this same host, so the absence of a `ce8e90d`-vs-HEAD gap is signal, not insensitivity.

Two real bugs *were* found during the classifier work — the `--classifier` no-op on Q2_0-native
models and the CPUID-advertised-but-faulting AVX-512VBMI SIGILL — but both affected only
explicit `--classifier` runs, not the default path that regressed from 2.74 to ~1.

## 3. Root cause

The sessions run on a Firecracker microVM whose **underlying physical host changed and remains
unstable**:

- **Different hardware between the two sweeps**: the engine's own profiler recorded L2
  2048 → 1024 KiB/core, L3 **260 → 33 MiB**, measured DRAM bandwidth 16.0 → 9.9–12.3 GB/s.
  A ~27B dense model reading ~1149 MB/token is bandwidth-bound, so the engine's calibrated
  ceiling fell 17.1 → 9.4–10.7 tok/s (~40% of the drop) purely from the ceiling moving.
- **Co-tenant contention on the new host**: achieved-vs-ceiling efficiency also fell
  (16% at 2.74/17.1 → 6–12% across host-B runs), and re-runs stalled minutes in first-touch of
  fresh memory (`mmap(MAP_POPULATE)` of the model, KV-cache `calloc`) while guest-side memory
  stats stayed healthy — the signature of hypervisor-level memory pressure invisible to the
  guest. Direct live evidence of the noise floor: within single calibration runs minutes apart,
  the same T=4 microbench measured 3.5 tok/s and 35.6 tok/s.
- Even the VBMI-faults-on-execute behavior appears host-state-dependent: `ce8e90d` (which
  executes VBMI unconditionally) crashed with SIGILL on the 07-16-evening host but completed
  calibration cleanly during the 07-17 bisection — consistent with continued host churn.

Full evidence chain: `docs/ai/mistakes.md` (2026-07-17 entries) and `docs/ai/decision-log.md`.

## 4. Solution

**Restore.** There is nothing to revert or patch: the current binary already produces
~2.7 tok/s-class results whenever it runs on host-A-class hardware. To confirm which class a
session landed on, read the startup Hardware Profile box (L3 size and measured DRAM bandwidth
are an effective host fingerprint) before trusting any absolute number. On this environment:

- Treat cross-session absolute tok/s as non-comparable; only same-session, back-to-back A/B
  sweeps are valid (already codified in `mistakes.md` and the README caveat).
- For regression checks, always include a known-slow control leg (as the bisection did) so
  "no gap" is distinguishable from "test can't see gaps."

**Improve (available today, no code change).** On the one clean post-fix same-session sweep,
`--classifier int8`/`int4` measured **2.60/2.62 tok/s vs 1.19 for the zero-copy default** —
the materialized INT8/INT4 LM head avoids the Q2_0 per-block unpack entirely and the VNNI
int8/int4 kernels are more compute-efficient per element for the 248320×5120 head, at the cost
of 0.6–1.2 GB extra RAM. Caveat: a later contended-host sweep did not reproduce the magnitude
or ordering, so verify with a same-session A/B on the target host before relying on it.

**Improve (code-side follow-ups — require a stable host to A/B first, per engineering rules).**

1. Profile the zero-copy Q2_0 LM head (`n=5120, d=248320` — confirmed to take the VNNI path,
   `TN_Q2_0V_MAX_N=20480`). The int8-vs-Q2_0 result implies the per-64-element unpack
   (~15-instruction SSE variant whenever verified VBMI is unavailable) dominates on weak cores;
   options include a row-blocked unpack-once-reuse scheme or auto-selecting the materialized
   INT8 head when `tn_get_free_ram()` shows headroom (would need a deliberate reversal of the
   opt-in-only decision in `decision-log.md` 2026-07-17).
2. General headroom: this model achieves 6–16% of the calibrated bandwidth ceiling vs 95% for
   BitNet on the same engine — the Qwen35/Q2_0 pipeline (hybrid attention + per-token Q2_0
   embedding/head dequant) has real optimization room, but none of it is measurable on the
   current host, so it is deliberately not attempted in this pass.
