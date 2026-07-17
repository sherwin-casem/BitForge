# Ceiling Calculation — how project-zero derives its tok/s ceiling

> Spec + audit of every place the engine computes "Data/token" and "Ceiling (at 100% BW)".
> Written 2026-07-17 during the Qwen throughput RCA, after the pre-load ceiling was found to
> be computed from hardcoded BitNet-2B constants for every model
> (`docs/ai/mistakes.md` 2026-07-17; fixed by `tn_hardware_profile_set_model_bytes()`).
> Companion documents: `docs/reports/RCA_QWEN_TOKS_DROP_2026-07.md` (why this was audited),
> `docs/PERFORMANCE_CEILING_REPORT.md` (historical BitNet-era ceiling work).

## 1. The model behind the number

Autoregressive CPU decode of a dense model is **DRAM-bandwidth-bound**: every generated token
must stream (nearly) all weight bytes through the cores once. The ceiling is therefore:

```
ceiling_tokps = measured_bandwidth_bytes_per_sec / weight_bytes_per_token
```

Everything else in this document is about how the two operands are obtained, where each of
the four computation sites gets them wrong or right, and how big the error bars are.

## 2. Operand 1 — measured DRAM bandwidth

`probe_dram_bandwidth()` — `src/core/hardware_profile.c` (worker: `bw_thread_fn`). Current
(fixed 2026-07-17) methodology:

- One thread per physical core (clamped to [1,8]); per-thread 512 MB buffer (grown to
  L3+64 MB if L3 > 512 MB; total capped at 4 GB). Pages faulted in by writing every 4096th
  byte before timing.
- Access pattern: each thread reads **every byte** of its buffer through 8 independent
  64-bit accumulator chains (one cache line per iteration — the compiler vectorizes, and
  independent chains let the core keep many line fills in flight), one pass per timed round.
- 3 rounds; each round's aggregate = total bytes across all threads / wall-clock time of the
  create-run-join round (deliberately includes bus contention; thread create/join overhead
  biases slightly conservative). Best round kept.

**History — two bugs fixed on 2026-07-17 that had compounded to ~3.4x under-measurement**
(full writeup in `docs/ai/mistakes.md`):

1. *Accounting (~3x):* each worker ran its 3 read passes **inside one timed round**, while
   the wall-clock aggregate divided only one pass worth of bytes by the elapsed time of all
   three. The correctly-accounted per-thread figures were computed and then discarded.
2. *Serialized reads:* one `volatile char` load + a volatile-sink store per 64-byte line — a
   dependency chain that cannot keep line fills in flight.

Same host, same session: old probe **12.0 GB/s** → fixed probe **41.2 GB/s**. Every
"DRAM bandwidth (measured)" figure printed before the fix (16.0, 14.0–17.3, 12.2, 10.9…)
is ~3x low, and every ceiling derived from one inherits that. Cross-checks that exposed it:
a plain memcpy loop measured ~22 GB/s *effective* (read+write ≈ 44 GB/s of bus traffic —
consistent with 41.2), and the Q2_0 kernel itself streams 29 GB/s of packed weights from a
larger-than-L3 buffer (`tools/bench_q2_0`), which no honest 12 GB/s host could do.
The strongest tell in hindsight: under the old probe, SmolLM2 measured 56.5 tok/s against a
44.1 tok/s "100% of bandwidth" ceiling — a physical impossibility for a >L3 working set,
which should have been read as evidence the denominator was wrong, not as "prefetch
effectiveness" (the in-code comment's explanation). Under the fixed probe the ceiling is a
genuine upper bound again (SmolLM2: 56.5 measured vs ~152 ceiling, 37%).

## 3. Operand 2 — weight bytes per token, per model class

| Model class | Per-token weight traffic | Accuracy |
|---|---|---|
| BitNet-2B native | ternary layers (522 MB) + classifier (BF16 656 / INT8 328 / INT4 164 MB) + norms (~1 MB) | Correct — the compile-time constants were written for exactly this model |
| Generic GGUF dense (e.g. SmolLM2 F16) | GGUF file size (upper bound: over-counts the embedding table, which is read one row/token; for tied-embedding models the "embedding" *is* the streamed LM head, so the bound is tight) | Good |
| Q2_0-native GGUF (Ternary-Bonsai-27B) | file size − embedding (vocab·dim·34/128 B) − in-file Q2_0 head + head actually used (raw Q2_0 head again, or the materialized BF16/INT8/INT4 classifier at vocab·dim·{2,1,0.5} B when `--classifier` was passed) | Good |
| MoE GGUF (DeepSeek-V2) | file size (over-counts: inactive experts are not read) | **TODO** — subtract inactive-expert bytes via `MoEConfig` |

**Excluded on purpose: KV-cache traffic.** It is position-dependent, not constant per token.
Worked bound for Ternary-Bonsai-27B (I8 KV, 16 full-attention layers of 64, 4 KV heads ×
256 head_dim): 2·4·256·1 B × 16 layers = **32 KB per cached position**, so attention reads
~134 MB/token at 4K context (~2% of the 6.8 GB weight stream) and ~713 MB/token at the 22K
max context (~10%). The ceiling is thus slightly optimistic at long contexts; the DeltaNet
layers hold O(1) state and add nothing that scales with position.

## 4. The four computation sites

1. **`tn_hardware_profile_init()`** — `src/core/hardware_profile.c` (constants at :23-27,
   projection at :427+). Runs **before the model file is opened** — it cannot know the real
   model, so it seeds `weight_bytes_per_tok` from the BitNet-2B compile-time constants and
   applies an **L3-retention heuristic**: classifier counted as cached if it fits in L3, then
   `min(L3 − classifier, ternary_bytes)` of layer weights also counted cached, remainder
   billed to DRAM; if everything fits, ceiling prints 999 ("compute-limited"). Note the
   header comment says "~half stay cached" but the code caches the `min()` expression — the
   literal ×0.5 heuristic lives only in calibration (site 4). Until 2026-07-17 this pre-load
   estimate was never corrected afterwards — the root cause of the "1149 MB / 17.1 tok/s"
   fiction for the 27B model. The startup box now labels these two lines "(pre-load est.)".
2. **`tn_hardware_profile_set_classifier()`** — :497. Re-derives classifier bytes for the
   chosen format (still from `MODEL_VOCAB`·`MODEL_DIM` constants — pre-load by necessity,
   same caveat) and recomputes the ceiling as a **plain division**, no L3 heuristic.
3. **`tn_hardware_profile_set_model_bytes()`** — :523, called from `src/cli/main.c` (≈:290)
   after GGUF weights load. The correction: real file size with the Q2_0 embedding/head
   adjustment from §3, plain division, updates `model_fits_l3` and the summary, and main.c
   prints `[profile] Data/token (loaded model): … -> ceiling …` (`main.c:321`). The L3
   heuristic is deliberately dropped here: at multi-GB per-token streams, L3 covers <1% and
   the adjustment is noise; for small models it would only widen an already-conservative
   lower bound.
4. **`tn_calibrate()` classifier estimates** — `src/core/calibration.c:403-423`. An
   *analytic* (not measured) per-format tok/s estimate feeding `--classifier auto-fast`:
   hardcoded 522 MB ternary + BitNet classifier dims, L3 retention modeled as
   `min(L3, ternary)·0.5` when the model doesn't fully fit. Same pre-load limitation as
   site 1, hardcoded independently (second copy). As of this change the result box prints
   the actually-computed best format and percentage, labeled "est." — previously both were
   a hardcoded `(INT8, ~+36%)` string literal regardless of the computation. Making
   calibration model-aware is TODO (it runs pre-load by design; the calibration cache is
   keyed on hardware, not model).

Consumers: CLI stdout only (startup box, `Active:` summary line, post-load `[profile]` line,
calibration box). Nothing in `src/api/` (`/metrics`, web UI) reads any of these fields.

## 5. Worked examples (verified live on 2026-07-17, host-B-class VM; **fixed probe: 41.2 GB/s**)

| Model | weight_bytes/token | Ceiling @ 41.2 GB/s | Measured | Utilization |
|---|---|---|---|---|
| SmolLM2-135M F16 | 258 MiB (real file; old constants said 1149 — 4.5x over) | ~152 tok/s | 56.5 tok/s | ~37% |
| Ternary-Bonsai-27B Q2_0 | 6511 MiB (7.165 GB file − embed − in-file head + zero-copy head) | **6.0 tok/s** | **3.56 tok/s** (optimized kernel; 2.74-2.80 pre-optimization, same session interleaved) | **59%** (was 45%) |
| BitNet-2B native, BF16 cls | ~1179 MB pre-L3-heuristic (constants correct for this model) | (host-dependent) | — | Historical "95% of ceiling" claims used the pre-fix probe, so true utilization was ~3x lower; re-measure before citing |

Historical translation: host A's famous 2.74 tok/s ≈ 18.6 GB/s of effective weight streaming.
Its probe-reported "16.0 GB/s" was ~3x low (true ~48 GB/s), so that run sat at roughly **40%
of its real ceiling (~7 tok/s)** — not "at the bandwidth wall" as the pre-fix numbers
suggested. The optimized kernel's 3.56 tok/s on today's host already exceeds it.

## 6. Error sources, bounded

| Source | Direction | Magnitude |
|---|---|---|
| Probe under-measurement (**fixed 2026-07-17**) | Ceiling was too LOW | was ~3.4x (12.0 → 41.2 GB/s same host); residual bias now limited to thread create/join overhead (small, conservative) |
| KV traffic excluded | Ceiling too HIGH at long context | ~2% @4K ctx, ~10% @22K (27B model; §3) |
| Generic-GGUF embedding over-count | Ceiling too LOW | ≤ embed/file share (few % for large models; ~0 for tied-embedding) |
| MoE over-count | Ceiling too LOW | Up to (total−active)/total expert share — unquantified, TODO |
| Cross-session host drift | Both (invalidates comparisons) | Documented at length in the RCA; compare same-session only |

## 7. Bridging the gap between measured tok/s and the ceiling

Current state on this host for Ternary-Bonsai-27B, after this pass: **3.56 tok/s measured vs
6.0 tok/s ceiling (59%)**, up from 2.74-2.80 (45%) pre-optimization. Options considered:

- **A. Kernel efficiency (IMPLEMENTED, verified)**:
  `dot_q2_0_row_vnni()` (`src/math/matmul_q2_0_vnni.c`) paid a full cross-lane
  `_mm512_reduce_add_epi32` **per 128-element block** plus a branchy scalar fp16 scale
  conversion per block. A1: per-row float vector accumulator (`_mm512_cvtepi32_ps` + FMA by
  broadcast scale, one horizontal reduce per row, `Σ sum_qx·d` correction accumulated
  scalar-side). A2: F16C scale decode. A3 (stretch, NOT yet done): wider AVX-512BW 2-bit
  unpack for hosts without (working) VBMI; multi-row blocking. **Results** —
  `tools/bench_q2_0` (both variants in one binary, host-drift-immune, 4 threads, medians of
  7 interleaved reps): attn 5120×5120 **1.32x**, ffn-down 17408×5120 **1.48x**, lm-head
  5120×248320 **1.68x** (17.3 → 29.0 GB/s), max output diff ~1e-6; `tests/test_q2_0_matmul`
  green. End-to-end (real model, interleaved same-session, 60-token greedy, 4 threads):
  baseline 2.74/2.80 → optimized **3.56/3.54 tok/s (+28%)**, generated text
  token-identical; `TN_STEP_TIMING=1`: Dense FFN 17.7 → 13.1 s, LM head 1.19 → 0.83 s per
  61-token generation.
- **B. Measurement honesty (IMPLEMENTED)**: fixed the probe's ~3x accounting error and its
  serialized read loop (§2). Raises the *target*, not the throughput: same host re-measured
  12.0 → 41.2 GB/s, giving the honest 6.0 tok/s ceiling above.
- **C. Configuration (exists)**: `--classifier int8/int4` trades RAM for LM-head kernel
  efficiency (2.60/2.62 vs 1.19 tok/s in the one clean same-session sweep; re-evaluate after
  A — a fixed Q2_0 kernel should let zero-copy win, since it streams 3.6x fewer head bytes).
  Calibrated thread count (t=4 confirmed optimal on 4-core hosts).
- **D. System-level: not applicable here.** The first-touch stalls documented in the RCA are
  host-side (hypervisor memory pressure); hugepages/`MAP_POPULATE` tuning can't fix a
  contended underlying host, and this 4-core VM has no NUMA dimension.
- **E. Architectural (future work, out of scope)**: batched decode / speculative decoding —
  amortize one weight stream across several tokens; the only route *above* the single-stream
  bandwidth wall. At 2.125 bits/weight there is no smaller format left to stream.
- **F. Hardware**: a host-A-class machine (260 MB L3, 16+ GB/s) restores ~2.7 tok/s with no
  code change at all — the cheapest 2.2x available.

## 8. Review log

Independently reviewed on 2026-07-17 by a fresh-context reviewer (no access to the authoring
session's derivation) — findings and dispositions:

- *(populated after the review pass)*
