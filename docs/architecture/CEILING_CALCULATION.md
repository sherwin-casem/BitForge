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

`probe_dram_bandwidth()` — `src/core/hardware_profile.c:130` (worker: `bw_thread_fn`, :106).

- One thread per physical core (clamped to [1,8]); per-thread 512 MB buffer (grown to
  L3+64 MB if L3 > 512 MB; total capped at 4 GB). Pages faulted in by writing every 4096th
  byte before timing.
- Access pattern: sequential **read of one `volatile char` per 64-byte cache line**,
  accumulated into a `volatile int64_t` sink. Bytes counted = full buffer size (each touched
  line is a full line fill from DRAM, so the accounting is correct).
- 3 passes, best kept, `CLOCK_MONOTONIC` wall time; the returned figure is the *aggregate*
  across concurrently running threads (deliberately includes bus contention).

**Known conservative bias.** The `volatile` scalar chain executes one serialized 1-byte load
+ 1 store (to the volatile sink) per line and cannot keep many line fills in flight; the
measurement leans on the hardware prefetcher alone. Measured on a host-B-class VM
(2026-07-17): probe = **12.0 GB/s**, while a plain 512 MB `memcpy` loop on the same host
measured **~22 GB/s effective** (read+write counted). True streaming-read bandwidth is
therefore materially higher than the probe reports, and every ceiling derived from it is a
**lower-bound estimate** — the engine legitimately exceeding a printed ceiling (e.g. host A's
2.74 tok/s vs a 16.0 GB/s-derived ~2.3 ceiling, or SmolLM2's 56.5 vs 44.1) is expected, not
anomalous. Improvement path: multi-accumulator vectorized read loop (§7 option B).

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

## 5. Worked examples (all verified live on 2026-07-17, host-B-class VM, probe 12.0-12.2 GB/s)

| Model | weight_bytes/token | Ceiling @ probe BW | Measured | Notes |
|---|---|---|---|---|
| BitNet-2B native, BF16 cls | ~1179 MB pre-L3-heuristic | (host-dependent) | 95% of ceiling historically | The class the constants were built for |
| SmolLM2-135M F16 | 258 MB (real file) vs 1149 MB (old constants) | 44.1 tok/s | 56.5 tok/s | Exceeds ceiling: probe bias + partial cache residency; old constants *under*-stated 4.5x |
| Ternary-Bonsai-27B Q2_0 | ~6.8 GB (7.165 GB file − 338 MB embed) vs "1149 MB" (old) | ~1.7-1.8 tok/s | 0.95-1.24 tok/s (this host class); 2.74 on the 16 GB/s host A | Old ceiling overstated ~6x; host A's 2.74 ≈ 18.6 GB/s effective — at/above its probe figure |

## 6. Error sources, bounded

| Source | Direction | Magnitude |
|---|---|---|
| Probe under-measures streaming bandwidth | Ceiling too LOW | ~1.3-1.8x on measured hosts (12.0 probe vs ~22 GB/s memcpy-effective) |
| KV traffic excluded | Ceiling too HIGH at long context | ~2% @4K ctx, ~10% @22K (27B model; §3) |
| Generic-GGUF embedding over-count | Ceiling too LOW | ≤ embed/file share (few % for large models; ~0 for tied-embedding) |
| MoE over-count | Ceiling too LOW | Up to (total−active)/total expert share — unquantified, TODO |
| Cross-session host drift | Both (invalidates comparisons) | Documented at length in the RCA; compare same-session only |

## 7. Bridging the gap between measured tok/s and the ceiling

Current gap on host-B-class hardware for Ternary-Bonsai-27B: measured 0.95-1.24 vs ~1.7
ceiling (~60-70% utilization). Options considered:

- **A. Kernel efficiency (implemented this pass — see Review log / benchmark records)**:
  `dot_q2_0_row_vnni()` (`src/math/matmul_q2_0_vnni.c`) paid a full cross-lane
  `_mm512_reduce_add_epi32` **per 128-element block** plus a branchy scalar fp16 scale
  conversion per block. A1: per-row float vector accumulator (`_mm512_cvtepi32_ps` + FMA by
  broadcast scale, one horizontal reduce per row, `Σ sum_qx·d` correction accumulated
  scalar-side). A2: F16C scale decode. A3 (stretch): wider AVX-512BW 2-bit unpack for hosts
  without (working) VBMI; multi-row blocking. Verified by `tools/bench_q2_0` (both kernel
  variants in one binary — host-drift-immune A/B) + `tests/test_q2_0_matmul.c` correctness
  + a same-session end-to-end model A/B.
- **B. Measurement honesty**: fix the probe's serialized read loop (multi-accumulator,
  vectorized) so the reported ceiling stops under-stating reality. Raises the *target*, not
  the throughput; changes calibration-cache hardware fingerprint inputs (acceptable — that
  cache re-triggers on hardware change by design).
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
