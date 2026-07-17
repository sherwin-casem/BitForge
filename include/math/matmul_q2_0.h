#pragma once
/*
 * parallel_matmul_q2_0 — fused Q2_0 × F32 activation matmul (gemv).
 *
 * Q2_0 block layout (18 bytes / 64 elements, ggml block_q2_0):
 *   [d:  fp16  2B] — scale
 *   [qs: u8[16]  ] — 64 × 2-bit codes, 4 per byte
 *
 * Element j: code = (qs[j/4] >> ((j%4)*2)) & 0x3   (code in {0,1,2,3})
 * Value = (code - 1) * d      (00=-1, 01=0, 10=+1, 11=+2, all scaled by d)
 *
 * Used by PrismML's Qwen3.6-based "Bonsai" ternary GGUF releases, where
 * embeddings/attention/MLP/LM-head projections are all stored as Q2_0.
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void parallel_matmul_q2_0(float *out, const float *x, const uint8_t *w_q2_0,
                           int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q2_0_batch — batched Q2_0 × F32 GEMV for k weight matrices.
 */
void parallel_matmul_q2_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
