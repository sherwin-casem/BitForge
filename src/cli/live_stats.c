/*
 * Phase 22.3 — src/cli/live_stats.c
 * See include/cli/live_stats.h for responsibility.
 */

#include "cli/live_stats.h"
#include "cli/timer.h"
#include <stdio.h>

void tn_live_stats_init(TnLiveStats *ls) {
    if (!ls) return;
    ls->start_us = 0;
    ls->count = 0;
}

void tn_live_stats_tick(TnLiveStats *ls) {
    if (!ls) return;
    if (ls->count == 0) ls->start_us = timer_now_us();
    ls->count++;
}

void tn_live_stats_render(const TnLiveStats *ls, int is_tty) {
    if (!ls || !is_tty) return;
    if (ls->count < 2) return;
    double rate = timer_tokens_per_sec(ls->start_us, timer_now_us(), (int)ls->count);
    fprintf(stderr, "\r\x1b[2m[%lld tok, %.1f tok/s]\x1b[K\x1b[0m", ls->count, rate);
    fflush(stderr);
}
