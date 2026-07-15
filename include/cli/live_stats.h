#ifndef TN_CLI_LIVE_STATS_H
#define TN_CLI_LIVE_STATS_H

/*
 * Phase 22.3 — Live tok/s indicator for the REPL, updated once per streamed
 * token. Reuses cli/timer.h (timer_now_us/timer_tokens_per_sec) rather than
 * reimplementing rate math. TTY-gated (real call sites in repl.c pass
 * isatty(fileno(stdout))); non-TTY runs keep the existing end-of-generation
 * stderr summary line from generate_with_callback, so this is purely
 * additive.
 */

#include <stdint.h>

typedef struct {
    int64_t start_us; /* set on first tick */
    long long count;
} TnLiveStats;

void tn_live_stats_init(TnLiveStats *ls);

/* Call once per generated token. */
void tn_live_stats_tick(TnLiveStats *ls);

/* Renders the current tok/s (via timer_tokens_per_sec) to stderr as an
 * in-place (\r) update. No-op when is_tty is false or fewer than 2 tokens
 * have ticked yet. */
void tn_live_stats_render(const TnLiveStats *ls, int is_tty);

#endif /* TN_CLI_LIVE_STATS_H */
