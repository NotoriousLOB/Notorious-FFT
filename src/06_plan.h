/*
 * Notorious FFT - Plan Creation with Runtime Algorithm Selection
 */

#ifndef NOTORIOUS_FFT_PLAN_H
#define NOTORIOUS_FFT_PLAN_H

#include "05_algorithms.h"

/* Forward declarations */
static notorious_fft_plan* notorious_fft_create_plan_power2(size_t n);
static void notorious_fft_destroy_plan(notorious_fft_plan* plan);

/* ============================================================================
 * Slab sizing helpers
 *
 * Every field that will be bump-allocated is rounded up to NOTORIOUS_FFT_BUMP_ALIGN
 * bytes.  We define a single macro so the accounting in the "compute total"
 * and the "bump alloc" steps stay in sync.
 * ============================================================================ */

#define NOTORIOUS_FFT_SLAB_FIELD(bytes) NOTORIOUS_FFT_BUMP_ROUND(bytes)

/* ============================================================================
 * Bluestein Plan Creation
 * ============================================================================ */

static notorious_fft_plan* notorious_fft_create_plan_bluestein(size_t n, int inverse) {
    if (n == 0) return NULL;

    /* Next power of 2 >= 2*n-1 */
    size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;

    /* Slab layout (high→low, allocated by decrementing bump pointer):
     *
     *   [notorious_fft_plan struct]           ← slab base (low address)
     *   ~~~~ padding to 64 bytes ~~~~
     *   bluestein_chirp_re  [m reals]
     *   bluestein_chirp_im  [m reals]
     *   bluestein_chirp_fft_re [m reals]
     *   bluestein_chirp_fft_im [m reals]
     *   bluestein_buf_re    [m reals]
     *   bluestein_buf_im    [m reals]
     *   bluestein_fft_buf_re [m reals]
     *   bluestein_fft_buf_im [m reals]
     *   work_re             [m reals]
     *   work_im             [m reals]   ← slab end (high address, bump starts here)
     */
    size_t real_bytes = m * sizeof(notorious_fft_real);
    size_t total = NOTORIOUS_FFT_SLAB_FIELD(sizeof(notorious_fft_plan))
                 + NOTORIOUS_FFT_SLAB_FIELD(real_bytes) * 10;  /* 10 arrays of m reals */

    void* slab = notorious_fft_malloc(total);
    if (!slab) return NULL;

    /* Plan lives at the base of the slab */
    notorious_fft_plan* plan = (notorious_fft_plan*)slab;
    memset(plan, 0, sizeof(notorious_fft_plan));

    plan->slab         = slab;
    plan->n            = n;
    plan->is_inverse   = inverse;
    plan->execute_func = notorious_fft_execute_bluestein;
    plan->bluestein_n  = m;

    /* Bump pointer starts at the end of the slab and decrements downward */
    char* bump = (char*)slab + total;

    plan->work_im              = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->work_re              = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_fft_buf_im = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_fft_buf_re = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_buf_im     = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_buf_re     = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_chirp_fft_im = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_chirp_fft_re = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_chirp_im   = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    plan->bluestein_chirp_re   = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, real_bytes);
    /* bump now points just above sizeof(notorious_fft_plan) — slab is fully packed */

    /* Inner power-of-2 plan (owns its own slab, freed separately) */
    plan->bluestein_plan = notorious_fft_create_plan_power2(m);
    if (!plan->bluestein_plan) {
        goto cleanup;
    }

    /* Create filter h[n] = exp(πin²/N) for n = -(N-1)..(N-1) in work buffers */
    for (size_t k = 0; k <= 2*(n-1); k++) {
        int idx = (int)k - (int)(n - 1);
        notorious_fft_real angle = NOTORIOUS_FFT_PI * (notorious_fft_real)idx * (notorious_fft_real)idx / (notorious_fft_real)n;
        plan->work_re[k] = notorious_fft_cos(angle);
        plan->work_im[k] = notorious_fft_sin(angle);
    }
    for (size_t k = 2*n-1; k < m; k++) {
        plan->work_re[k] = 0;
        plan->work_im[k] = 0;
    }

    /* FFT the filter h → bluestein_chirp_fft */
    notorious_fft_execute_iterative(plan->bluestein_plan,
                             plan->work_re, plan->work_im,
                             plan->bluestein_chirp_fft_re, plan->bluestein_chirp_fft_im);

    /* Chirp a[n] = exp(-πin²/N) for n = 0..N-1 */
    for (size_t i = 0; i < n; i++) {
        notorious_fft_real angle = -NOTORIOUS_FFT_PI * (notorious_fft_real)i * (notorious_fft_real)i / (notorious_fft_real)n;
        plan->bluestein_chirp_re[i] = notorious_fft_cos(angle);
        plan->bluestein_chirp_im[i] = notorious_fft_sin(angle);
    }

    return plan;

cleanup:
    if (plan->bluestein_plan)
        notorious_fft_destroy_plan(plan->bluestein_plan);
    notorious_fft_free(slab);
    return NULL;
}

/* ============================================================================
 * Power-of-2 Plan Creation
 * ============================================================================ */

static notorious_fft_plan* notorious_fft_create_plan_power2(size_t n) {
    if (n == 0 || (n & (n - 1))) return NULL;

    size_t real_bytes  = sizeof(notorious_fft_real);
    size_t int_bytes   = sizeof(int);

    size_t total;

    if (n <= NOTORIOUS_FFT_SMALL_SIZE) {
        /* Small plan slab layout (high→low):
         *   [notorious_fft_plan]
         *   work_im    [n reals]
         *   work_re    [n reals]
         *   tw_im      [n/2+1 reals]
         *   tw_re      [n/2+1 reals]
         *   bitrev     [n ints]
         */
        total = NOTORIOUS_FFT_SLAB_FIELD(sizeof(notorious_fft_plan))
              + NOTORIOUS_FFT_SLAB_FIELD(n * int_bytes)
              + NOTORIOUS_FFT_SLAB_FIELD((n / 2 + 1) * real_bytes) * 2
              + NOTORIOUS_FFT_SLAB_FIELD(n * real_bytes) * 2;
    } else {
        /* Large plan slab layout (high→low):
         *   [notorious_fft_plan]
         *   sr_t       [2n reals]
         *   sr_e       [2n reals]
         *   work_re    [2n reals]  ← work_im = work_re + n (no extra field)
         *   tw_im      [n/2 reals]
         *   tw_re      [n/2 reals]
         *   bitrev     [n ints]
         */
        total = NOTORIOUS_FFT_SLAB_FIELD(sizeof(notorious_fft_plan))
              + NOTORIOUS_FFT_SLAB_FIELD(n * int_bytes)
              + NOTORIOUS_FFT_SLAB_FIELD((n / 2) * real_bytes) * 2
              + NOTORIOUS_FFT_SLAB_FIELD(2 * n * real_bytes)     /* work_re (2n, work_im aliased) */
              + NOTORIOUS_FFT_SLAB_FIELD(2 * n * real_bytes) * 2;/* sr_e, sr_t */
    }

    void* slab = notorious_fft_malloc(total);
    if (!slab) return NULL;

    notorious_fft_plan* plan = (notorious_fft_plan*)slab;
    memset(plan, 0, sizeof(notorious_fft_plan));

    plan->slab          = slab;
    plan->n             = n;

    char* bump = (char*)slab + total;

    if (n <= NOTORIOUS_FFT_SMALL_SIZE) {
        plan->work_im  = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, n * real_bytes);
        plan->work_re  = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, n * real_bytes);
        plan->tw_im    = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, (n / 2 + 1) * real_bytes);
        plan->tw_re    = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, (n / 2 + 1) * real_bytes);
        plan->bitrev   = (int*)        notorious_fft_bump_alloc(&bump, n * int_bytes);

        plan->execute_func = notorious_fft_execute_iterative;

        notorious_fft_compute_bitrev(plan->bitrev, n);
        for (size_t i = 0; i < n / 2; i++) {
            notorious_fft_real angle = -NOTORIOUS_FFT_2PI * (notorious_fft_real)i / (notorious_fft_real)n;
            plan->tw_re[i] = notorious_fft_cos(angle);
            plan->tw_im[i] = notorious_fft_sin(angle);
        }
    } else {
        plan->sr_t     = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, 2 * n * real_bytes);
        plan->sr_e     = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, 2 * n * real_bytes);
        plan->work_re  = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, 2 * n * real_bytes);
        plan->work_im  = plan->work_re + n;  /* alias into same block */
        plan->tw_im    = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, (n / 2) * real_bytes);
        plan->tw_re    = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, (n / 2) * real_bytes);
        plan->bitrev   = (int*)        notorious_fft_bump_alloc(&bump, n * int_bytes);

        plan->execute_func = notorious_fft_execute_iterative;

        notorious_fft_compute_bitrev(plan->bitrev, n);
        for (size_t i = 0; i < n / 2; i++) {
            notorious_fft_real angle = -NOTORIOUS_FFT_2PI * (notorious_fft_real)i / (notorious_fft_real)n;
            plan->tw_re[i] = notorious_fft_cos(angle);
            plan->tw_im[i] = notorious_fft_sin(angle);
        }

        /* Split-radix DIF twiddles */
        notorious_fft_real* ep = plan->sr_e;
        size_t sz = n;
        while (sz >= 16) {
            for (size_t k = 0; k < sz / 4; k++) {
                *ep++ = notorious_fft_cos(-NOTORIOUS_FFT_2PI * (notorious_fft_real)k / (notorious_fft_real)sz);
                *ep++ = notorious_fft_sin(-NOTORIOUS_FFT_2PI * (notorious_fft_real)k / (notorious_fft_real)sz);
                *ep++ = notorious_fft_cos(-NOTORIOUS_FFT_2PI * 3.0 * (notorious_fft_real)k / (notorious_fft_real)sz);
                *ep++ = notorious_fft_sin(-NOTORIOUS_FFT_2PI * 3.0 * (notorious_fft_real)k / (notorious_fft_real)sz);
            }
            sz >>= 1;
        }
    }

    return plan;
}

/* ============================================================================
 * Main Plan API
 * ============================================================================ */

static notorious_fft_plan* notorious_fft_create_plan(size_t n, int inverse) {
    if (n == 0) return NULL;
    if ((n & (n - 1)) == 0)
        return notorious_fft_create_plan_power2(n);
    return notorious_fft_create_plan_bluestein(n, inverse);
}

static void notorious_fft_destroy_plan(notorious_fft_plan* plan) {
    if (!plan) return;
    /* Bluestein inner plan owns its own slab */
    if (plan->bluestein_plan)
        notorious_fft_destroy_plan(plan->bluestein_plan);
    /* Everything else is in the single slab — one free to rule them all */
    notorious_fft_free(plan->slab);
}


#endif /* NOTORIOUS_FFT_PLAN_H */
