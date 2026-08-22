/*
 * Notorious FFT - Plan Creation with Runtime Algorithm Selection
 */

#ifndef NOTORIOUS_FFT_PLAN_H
#define NOTORIOUS_FFT_PLAN_H

#include "05_algorithms.h"

/* Forward declarations */
static notorious_fft_plan* notorious_fft_create_plan_power2(size_t n);
static notorious_fft_plan* notorious_fft_create_plan_mixed(size_t n);
static notorious_fft_plan* notorious_fft_create_plan_rader(size_t n);
static notorious_fft_plan* notorious_fft_create_plan(size_t n, int inverse);
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
    /* 2n-1 must fit in size_t; convolution pad is the next power of two. */
    if (n > (SIZE_MAX - 1) / 2) return NULL;

    size_t m = notorious_fft_next_pow2(2 * n - 1);
    if (m == 0) return NULL;

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
    /* Round total to alignment - ensures bump starts at aligned address */
    total = NOTORIOUS_FFT_BUMP_ROUND(total);

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
    for (size_t k = 0; k <= 2 * (n - 1); k++) {
        int64_t idx = (int64_t)k - (int64_t)(n - 1);
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
        /* Round total to alignment - ensures bump starts at aligned address */
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
        if (n >= (size_t)NOTORIOUS_FFT_FOURSTEP_MIN)
            total += NOTORIOUS_FFT_SLAB_FIELD(n * real_bytes) * 2; /* four-step twiddles */
        total = NOTORIOUS_FFT_BUMP_ROUND(total);
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
        if (n >= (size_t)NOTORIOUS_FFT_FOURSTEP_MIN) {
            plan->four_tw_im = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, n * real_bytes);
            plan->four_tw_re = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, n * real_bytes);
        }
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

        if (n >= (size_t)NOTORIOUS_FFT_FOURSTEP_MIN && plan->four_tw_re) {
            int lg = 0;
            for (size_t t = n; t > 1; t >>= 1) lg++;
            int n1 = 1 << (lg / 2);
            int n2 = (int)(n / (size_t)n1);
            plan->four_n1 = n1;
            plan->four_n2 = n2;
            plan->four_sub1 = notorious_fft_create_plan_power2((size_t)n1);
            if (!plan->four_sub1) {
                notorious_fft_destroy_plan(plan);
                return NULL;
            }
            if (n2 != n1) {
                plan->four_sub2 = notorious_fft_create_plan_power2((size_t)n2);
                if (!plan->four_sub2) {
                    notorious_fft_destroy_plan(plan);
                    return NULL;
                }
            }
            for (int q = 0; q < n2; q++) {
                for (int p = 0; p < n1; p++) {
                    size_t i = (size_t)q * (size_t)n1 + (size_t)p;
                    notorious_fft_real angle = -NOTORIOUS_FFT_2PI * (notorious_fft_real)p * (notorious_fft_real)q
                                               / (notorious_fft_real)n;
                    plan->four_tw_re[i] = notorious_fft_cos(angle);
                    plan->four_tw_im[i] = notorious_fft_sin(angle);
                }
            }
        }
    }

    return plan;
}

/* ============================================================================
 * Main Plan API
 * ============================================================================ */

static notorious_fft_plan* notorious_fft_create_plan_mixed(size_t n) {
    int radix = (n % 7u == 0) ? 7 : (n % 5u == 0) ? 5 : 3;
    if (n % (size_t)radix != 0) return NULL;
    size_t m = n / (size_t)radix;

    notorious_fft_plan* sub = notorious_fft_create_plan(m, 0);
    if (!sub) return NULL;

    size_t tw_count = (size_t)(radix - 1) * m;
    size_t real_bytes = sizeof(notorious_fft_real);
    size_t total = NOTORIOUS_FFT_SLAB_FIELD(sizeof(notorious_fft_plan))
                 + NOTORIOUS_FFT_SLAB_FIELD(2 * n * real_bytes)          /* sr_t work */
                 + NOTORIOUS_FFT_SLAB_FIELD(tw_count * real_bytes) * 2;  /* tw_re, tw_im */
    total = NOTORIOUS_FFT_BUMP_ROUND(total);

    void* slab = notorious_fft_malloc(total);
    if (!slab) {
        notorious_fft_destroy_plan(sub);
        return NULL;
    }

    notorious_fft_plan* plan = (notorious_fft_plan*)slab;
    memset(plan, 0, sizeof(*plan));
    plan->slab = slab;
    plan->n = n;
    plan->mixed_radix = radix;
    plan->mixed_sub = sub;

    char* bump = (char*)slab + total;
    plan->tw_im = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, tw_count * real_bytes);
    plan->tw_re = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, tw_count * real_bytes);
    plan->sr_t  = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, 2 * n * real_bytes);

    for (int n1 = 1; n1 < radix; n1++) {
        for (size_t k = 0; k < m; k++) {
            size_t ti = (size_t)(n1 - 1) * m + k;
            notorious_fft_real angle = -NOTORIOUS_FFT_2PI * (notorious_fft_real)n1 * (notorious_fft_real)k
                                       / (notorious_fft_real)n;
            plan->tw_re[ti] = notorious_fft_cos(angle);
            plan->tw_im[ti] = notorious_fft_sin(angle);
        }
    }
    return plan;
}

static notorious_fft_plan* notorious_fft_create_plan_rader(size_t n) {
    /* Prime N, (N−1) 2·3·5·7-smooth → (N−1)-point FFT instead of Bluestein ~2N.
     * Mersenne N=2^p−1 is (n&(n+1))==0; do NOT pad a zero onto the 2^p path —
     * that is a different DFT. Small Mersenne primes (3,7,31,127) have smooth N−1. */
    if (n < 11 || n > (size_t)0x7fffffff) return NULL;
    /* Mersenne N=2^p−1 is (n&(n+1))==0; still need primality (15, 63, 255
     * are composite) and a smooth N−1. Do not pad onto the 2^p path. */
    (void)notorious_fft_is_mersenne_number(n);
    if (!notorious_fft_is_prime(n) || !notorious_fft_is_2357_smooth(n - 1))
        return NULL;
    size_t g = notorious_fft_primitive_root(n);
    if (g == 0) return NULL;
    size_t ginv = notorious_fft_modpow(g, n - 2, n);
    if (ginv == 0) return NULL;
    size_t m = n - 1;

    notorious_fft_plan* sub = notorious_fft_create_plan(m, 0);
    if (!sub) return NULL;

    size_t real_bytes = sizeof(notorious_fft_real);
    size_t int_bytes = sizeof(int);
    size_t total = NOTORIOUS_FFT_SLAB_FIELD(sizeof(notorious_fft_plan))
                 + NOTORIOUS_FFT_SLAB_FIELD(m * int_bytes) * 2
                 + NOTORIOUS_FFT_SLAB_FIELD(m * real_bytes) * 2
                 + NOTORIOUS_FFT_SLAB_FIELD(2 * n * real_bytes);
    total = NOTORIOUS_FFT_BUMP_ROUND(total);

    void* slab = notorious_fft_malloc(total);
    if (!slab) {
        notorious_fft_destroy_plan(sub);
        return NULL;
    }

    notorious_fft_plan* plan = (notorious_fft_plan*)slab;
    memset(plan, 0, sizeof(*plan));
    plan->slab = slab;
    plan->n = n;
    plan->rader_sub = sub;

    char* bump = (char*)slab + total;
    plan->sr_t      = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, 2 * n * real_bytes);
    plan->rader_b_im = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, m * real_bytes);
    plan->rader_b_re = (notorious_fft_real*)notorious_fft_bump_alloc(&bump, m * real_bytes);
    plan->rader_out = (int*)notorious_fft_bump_alloc(&bump, m * int_bytes);
    plan->rader_in  = (int*)notorious_fft_bump_alloc(&bump, m * int_bytes);

    size_t p = 1, q = 1;
    for (size_t j = 0; j < m; j++) {
        plan->rader_in[j]  = (int)p;
        plan->rader_out[j] = (int)q;
        p = (size_t)(((uint64_t)p * (uint64_t)g) % (uint64_t)n);
        q = (size_t)(((uint64_t)q * (uint64_t)ginv) % (uint64_t)n);
    }

    for (size_t j = 0; j < m; j++) {
        notorious_fft_real angle = -NOTORIOUS_FFT_2PI
            * (notorious_fft_real)plan->rader_out[j] / (notorious_fft_real)n;
        plan->sr_t[2 * j]     = notorious_fft_cos(angle);
        plan->sr_t[2 * j + 1] = notorious_fft_sin(angle);
    }
    notorious_fft_execute_cx(sub, plan->sr_t, plan->sr_t, 0);
    for (size_t j = 0; j < m; j++) {
        plan->rader_b_re[j] = plan->sr_t[2 * j];
        plan->rader_b_im[j] = plan->sr_t[2 * j + 1];
    }
    return plan;
}

static notorious_fft_plan* notorious_fft_create_plan(size_t n, int inverse) {
    if (n == 0) return NULL;
    if ((n & (n - 1)) == 0)
        return notorious_fft_create_plan_power2(n);
    /* Peel 3/5/7 even when the cofactor is prime (Rader) or awkward (Bluestein). */
    if (n % 7u == 0 || n % 5u == 0 || n % 3u == 0) {
        notorious_fft_plan* p = notorious_fft_create_plan_mixed(n);
        if (p) return p;
    }
    if (n >= 11 && notorious_fft_is_prime(n) && notorious_fft_is_2357_smooth(n - 1)) {
        notorious_fft_plan* p = notorious_fft_create_plan_rader(n);
        if (p) return p;
    }
    return notorious_fft_create_plan_bluestein(n, inverse);
}

static void notorious_fft_tune_plan(notorious_fft_plan* plan) {
    if (!plan || !plan->sr_e || !plan->sr_t || !plan->bitrev || plan->n < 32)
        return;
    size_t n = plan->n;
    size_t bytes = 2 * n * sizeof(notorious_fft_real);
    notorious_fft_real* a = (notorious_fft_real*)notorious_fft_malloc(bytes);
    notorious_fft_real* b = (notorious_fft_real*)notorious_fft_malloc(bytes);
    if (!a || !b) {
        notorious_fft_free(a);
        notorious_fft_free(b);
        return;
    }
    for (size_t i = 0; i < 2 * n; i++)
        a[i] = (notorious_fft_real)((int)i * 0.017 + 0.3);

    const int reps = n >= 4096 ? 3 : 8;
    uint64_t dif_dt, dit_dt, it_dt, t0, t1;

    memcpy(b, a, bytes);
    t0 = notorious_fft_rdtsc();
    for (int i = 0; i < reps; i++) {
        memcpy(b, a, bytes);
        notorious_fft_execute_sr_dif(plan, b, b, 0);
    }
    t1 = notorious_fft_rdtsc();
    dif_dt = t1 - t0;

    dit_dt = ~(uint64_t)0;
    if (n >= 64) {
        t0 = notorious_fft_rdtsc();
        for (int i = 0; i < reps; i++) {
            memcpy(b, a, bytes);
            notorious_fft_execute_sr_dit(plan, b, b, 0);
        }
        t1 = notorious_fft_rdtsc();
        dit_dt = t1 - t0;
    }

    t0 = notorious_fft_rdtsc();
    for (int i = 0; i < reps; i++) {
        memcpy(b, a, bytes);
        notorious_fft_iterative_inplace_cx(b, plan->bitrev, plan->tw_re, plan->tw_im, n, 0);
    }
    t1 = notorious_fft_rdtsc();
    it_dt = t1 - t0;

    plan->prefer_iterative = 0;
    plan->prefer_dit = 0;
    if (it_dt < dif_dt && it_dt < dit_dt)
        plan->prefer_iterative = 1;
    else if (dit_dt < dif_dt)
        plan->prefer_dit = 1;
    notorious_fft_free(a);
    notorious_fft_free(b);
}

static void notorious_fft_destroy_plan(notorious_fft_plan* plan) {
    if (!plan) return;
    if (plan->mixed_sub)
        notorious_fft_destroy_plan(plan->mixed_sub);
    if (plan->rader_sub)
        notorious_fft_destroy_plan(plan->rader_sub);
    if (plan->four_sub1)
        notorious_fft_destroy_plan(plan->four_sub1);
    if (plan->four_sub2 && plan->four_sub2 != plan->four_sub1)
        notorious_fft_destroy_plan(plan->four_sub2);
    if (plan->bluestein_plan)
        notorious_fft_destroy_plan(plan->bluestein_plan);
    notorious_fft_free(plan->slab);
}


#endif /* NOTORIOUS_FFT_PLAN_H */
