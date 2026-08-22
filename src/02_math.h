/*
 * Notorious FFT - Math Utilities and Memory Management
 */

#ifndef NOTORIOUS_FFT_MATH_H
#define NOTORIOUS_FFT_MATH_H

/* Include core types */
#include "01_core.h"

/* ============================================================================
 * Fast Math Approximations
 * ============================================================================ */

/* Bhaskara I sine approximation: sin(x) ≈ (16x(π-x))/(5π² - 4x(π-x))
 * Extended for full [-π, π] range using symmetries
 * Max error ~0.001 (0.1%) - sufficient for twiddle factors */
static NOTORIOUS_FFT_INLINE notorious_fft_real notorious_fft_sin_fast(notorious_fft_real x) {
    /* Reduce to [-π, π] */
    x = x - NOTORIOUS_FFT_2PI * floor(x * NOTORIOUS_FFT_INV_2PI + 0.5);
    
    /* Use symmetry: sin(-x) = -sin(x) */
    notorious_fft_real sign = 1.0;
    if (x < 0) {
        sign = -1.0;
        x = -x;
    }
    
    /* Reduce to [0, π] using sin(π - x) = sin(x) */
    if (x > NOTORIOUS_FFT_PI) {
        x = NOTORIOUS_FFT_2PI - x;
    }
    
    /* Reduce to [0, π/2] using sin(π - x) = sin(x) */
    if (x > NOTORIOUS_FFT_PI / 2) {
        x = NOTORIOUS_FFT_PI - x;
    }
    
    /* Bhaskara I approximation on [0, π/2] */
    notorious_fft_real x_by_pi = x / NOTORIOUS_FFT_PI;
    notorious_fft_real num = 16.0 * x_by_pi * (1.0 - x_by_pi);
    notorious_fft_real den = 5.0 - 4.0 * x_by_pi * (1.0 - x_by_pi);
    
    return sign * num / den;
}

static NOTORIOUS_FFT_INLINE notorious_fft_real notorious_fft_cos_fast(notorious_fft_real x) {
    return notorious_fft_sin_fast(x + NOTORIOUS_FFT_PI / 2.0);
}

/* Accurate versions using standard library */
static NOTORIOUS_FFT_INLINE notorious_fft_real notorious_fft_sin_accurate(notorious_fft_real x) {
    return (notorious_fft_real)sin((double)x);
}

static NOTORIOUS_FFT_INLINE notorious_fft_real notorious_fft_cos_accurate(notorious_fft_real x) {
    return (notorious_fft_real)cos((double)x);
}

/* Always use accurate sin/cos for twiddle factor precomputation.
 * The fast approximation (0.1% error) accumulates across FFT stages and
 * produces unacceptable numerical error.  NOTORIOUS_FFT_FAST_MATH enables it as
 * an opt-in for applications that can tolerate ~1% output error. */
#ifdef NOTORIOUS_FFT_FAST_MATH
    #define notorious_fft_sin notorious_fft_sin_fast
    #define notorious_fft_cos notorious_fft_cos_fast
#else
    #define notorious_fft_sin notorious_fft_sin_accurate
    #define notorious_fft_cos notorious_fft_cos_accurate
#endif

/* ============================================================================
 * Memory Management
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void* notorious_fft_malloc(size_t size) {
    void* ptr = NULL;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, 64);
#elif defined(__APPLE__)
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
#else
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
#endif
    return ptr;
}

static NOTORIOUS_FFT_INLINE void notorious_fft_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static NOTORIOUS_FFT_INLINE int notorious_fft_alignment(void) {
    return NOTORIOUS_FFT_ALIGNMENT;
}

/* Next power of two >= v. Returns 0 on overflow (v too large). */
static NOTORIOUS_FFT_INLINE size_t notorious_fft_next_pow2(size_t v) {
    if (v == 0) return 1;
    if (v > ((size_t)1 << (sizeof(size_t) * 8 - 1))) return 0;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if SIZE_MAX > 0xffffffffu
    v |= v >> 32;
#endif
    return v + 1;
}

/* True iff n = 2^a * 3^b * 5^c * 7^d (and n > 0). */
static NOTORIOUS_FFT_INLINE int notorious_fft_is_2357_smooth(size_t n) {
    if (n == 0) return 0;
    while ((n & 1u) == 0) n >>= 1;
    while (n % 3u == 0) n /= 3u;
    while (n % 5u == 0) n /= 5u;
    while (n % 7u == 0) n /= 7u;
    return n == 1;
}

/* True iff n = 2^p − 1 (all bits set). Cheap: (n & (n+1)) == 0.
 * Mersenne primes are the prime subset (3, 7, 31, 127, …).
 * Padding one zero and taking a 2^p FFT is NOT an n-point DFT — bin
 * frequencies are 2πk/2^p, not 2πk/n. Rader is the valid fast path. */
static NOTORIOUS_FFT_INLINE int notorious_fft_is_mersenne_number(size_t n) {
    return n != 0 && (n & (n + (size_t)1)) == 0;
}

static NOTORIOUS_FFT_INLINE int notorious_fft_is_prime(size_t n) {
    if (n < 2) return 0;
    if ((n & 1u) == 0) return n == 2;
    if (n % 3u == 0) return n == 3;
    if (n % 5u == 0) return n == 5;
    if (n % 7u == 0) return n == 7;
    /* i <= n/i avoids overflow on i*i. */
    for (size_t i = 11, j = 13; i <= n / i; i += 6, j += 6) {
        if (n % i == 0 || n % j == 0) return 0;
    }
    return 1;
}

/* (b^e) mod m. Residues fit in uint64 when m ≤ 2^32; __int128 otherwise. */
static NOTORIOUS_FFT_INLINE size_t notorious_fft_modpow(size_t b, size_t e, size_t m) {
    if (m <= 1) return 0;
    uint64_t r = 1, bb = (uint64_t)b % (uint64_t)m, mm = (uint64_t)m;
#if defined(__SIZEOF_INT128__)
    while (e) {
        if (e & 1) r = (uint64_t)(((__uint128_t)r * bb) % mm);
        bb = (uint64_t)(((__uint128_t)bb * bb) % mm);
        e >>= 1;
    }
#else
    if (mm > 0xffffffffu) return 0;
    while (e) {
        if (e & 1) r = (r * bb) % mm;
        bb = (bb * bb) % mm;
        e >>= 1;
    }
#endif
    return (size_t)r;
}

/* Primitive root modulo prime n, or 0. Requires n−1 to be 2·3·5·7-smooth. */
static NOTORIOUS_FFT_INLINE size_t notorious_fft_primitive_root(size_t n) {
    if (n < 3) return 0;
    size_t factors[4];
    int nf = 0;
    size_t m = n - 1;
    if ((m & 1u) == 0) { factors[nf++] = 2; while ((m & 1u) == 0) m >>= 1; }
    if (m % 3u == 0) { factors[nf++] = 3; while (m % 3u == 0) m /= 3u; }
    if (m % 5u == 0) { factors[nf++] = 5; while (m % 5u == 0) m /= 5u; }
    if (m % 7u == 0) { factors[nf++] = 7; while (m % 7u == 0) m /= 7u; }
    if (m != 1) return 0;
    for (size_t g = 2; g < n; g++) {
        int ok = 1;
        for (int i = 0; i < nf; i++) {
            if (notorious_fft_modpow(g, (n - 1) / factors[i], n) == 1) {
                ok = 0;
                break;
            }
        }
        if (ok) return g;
    }
    return 0;
}

#ifdef NOTORIOUS_FFT_DEBUG
#include <assert.h>
#define NOTORIOUS_FFT_ASSERT_ALIGNED(p) \
    assert(((uintptr_t)(p) & (NOTORIOUS_FFT_ALIGNMENT - 1)) == 0)
#else
#define NOTORIOUS_FFT_ASSERT_ALIGNED(p) ((void)0)
#endif

/* ============================================================================
 * Bump Allocator — decrement-from-end, no per-free bookkeeping
 *
 * Usage:
 *   1. Compute total bytes needed (sum of NOTORIOUS_FFT_BUMP_ALIGN-rounded sizes
 *      plus sizeof(notorious_fft_plan)).
 *   2. Call notorious_fft_malloc(total) to get the slab.
 *   3. Place notorious_fft_plan at slab[0..sizeof(notorious_fft_plan)-1].
 *   4. Set plan->bump = (char*)slab + total.
 *   5. Each notorious_fft_bump_alloc(plan, n) decrements plan->bump by the
 *      aligned size and returns the new pointer — no branch, no NULL check
 *      (the slab is pre-sized to fit all allocations exactly).
 *   6. notorious_fft_destroy_plan frees plan->slab in one call.
 *
 * All sub-allocations are 64-byte aligned because:
 *   - The slab base is 64-byte aligned (from notorious_fft_malloc).
 *   - Every allocation is rounded up to NOTORIOUS_FFT_BUMP_ALIGN (64) bytes.
 *   - Decrementing by a multiple of 64 from a 64-aligned end preserves
 *     alignment for every pointer.
 * ============================================================================ */

#define NOTORIOUS_FFT_BUMP_ALIGN 64u

/* Round n up to the next multiple of NOTORIOUS_FFT_BUMP_ALIGN */
#define NOTORIOUS_FFT_BUMP_ROUND(n) \
    (((size_t)(n) + NOTORIOUS_FFT_BUMP_ALIGN - 1u) & ~(size_t)(NOTORIOUS_FFT_BUMP_ALIGN - 1u))

/* Suballocate from the high end of the slab, decrementing the bump pointer.
 * 'bump' is a (char**) pointing to the current high-water mark. */
static NOTORIOUS_FFT_INLINE void* notorious_fft_bump_alloc(char** bump, size_t bytes) {
    *bump -= NOTORIOUS_FFT_BUMP_ROUND(bytes);
    return (void*)*bump;
}

/* ============================================================================
 * Bit Reversal
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_compute_bitrev(int* rev, size_t n) {
    int log_n = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 1;
        log_n++;
    }
    
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        for (int k = 0; k < log_n; k++) {
            if ((i >> k) & 1) {
                j |= (1 << (log_n - 1 - k));
            }
        }
        rev[i] = (int)j;
    }
}

/* ============================================================================
 * Butterfly Macros
 * ============================================================================ */

#define NOTORIOUS_FFT_BUTTERFLY(re, im, i, j, wr, wi) do { \
    notorious_fft_real ur = re[i], ui = im[i]; \
    notorious_fft_real vr = re[j] * (wr) - im[j] * (wi); \
    notorious_fft_real vi = re[j] * (wi) + im[j] * (wr); \
    re[i] = ur + vr; im[i] = ui + vi; \
    re[j] = ur - vr; im[j] = ui - vi; \
} while(0)

static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly_scalar(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT re, notorious_fft_real* NOTORIOUS_FFT_RESTRICT im,
    size_t i, size_t j, notorious_fft_real wr, notorious_fft_real wi
) {
    notorious_fft_real ur = re[i], ui = im[i];
    notorious_fft_real vr = re[j] * wr - im[j] * wi;
    notorious_fft_real vi = re[j] * wi + im[j] * wr;
    re[i] = ur + vr; im[i] = ui + vi;
    re[j] = ur - vr; im[j] = ui - vi;
}

#endif /* NOTORIOUS_FFT_MATH_H */
