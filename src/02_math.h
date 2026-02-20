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
    if (posix_memalign(&ptr, 64, size) != 0) ptr = malloc(size);
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
