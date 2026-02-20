/*
 * Notorious FFT - Core Types and Platform Detection
 */

#ifndef NOTORIOUS_FFT_CORE_H
#define NOTORIOUS_FFT_CORE_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

/* ============================================================================
 * Configuration
 * ============================================================================ */

#ifndef NOTORIOUS_FFT_SMALL_SIZE
#define NOTORIOUS_FFT_SMALL_SIZE 8  /* Use iterative path for N ≤ this; N ≥ 16 uses split-radix DIF */
#endif

#ifndef NOTORIOUS_FFT_BLOCK_SIZE
#define NOTORIOUS_FFT_BLOCK_SIZE 4096  /* Cache block size for tiling */
#endif

#ifndef NOTORIOUS_FFT_TIMING_RUNS
#define NOTORIOUS_FFT_TIMING_RUNS 3  /* Runs for timing-based selection */
#endif

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

#ifdef NOTORIOUS_FFT_SINGLE
typedef float notorious_fft_real;
#else
typedef double notorious_fft_real;
#endif

/* Split-complex: separate real and imaginary arrays */
typedef struct {
    notorious_fft_real *re;
    notorious_fft_real *im;
} notorious_fft_complex;

/* Forward declaration */
struct notorious_fft_plan;

/* Function pointer type for execute functions */
typedef void (*notorious_fft_execute_func_t)(const struct notorious_fft_plan*, 
                                       const notorious_fft_real*, const notorious_fft_real*,
                                       notorious_fft_real*, notorious_fft_real*);

/* FFT Plan - precomputed data for a specific size */
typedef struct notorious_fft_plan {
    size_t n;               /* Transform size (any positive integer) */

    notorious_fft_execute_func_t execute_func;

    /* Bump allocator: one slab holds the struct + all sub-allocations.
     * Suballocations decrement from the end — no per-free bookkeeping.
     * notorious_fft_destroy_plan frees just this one pointer. */
    void *slab;             /* Base of the single allocation */

    /* For power-of-2: bit-reversal and twiddles */
    int *bitrev;            /* Bit-reversal permutation table */
    notorious_fft_real *tw_re;     /* Twiddle factors (real part) */
    notorious_fft_real *tw_im;     /* Twiddle factors (imag part) */

    /* Working buffers (allocated at plan creation) */
    notorious_fft_real *work_re;   /* Working buffer (real) */
    notorious_fft_real *work_im;   /* Working buffer (imag) */

    /* Split-radix DIF twiddles (interleaved: e[4k], e[4k+1] = w1, e[4k+2], e[4k+3] = w3) */
    notorious_fft_real *sr_e;      /* Split-radix exponent table — same layout as minfft */
    notorious_fft_real *sr_t;      /* Temp buffer for split-radix (n complex = 2*n reals) */

    /* For Bluestein algorithm (non-power-of-2) */
    size_t bluestein_n;     /* Next power of 2 >= 2*n-1 */
    int is_inverse;         /* 1 for inverse FFT, 0 for forward */
    notorious_fft_real *bluestein_chirp_re;  /* Chirp factors exp(-i*pi*k^2/n) */
    notorious_fft_real *bluestein_chirp_im;
    notorious_fft_real *bluestein_chirp_fft_re;  /* FFT of chirp (conjugated) */
    notorious_fft_real *bluestein_chirp_fft_im;
    struct notorious_fft_plan *bluestein_plan;  /* Internal FFT plan (owns its own slab) */
    notorious_fft_real *bluestein_buf_re;  /* Convolution buffer for input/output */
    notorious_fft_real *bluestein_buf_im;
    notorious_fft_real *bluestein_fft_buf_re;  /* FFT output buffer */
    notorious_fft_real *bluestein_fft_buf_im;

} notorious_fft_plan;

/* Legacy aux structure for API compatibility */
typedef struct notorious_fft_aux {
    int N;
    void *t;
    void *e;
    struct notorious_fft_aux *sub1, *sub2;
    notorious_fft_plan *plan;
    /* Pre-allocated scratch buffers to avoid per-call malloc in DFT wrapper */
    notorious_fft_real *scratch_re;
    notorious_fft_real *scratch_im;
} notorious_fft_aux;

/* ============================================================================
 * Platform Detection
 * ============================================================================ */

/* MSVC compatibility */
#ifdef _MSC_VER
    #include <intrin.h>
    #define NOTORIOUS_FFT_INLINE __inline
    #define NOTORIOUS_FFT_ALIGN(x) __declspec(align(x))
    #define NOTORIOUS_FFT_RESTRICT __restrict
#else
    #define NOTORIOUS_FFT_INLINE inline __attribute__((always_inline))
    #define NOTORIOUS_FFT_ALIGN(x) __attribute__((aligned(x)))
    #ifdef __cplusplus
        #define NOTORIOUS_FFT_RESTRICT __restrict
    #else
        #define NOTORIOUS_FFT_RESTRICT restrict
    #endif
#endif

/* SIMD Detection */
#ifndef NOTORIOUS_FFT_HAS_NEON
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
        #define NOTORIOUS_FFT_HAS_NEON 1
    #else
        #define NOTORIOUS_FFT_HAS_NEON 0
    #endif
#endif

#ifndef NOTORIOUS_FFT_HAS_AVX512
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        #include <immintrin.h>
        #define NOTORIOUS_FFT_HAS_AVX512 1
    #else
        #define NOTORIOUS_FFT_HAS_AVX512 0
    #endif
#endif

#ifndef NOTORIOUS_FFT_HAS_AVX2
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define NOTORIOUS_FFT_HAS_AVX2 1
    #else
        #define NOTORIOUS_FFT_HAS_AVX2 0
    #endif
#endif

#ifndef NOTORIOUS_FFT_HAS_SSE2
    #if defined(__SSE2__) || (defined(_M_X64) || defined(_M_AMD64))
        #include <emmintrin.h>
        #define NOTORIOUS_FFT_HAS_SSE2 1
    #else
        #define NOTORIOUS_FFT_HAS_SSE2 0
    #endif
#endif

/* OpenMP Detection */
#ifdef _OPENMP
    #include <omp.h>
    #define NOTORIOUS_FFT_HAS_OPENMP 1
#else
    #define NOTORIOUS_FFT_HAS_OPENMP 0
#endif

/* High-resolution timer */
#if defined(_MSC_VER)
    static NOTORIOUS_FFT_INLINE uint64_t notorious_fft_rdtsc(void) {
        return __rdtsc();
    }
#elif defined(__x86_64__) || defined(__i386__)
    static NOTORIOUS_FFT_INLINE uint64_t notorious_fft_rdtsc(void) {
        unsigned int lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | lo;
    }
#elif defined(__aarch64__)
    static NOTORIOUS_FFT_INLINE uint64_t notorious_fft_rdtsc(void) {
        uint64_t val;
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r" (val));
        return val;
    }
#else
    static NOTORIOUS_FFT_INLINE uint64_t notorious_fft_rdtsc(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }
#endif

/* Constants */
static const notorious_fft_real NOTORIOUS_FFT_PI = 3.141592653589793238462643383279502884L;
static const notorious_fft_real NOTORIOUS_FFT_2PI = 6.283185307179586476925286766559005768L;
static const notorious_fft_real NOTORIOUS_FFT_INV_2PI = 0.159154943091895335768883763372514362L;
static const notorious_fft_real NOTORIOUS_FFT_SQRT2 = 1.414213562373095048801688724209698079L;
static const notorious_fft_real NOTORIOUS_FFT_INV_SQRT2 = 0.707106781186547524400844362104849039L;

#endif /* NOTORIOUS_FFT_CORE_H */
