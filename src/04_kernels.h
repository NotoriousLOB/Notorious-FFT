/*
 * Notorious FFT - Hardcoded FFT Kernels for Small N
 * Fully unrolled for optimal performance
 */

#ifndef NOTORIOUS_FFT_KERNELS_H
#define NOTORIOUS_FFT_KERNELS_H

#include "03_simd.h"

/* ============================================================================
 * N = 2 Kernel
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_kernel_2(notorious_fft_real* re, notorious_fft_real* im) {
    notorious_fft_real t0 = re[0] + re[1];
    notorious_fft_real t1 = re[0] - re[1];
    re[0] = t0; re[1] = t1;
    
    t0 = im[0] + im[1];
    t1 = im[0] - im[1];
    im[0] = t0; im[1] = t1;
}

/* ============================================================================
 * N = 4 Kernel - Fully Unrolled
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_kernel_4(notorious_fft_real* re, notorious_fft_real* im) {
    /* Stage 1: Distance 2 butterflies (twiddle = 1) */
    notorious_fft_real t0 = re[0] + re[2];
    notorious_fft_real t1 = re[1] + re[3];
    notorious_fft_real t2 = re[0] - re[2];
    notorious_fft_real t3 = re[1] - re[3];
    re[0] = t0 + t1;
    re[2] = t0 - t1;
    re[1] = t2;
    re[3] = t3;
    
    t0 = im[0] + im[2];
    t1 = im[1] + im[3];
    t2 = im[0] - im[2];
    t3 = im[1] - im[3];
    im[0] = t0 + t1;
    im[2] = t0 - t1;
    im[1] = t2;
    im[3] = t3;
    
    /* Stage 2: Distance 1 butterflies
     * Pair (0,1): twiddle = 1
     * Pair (2,3): twiddle = -i (swap and negate)
     */
    t0 = re[1];
    t1 = im[1];
    t2 = re[3];  /* save re[3] before overwrite */
    t3 = im[3];  /* save im[3] before overwrite */
    re[1] = t0 + t3;
    im[1] = t1 - t2;
    re[3] = t0 - t3;
    im[3] = t1 + t2;
}

/* ============================================================================
 * N = 8 Kernel - Optimized with explicit twiddles (FORWARD)
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_kernel_8(notorious_fft_real* re, notorious_fft_real* im) {
    const notorious_fft_real c4 = NOTORIOUS_FFT_INV_SQRT2;  /* cos(π/4) = sin(π/4) */
    
    /* Stage 1: Distance 4 butterflies (twiddle = 1) */
    notorious_fft_real ar0 = re[0] + re[4], ar1 = re[1] + re[5];
    notorious_fft_real ar2 = re[2] + re[6], ar3 = re[3] + re[7];
    notorious_fft_real br0 = re[0] - re[4], br1 = re[1] - re[5];
    notorious_fft_real br2 = re[2] - re[6], br3 = re[3] - re[7];
    
    notorious_fft_real ai0 = im[0] + im[4], ai1 = im[1] + im[5];
    notorious_fft_real ai2 = im[2] + im[6], ai3 = im[3] + im[7];
    notorious_fft_real bi0 = im[0] - im[4], bi1 = im[1] - im[5];
    notorious_fft_real bi2 = im[2] - im[6], bi3 = im[3] - im[7];
    
    /* Stage 2: Distance 2 butterflies
     * Groups 0-3: twiddle = 1
     * Groups 4-7: twiddle = -i
     */
    notorious_fft_real cr0 = ar0 + ar2, cr1 = ar1 + ar3;
    notorious_fft_real cr2 = ar0 - ar2, cr3 = ar1 - ar3;
    notorious_fft_real ci0 = ai0 + ai2, ci1 = ai1 + ai3;
    notorious_fft_real ci2 = ai0 - ai2, ci3 = ai1 - ai3;
    
    notorious_fft_real dr0 = br0 + bi2, dr1 = br1 + bi3;
    notorious_fft_real dr2 = br0 - bi2, dr3 = br1 - bi3;
    notorious_fft_real di0 = bi0 - br2, di1 = bi1 - br3;
    notorious_fft_real di2 = bi0 + br2, di3 = bi1 + br3;
    
    /* Stage 3: Distance 1 butterflies with various twiddles */
    /* Index 0,1: twiddle = 1 */
    re[0] = cr0 + cr1; im[0] = ci0 + ci1;
    re[1] = cr0 - cr1; im[1] = ci0 - ci1;
    
    /* Index 2,3: twiddle = exp(-iπ/4) = c4 - i*c4 */
    /* (cr2 + i*ci2) + (cr3 + i*ci3)*(c4 - i*c4) */
    notorious_fft_real tr = cr3 * c4 + ci3 * c4;  /* Real part of product */
    notorious_fft_real ti = ci3 * c4 - cr3 * c4;  /* Imag part of product */
    re[2] = cr2 + tr; im[2] = ci2 + ti;
    re[3] = cr2 - tr; im[3] = ci2 - ti;
    
    /* Index 4,5: twiddle = exp(-iπ/2) = -i */
    re[4] = dr0 + di1; im[4] = di0 - dr1;
    re[5] = dr0 - di1; im[5] = di0 + dr1;
    
    /* Index 6,7: twiddle = exp(-i3π/4) = -c4 - i*c4 */
    tr = -dr3 * c4 + di3 * c4;
    ti = -dr3 * c4 - di3 * c4;
    re[6] = dr2 + tr; im[6] = di2 + ti;
    re[7] = dr2 - tr; im[7] = di2 - ti;
}


#endif /* NOTORIOUS_FFT_KERNELS_H */
