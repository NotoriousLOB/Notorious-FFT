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

/* ============================================================================
 * In-place radix-3 / radix-5 butterflies on interleaved data, stride `s`
 * complexes (i.e. 2*s reals between elements).
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_radix3_stride(
    notorious_fft_real* x, int s, int inverse)
{
    const notorious_fft_real half = (notorious_fft_real)0.5;
    const notorious_fft_real s3 = (notorious_fft_real)0.866025403784438646763723170752936183;
    const notorious_fft_real sig = inverse ? s3 : -s3;
    const int s2 = 2 * s;
    notorious_fft_real x0r = x[0],     x0i = x[1];
    notorious_fft_real x1r = x[s2],    x1i = x[s2 + 1];
    notorious_fft_real x2r = x[2 * s2], x2i = x[2 * s2 + 1];
    notorious_fft_real ur = x1r + x2r, ui = x1i + x2i;
    notorious_fft_real vr = x1r - x2r, vi = x1i - x2i;
    x[0]      = x0r + ur;
    x[1]      = x0i + ui;
    x[s2]     = x0r - half * ur - sig * vi;
    x[s2 + 1] = x0i - half * ui + sig * vr;
    x[2 * s2]     = x0r - half * ur + sig * vi;
    x[2 * s2 + 1] = x0i - half * ui - sig * vr;
}

static NOTORIOUS_FFT_INLINE void notorious_fft_radix5_stride(
    notorious_fft_real* x, int s, int inverse)
{
    /* Direct 5-point DFT with W = exp(∓2πi/5) */
    const notorious_fft_real c1 = (notorious_fft_real)0.309016994374947424102293417182819059;  /* cos(2π/5) */
    const notorious_fft_real s1 = (notorious_fft_real)0.951056516295153572116439333379382143;  /* sin(2π/5) */
    const notorious_fft_real c2 = (notorious_fft_real)-0.809016994374947424102293417182819059; /* cos(4π/5) */
    const notorious_fft_real s2 = (notorious_fft_real)0.587785252292473129168705954639072768;  /* sin(4π/5) */
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    const int stride = 2 * s;
    notorious_fft_real xr[5], xi[5];
    for (int n = 0; n < 5; n++) {
        xr[n] = x[n * stride];
        xi[n] = x[n * stride + 1];
    }
    notorious_fft_real y0r = xr[0] + xr[1] + xr[2] + xr[3] + xr[4];
    notorious_fft_real y0i = xi[0] + xi[1] + xi[2] + xi[3] + xi[4];
    notorious_fft_real w1r = c1, w1i = sg * s1;
    notorious_fft_real w2r = c2, w2i = sg * s2;
    notorious_fft_real w3r = c2, w3i = -sg * s2; /* W^3 = W^{-2} = conj(W^2) for |W|=1, forward W=e^{-2πi/5} */
    notorious_fft_real w4r = c1, w4i = -sg * s1;
    /* For inverse W=e^{+2πi/5}, W^3 = e^{6πi/5} = e^{-4πi/5} = conj of e^{4πi/5} = (c2, -s2) with sg=+1
     * w3 = (c2, -sg*s2) works for both. W^4 = W^{-1} = conj(W) = (c1, -sg*s1). */
    notorious_fft_real y1r = xr[0] + xr[1]*w1r - xi[1]*w1i + xr[2]*w2r - xi[2]*w2i
                           + xr[3]*w3r - xi[3]*w3i + xr[4]*w4r - xi[4]*w4i;
    notorious_fft_real y1i = xi[0] + xr[1]*w1i + xi[1]*w1r + xr[2]*w2i + xi[2]*w2r
                           + xr[3]*w3i + xi[3]*w3r + xr[4]*w4i + xi[4]*w4r;
    notorious_fft_real y2r = xr[0] + xr[1]*w2r - xi[1]*w2i + xr[2]*w4r - xi[2]*w4i
                           + xr[3]*w1r - xi[3]*w1i + xr[4]*w3r - xi[4]*w3i;
    notorious_fft_real y2i = xi[0] + xr[1]*w2i + xi[1]*w2r + xr[2]*w4i + xi[2]*w4r
                           + xr[3]*w1i + xi[3]*w1r + xr[4]*w3i + xi[4]*w3r;
    notorious_fft_real y3r = xr[0] + xr[1]*w3r - xi[1]*w3i + xr[2]*w1r - xi[2]*w1i
                           + xr[3]*w4r - xi[3]*w4i + xr[4]*w2r - xi[4]*w2i;
    notorious_fft_real y3i = xi[0] + xr[1]*w3i + xi[1]*w3r + xr[2]*w1i + xi[2]*w1r
                           + xr[3]*w4i + xi[3]*w4r + xr[4]*w2i + xi[4]*w2r;
    notorious_fft_real y4r = xr[0] + xr[1]*w4r - xi[1]*w4i + xr[2]*w3r - xi[2]*w3i
                           + xr[3]*w2r - xi[3]*w2i + xr[4]*w1r - xi[4]*w1i;
    notorious_fft_real y4i = xi[0] + xr[1]*w4i + xi[1]*w4r + xr[2]*w3i + xi[2]*w3r
                           + xr[3]*w2i + xi[3]*w2r + xr[4]*w1i + xi[4]*w1r;
    x[0]              = y0r; x[1]                  = y0i;
    x[stride]         = y1r; x[stride + 1]         = y1i;
    x[2 * stride]     = y2r; x[2 * stride + 1]     = y2i;
    x[3 * stride]     = y3r; x[3 * stride + 1]     = y3i;
    x[4 * stride]     = y4r; x[4 * stride + 1]     = y4i;
}

/* 7-point DFT (FFmpeg tx has a dedicated fft7; we use the same idea: native
 * radix-7 instead of Bluestein). W^k stored as (cos(2πk/7), ±sin). */
static NOTORIOUS_FFT_INLINE void notorious_fft_radix7_stride(
    notorious_fft_real* x, int s, int inverse)
{
    const notorious_fft_real c1 = (notorious_fft_real)0.623489801858733530525004884004239737; /* cos(2π/7) */
    const notorious_fft_real s1 = (notorious_fft_real)0.781831482468029808708444526674057751;
    const notorious_fft_real c2 = (notorious_fft_real)-0.222520933956314404288902564496794972; /* cos(4π/7) */
    const notorious_fft_real s2 = (notorious_fft_real)0.974927912181823607018131682993903878;
    const notorious_fft_real c3 = (notorious_fft_real)-0.900968867902419126236102319507445351; /* cos(6π/7) */
    const notorious_fft_real s3 = (notorious_fft_real)0.433883739117558120475768332848358754;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    const int st = 2 * s;
    notorious_fft_real xr[7], xi[7];
    notorious_fft_real wr[7], wi[7];
    wr[0] = 1; wi[0] = 0;
    wr[1] = c1; wi[1] = sg * s1;
    wr[2] = c2; wi[2] = sg * s2;
    wr[3] = c3; wi[3] = sg * s3;
    wr[4] = c3; wi[4] = -sg * s3;
    wr[5] = c2; wi[5] = -sg * s2;
    wr[6] = c1; wi[6] = -sg * s1;
    for (int n = 0; n < 7; n++) {
        xr[n] = x[n * st];
        xi[n] = x[n * st + 1];
    }
    for (int k = 0; k < 7; k++) {
        notorious_fft_real sr = 0, si = 0;
        for (int n = 0; n < 7; n++) {
            int p = (k * n) % 7;
            sr += xr[n] * wr[p] - xi[n] * wi[p];
            si += xr[n] * wi[p] + xi[n] * wr[p];
        }
        x[k * st]     = sr;
        x[k * st + 1] = si;
    }
}

#endif /* NOTORIOUS_FFT_KERNELS_H */
