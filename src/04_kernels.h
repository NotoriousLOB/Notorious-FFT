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

/* 7-point DFT: conjugate-pair factorization (3 real cos/sin pairs). */
static void notorious_fft_radix7_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real ur1 = xr1 + xr6, ui1 = xi1 + xi6;
    notorious_fft_real vr1 = xr1 - xr6, vi1 = xi1 - xi6;
    notorious_fft_real ur2 = xr2 + xr5, ui2 = xi2 + xi5;
    notorious_fft_real vr2 = xr2 - xr5, vi2 = xi2 - xi5;
    notorious_fft_real ur3 = xr3 + xr4, ui3 = xi3 + xi4;
    notorious_fft_real vr3 = xr3 - xr4, vi3 = xi3 - xi4;
    x[0] = xr0 + ur1 + ur2 + ur3;
    x[1] = xi0 + ui1 + ui2 + ui3;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)6.234898018587335944e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)6.234898018587335944e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.818314824680298036e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.818314824680298036e-01;
        sr += ur2 * (notorious_fft_real)-2.225209339563143374e-01; si += ui2 * (notorious_fft_real)-2.225209339563143374e-01;
        tr += vi2 * (notorious_fft_real)9.749279121818236193e-01; ti += vr2 * (notorious_fft_real)9.749279121818236193e-01;
        sr += ur3 * (notorious_fft_real)-9.009688679024190350e-01; si += ui3 * (notorious_fft_real)-9.009688679024190350e-01;
        tr += vi3 * (notorious_fft_real)4.338837391175582314e-01; ti += vr3 * (notorious_fft_real)4.338837391175582314e-01;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[6*st]       = xr0 + sr + sg * tr;
        x[6*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-2.225209339563143374e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-2.225209339563143374e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.749279121818236193e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.749279121818236193e-01;
        sr += ur2 * (notorious_fft_real)-9.009688679024191460e-01; si += ui2 * (notorious_fft_real)-9.009688679024191460e-01;
        tr += vi2 * (notorious_fft_real)-4.338837391175580094e-01; ti += vr2 * (notorious_fft_real)-4.338837391175580094e-01;
        sr += ur3 * (notorious_fft_real)6.234898018587333723e-01; si += ui3 * (notorious_fft_real)6.234898018587333723e-01;
        tr += vi3 * (notorious_fft_real)-7.818314824680299147e-01; ti += vr3 * (notorious_fft_real)-7.818314824680299147e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[5*st]       = xr0 + sr + sg * tr;
        x[5*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.009688679024190350e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.009688679024190350e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)4.338837391175582314e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)4.338837391175582314e-01;
        sr += ur2 * (notorious_fft_real)6.234898018587333723e-01; si += ui2 * (notorious_fft_real)6.234898018587333723e-01;
        tr += vi2 * (notorious_fft_real)-7.818314824680299147e-01; ti += vr2 * (notorious_fft_real)-7.818314824680299147e-01;
        sr += ur3 * (notorious_fft_real)-2.225209339563140876e-01; si += ui3 * (notorious_fft_real)-2.225209339563140876e-01;
        tr += vi3 * (notorious_fft_real)9.749279121818237304e-01; ti += vr3 * (notorious_fft_real)9.749279121818237304e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[4*st]       = xr0 + sr + sg * tr;
        x[4*st + 1]   = xi0 + si - sg * ti;
    }
}

/* ============================================================================
 * Small-N DFT codelets (leaf plans and mixed-radix columns)
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE int notorious_fft_is_codelet(size_t n) {
    switch (n) {
    case 3: case 5: case 7: case 9: case 11: case 13:
    case 15: case 17: case 19: case 21: case 27: case 43:
        return 1;
    default:
        return 0;
    }
}

static NOTORIOUS_FFT_INLINE void notorious_fft_codelet_cmul(
    notorious_fft_real* b, int i, notorious_fft_real wr, notorious_fft_real wi)
{
    notorious_fft_real ur = b[2 * i], ui = b[2 * i + 1];
    b[2 * i]     = ur * wr - ui * wi;
    b[2 * i + 1] = ur * wi + ui * wr;
}

static const notorious_fft_real notorious_fft_c9[9] = {
    (notorious_fft_real)1.0,
    (notorious_fft_real)0.7660444431189780135,
    (notorious_fft_real)0.1736481776669304145,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.9396926207859083169,
    (notorious_fft_real)-0.9396926207859084279,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)0.1736481776669299704,
    (notorious_fft_real)0.7660444431189777914
};
static const notorious_fft_real notorious_fft_s9[9] = {
    (notorious_fft_real)0.0,
    (notorious_fft_real)0.6427876096865392519,
    (notorious_fft_real)0.9848077530122080203,
    (notorious_fft_real)0.8660254037844387076,
    (notorious_fft_real)0.3420201433256688794,
    (notorious_fft_real)-0.3420201433256686574,
    (notorious_fft_real)-0.8660254037844383745,
    (notorious_fft_real)-0.9848077530122081313,
    (notorious_fft_real)-0.6427876096865395850
};



static const notorious_fft_real notorious_fft_c15[15] = {
    (notorious_fft_real)1.0,
    (notorious_fft_real)0.9135454576426008666,
    (notorious_fft_real)0.6691306063588582376,
    (notorious_fft_real)0.3090169943749474513,
    (notorious_fft_real)-0.1045284632676533321,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.8090169943749473402,
    (notorious_fft_real)-0.9781476007338056888,
    (notorious_fft_real)-0.9781476007338056888,
    (notorious_fft_real)-0.8090169943749475623,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.1045284632676542341,
    (notorious_fft_real)0.3090169943749472292,
    (notorious_fft_real)0.6691306063588584596,
    (notorious_fft_real)0.9135454576426009776
};
static const notorious_fft_real notorious_fft_s15[15] = {
    (notorious_fft_real)0.0,
    (notorious_fft_real)0.4067366430758001528,
    (notorious_fft_real)0.7431448254773941331,
    (notorious_fft_real)0.9510565162951535312,
    (notorious_fft_real)0.9945218953682734009,
    (notorious_fft_real)0.8660254037844387076,
    (notorious_fft_real)0.5877852522924732481,
    (notorious_fft_real)0.2079116908177593148,
    (notorious_fft_real)-0.2079116908177590650,
    (notorious_fft_real)-0.5877852522924730261,
    (notorious_fft_real)-0.8660254037844383745,
    (notorious_fft_real)-0.9945218953682732899,
    (notorious_fft_real)-0.9510565162951536422,
    (notorious_fft_real)-0.7431448254773940221,
    (notorious_fft_real)-0.4067366430758001528
};


static const notorious_fft_real notorious_fft_c21[21] = {
    (notorious_fft_real)1.0,
    (notorious_fft_real)0.9555728057861406777,
    (notorious_fft_real)0.8262387743159949061,
    (notorious_fft_real)0.6234898018587335944,
    (notorious_fft_real)0.3653410243663949841,
    (notorious_fft_real)0.07473009358642439259,
    (notorious_fft_real)-0.2225209339563143374,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.7330518718298263403,
    (notorious_fft_real)-0.9009688679024190350,
    (notorious_fft_real)-0.9888308262251285230,
    (notorious_fft_real)-0.9888308262251286340,
    (notorious_fft_real)-0.9009688679024191460,
    (notorious_fft_real)-0.7330518718298262293,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.2225209339563145872,
    (notorious_fft_real)0.07473009358642436484,
    (notorious_fft_real)0.3653410243663953727,
    (notorious_fft_real)0.6234898018587333723,
    (notorious_fft_real)0.8262387743159944620,
    (notorious_fft_real)0.9555728057861405667
};
static const notorious_fft_real notorious_fft_s21[21] = {
    (notorious_fft_real)0.0,
    (notorious_fft_real)0.2947551744109042082,
    (notorious_fft_real)0.5633200580636220645,
    (notorious_fft_real)0.7818314824680298036,
    (notorious_fft_real)0.9308737486442042464,
    (notorious_fft_real)0.9972037971811801293,
    (notorious_fft_real)0.9749279121818236193,
    (notorious_fft_real)0.8660254037844387076,
    (notorious_fft_real)0.6801727377709193556,
    (notorious_fft_real)0.4338837391175582314,
    (notorious_fft_real)0.1490422661761747203,
    (notorious_fft_real)-0.1490422661761740264,
    (notorious_fft_real)-0.4338837391175580094,
    (notorious_fft_real)-0.6801727377709194666,
    (notorious_fft_real)-0.8660254037844383745,
    (notorious_fft_real)-0.9749279121818236193,
    (notorious_fft_real)-0.9972037971811801293,
    (notorious_fft_real)-0.9308737486442041353,
    (notorious_fft_real)-0.7818314824680299147,
    (notorious_fft_real)-0.5633200580636227306,
    (notorious_fft_real)-0.2947551744109047078
};

static const notorious_fft_real notorious_fft_c27[27] = {
    (notorious_fft_real)1.0,
    (notorious_fft_real)0.9730448705798238063,
    (notorious_fft_real)0.8936326403234122751,
    (notorious_fft_real)0.7660444431189780135,
    (notorious_fft_real)0.5971585917027861790,
    (notorious_fft_real)0.3960797660391568997,
    (notorious_fft_real)0.1736481776669304145,
    (notorious_fft_real)-0.05814482891047577373,
    (notorious_fft_real)-0.2868032327110902058,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.6862416378687334895,
    (notorious_fft_real)-0.8354878114129362654,
    (notorious_fft_real)-0.9396926207859083169,
    (notorious_fft_real)-0.9932383577419430232,
    (notorious_fft_real)-0.9932383577419430232,
    (notorious_fft_real)-0.9396926207859085389,
    (notorious_fft_real)-0.8354878114129364874,
    (notorious_fft_real)-0.6862416378687336005,
    (notorious_fft_real)-0.5,
    (notorious_fft_real)-0.2868032327110906499,
    (notorious_fft_real)-0.05814482891047623864,
    (notorious_fft_real)0.1736481776669308308,
    (notorious_fft_real)0.3960797660391565111,
    (notorious_fft_real)0.5971585917027858459,
    (notorious_fft_real)0.7660444431189777914,
    (notorious_fft_real)0.8936326403234121640,
    (notorious_fft_real)0.9730448705798240283
};
static const notorious_fft_real notorious_fft_s27[27] = {
    (notorious_fft_real)0.0,
    (notorious_fft_real)0.2306158707424401655,
    (notorious_fft_real)0.4487991802004621666,
    (notorious_fft_real)0.6427876096865392519,
    (notorious_fft_real)0.8021231927550437346,
    (notorious_fft_real)0.9182161068802739967,
    (notorious_fft_real)0.9848077530122080203,
    (notorious_fft_real)0.9983081582712681756,
    (notorious_fft_real)0.9579895123154889003,
    (notorious_fft_real)0.8660254037844387076,
    (notorious_fft_real)0.7273736415730488458,
    (notorious_fft_real)0.5495089780708062310,
    (notorious_fft_real)0.3420201433256688794,
    (notorious_fft_real)0.1160929141252299290,
    (notorious_fft_real)-0.1160929141252301233,
    (notorious_fft_real)-0.3420201433256682133,
    (notorious_fft_real)-0.5495089780708060090,
    (notorious_fft_real)-0.7273736415730486238,
    (notorious_fft_real)-0.8660254037844383745,
    (notorious_fft_real)-0.9579895123154887893,
    (notorious_fft_real)-0.9983081582712681756,
    (notorious_fft_real)-0.9848077530122080203,
    (notorious_fft_real)-0.9182161068802741077,
    (notorious_fft_real)-0.8021231927550439567,
    (notorious_fft_real)-0.6427876096865395850,
    (notorious_fft_real)-0.4487991802004624442,
    (notorious_fft_real)-0.2306158707424395826
};

static void notorious_fft_fft9_stride(notorious_fft_real* x, int s, int inverse) {
    notorious_fft_real b[18];
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    for (int n1 = 0; n1 < 3; n1++) {
        for (int n2 = 0; n2 < 3; n2++) {
            int src = (n1 + 3 * n2) * st;
            int dst = (n1 * 3 + n2) * 2;
            b[dst]     = x[src];
            b[dst + 1] = x[src + 1];
        }
    }
    for (int n1 = 0; n1 < 3; n1++)
        notorious_fft_radix3_stride(b + 6 * n1, 1, inverse);
    for (int n1 = 1; n1 < 3; n1++) {
        for (int k2 = 1; k2 < 3; k2++) {
            int p = n1 * k2;
            notorious_fft_codelet_cmul(b, n1 * 3 + k2, notorious_fft_c9[p], sg * notorious_fft_s9[p]);
        }
    }
    for (int k2 = 0; k2 < 3; k2++)
        notorious_fft_radix3_stride(b + 2 * k2, 3, inverse);
    for (int i = 0; i < 9; i++) {
        x[i * st]     = b[2 * i];
        x[i * st + 1] = b[2 * i + 1];
    }
}

static void notorious_fft_fft15_stride(notorious_fft_real* x, int s, int inverse) {
    notorious_fft_real b[30];
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    for (int n1 = 0; n1 < 3; n1++) {
        for (int n2 = 0; n2 < 5; n2++) {
            int src = (n1 + 3 * n2) * st;
            int dst = (n1 * 5 + n2) * 2;
            b[dst]     = x[src];
            b[dst + 1] = x[src + 1];
        }
    }
    for (int n1 = 0; n1 < 3; n1++)
        notorious_fft_radix5_stride(b + 10 * n1, 1, inverse);
    for (int n1 = 1; n1 < 3; n1++) {
        for (int k2 = 1; k2 < 5; k2++) {
            int p = (n1 * k2) % 15;
            notorious_fft_codelet_cmul(b, n1 * 5 + k2, notorious_fft_c15[p], sg * notorious_fft_s15[p]);
        }
    }
    for (int k2 = 0; k2 < 5; k2++)
        notorious_fft_radix3_stride(b + 2 * k2, 5, inverse);
    for (int i = 0; i < 15; i++) {
        x[i * st]     = b[2 * i];
        x[i * st + 1] = b[2 * i + 1];
    }
}

static void notorious_fft_fft21_stride(notorious_fft_real* x, int s, int inverse) {
    notorious_fft_real b[42];
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    for (int n1 = 0; n1 < 3; n1++) {
        for (int n2 = 0; n2 < 7; n2++) {
            int src = (n1 + 3 * n2) * st;
            int dst = (n1 * 7 + n2) * 2;
            b[dst]     = x[src];
            b[dst + 1] = x[src + 1];
        }
    }
    for (int n1 = 0; n1 < 3; n1++)
        notorious_fft_radix7_stride(b + 14 * n1, 1, inverse);
    for (int n1 = 1; n1 < 3; n1++) {
        for (int k2 = 1; k2 < 7; k2++) {
            int p = (n1 * k2) % 21;
            notorious_fft_codelet_cmul(b, n1 * 7 + k2, notorious_fft_c21[p], sg * notorious_fft_s21[p]);
        }
    }
    for (int k2 = 0; k2 < 7; k2++)
        notorious_fft_radix3_stride(b + 2 * k2, 7, inverse);
    for (int i = 0; i < 21; i++) {
        x[i * st]     = b[2 * i];
        x[i * st + 1] = b[2 * i + 1];
    }
}

static void notorious_fft_fft27_stride(notorious_fft_real* x, int s, int inverse) {
    notorious_fft_real b[54];
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    for (int n1 = 0; n1 < 3; n1++) {
        for (int n2 = 0; n2 < 9; n2++) {
            int src = (n1 + 3 * n2) * st;
            int dst = (n1 * 9 + n2) * 2;
            b[dst]     = x[src];
            b[dst + 1] = x[src + 1];
        }
    }
    for (int n1 = 0; n1 < 3; n1++)
        notorious_fft_fft9_stride(b + 18 * n1, 1, inverse);
    for (int n1 = 1; n1 < 3; n1++) {
        for (int k2 = 1; k2 < 9; k2++) {
            int p = n1 * k2;
            notorious_fft_codelet_cmul(b, n1 * 9 + k2, notorious_fft_c27[p], sg * notorious_fft_s27[p]);
        }
    }
    for (int k2 = 0; k2 < 9; k2++)
        notorious_fft_radix3_stride(b + 2 * k2, 9, inverse);
    for (int i = 0; i < 27; i++) {
        x[i * st]     = b[2 * i];
        x[i * st + 1] = b[2 * i + 1];
    }
}

/* Conjugate-pair prime codelets (11, 13, 19). */
static void notorious_fft_fft11_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real xr7 = x[7*st], xi7 = x[7*st + 1];
    notorious_fft_real xr8 = x[8*st], xi8 = x[8*st + 1];
    notorious_fft_real xr9 = x[9*st], xi9 = x[9*st + 1];
    notorious_fft_real xr10 = x[10*st], xi10 = x[10*st + 1];
    notorious_fft_real ur1 = xr1 + xr10, ui1 = xi1 + xi10;
    notorious_fft_real vr1 = xr1 - xr10, vi1 = xi1 - xi10;
    notorious_fft_real ur2 = xr2 + xr9, ui2 = xi2 + xi9;
    notorious_fft_real vr2 = xr2 - xr9, vi2 = xi2 - xi9;
    notorious_fft_real ur3 = xr3 + xr8, ui3 = xi3 + xi8;
    notorious_fft_real vr3 = xr3 - xr8, vi3 = xi3 - xi8;
    notorious_fft_real ur4 = xr4 + xr7, ui4 = xi4 + xi7;
    notorious_fft_real vr4 = xr4 - xr7, vi4 = xi4 - xi7;
    notorious_fft_real ur5 = xr5 + xr6, ui5 = xi5 + xi6;
    notorious_fft_real vr5 = xr5 - xr6, vi5 = xi5 - xi6;
    x[0] = xr0 + ur1 + ur2 + ur3 + ur4 + ur5;
    x[1] = xi0 + ui1 + ui2 + ui3 + ui4 + ui5;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)8.412535328311812055e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)8.412535328311812055e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)5.406408174555975554e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)5.406408174555975554e-01;
        sr += ur2 * (notorious_fft_real)4.154150130018864351e-01; si += ui2 * (notorious_fft_real)4.154150130018864351e-01;
        tr += vi2 * (notorious_fft_real)9.096319953545183301e-01; ti += vr2 * (notorious_fft_real)9.096319953545183301e-01;
        sr += ur3 * (notorious_fft_real)-1.423148382732850048e-01; si += ui3 * (notorious_fft_real)-1.423148382732850048e-01;
        tr += vi3 * (notorious_fft_real)9.898214418809327952e-01; ti += vr3 * (notorious_fft_real)9.898214418809327952e-01;
        sr += ur4 * (notorious_fft_real)-6.548607339452849896e-01; si += ui4 * (notorious_fft_real)-6.548607339452849896e-01;
        tr += vi4 * (notorious_fft_real)7.557495743542582689e-01; ti += vr4 * (notorious_fft_real)7.557495743542582689e-01;
        sr += ur5 * (notorious_fft_real)-9.594929736144973687e-01; si += ui5 * (notorious_fft_real)-9.594929736144973687e-01;
        tr += vi5 * (notorious_fft_real)2.817325568414296710e-01; ti += vr5 * (notorious_fft_real)2.817325568414296710e-01;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[10*st]       = xr0 + sr + sg * tr;
        x[10*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)4.154150130018864351e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)4.154150130018864351e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.096319953545183301e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.096319953545183301e-01;
        sr += ur2 * (notorious_fft_real)-6.548607339452849896e-01; si += ui2 * (notorious_fft_real)-6.548607339452849896e-01;
        tr += vi2 * (notorious_fft_real)7.557495743542582689e-01; ti += vr2 * (notorious_fft_real)7.557495743542582689e-01;
        sr += ur3 * (notorious_fft_real)-9.594929736144974797e-01; si += ui3 * (notorious_fft_real)-9.594929736144974797e-01;
        tr += vi3 * (notorious_fft_real)-2.817325568414293935e-01; ti += vr3 * (notorious_fft_real)-2.817325568414293935e-01;
        sr += ur4 * (notorious_fft_real)-1.423148382732852268e-01; si += ui4 * (notorious_fft_real)-1.423148382732852268e-01;
        tr += vi4 * (notorious_fft_real)-9.898214418809326842e-01; ti += vr4 * (notorious_fft_real)-9.898214418809326842e-01;
        sr += ur5 * (notorious_fft_real)8.412535328311812055e-01; si += ui5 * (notorious_fft_real)8.412535328311812055e-01;
        tr += vi5 * (notorious_fft_real)-5.406408174555974444e-01; ti += vr5 * (notorious_fft_real)-5.406408174555974444e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[9*st]       = xr0 + sr + sg * tr;
        x[9*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-1.423148382732850048e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-1.423148382732850048e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.898214418809327952e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.898214418809327952e-01;
        sr += ur2 * (notorious_fft_real)-9.594929736144974797e-01; si += ui2 * (notorious_fft_real)-9.594929736144974797e-01;
        tr += vi2 * (notorious_fft_real)-2.817325568414293935e-01; ti += vr2 * (notorious_fft_real)-2.817325568414293935e-01;
        sr += ur3 * (notorious_fft_real)4.154150130018860465e-01; si += ui3 * (notorious_fft_real)4.154150130018860465e-01;
        tr += vi3 * (notorious_fft_real)-9.096319953545185522e-01; ti += vr3 * (notorious_fft_real)-9.096319953545185522e-01;
        sr += ur4 * (notorious_fft_real)8.412535328311815386e-01; si += ui4 * (notorious_fft_real)8.412535328311815386e-01;
        tr += vi4 * (notorious_fft_real)5.406408174555971113e-01; ti += vr4 * (notorious_fft_real)5.406408174555971113e-01;
        sr += ur5 * (notorious_fft_real)-6.548607339452838794e-01; si += ui5 * (notorious_fft_real)-6.548607339452838794e-01;
        tr += vi5 * (notorious_fft_real)7.557495743542593791e-01; ti += vr5 * (notorious_fft_real)7.557495743542593791e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[8*st]       = xr0 + sr + sg * tr;
        x[8*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-6.548607339452849896e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-6.548607339452849896e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.557495743542582689e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.557495743542582689e-01;
        sr += ur2 * (notorious_fft_real)-1.423148382732852268e-01; si += ui2 * (notorious_fft_real)-1.423148382732852268e-01;
        tr += vi2 * (notorious_fft_real)-9.898214418809326842e-01; ti += vr2 * (notorious_fft_real)-9.898214418809326842e-01;
        sr += ur3 * (notorious_fft_real)8.412535328311815386e-01; si += ui3 * (notorious_fft_real)8.412535328311815386e-01;
        tr += vi3 * (notorious_fft_real)5.406408174555971113e-01; ti += vr3 * (notorious_fft_real)5.406408174555971113e-01;
        sr += ur4 * (notorious_fft_real)-9.594929736144973687e-01; si += ui4 * (notorious_fft_real)-9.594929736144973687e-01;
        tr += vi4 * (notorious_fft_real)2.817325568414298931e-01; ti += vr4 * (notorious_fft_real)2.817325568414298931e-01;
        sr += ur5 * (notorious_fft_real)4.154150130018866016e-01; si += ui5 * (notorious_fft_real)4.154150130018866016e-01;
        tr += vi5 * (notorious_fft_real)-9.096319953545183301e-01; ti += vr5 * (notorious_fft_real)-9.096319953545183301e-01;
        x[4*st]         = xr0 + sr - sg * tr;
        x[4*st + 1]     = xi0 + si + sg * ti;
        x[7*st]       = xr0 + sr + sg * tr;
        x[7*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.594929736144973687e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.594929736144973687e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)2.817325568414296710e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)2.817325568414296710e-01;
        sr += ur2 * (notorious_fft_real)8.412535328311812055e-01; si += ui2 * (notorious_fft_real)8.412535328311812055e-01;
        tr += vi2 * (notorious_fft_real)-5.406408174555974444e-01; ti += vr2 * (notorious_fft_real)-5.406408174555974444e-01;
        sr += ur3 * (notorious_fft_real)-6.548607339452838794e-01; si += ui3 * (notorious_fft_real)-6.548607339452838794e-01;
        tr += vi3 * (notorious_fft_real)7.557495743542593791e-01; ti += vr3 * (notorious_fft_real)7.557495743542593791e-01;
        sr += ur4 * (notorious_fft_real)4.154150130018866016e-01; si += ui4 * (notorious_fft_real)4.154150130018866016e-01;
        tr += vi4 * (notorious_fft_real)-9.096319953545183301e-01; ti += vr4 * (notorious_fft_real)-9.096319953545183301e-01;
        sr += ur5 * (notorious_fft_real)-1.423148382732853934e-01; si += ui5 * (notorious_fft_real)-1.423148382732853934e-01;
        tr += vi5 * (notorious_fft_real)9.898214418809326842e-01; ti += vr5 * (notorious_fft_real)9.898214418809326842e-01;
        x[5*st]         = xr0 + sr - sg * tr;
        x[5*st + 1]     = xi0 + si + sg * ti;
        x[6*st]       = xr0 + sr + sg * tr;
        x[6*st + 1]   = xi0 + si - sg * ti;
    }
}

static void notorious_fft_fft13_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real xr7 = x[7*st], xi7 = x[7*st + 1];
    notorious_fft_real xr8 = x[8*st], xi8 = x[8*st + 1];
    notorious_fft_real xr9 = x[9*st], xi9 = x[9*st + 1];
    notorious_fft_real xr10 = x[10*st], xi10 = x[10*st + 1];
    notorious_fft_real xr11 = x[11*st], xi11 = x[11*st + 1];
    notorious_fft_real xr12 = x[12*st], xi12 = x[12*st + 1];
    notorious_fft_real ur1 = xr1 + xr12, ui1 = xi1 + xi12;
    notorious_fft_real vr1 = xr1 - xr12, vi1 = xi1 - xi12;
    notorious_fft_real ur2 = xr2 + xr11, ui2 = xi2 + xi11;
    notorious_fft_real vr2 = xr2 - xr11, vi2 = xi2 - xi11;
    notorious_fft_real ur3 = xr3 + xr10, ui3 = xi3 + xi10;
    notorious_fft_real vr3 = xr3 - xr10, vi3 = xi3 - xi10;
    notorious_fft_real ur4 = xr4 + xr9, ui4 = xi4 + xi9;
    notorious_fft_real vr4 = xr4 - xr9, vi4 = xi4 - xi9;
    notorious_fft_real ur5 = xr5 + xr8, ui5 = xi5 + xi8;
    notorious_fft_real vr5 = xr5 - xr8, vi5 = xi5 - xi8;
    notorious_fft_real ur6 = xr6 + xr7, ui6 = xi6 + xi7;
    notorious_fft_real vr6 = xr6 - xr7, vi6 = xi6 - xi7;
    x[0] = xr0 + ur1 + ur2 + ur3 + ur4 + ur5 + ur6;
    x[1] = xi0 + ui1 + ui2 + ui3 + ui4 + ui5 + ui6;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)8.854560256532099105e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)8.854560256532099105e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)4.647231720437685065e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)4.647231720437685065e-01;
        sr += ur2 * (notorious_fft_real)5.680647467311559229e-01; si += ui2 * (notorious_fft_real)5.680647467311559229e-01;
        tr += vi2 * (notorious_fft_real)8.229838658936563522e-01; ti += vr2 * (notorious_fft_real)8.229838658936563522e-01;
        sr += ur3 * (notorious_fft_real)1.205366802553230060e-01; si += ui3 * (notorious_fft_real)1.205366802553230060e-01;
        tr += vi3 * (notorious_fft_real)9.927088740980539727e-01; ti += vr3 * (notorious_fft_real)9.927088740980539727e-01;
        sr += ur4 * (notorious_fft_real)-3.546048870425354549e-01; si += ui4 * (notorious_fft_real)-3.546048870425354549e-01;
        tr += vi4 * (notorious_fft_real)9.350162426854148334e-01; ti += vr4 * (notorious_fft_real)9.350162426854148334e-01;
        sr += ur5 * (notorious_fft_real)-7.485107481711011923e-01; si += ui5 * (notorious_fft_real)-7.485107481711011923e-01;
        tr += vi5 * (notorious_fft_real)6.631226582407951931e-01; ti += vr5 * (notorious_fft_real)6.631226582407951931e-01;
        sr += ur6 * (notorious_fft_real)-9.709418174260520118e-01; si += ui6 * (notorious_fft_real)-9.709418174260520118e-01;
        tr += vi6 * (notorious_fft_real)2.393156642875576834e-01; ti += vr6 * (notorious_fft_real)2.393156642875576834e-01;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[12*st]       = xr0 + sr + sg * tr;
        x[12*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)5.680647467311559229e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)5.680647467311559229e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.229838658936563522e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.229838658936563522e-01;
        sr += ur2 * (notorious_fft_real)-3.546048870425354549e-01; si += ui2 * (notorious_fft_real)-3.546048870425354549e-01;
        tr += vi2 * (notorious_fft_real)9.350162426854148334e-01; ti += vr2 * (notorious_fft_real)9.350162426854148334e-01;
        sr += ur3 * (notorious_fft_real)-9.709418174260520118e-01; si += ui3 * (notorious_fft_real)-9.709418174260520118e-01;
        tr += vi3 * (notorious_fft_real)2.393156642875576834e-01; ti += vr3 * (notorious_fft_real)2.393156642875576834e-01;
        sr += ur4 * (notorious_fft_real)-7.485107481711013033e-01; si += ui4 * (notorious_fft_real)-7.485107481711013033e-01;
        tr += vi4 * (notorious_fft_real)-6.631226582407949710e-01; ti += vr4 * (notorious_fft_real)-6.631226582407949710e-01;
        sr += ur5 * (notorious_fft_real)1.205366802553232003e-01; si += ui5 * (notorious_fft_real)1.205366802553232003e-01;
        tr += vi5 * (notorious_fft_real)-9.927088740980539727e-01; ti += vr5 * (notorious_fft_real)-9.927088740980539727e-01;
        sr += ur6 * (notorious_fft_real)8.854560256532100215e-01; si += ui6 * (notorious_fft_real)8.854560256532100215e-01;
        tr += vi6 * (notorious_fft_real)-4.647231720437683955e-01; ti += vr6 * (notorious_fft_real)-4.647231720437683955e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[11*st]       = xr0 + sr + sg * tr;
        x[11*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)1.205366802553230060e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)1.205366802553230060e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.927088740980539727e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.927088740980539727e-01;
        sr += ur2 * (notorious_fft_real)-9.709418174260520118e-01; si += ui2 * (notorious_fft_real)-9.709418174260520118e-01;
        tr += vi2 * (notorious_fft_real)2.393156642875576834e-01; ti += vr2 * (notorious_fft_real)2.393156642875576834e-01;
        sr += ur3 * (notorious_fft_real)-3.546048870425358990e-01; si += ui3 * (notorious_fft_real)-3.546048870425358990e-01;
        tr += vi3 * (notorious_fft_real)-9.350162426854147224e-01; ti += vr3 * (notorious_fft_real)-9.350162426854147224e-01;
        sr += ur4 * (notorious_fft_real)8.854560256532100215e-01; si += ui4 * (notorious_fft_real)8.854560256532100215e-01;
        tr += vi4 * (notorious_fft_real)-4.647231720437683955e-01; ti += vr4 * (notorious_fft_real)-4.647231720437683955e-01;
        sr += ur5 * (notorious_fft_real)5.680647467311567000e-01; si += ui5 * (notorious_fft_real)5.680647467311567000e-01;
        tr += vi5 * (notorious_fft_real)8.229838658936557971e-01; ti += vr5 * (notorious_fft_real)8.229838658936557971e-01;
        sr += ur6 * (notorious_fft_real)-7.485107481711006372e-01; si += ui6 * (notorious_fft_real)-7.485107481711006372e-01;
        tr += vi6 * (notorious_fft_real)6.631226582407956371e-01; ti += vr6 * (notorious_fft_real)6.631226582407956371e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[10*st]       = xr0 + sr + sg * tr;
        x[10*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-3.546048870425354549e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-3.546048870425354549e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.350162426854148334e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.350162426854148334e-01;
        sr += ur2 * (notorious_fft_real)-7.485107481711013033e-01; si += ui2 * (notorious_fft_real)-7.485107481711013033e-01;
        tr += vi2 * (notorious_fft_real)-6.631226582407949710e-01; ti += vr2 * (notorious_fft_real)-6.631226582407949710e-01;
        sr += ur3 * (notorious_fft_real)8.854560256532100215e-01; si += ui3 * (notorious_fft_real)8.854560256532100215e-01;
        tr += vi3 * (notorious_fft_real)-4.647231720437683955e-01; ti += vr3 * (notorious_fft_real)-4.647231720437683955e-01;
        sr += ur4 * (notorious_fft_real)1.205366802553236860e-01; si += ui4 * (notorious_fft_real)1.205366802553236860e-01;
        tr += vi4 * (notorious_fft_real)9.927088740980538617e-01; ti += vr4 * (notorious_fft_real)9.927088740980538617e-01;
        sr += ur5 * (notorious_fft_real)-9.709418174260519008e-01; si += ui5 * (notorious_fft_real)-9.709418174260519008e-01;
        tr += vi5 * (notorious_fft_real)-2.393156642875580720e-01; ti += vr5 * (notorious_fft_real)-2.393156642875580720e-01;
        sr += ur6 * (notorious_fft_real)5.680647467311561449e-01; si += ui6 * (notorious_fft_real)5.680647467311561449e-01;
        tr += vi6 * (notorious_fft_real)-8.229838658936561302e-01; ti += vr6 * (notorious_fft_real)-8.229838658936561302e-01;
        x[4*st]         = xr0 + sr - sg * tr;
        x[4*st + 1]     = xi0 + si + sg * ti;
        x[9*st]       = xr0 + sr + sg * tr;
        x[9*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-7.485107481711011923e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-7.485107481711011923e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)6.631226582407951931e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)6.631226582407951931e-01;
        sr += ur2 * (notorious_fft_real)1.205366802553232003e-01; si += ui2 * (notorious_fft_real)1.205366802553232003e-01;
        tr += vi2 * (notorious_fft_real)-9.927088740980539727e-01; ti += vr2 * (notorious_fft_real)-9.927088740980539727e-01;
        sr += ur3 * (notorious_fft_real)5.680647467311567000e-01; si += ui3 * (notorious_fft_real)5.680647467311567000e-01;
        tr += vi3 * (notorious_fft_real)8.229838658936557971e-01; ti += vr3 * (notorious_fft_real)8.229838658936557971e-01;
        sr += ur4 * (notorious_fft_real)-9.709418174260519008e-01; si += ui4 * (notorious_fft_real)-9.709418174260519008e-01;
        tr += vi4 * (notorious_fft_real)-2.393156642875580720e-01; ti += vr4 * (notorious_fft_real)-2.393156642875580720e-01;
        sr += ur5 * (notorious_fft_real)8.854560256532099105e-01; si += ui5 * (notorious_fft_real)8.854560256532099105e-01;
        tr += vi5 * (notorious_fft_real)-4.647231720437686175e-01; ti += vr5 * (notorious_fft_real)-4.647231720437686175e-01;
        sr += ur6 * (notorious_fft_real)-3.546048870425335675e-01; si += ui6 * (notorious_fft_real)-3.546048870425335675e-01;
        tr += vi6 * (notorious_fft_real)9.350162426854156106e-01; ti += vr6 * (notorious_fft_real)9.350162426854156106e-01;
        x[5*st]         = xr0 + sr - sg * tr;
        x[5*st + 1]     = xi0 + si + sg * ti;
        x[8*st]       = xr0 + sr + sg * tr;
        x[8*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.709418174260520118e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.709418174260520118e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)2.393156642875576834e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)2.393156642875576834e-01;
        sr += ur2 * (notorious_fft_real)8.854560256532100215e-01; si += ui2 * (notorious_fft_real)8.854560256532100215e-01;
        tr += vi2 * (notorious_fft_real)-4.647231720437683955e-01; ti += vr2 * (notorious_fft_real)-4.647231720437683955e-01;
        sr += ur3 * (notorious_fft_real)-7.485107481711006372e-01; si += ui3 * (notorious_fft_real)-7.485107481711006372e-01;
        tr += vi3 * (notorious_fft_real)6.631226582407956371e-01; ti += vr3 * (notorious_fft_real)6.631226582407956371e-01;
        sr += ur4 * (notorious_fft_real)5.680647467311561449e-01; si += ui4 * (notorious_fft_real)5.680647467311561449e-01;
        tr += vi4 * (notorious_fft_real)-8.229838658936561302e-01; ti += vr4 * (notorious_fft_real)-8.229838658936561302e-01;
        sr += ur5 * (notorious_fft_real)-3.546048870425335675e-01; si += ui5 * (notorious_fft_real)-3.546048870425335675e-01;
        tr += vi5 * (notorious_fft_real)9.350162426854156106e-01; ti += vr5 * (notorious_fft_real)9.350162426854156106e-01;
        sr += ur6 * (notorious_fft_real)1.205366802553218403e-01; si += ui6 * (notorious_fft_real)1.205366802553218403e-01;
        tr += vi6 * (notorious_fft_real)-9.927088740980541948e-01; ti += vr6 * (notorious_fft_real)-9.927088740980541948e-01;
        x[6*st]         = xr0 + sr - sg * tr;
        x[6*st + 1]     = xi0 + si + sg * ti;
        x[7*st]       = xr0 + sr + sg * tr;
        x[7*st + 1]   = xi0 + si - sg * ti;
    }
}

static void notorious_fft_fft19_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real xr7 = x[7*st], xi7 = x[7*st + 1];
    notorious_fft_real xr8 = x[8*st], xi8 = x[8*st + 1];
    notorious_fft_real xr9 = x[9*st], xi9 = x[9*st + 1];
    notorious_fft_real xr10 = x[10*st], xi10 = x[10*st + 1];
    notorious_fft_real xr11 = x[11*st], xi11 = x[11*st + 1];
    notorious_fft_real xr12 = x[12*st], xi12 = x[12*st + 1];
    notorious_fft_real xr13 = x[13*st], xi13 = x[13*st + 1];
    notorious_fft_real xr14 = x[14*st], xi14 = x[14*st + 1];
    notorious_fft_real xr15 = x[15*st], xi15 = x[15*st + 1];
    notorious_fft_real xr16 = x[16*st], xi16 = x[16*st + 1];
    notorious_fft_real xr17 = x[17*st], xi17 = x[17*st + 1];
    notorious_fft_real xr18 = x[18*st], xi18 = x[18*st + 1];
    notorious_fft_real ur1 = xr1 + xr18, ui1 = xi1 + xi18;
    notorious_fft_real vr1 = xr1 - xr18, vi1 = xi1 - xi18;
    notorious_fft_real ur2 = xr2 + xr17, ui2 = xi2 + xi17;
    notorious_fft_real vr2 = xr2 - xr17, vi2 = xi2 - xi17;
    notorious_fft_real ur3 = xr3 + xr16, ui3 = xi3 + xi16;
    notorious_fft_real vr3 = xr3 - xr16, vi3 = xi3 - xi16;
    notorious_fft_real ur4 = xr4 + xr15, ui4 = xi4 + xi15;
    notorious_fft_real vr4 = xr4 - xr15, vi4 = xi4 - xi15;
    notorious_fft_real ur5 = xr5 + xr14, ui5 = xi5 + xi14;
    notorious_fft_real vr5 = xr5 - xr14, vi5 = xi5 - xi14;
    notorious_fft_real ur6 = xr6 + xr13, ui6 = xi6 + xi13;
    notorious_fft_real vr6 = xr6 - xr13, vi6 = xi6 - xi13;
    notorious_fft_real ur7 = xr7 + xr12, ui7 = xi7 + xi12;
    notorious_fft_real vr7 = xr7 - xr12, vi7 = xi7 - xi12;
    notorious_fft_real ur8 = xr8 + xr11, ui8 = xi8 + xi11;
    notorious_fft_real vr8 = xr8 - xr11, vi8 = xi8 - xi11;
    notorious_fft_real ur9 = xr9 + xr10, ui9 = xi9 + xi10;
    notorious_fft_real vr9 = xr9 - xr10, vi9 = xi9 - xi10;
    x[0] = xr0 + ur1 + ur2 + ur3 + ur4 + ur5 + ur6 + ur7 + ur8 + ur9;
    x[1] = xi0 + ui1 + ui2 + ui3 + ui4 + ui5 + ui6 + ui7 + ui8 + ui9;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.458172417006346366e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.458172417006346366e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)3.246994692046834552e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)3.246994692046834552e-01;
        sr += ur2 * (notorious_fft_real)7.891405093963935746e-01; si += ui2 * (notorious_fft_real)7.891405093963935746e-01;
        tr += vi2 * (notorious_fft_real)6.142127126896678169e-01; ti += vr2 * (notorious_fft_real)6.142127126896678169e-01;
        sr += ur3 * (notorious_fft_real)5.469481581224269195e-01; si += ui3 * (notorious_fft_real)5.469481581224269195e-01;
        tr += vi3 * (notorious_fft_real)8.371664782625285461e-01; ti += vr3 * (notorious_fft_real)8.371664782625285461e-01;
        sr += ur4 * (notorious_fft_real)2.454854871407992356e-01; si += ui4 * (notorious_fft_real)2.454854871407992356e-01;
        tr += vi4 * (notorious_fft_real)9.694002659393303745e-01; ti += vr4 * (notorious_fft_real)9.694002659393303745e-01;
        sr += ur5 * (notorious_fft_real)-8.257934547233226896e-02; si += ui5 * (notorious_fft_real)-8.257934547233226896e-02;
        tr += vi5 * (notorious_fft_real)9.965844930066698470e-01; ti += vr5 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur6 * (notorious_fft_real)-4.016954246529694239e-01; si += ui6 * (notorious_fft_real)-4.016954246529694239e-01;
        tr += vi6 * (notorious_fft_real)9.157733266550573958e-01; ti += vr6 * (notorious_fft_real)9.157733266550573958e-01;
        sr += ur7 * (notorious_fft_real)-6.772815716257408747e-01; si += ui7 * (notorious_fft_real)-6.772815716257408747e-01;
        tr += vi7 * (notorious_fft_real)7.357239106731318090e-01; ti += vr7 * (notorious_fft_real)7.357239106731318090e-01;
        sr += ur8 * (notorious_fft_real)-8.794737512064889629e-01; si += ui8 * (notorious_fft_real)-8.794737512064889629e-01;
        tr += vi8 * (notorious_fft_real)4.759473930370736738e-01; ti += vr8 * (notorious_fft_real)4.759473930370736738e-01;
        sr += ur9 * (notorious_fft_real)-9.863613034027223225e-01; si += ui9 * (notorious_fft_real)-9.863613034027223225e-01;
        tr += vi9 * (notorious_fft_real)1.645945902807340322e-01; ti += vr9 * (notorious_fft_real)1.645945902807340322e-01;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[18*st]       = xr0 + sr + sg * tr;
        x[18*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)7.891405093963935746e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)7.891405093963935746e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)6.142127126896678169e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)6.142127126896678169e-01;
        sr += ur2 * (notorious_fft_real)2.454854871407992356e-01; si += ui2 * (notorious_fft_real)2.454854871407992356e-01;
        tr += vi2 * (notorious_fft_real)9.694002659393303745e-01; ti += vr2 * (notorious_fft_real)9.694002659393303745e-01;
        sr += ur3 * (notorious_fft_real)-4.016954246529694239e-01; si += ui3 * (notorious_fft_real)-4.016954246529694239e-01;
        tr += vi3 * (notorious_fft_real)9.157733266550573958e-01; ti += vr3 * (notorious_fft_real)9.157733266550573958e-01;
        sr += ur4 * (notorious_fft_real)-8.794737512064889629e-01; si += ui4 * (notorious_fft_real)-8.794737512064889629e-01;
        tr += vi4 * (notorious_fft_real)4.759473930370736738e-01; ti += vr4 * (notorious_fft_real)4.759473930370736738e-01;
        sr += ur5 * (notorious_fft_real)-9.863613034027224336e-01; si += ui5 * (notorious_fft_real)-9.863613034027224336e-01;
        tr += vi5 * (notorious_fft_real)-1.645945902807337824e-01; ti += vr5 * (notorious_fft_real)-1.645945902807337824e-01;
        sr += ur6 * (notorious_fft_real)-6.772815716257410967e-01; si += ui6 * (notorious_fft_real)-6.772815716257410967e-01;
        tr += vi6 * (notorious_fft_real)-7.357239106731315870e-01; ti += vr6 * (notorious_fft_real)-7.357239106731315870e-01;
        sr += ur7 * (notorious_fft_real)-8.257934547233274081e-02; si += ui7 * (notorious_fft_real)-8.257934547233274081e-02;
        tr += vi7 * (notorious_fft_real)-9.965844930066698470e-01; ti += vr7 * (notorious_fft_real)-9.965844930066698470e-01;
        sr += ur8 * (notorious_fft_real)5.469481581224265865e-01; si += ui8 * (notorious_fft_real)5.469481581224265865e-01;
        tr += vi8 * (notorious_fft_real)-8.371664782625287682e-01; ti += vr8 * (notorious_fft_real)-8.371664782625287682e-01;
        sr += ur9 * (notorious_fft_real)9.458172417006346366e-01; si += ui9 * (notorious_fft_real)9.458172417006346366e-01;
        tr += vi9 * (notorious_fft_real)-3.246994692046837327e-01; ti += vr9 * (notorious_fft_real)-3.246994692046837327e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[17*st]       = xr0 + sr + sg * tr;
        x[17*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)5.469481581224269195e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)5.469481581224269195e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.371664782625285461e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.371664782625285461e-01;
        sr += ur2 * (notorious_fft_real)-4.016954246529694239e-01; si += ui2 * (notorious_fft_real)-4.016954246529694239e-01;
        tr += vi2 * (notorious_fft_real)9.157733266550573958e-01; ti += vr2 * (notorious_fft_real)9.157733266550573958e-01;
        sr += ur3 * (notorious_fft_real)-9.863613034027223225e-01; si += ui3 * (notorious_fft_real)-9.863613034027223225e-01;
        tr += vi3 * (notorious_fft_real)1.645945902807340322e-01; ti += vr3 * (notorious_fft_real)1.645945902807340322e-01;
        sr += ur4 * (notorious_fft_real)-6.772815716257410967e-01; si += ui4 * (notorious_fft_real)-6.772815716257410967e-01;
        tr += vi4 * (notorious_fft_real)-7.357239106731315870e-01; ti += vr4 * (notorious_fft_real)-7.357239106731315870e-01;
        sr += ur5 * (notorious_fft_real)2.454854871407987915e-01; si += ui5 * (notorious_fft_real)2.454854871407987915e-01;
        tr += vi5 * (notorious_fft_real)-9.694002659393304855e-01; ti += vr5 * (notorious_fft_real)-9.694002659393304855e-01;
        sr += ur6 * (notorious_fft_real)9.458172417006346366e-01; si += ui6 * (notorious_fft_real)9.458172417006346366e-01;
        tr += vi6 * (notorious_fft_real)-3.246994692046837327e-01; ti += vr6 * (notorious_fft_real)-3.246994692046837327e-01;
        sr += ur7 * (notorious_fft_real)7.891405093963936856e-01; si += ui7 * (notorious_fft_real)7.891405093963936856e-01;
        tr += vi7 * (notorious_fft_real)6.142127126896677058e-01; ti += vr7 * (notorious_fft_real)6.142127126896677058e-01;
        sr += ur8 * (notorious_fft_real)-8.257934547233225508e-02; si += ui8 * (notorious_fft_real)-8.257934547233225508e-02;
        tr += vi8 * (notorious_fft_real)9.965844930066698470e-01; ti += vr8 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur9 * (notorious_fft_real)-8.794737512064890739e-01; si += ui9 * (notorious_fft_real)-8.794737512064890739e-01;
        tr += vi9 * (notorious_fft_real)4.759473930370735073e-01; ti += vr9 * (notorious_fft_real)4.759473930370735073e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[16*st]       = xr0 + sr + sg * tr;
        x[16*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)2.454854871407992356e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)2.454854871407992356e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.694002659393303745e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.694002659393303745e-01;
        sr += ur2 * (notorious_fft_real)-8.794737512064889629e-01; si += ui2 * (notorious_fft_real)-8.794737512064889629e-01;
        tr += vi2 * (notorious_fft_real)4.759473930370736738e-01; ti += vr2 * (notorious_fft_real)4.759473930370736738e-01;
        sr += ur3 * (notorious_fft_real)-6.772815716257410967e-01; si += ui3 * (notorious_fft_real)-6.772815716257410967e-01;
        tr += vi3 * (notorious_fft_real)-7.357239106731315870e-01; ti += vr3 * (notorious_fft_real)-7.357239106731315870e-01;
        sr += ur4 * (notorious_fft_real)5.469481581224265865e-01; si += ui4 * (notorious_fft_real)5.469481581224265865e-01;
        tr += vi4 * (notorious_fft_real)-8.371664782625287682e-01; ti += vr4 * (notorious_fft_real)-8.371664782625287682e-01;
        sr += ur5 * (notorious_fft_real)9.458172417006347477e-01; si += ui5 * (notorious_fft_real)9.458172417006347477e-01;
        tr += vi5 * (notorious_fft_real)3.246994692046832887e-01; ti += vr5 * (notorious_fft_real)3.246994692046832887e-01;
        sr += ur6 * (notorious_fft_real)-8.257934547233225508e-02; si += ui6 * (notorious_fft_real)-8.257934547233225508e-02;
        tr += vi6 * (notorious_fft_real)9.965844930066698470e-01; ti += vr6 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur7 * (notorious_fft_real)-9.863613034027222115e-01; si += ui7 * (notorious_fft_real)-9.863613034027222115e-01;
        tr += vi7 * (notorious_fft_real)1.645945902807346983e-01; ti += vr7 * (notorious_fft_real)1.645945902807346983e-01;
        sr += ur8 * (notorious_fft_real)-4.016954246529700900e-01; si += ui8 * (notorious_fft_real)-4.016954246529700900e-01;
        tr += vi8 * (notorious_fft_real)-9.157733266550571738e-01; ti += vr8 * (notorious_fft_real)-9.157733266550571738e-01;
        sr += ur9 * (notorious_fft_real)7.891405093963932416e-01; si += ui9 * (notorious_fft_real)7.891405093963932416e-01;
        tr += vi9 * (notorious_fft_real)-6.142127126896682610e-01; ti += vr9 * (notorious_fft_real)-6.142127126896682610e-01;
        x[4*st]         = xr0 + sr - sg * tr;
        x[4*st + 1]     = xi0 + si + sg * ti;
        x[15*st]       = xr0 + sr + sg * tr;
        x[15*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-8.257934547233226896e-02;
        notorious_fft_real si = ui1 * (notorious_fft_real)-8.257934547233226896e-02;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.965844930066698470e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur2 * (notorious_fft_real)-9.863613034027224336e-01; si += ui2 * (notorious_fft_real)-9.863613034027224336e-01;
        tr += vi2 * (notorious_fft_real)-1.645945902807337824e-01; ti += vr2 * (notorious_fft_real)-1.645945902807337824e-01;
        sr += ur3 * (notorious_fft_real)2.454854871407987915e-01; si += ui3 * (notorious_fft_real)2.454854871407987915e-01;
        tr += vi3 * (notorious_fft_real)-9.694002659393304855e-01; ti += vr3 * (notorious_fft_real)-9.694002659393304855e-01;
        sr += ur4 * (notorious_fft_real)9.458172417006347477e-01; si += ui4 * (notorious_fft_real)9.458172417006347477e-01;
        tr += vi4 * (notorious_fft_real)3.246994692046832887e-01; ti += vr4 * (notorious_fft_real)3.246994692046832887e-01;
        sr += ur5 * (notorious_fft_real)-4.016954246529694239e-01; si += ui5 * (notorious_fft_real)-4.016954246529694239e-01;
        tr += vi5 * (notorious_fft_real)9.157733266550575069e-01; ti += vr5 * (notorious_fft_real)9.157733266550575069e-01;
        sr += ur6 * (notorious_fft_real)-8.794737512064894069e-01; si += ui6 * (notorious_fft_real)-8.794737512064894069e-01;
        tr += vi6 * (notorious_fft_real)-4.759473930370728967e-01; ti += vr6 * (notorious_fft_real)-4.759473930370728967e-01;
        sr += ur7 * (notorious_fft_real)5.469481581224255873e-01; si += ui7 * (notorious_fft_real)5.469481581224255873e-01;
        tr += vi7 * (notorious_fft_real)-8.371664782625294343e-01; ti += vr7 * (notorious_fft_real)-8.371664782625294343e-01;
        sr += ur8 * (notorious_fft_real)7.891405093963939077e-01; si += ui8 * (notorious_fft_real)7.891405093963939077e-01;
        tr += vi8 * (notorious_fft_real)6.142127126896674838e-01; ti += vr8 * (notorious_fft_real)6.142127126896674838e-01;
        sr += ur9 * (notorious_fft_real)-6.772815716257415408e-01; si += ui9 * (notorious_fft_real)-6.772815716257415408e-01;
        tr += vi9 * (notorious_fft_real)7.357239106731311429e-01; ti += vr9 * (notorious_fft_real)7.357239106731311429e-01;
        x[5*st]         = xr0 + sr - sg * tr;
        x[5*st + 1]     = xi0 + si + sg * ti;
        x[14*st]       = xr0 + sr + sg * tr;
        x[14*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-4.016954246529694239e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-4.016954246529694239e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.157733266550573958e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.157733266550573958e-01;
        sr += ur2 * (notorious_fft_real)-6.772815716257410967e-01; si += ui2 * (notorious_fft_real)-6.772815716257410967e-01;
        tr += vi2 * (notorious_fft_real)-7.357239106731315870e-01; ti += vr2 * (notorious_fft_real)-7.357239106731315870e-01;
        sr += ur3 * (notorious_fft_real)9.458172417006346366e-01; si += ui3 * (notorious_fft_real)9.458172417006346366e-01;
        tr += vi3 * (notorious_fft_real)-3.246994692046837327e-01; ti += vr3 * (notorious_fft_real)-3.246994692046837327e-01;
        sr += ur4 * (notorious_fft_real)-8.257934547233225508e-02; si += ui4 * (notorious_fft_real)-8.257934547233225508e-02;
        tr += vi4 * (notorious_fft_real)9.965844930066698470e-01; ti += vr4 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur5 * (notorious_fft_real)-8.794737512064894069e-01; si += ui5 * (notorious_fft_real)-8.794737512064894069e-01;
        tr += vi5 * (notorious_fft_real)-4.759473930370728967e-01; ti += vr5 * (notorious_fft_real)-4.759473930370728967e-01;
        sr += ur6 * (notorious_fft_real)7.891405093963932416e-01; si += ui6 * (notorious_fft_real)7.891405093963932416e-01;
        tr += vi6 * (notorious_fft_real)-6.142127126896682610e-01; ti += vr6 * (notorious_fft_real)-6.142127126896682610e-01;
        sr += ur7 * (notorious_fft_real)2.454854871407994854e-01; si += ui7 * (notorious_fft_real)2.454854871407994854e-01;
        tr += vi7 * (notorious_fft_real)9.694002659393303745e-01; ti += vr7 * (notorious_fft_real)9.694002659393303745e-01;
        sr += ur8 * (notorious_fft_real)-9.863613034027224336e-01; si += ui8 * (notorious_fft_real)-9.863613034027224336e-01;
        tr += vi8 * (notorious_fft_real)-1.645945902807337546e-01; ti += vr8 * (notorious_fft_real)-1.645945902807337546e-01;
        sr += ur9 * (notorious_fft_real)5.469481581224269195e-01; si += ui9 * (notorious_fft_real)5.469481581224269195e-01;
        tr += vi9 * (notorious_fft_real)-8.371664782625285461e-01; ti += vr9 * (notorious_fft_real)-8.371664782625285461e-01;
        x[6*st]         = xr0 + sr - sg * tr;
        x[6*st + 1]     = xi0 + si + sg * ti;
        x[13*st]       = xr0 + sr + sg * tr;
        x[13*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-6.772815716257408747e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-6.772815716257408747e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.357239106731318090e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.357239106731318090e-01;
        sr += ur2 * (notorious_fft_real)-8.257934547233274081e-02; si += ui2 * (notorious_fft_real)-8.257934547233274081e-02;
        tr += vi2 * (notorious_fft_real)-9.965844930066698470e-01; ti += vr2 * (notorious_fft_real)-9.965844930066698470e-01;
        sr += ur3 * (notorious_fft_real)7.891405093963936856e-01; si += ui3 * (notorious_fft_real)7.891405093963936856e-01;
        tr += vi3 * (notorious_fft_real)6.142127126896677058e-01; ti += vr3 * (notorious_fft_real)6.142127126896677058e-01;
        sr += ur4 * (notorious_fft_real)-9.863613034027222115e-01; si += ui4 * (notorious_fft_real)-9.863613034027222115e-01;
        tr += vi4 * (notorious_fft_real)1.645945902807346983e-01; ti += vr4 * (notorious_fft_real)1.645945902807346983e-01;
        sr += ur5 * (notorious_fft_real)5.469481581224255873e-01; si += ui5 * (notorious_fft_real)5.469481581224255873e-01;
        tr += vi5 * (notorious_fft_real)-8.371664782625294343e-01; ti += vr5 * (notorious_fft_real)-8.371664782625294343e-01;
        sr += ur6 * (notorious_fft_real)2.454854871407994854e-01; si += ui6 * (notorious_fft_real)2.454854871407994854e-01;
        tr += vi6 * (notorious_fft_real)9.694002659393303745e-01; ti += vr6 * (notorious_fft_real)9.694002659393303745e-01;
        sr += ur7 * (notorious_fft_real)-8.794737512064895180e-01; si += ui7 * (notorious_fft_real)-8.794737512064895180e-01;
        tr += vi7 * (notorious_fft_real)-4.759473930370726746e-01; ti += vr7 * (notorious_fft_real)-4.759473930370726746e-01;
        sr += ur8 * (notorious_fft_real)9.458172417006341925e-01; si += ui8 * (notorious_fft_real)9.458172417006341925e-01;
        tr += vi8 * (notorious_fft_real)-3.246994692046850650e-01; ti += vr8 * (notorious_fft_real)-3.246994692046850650e-01;
        sr += ur9 * (notorious_fft_real)-4.016954246529705896e-01; si += ui9 * (notorious_fft_real)-4.016954246529705896e-01;
        tr += vi9 * (notorious_fft_real)9.157733266550569518e-01; ti += vr9 * (notorious_fft_real)9.157733266550569518e-01;
        x[7*st]         = xr0 + sr - sg * tr;
        x[7*st + 1]     = xi0 + si + sg * ti;
        x[12*st]       = xr0 + sr + sg * tr;
        x[12*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-8.794737512064889629e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-8.794737512064889629e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)4.759473930370736738e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)4.759473930370736738e-01;
        sr += ur2 * (notorious_fft_real)5.469481581224265865e-01; si += ui2 * (notorious_fft_real)5.469481581224265865e-01;
        tr += vi2 * (notorious_fft_real)-8.371664782625287682e-01; ti += vr2 * (notorious_fft_real)-8.371664782625287682e-01;
        sr += ur3 * (notorious_fft_real)-8.257934547233225508e-02; si += ui3 * (notorious_fft_real)-8.257934547233225508e-02;
        tr += vi3 * (notorious_fft_real)9.965844930066698470e-01; ti += vr3 * (notorious_fft_real)9.965844930066698470e-01;
        sr += ur4 * (notorious_fft_real)-4.016954246529700900e-01; si += ui4 * (notorious_fft_real)-4.016954246529700900e-01;
        tr += vi4 * (notorious_fft_real)-9.157733266550571738e-01; ti += vr4 * (notorious_fft_real)-9.157733266550571738e-01;
        sr += ur5 * (notorious_fft_real)7.891405093963939077e-01; si += ui5 * (notorious_fft_real)7.891405093963939077e-01;
        tr += vi5 * (notorious_fft_real)6.142127126896674838e-01; ti += vr5 * (notorious_fft_real)6.142127126896674838e-01;
        sr += ur6 * (notorious_fft_real)-9.863613034027224336e-01; si += ui6 * (notorious_fft_real)-9.863613034027224336e-01;
        tr += vi6 * (notorious_fft_real)-1.645945902807337546e-01; ti += vr6 * (notorious_fft_real)-1.645945902807337546e-01;
        sr += ur7 * (notorious_fft_real)9.458172417006341925e-01; si += ui7 * (notorious_fft_real)9.458172417006341925e-01;
        tr += vi7 * (notorious_fft_real)-3.246994692046850650e-01; ti += vr7 * (notorious_fft_real)-3.246994692046850650e-01;
        sr += ur8 * (notorious_fft_real)-6.772815716257400975e-01; si += ui8 * (notorious_fft_real)-6.772815716257400975e-01;
        tr += vi8 * (notorious_fft_real)7.357239106731325862e-01; ti += vr8 * (notorious_fft_real)7.357239106731325862e-01;
        sr += ur9 * (notorious_fft_real)2.454854871407980699e-01; si += ui9 * (notorious_fft_real)2.454854871407980699e-01;
        tr += vi9 * (notorious_fft_real)-9.694002659393307075e-01; ti += vr9 * (notorious_fft_real)-9.694002659393307075e-01;
        x[8*st]         = xr0 + sr - sg * tr;
        x[8*st + 1]     = xi0 + si + sg * ti;
        x[11*st]       = xr0 + sr + sg * tr;
        x[11*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.863613034027223225e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.863613034027223225e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)1.645945902807340322e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)1.645945902807340322e-01;
        sr += ur2 * (notorious_fft_real)9.458172417006346366e-01; si += ui2 * (notorious_fft_real)9.458172417006346366e-01;
        tr += vi2 * (notorious_fft_real)-3.246994692046837327e-01; ti += vr2 * (notorious_fft_real)-3.246994692046837327e-01;
        sr += ur3 * (notorious_fft_real)-8.794737512064890739e-01; si += ui3 * (notorious_fft_real)-8.794737512064890739e-01;
        tr += vi3 * (notorious_fft_real)4.759473930370735073e-01; ti += vr3 * (notorious_fft_real)4.759473930370735073e-01;
        sr += ur4 * (notorious_fft_real)7.891405093963932416e-01; si += ui4 * (notorious_fft_real)7.891405093963932416e-01;
        tr += vi4 * (notorious_fft_real)-6.142127126896682610e-01; ti += vr4 * (notorious_fft_real)-6.142127126896682610e-01;
        sr += ur5 * (notorious_fft_real)-6.772815716257415408e-01; si += ui5 * (notorious_fft_real)-6.772815716257415408e-01;
        tr += vi5 * (notorious_fft_real)7.357239106731311429e-01; ti += vr5 * (notorious_fft_real)7.357239106731311429e-01;
        sr += ur6 * (notorious_fft_real)5.469481581224269195e-01; si += ui6 * (notorious_fft_real)5.469481581224269195e-01;
        tr += vi6 * (notorious_fft_real)-8.371664782625285461e-01; ti += vr6 * (notorious_fft_real)-8.371664782625285461e-01;
        sr += ur7 * (notorious_fft_real)-4.016954246529705896e-01; si += ui7 * (notorious_fft_real)-4.016954246529705896e-01;
        tr += vi7 * (notorious_fft_real)9.157733266550569518e-01; ti += vr7 * (notorious_fft_real)9.157733266550569518e-01;
        sr += ur8 * (notorious_fft_real)2.454854871407980699e-01; si += ui8 * (notorious_fft_real)2.454854871407980699e-01;
        tr += vi8 * (notorious_fft_real)-9.694002659393307075e-01; ti += vr8 * (notorious_fft_real)-9.694002659393307075e-01;
        sr += ur9 * (notorious_fft_real)-8.257934547232885503e-02; si += ui9 * (notorious_fft_real)-8.257934547232885503e-02;
        tr += vi9 * (notorious_fft_real)9.965844930066701801e-01; ti += vr9 * (notorious_fft_real)9.965844930066701801e-01;
        x[9*st]         = xr0 + sr - sg * tr;
        x[9*st + 1]     = xi0 + si + sg * ti;
        x[10*st]       = xr0 + sr + sg * tr;
        x[10*st + 1]   = xi0 + si - sg * ti;
    }
}

static void notorious_fft_fft17_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real xr7 = x[7*st], xi7 = x[7*st + 1];
    notorious_fft_real xr8 = x[8*st], xi8 = x[8*st + 1];
    notorious_fft_real xr9 = x[9*st], xi9 = x[9*st + 1];
    notorious_fft_real xr10 = x[10*st], xi10 = x[10*st + 1];
    notorious_fft_real xr11 = x[11*st], xi11 = x[11*st + 1];
    notorious_fft_real xr12 = x[12*st], xi12 = x[12*st + 1];
    notorious_fft_real xr13 = x[13*st], xi13 = x[13*st + 1];
    notorious_fft_real xr14 = x[14*st], xi14 = x[14*st + 1];
    notorious_fft_real xr15 = x[15*st], xi15 = x[15*st + 1];
    notorious_fft_real xr16 = x[16*st], xi16 = x[16*st + 1];
    notorious_fft_real ur1 = xr1 + xr16, ui1 = xi1 + xi16;
    notorious_fft_real vr1 = xr1 - xr16, vi1 = xi1 - xi16;
    notorious_fft_real ur2 = xr2 + xr15, ui2 = xi2 + xi15;
    notorious_fft_real vr2 = xr2 - xr15, vi2 = xi2 - xi15;
    notorious_fft_real ur3 = xr3 + xr14, ui3 = xi3 + xi14;
    notorious_fft_real vr3 = xr3 - xr14, vi3 = xi3 - xi14;
    notorious_fft_real ur4 = xr4 + xr13, ui4 = xi4 + xi13;
    notorious_fft_real vr4 = xr4 - xr13, vi4 = xi4 - xi13;
    notorious_fft_real ur5 = xr5 + xr12, ui5 = xi5 + xi12;
    notorious_fft_real vr5 = xr5 - xr12, vi5 = xi5 - xi12;
    notorious_fft_real ur6 = xr6 + xr11, ui6 = xi6 + xi11;
    notorious_fft_real vr6 = xr6 - xr11, vi6 = xi6 - xi11;
    notorious_fft_real ur7 = xr7 + xr10, ui7 = xi7 + xi10;
    notorious_fft_real vr7 = xr7 - xr10, vi7 = xi7 - xi10;
    notorious_fft_real ur8 = xr8 + xr9, ui8 = xi8 + xi9;
    notorious_fft_real vr8 = xr8 - xr9, vi8 = xi8 - xi9;
    x[0] = xr0 + ur1 + ur2 + ur3 + ur4 + ur5 + ur6 + ur7 + ur8;
    x[1] = xi0 + ui1 + ui2 + ui3 + ui4 + ui5 + ui6 + ui7 + ui8;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.324722294043558124e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.324722294043558124e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)3.612416661871529211e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)3.612416661871529211e-01;
        sr += ur2 * (notorious_fft_real)7.390089172206590895e-01; si += ui2 * (notorious_fft_real)7.390089172206590895e-01;
        tr += vi2 * (notorious_fft_real)6.736956436465572073e-01; ti += vr2 * (notorious_fft_real)6.736956436465572073e-01;
        sr += ur3 * (notorious_fft_real)4.457383557765383086e-01; si += ui3 * (notorious_fft_real)4.457383557765383086e-01;
        tr += vi3 * (notorious_fft_real)8.951632913550623405e-01; ti += vr3 * (notorious_fft_real)8.951632913550623405e-01;
        sr += ur4 * (notorious_fft_real)9.226835946330201599e-02; si += ui4 * (notorious_fft_real)9.226835946330201599e-02;
        tr += vi4 * (notorious_fft_real)9.957341762950344677e-01; ti += vr4 * (notorious_fft_real)9.957341762950344677e-01;
        sr += ur5 * (notorious_fft_real)-2.736629900720828901e-01; si += ui5 * (notorious_fft_real)-2.736629900720828901e-01;
        tr += vi5 * (notorious_fft_real)9.618256431728190448e-01; ti += vr5 * (notorious_fft_real)9.618256431728190448e-01;
        sr += ur6 * (notorious_fft_real)-6.026346363792562943e-01; si += ui6 * (notorious_fft_real)-6.026346363792562943e-01;
        tr += vi6 * (notorious_fft_real)7.980172272802396050e-01; ti += vr6 * (notorious_fft_real)7.980172272802396050e-01;
        sr += ur7 * (notorious_fft_real)-8.502171357296139931e-01; si += ui7 * (notorious_fft_real)-8.502171357296139931e-01;
        tr += vi7 * (notorious_fft_real)5.264321628773560580e-01; ti += vr7 * (notorious_fft_real)5.264321628773560580e-01;
        sr += ur8 * (notorious_fft_real)-9.829730996839017854e-01; si += ui8 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi8 * (notorious_fft_real)1.837495178165703680e-01; ti += vr8 * (notorious_fft_real)1.837495178165703680e-01;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[16*st]       = xr0 + sr + sg * tr;
        x[16*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)7.390089172206590895e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)7.390089172206590895e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)6.736956436465572073e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)6.736956436465572073e-01;
        sr += ur2 * (notorious_fft_real)9.226835946330201599e-02; si += ui2 * (notorious_fft_real)9.226835946330201599e-02;
        tr += vi2 * (notorious_fft_real)9.957341762950344677e-01; ti += vr2 * (notorious_fft_real)9.957341762950344677e-01;
        sr += ur3 * (notorious_fft_real)-6.026346363792562943e-01; si += ui3 * (notorious_fft_real)-6.026346363792562943e-01;
        tr += vi3 * (notorious_fft_real)7.980172272802396050e-01; ti += vr3 * (notorious_fft_real)7.980172272802396050e-01;
        sr += ur4 * (notorious_fft_real)-9.829730996839017854e-01; si += ui4 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi4 * (notorious_fft_real)1.837495178165703680e-01; ti += vr4 * (notorious_fft_real)1.837495178165703680e-01;
        sr += ur5 * (notorious_fft_real)-8.502171357296141041e-01; si += ui5 * (notorious_fft_real)-8.502171357296141041e-01;
        tr += vi5 * (notorious_fft_real)-5.264321628773558359e-01; ti += vr5 * (notorious_fft_real)-5.264321628773558359e-01;
        sr += ur6 * (notorious_fft_real)-2.736629900720831121e-01; si += ui6 * (notorious_fft_real)-2.736629900720831121e-01;
        tr += vi6 * (notorious_fft_real)-9.618256431728190448e-01; ti += vr6 * (notorious_fft_real)-9.618256431728190448e-01;
        sr += ur7 * (notorious_fft_real)4.457383557765376980e-01; si += ui7 * (notorious_fft_real)4.457383557765376980e-01;
        tr += vi7 * (notorious_fft_real)-8.951632913550625625e-01; ti += vr7 * (notorious_fft_real)-8.951632913550625625e-01;
        sr += ur8 * (notorious_fft_real)9.324722294043558124e-01; si += ui8 * (notorious_fft_real)9.324722294043558124e-01;
        tr += vi8 * (notorious_fft_real)-3.612416661871530321e-01; ti += vr8 * (notorious_fft_real)-3.612416661871530321e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[15*st]       = xr0 + sr + sg * tr;
        x[15*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)4.457383557765383086e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)4.457383557765383086e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.951632913550623405e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.951632913550623405e-01;
        sr += ur2 * (notorious_fft_real)-6.026346363792562943e-01; si += ui2 * (notorious_fft_real)-6.026346363792562943e-01;
        tr += vi2 * (notorious_fft_real)7.980172272802396050e-01; ti += vr2 * (notorious_fft_real)7.980172272802396050e-01;
        sr += ur3 * (notorious_fft_real)-9.829730996839017854e-01; si += ui3 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi3 * (notorious_fft_real)-1.837495178165701182e-01; ti += vr3 * (notorious_fft_real)-1.837495178165701182e-01;
        sr += ur4 * (notorious_fft_real)-2.736629900720831121e-01; si += ui4 * (notorious_fft_real)-2.736629900720831121e-01;
        tr += vi4 * (notorious_fft_real)-9.618256431728190448e-01; ti += vr4 * (notorious_fft_real)-9.618256431728190448e-01;
        sr += ur5 * (notorious_fft_real)7.390089172206585344e-01; si += ui5 * (notorious_fft_real)7.390089172206585344e-01;
        tr += vi5 * (notorious_fft_real)-6.736956436465577625e-01; ti += vr5 * (notorious_fft_real)-6.736956436465577625e-01;
        sr += ur6 * (notorious_fft_real)9.324722294043559234e-01; si += ui6 * (notorious_fft_real)9.324722294043559234e-01;
        tr += vi6 * (notorious_fft_real)3.612416661871525880e-01; ti += vr6 * (notorious_fft_real)3.612416661871525880e-01;
        sr += ur7 * (notorious_fft_real)9.226835946330202987e-02; si += ui7 * (notorious_fft_real)9.226835946330202987e-02;
        tr += vi7 * (notorious_fft_real)9.957341762950344677e-01; ti += vr7 * (notorious_fft_real)9.957341762950344677e-01;
        sr += ur8 * (notorious_fft_real)-8.502171357296138821e-01; si += ui8 * (notorious_fft_real)-8.502171357296138821e-01;
        tr += vi8 * (notorious_fft_real)5.264321628773562800e-01; ti += vr8 * (notorious_fft_real)5.264321628773562800e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[14*st]       = xr0 + sr + sg * tr;
        x[14*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.226835946330201599e-02;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.226835946330201599e-02;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.957341762950344677e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.957341762950344677e-01;
        sr += ur2 * (notorious_fft_real)-9.829730996839017854e-01; si += ui2 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi2 * (notorious_fft_real)1.837495178165703680e-01; ti += vr2 * (notorious_fft_real)1.837495178165703680e-01;
        sr += ur3 * (notorious_fft_real)-2.736629900720831121e-01; si += ui3 * (notorious_fft_real)-2.736629900720831121e-01;
        tr += vi3 * (notorious_fft_real)-9.618256431728190448e-01; ti += vr3 * (notorious_fft_real)-9.618256431728190448e-01;
        sr += ur4 * (notorious_fft_real)9.324722294043558124e-01; si += ui4 * (notorious_fft_real)9.324722294043558124e-01;
        tr += vi4 * (notorious_fft_real)-3.612416661871530321e-01; ti += vr4 * (notorious_fft_real)-3.612416661871530321e-01;
        sr += ur5 * (notorious_fft_real)4.457383557765381421e-01; si += ui5 * (notorious_fft_real)4.457383557765381421e-01;
        tr += vi5 * (notorious_fft_real)8.951632913550623405e-01; ti += vr5 * (notorious_fft_real)8.951632913550623405e-01;
        sr += ur6 * (notorious_fft_real)-8.502171357296138821e-01; si += ui6 * (notorious_fft_real)-8.502171357296138821e-01;
        tr += vi6 * (notorious_fft_real)5.264321628773562800e-01; ti += vr6 * (notorious_fft_real)5.264321628773562800e-01;
        sr += ur7 * (notorious_fft_real)-6.026346363792574046e-01; si += ui7 * (notorious_fft_real)-6.026346363792574046e-01;
        tr += vi7 * (notorious_fft_real)-7.980172272802387168e-01; ti += vr7 * (notorious_fft_real)-7.980172272802387168e-01;
        sr += ur8 * (notorious_fft_real)7.390089172206589785e-01; si += ui8 * (notorious_fft_real)7.390089172206589785e-01;
        tr += vi8 * (notorious_fft_real)-6.736956436465573184e-01; ti += vr8 * (notorious_fft_real)-6.736956436465573184e-01;
        x[4*st]         = xr0 + sr - sg * tr;
        x[4*st + 1]     = xi0 + si + sg * ti;
        x[13*st]       = xr0 + sr + sg * tr;
        x[13*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-2.736629900720828901e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-2.736629900720828901e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.618256431728190448e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.618256431728190448e-01;
        sr += ur2 * (notorious_fft_real)-8.502171357296141041e-01; si += ui2 * (notorious_fft_real)-8.502171357296141041e-01;
        tr += vi2 * (notorious_fft_real)-5.264321628773558359e-01; ti += vr2 * (notorious_fft_real)-5.264321628773558359e-01;
        sr += ur3 * (notorious_fft_real)7.390089172206585344e-01; si += ui3 * (notorious_fft_real)7.390089172206585344e-01;
        tr += vi3 * (notorious_fft_real)-6.736956436465577625e-01; ti += vr3 * (notorious_fft_real)-6.736956436465577625e-01;
        sr += ur4 * (notorious_fft_real)4.457383557765381421e-01; si += ui4 * (notorious_fft_real)4.457383557765381421e-01;
        tr += vi4 * (notorious_fft_real)8.951632913550623405e-01; ti += vr4 * (notorious_fft_real)8.951632913550623405e-01;
        sr += ur5 * (notorious_fft_real)-9.829730996839017854e-01; si += ui5 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi5 * (notorious_fft_real)1.837495178165701737e-01; ti += vr5 * (notorious_fft_real)1.837495178165701737e-01;
        sr += ur6 * (notorious_fft_real)9.226835946330042004e-02; si += ui6 * (notorious_fft_real)9.226835946330042004e-02;
        tr += vi6 * (notorious_fft_real)-9.957341762950346897e-01; ti += vr6 * (notorious_fft_real)-9.957341762950346897e-01;
        sr += ur7 * (notorious_fft_real)9.324722294043563675e-01; si += ui7 * (notorious_fft_real)9.324722294043563675e-01;
        tr += vi7 * (notorious_fft_real)3.612416661871514778e-01; ti += vr7 * (notorious_fft_real)3.612416661871514778e-01;
        sr += ur8 * (notorious_fft_real)-6.026346363792566274e-01; si += ui8 * (notorious_fft_real)-6.026346363792566274e-01;
        tr += vi8 * (notorious_fft_real)7.980172272802393829e-01; ti += vr8 * (notorious_fft_real)7.980172272802393829e-01;
        x[5*st]         = xr0 + sr - sg * tr;
        x[5*st + 1]     = xi0 + si + sg * ti;
        x[12*st]       = xr0 + sr + sg * tr;
        x[12*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-6.026346363792562943e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-6.026346363792562943e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.980172272802396050e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.980172272802396050e-01;
        sr += ur2 * (notorious_fft_real)-2.736629900720831121e-01; si += ui2 * (notorious_fft_real)-2.736629900720831121e-01;
        tr += vi2 * (notorious_fft_real)-9.618256431728190448e-01; ti += vr2 * (notorious_fft_real)-9.618256431728190448e-01;
        sr += ur3 * (notorious_fft_real)9.324722294043559234e-01; si += ui3 * (notorious_fft_real)9.324722294043559234e-01;
        tr += vi3 * (notorious_fft_real)3.612416661871525880e-01; ti += vr3 * (notorious_fft_real)3.612416661871525880e-01;
        sr += ur4 * (notorious_fft_real)-8.502171357296138821e-01; si += ui4 * (notorious_fft_real)-8.502171357296138821e-01;
        tr += vi4 * (notorious_fft_real)5.264321628773562800e-01; ti += vr4 * (notorious_fft_real)5.264321628773562800e-01;
        sr += ur5 * (notorious_fft_real)9.226835946330042004e-02; si += ui5 * (notorious_fft_real)9.226835946330042004e-02;
        tr += vi5 * (notorious_fft_real)-9.957341762950346897e-01; ti += vr5 * (notorious_fft_real)-9.957341762950346897e-01;
        sr += ur6 * (notorious_fft_real)7.390089172206596446e-01; si += ui6 * (notorious_fft_real)7.390089172206596446e-01;
        tr += vi6 * (notorious_fft_real)6.736956436465566522e-01; ti += vr6 * (notorious_fft_real)6.736956436465566522e-01;
        sr += ur7 * (notorious_fft_real)-9.829730996839017854e-01; si += ui7 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi7 * (notorious_fft_real)1.837495178165704235e-01; ti += vr7 * (notorious_fft_real)1.837495178165704235e-01;
        sr += ur8 * (notorious_fft_real)4.457383557765372539e-01; si += ui8 * (notorious_fft_real)4.457383557765372539e-01;
        tr += vi8 * (notorious_fft_real)-8.951632913550627846e-01; ti += vr8 * (notorious_fft_real)-8.951632913550627846e-01;
        x[6*st]         = xr0 + sr - sg * tr;
        x[6*st + 1]     = xi0 + si + sg * ti;
        x[11*st]       = xr0 + sr + sg * tr;
        x[11*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-8.502171357296139931e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-8.502171357296139931e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)5.264321628773560580e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)5.264321628773560580e-01;
        sr += ur2 * (notorious_fft_real)4.457383557765376980e-01; si += ui2 * (notorious_fft_real)4.457383557765376980e-01;
        tr += vi2 * (notorious_fft_real)-8.951632913550625625e-01; ti += vr2 * (notorious_fft_real)-8.951632913550625625e-01;
        sr += ur3 * (notorious_fft_real)9.226835946330202987e-02; si += ui3 * (notorious_fft_real)9.226835946330202987e-02;
        tr += vi3 * (notorious_fft_real)9.957341762950344677e-01; ti += vr3 * (notorious_fft_real)9.957341762950344677e-01;
        sr += ur4 * (notorious_fft_real)-6.026346363792574046e-01; si += ui4 * (notorious_fft_real)-6.026346363792574046e-01;
        tr += vi4 * (notorious_fft_real)-7.980172272802387168e-01; ti += vr4 * (notorious_fft_real)-7.980172272802387168e-01;
        sr += ur5 * (notorious_fft_real)9.324722294043563675e-01; si += ui5 * (notorious_fft_real)9.324722294043563675e-01;
        tr += vi5 * (notorious_fft_real)3.612416661871514778e-01; ti += vr5 * (notorious_fft_real)3.612416661871514778e-01;
        sr += ur6 * (notorious_fft_real)-9.829730996839017854e-01; si += ui6 * (notorious_fft_real)-9.829730996839017854e-01;
        tr += vi6 * (notorious_fft_real)1.837495178165704235e-01; ti += vr6 * (notorious_fft_real)1.837495178165704235e-01;
        sr += ur7 * (notorious_fft_real)7.390089172206576462e-01; si += ui7 * (notorious_fft_real)7.390089172206576462e-01;
        tr += vi7 * (notorious_fft_real)-6.736956436465587617e-01; ti += vr7 * (notorious_fft_real)-6.736956436465587617e-01;
        sr += ur8 * (notorious_fft_real)-2.736629900720805031e-01; si += ui8 * (notorious_fft_real)-2.736629900720805031e-01;
        tr += vi8 * (notorious_fft_real)9.618256431728197109e-01; ti += vr8 * (notorious_fft_real)9.618256431728197109e-01;
        x[7*st]         = xr0 + sr - sg * tr;
        x[7*st + 1]     = xi0 + si + sg * ti;
        x[10*st]       = xr0 + sr + sg * tr;
        x[10*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.829730996839017854e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.829730996839017854e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)1.837495178165703680e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)1.837495178165703680e-01;
        sr += ur2 * (notorious_fft_real)9.324722294043558124e-01; si += ui2 * (notorious_fft_real)9.324722294043558124e-01;
        tr += vi2 * (notorious_fft_real)-3.612416661871530321e-01; ti += vr2 * (notorious_fft_real)-3.612416661871530321e-01;
        sr += ur3 * (notorious_fft_real)-8.502171357296138821e-01; si += ui3 * (notorious_fft_real)-8.502171357296138821e-01;
        tr += vi3 * (notorious_fft_real)5.264321628773562800e-01; ti += vr3 * (notorious_fft_real)5.264321628773562800e-01;
        sr += ur4 * (notorious_fft_real)7.390089172206589785e-01; si += ui4 * (notorious_fft_real)7.390089172206589785e-01;
        tr += vi4 * (notorious_fft_real)-6.736956436465573184e-01; ti += vr4 * (notorious_fft_real)-6.736956436465573184e-01;
        sr += ur5 * (notorious_fft_real)-6.026346363792566274e-01; si += ui5 * (notorious_fft_real)-6.026346363792566274e-01;
        tr += vi5 * (notorious_fft_real)7.980172272802393829e-01; ti += vr5 * (notorious_fft_real)7.980172272802393829e-01;
        sr += ur6 * (notorious_fft_real)4.457383557765372539e-01; si += ui6 * (notorious_fft_real)4.457383557765372539e-01;
        tr += vi6 * (notorious_fft_real)-8.951632913550627846e-01; ti += vr6 * (notorious_fft_real)-8.951632913550627846e-01;
        sr += ur7 * (notorious_fft_real)-2.736629900720805031e-01; si += ui7 * (notorious_fft_real)-2.736629900720805031e-01;
        tr += vi7 * (notorious_fft_real)9.618256431728197109e-01; ti += vr7 * (notorious_fft_real)9.618256431728197109e-01;
        sr += ur8 * (notorious_fft_real)9.226835946330169680e-02; si += ui8 * (notorious_fft_real)9.226835946330169680e-02;
        tr += vi8 * (notorious_fft_real)-9.957341762950345787e-01; ti += vr8 * (notorious_fft_real)-9.957341762950345787e-01;
        x[8*st]         = xr0 + sr - sg * tr;
        x[8*st + 1]     = xi0 + si + sg * ti;
        x[9*st]       = xr0 + sr + sg * tr;
        x[9*st + 1]   = xi0 + si - sg * ti;
    }
}

static void notorious_fft_fft43_stride(notorious_fft_real* x, int s, int inverse)
{
    const int st = 2 * s;
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    notorious_fft_real xr0 = x[0*st], xi0 = x[0*st + 1];
    notorious_fft_real xr1 = x[1*st], xi1 = x[1*st + 1];
    notorious_fft_real xr2 = x[2*st], xi2 = x[2*st + 1];
    notorious_fft_real xr3 = x[3*st], xi3 = x[3*st + 1];
    notorious_fft_real xr4 = x[4*st], xi4 = x[4*st + 1];
    notorious_fft_real xr5 = x[5*st], xi5 = x[5*st + 1];
    notorious_fft_real xr6 = x[6*st], xi6 = x[6*st + 1];
    notorious_fft_real xr7 = x[7*st], xi7 = x[7*st + 1];
    notorious_fft_real xr8 = x[8*st], xi8 = x[8*st + 1];
    notorious_fft_real xr9 = x[9*st], xi9 = x[9*st + 1];
    notorious_fft_real xr10 = x[10*st], xi10 = x[10*st + 1];
    notorious_fft_real xr11 = x[11*st], xi11 = x[11*st + 1];
    notorious_fft_real xr12 = x[12*st], xi12 = x[12*st + 1];
    notorious_fft_real xr13 = x[13*st], xi13 = x[13*st + 1];
    notorious_fft_real xr14 = x[14*st], xi14 = x[14*st + 1];
    notorious_fft_real xr15 = x[15*st], xi15 = x[15*st + 1];
    notorious_fft_real xr16 = x[16*st], xi16 = x[16*st + 1];
    notorious_fft_real xr17 = x[17*st], xi17 = x[17*st + 1];
    notorious_fft_real xr18 = x[18*st], xi18 = x[18*st + 1];
    notorious_fft_real xr19 = x[19*st], xi19 = x[19*st + 1];
    notorious_fft_real xr20 = x[20*st], xi20 = x[20*st + 1];
    notorious_fft_real xr21 = x[21*st], xi21 = x[21*st + 1];
    notorious_fft_real xr22 = x[22*st], xi22 = x[22*st + 1];
    notorious_fft_real xr23 = x[23*st], xi23 = x[23*st + 1];
    notorious_fft_real xr24 = x[24*st], xi24 = x[24*st + 1];
    notorious_fft_real xr25 = x[25*st], xi25 = x[25*st + 1];
    notorious_fft_real xr26 = x[26*st], xi26 = x[26*st + 1];
    notorious_fft_real xr27 = x[27*st], xi27 = x[27*st + 1];
    notorious_fft_real xr28 = x[28*st], xi28 = x[28*st + 1];
    notorious_fft_real xr29 = x[29*st], xi29 = x[29*st + 1];
    notorious_fft_real xr30 = x[30*st], xi30 = x[30*st + 1];
    notorious_fft_real xr31 = x[31*st], xi31 = x[31*st + 1];
    notorious_fft_real xr32 = x[32*st], xi32 = x[32*st + 1];
    notorious_fft_real xr33 = x[33*st], xi33 = x[33*st + 1];
    notorious_fft_real xr34 = x[34*st], xi34 = x[34*st + 1];
    notorious_fft_real xr35 = x[35*st], xi35 = x[35*st + 1];
    notorious_fft_real xr36 = x[36*st], xi36 = x[36*st + 1];
    notorious_fft_real xr37 = x[37*st], xi37 = x[37*st + 1];
    notorious_fft_real xr38 = x[38*st], xi38 = x[38*st + 1];
    notorious_fft_real xr39 = x[39*st], xi39 = x[39*st + 1];
    notorious_fft_real xr40 = x[40*st], xi40 = x[40*st + 1];
    notorious_fft_real xr41 = x[41*st], xi41 = x[41*st + 1];
    notorious_fft_real xr42 = x[42*st], xi42 = x[42*st + 1];
    notorious_fft_real ur1 = xr1 + xr42, ui1 = xi1 + xi42;
    notorious_fft_real vr1 = xr1 - xr42, vi1 = xi1 - xi42;
    notorious_fft_real ur2 = xr2 + xr41, ui2 = xi2 + xi41;
    notorious_fft_real vr2 = xr2 - xr41, vi2 = xi2 - xi41;
    notorious_fft_real ur3 = xr3 + xr40, ui3 = xi3 + xi40;
    notorious_fft_real vr3 = xr3 - xr40, vi3 = xi3 - xi40;
    notorious_fft_real ur4 = xr4 + xr39, ui4 = xi4 + xi39;
    notorious_fft_real vr4 = xr4 - xr39, vi4 = xi4 - xi39;
    notorious_fft_real ur5 = xr5 + xr38, ui5 = xi5 + xi38;
    notorious_fft_real vr5 = xr5 - xr38, vi5 = xi5 - xi38;
    notorious_fft_real ur6 = xr6 + xr37, ui6 = xi6 + xi37;
    notorious_fft_real vr6 = xr6 - xr37, vi6 = xi6 - xi37;
    notorious_fft_real ur7 = xr7 + xr36, ui7 = xi7 + xi36;
    notorious_fft_real vr7 = xr7 - xr36, vi7 = xi7 - xi36;
    notorious_fft_real ur8 = xr8 + xr35, ui8 = xi8 + xi35;
    notorious_fft_real vr8 = xr8 - xr35, vi8 = xi8 - xi35;
    notorious_fft_real ur9 = xr9 + xr34, ui9 = xi9 + xi34;
    notorious_fft_real vr9 = xr9 - xr34, vi9 = xi9 - xi34;
    notorious_fft_real ur10 = xr10 + xr33, ui10 = xi10 + xi33;
    notorious_fft_real vr10 = xr10 - xr33, vi10 = xi10 - xi33;
    notorious_fft_real ur11 = xr11 + xr32, ui11 = xi11 + xi32;
    notorious_fft_real vr11 = xr11 - xr32, vi11 = xi11 - xi32;
    notorious_fft_real ur12 = xr12 + xr31, ui12 = xi12 + xi31;
    notorious_fft_real vr12 = xr12 - xr31, vi12 = xi12 - xi31;
    notorious_fft_real ur13 = xr13 + xr30, ui13 = xi13 + xi30;
    notorious_fft_real vr13 = xr13 - xr30, vi13 = xi13 - xi30;
    notorious_fft_real ur14 = xr14 + xr29, ui14 = xi14 + xi29;
    notorious_fft_real vr14 = xr14 - xr29, vi14 = xi14 - xi29;
    notorious_fft_real ur15 = xr15 + xr28, ui15 = xi15 + xi28;
    notorious_fft_real vr15 = xr15 - xr28, vi15 = xi15 - xi28;
    notorious_fft_real ur16 = xr16 + xr27, ui16 = xi16 + xi27;
    notorious_fft_real vr16 = xr16 - xr27, vi16 = xi16 - xi27;
    notorious_fft_real ur17 = xr17 + xr26, ui17 = xi17 + xi26;
    notorious_fft_real vr17 = xr17 - xr26, vi17 = xi17 - xi26;
    notorious_fft_real ur18 = xr18 + xr25, ui18 = xi18 + xi25;
    notorious_fft_real vr18 = xr18 - xr25, vi18 = xi18 - xi25;
    notorious_fft_real ur19 = xr19 + xr24, ui19 = xi19 + xi24;
    notorious_fft_real vr19 = xr19 - xr24, vi19 = xi19 - xi24;
    notorious_fft_real ur20 = xr20 + xr23, ui20 = xi20 + xi23;
    notorious_fft_real vr20 = xr20 - xr23, vi20 = xi20 - xi23;
    notorious_fft_real ur21 = xr21 + xr22, ui21 = xi21 + xi22;
    notorious_fft_real vr21 = xr21 - xr22, vi21 = xi21 - xi22;
    x[0] = xr0 + ur1 + ur2 + ur3 + ur4 + ur5 + ur6 + ur7 + ur8 + ur9 + ur10 + ur11 + ur12 + ur13 + ur14 + ur15 + ur16 + ur17 + ur18 + ur19 + ur20 + ur21;
    x[1] = xi0 + ui1 + ui2 + ui3 + ui4 + ui5 + ui6 + ui7 + ui8 + ui9 + ui10 + ui11 + ui12 + ui13 + ui14 + ui15 + ui16 + ui17 + ui18 + ui19 + ui20 + ui21;
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.893433680751102521e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.893433680751102521e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)1.456011677350048661e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)1.456011677350048661e-01;
        sr += ur2 * (notorious_fft_real)9.576005999084059583e-01; si += ui2 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi2 * (notorious_fft_real)2.880990993652375876e-01; ti += vr2 * (notorious_fft_real)2.880990993652375876e-01;
        sr += ur3 * (notorious_fft_real)9.054482374931466238e-01; si += ui3 * (notorious_fft_real)9.054482374931466238e-01;
        tr += vi3 * (notorious_fft_real)4.244566988758150528e-01; ti += vr3 * (notorious_fft_real)4.244566988758150528e-01;
        sr += ur4 * (notorious_fft_real)8.339978178898779237e-01; si += ui4 * (notorious_fft_real)8.339978178898779237e-01;
        tr += vi4 * (notorious_fft_real)5.517677407704458803e-01; ti += vr4 * (notorious_fft_real)5.517677407704458803e-01;
        sr += ur5 * (notorious_fft_real)7.447721827437818742e-01; si += ui5 * (notorious_fft_real)7.447721827437818742e-01;
        tr += vi5 * (notorious_fft_real)6.673188112222393498e-01; ti += vr5 * (notorious_fft_real)6.673188112222393498e-01;
        sr += ur6 * (notorious_fft_real)6.396730215588912749e-01; si += ui6 * (notorious_fft_real)6.396730215588912749e-01;
        tr += vi6 * (notorious_fft_real)7.686471397785320514e-01; ti += vr6 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur7 * (notorious_fft_real)5.209403404879302535e-01; si += ui7 * (notorious_fft_real)5.209403404879302535e-01;
        tr += vi7 * (notorious_fft_real)8.535930890373464308e-01; ti += vr7 * (notorious_fft_real)8.535930890373464308e-01;
        sr += ur8 * (notorious_fft_real)3.911047204901559882e-01; si += ui8 * (notorious_fft_real)3.911047204901559882e-01;
        tr += vi8 * (notorious_fft_real)9.203461835691594262e-01; ti += vr8 * (notorious_fft_real)9.203461835691594262e-01;
        sr += ur9 * (notorious_fft_real)2.529333823916806723e-01; si += ui9 * (notorious_fft_real)2.529333823916806723e-01;
        tr += vi9 * (notorious_fft_real)9.674836970574253181e-01; ti += vr9 * (notorious_fft_real)9.674836970574253181e-01;
        sr += ur10 * (notorious_fft_real)1.093712083778745209e-01; si += ui10 * (notorious_fft_real)1.093712083778745209e-01;
        tr += vi10 * (notorious_fft_real)9.940009752399459053e-01; ti += vr10 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur11 * (notorious_fft_real)-3.652202305765850365e-02; si += ui11 * (notorious_fft_real)-3.652202305765850365e-02;
        tr += vi11 * (notorious_fft_real)9.993328483702393905e-01; ti += vr11 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur12 * (notorious_fft_real)-1.816368509794363539e-01; si += ui12 * (notorious_fft_real)-1.816368509794363539e-01;
        tr += vi12 * (notorious_fft_real)9.833656768294661843e-01; ti += vr12 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur13 * (notorious_fft_real)-3.228804047714463610e-01; si += ui13 * (notorious_fft_real)-3.228804047714463610e-01;
        tr += vi13 * (notorious_fft_real)9.464397731576092987e-01; ti += vr13 * (notorious_fft_real)9.464397731576092987e-01;
        sr += ur14 * (notorious_fft_real)-4.572423233046384872e-01; si += ui14 * (notorious_fft_real)-4.572423233046384872e-01;
        tr += vi14 * (notorious_fft_real)8.893421488825189458e-01; ti += vr14 * (notorious_fft_real)8.893421488825189458e-01;
        sr += ur15 * (notorious_fft_real)-5.818589155579526606e-01; si += ui15 * (notorious_fft_real)-5.818589155579526606e-01;
        tr += vi15 * (notorious_fft_real)8.132897407355654806e-01; ti += vr15 * (notorious_fft_real)8.132897407355654806e-01;
        sr += ur16 * (notorious_fft_real)-6.940741952206339604e-01; si += ui16 * (notorious_fft_real)-6.940741952206339604e-01;
        tr += vi16 * (notorious_fft_real)7.199034737579957932e-01; ti += vr16 * (notorious_fft_real)7.199034737579957932e-01;
        sr += ur17 * (notorious_fft_real)-7.914964884292540193e-01; si += ui17 * (notorious_fft_real)-7.914964884292540193e-01;
        tr += vi17 * (notorious_fft_real)6.111737140978493743e-01; ti += vr17 * (notorious_fft_real)6.111737140978493743e-01;
        sr += ur18 * (notorious_fft_real)-8.720494081438077139e-01; si += ui18 * (notorious_fft_real)-8.720494081438077139e-01;
        tr += vi18 * (notorious_fft_real)4.894178478110854380e-01; ti += vr18 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur19 * (notorious_fft_real)-9.340161087325479228e-01; si += ui19 * (notorious_fft_real)-9.340161087325479228e-01;
        tr += vi19 * (notorious_fft_real)3.572308898011328782e-01; ti += vr19 * (notorious_fft_real)3.572308898011328782e-01;
        sr += ur20 * (notorious_fft_real)-9.760758775559270628e-01; si += ui20 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi20 * (notorious_fft_real)2.174301755815572501e-01; ti += vr20 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur21 * (notorious_fft_real)-9.973322836635516442e-01; si += ui21 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi21 * (notorious_fft_real)7.299531466090755649e-02; ti += vr21 * (notorious_fft_real)7.299531466090755649e-02;
        x[1*st]         = xr0 + sr - sg * tr;
        x[1*st + 1]     = xi0 + si + sg * ti;
        x[42*st]       = xr0 + sr + sg * tr;
        x[42*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.576005999084059583e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.576005999084059583e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)2.880990993652375876e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)2.880990993652375876e-01;
        sr += ur2 * (notorious_fft_real)8.339978178898779237e-01; si += ui2 * (notorious_fft_real)8.339978178898779237e-01;
        tr += vi2 * (notorious_fft_real)5.517677407704458803e-01; ti += vr2 * (notorious_fft_real)5.517677407704458803e-01;
        sr += ur3 * (notorious_fft_real)6.396730215588912749e-01; si += ui3 * (notorious_fft_real)6.396730215588912749e-01;
        tr += vi3 * (notorious_fft_real)7.686471397785320514e-01; ti += vr3 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur4 * (notorious_fft_real)3.911047204901559882e-01; si += ui4 * (notorious_fft_real)3.911047204901559882e-01;
        tr += vi4 * (notorious_fft_real)9.203461835691594262e-01; ti += vr4 * (notorious_fft_real)9.203461835691594262e-01;
        sr += ur5 * (notorious_fft_real)1.093712083778745209e-01; si += ui5 * (notorious_fft_real)1.093712083778745209e-01;
        tr += vi5 * (notorious_fft_real)9.940009752399459053e-01; ti += vr5 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur6 * (notorious_fft_real)-1.816368509794363539e-01; si += ui6 * (notorious_fft_real)-1.816368509794363539e-01;
        tr += vi6 * (notorious_fft_real)9.833656768294661843e-01; ti += vr6 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur7 * (notorious_fft_real)-4.572423233046384872e-01; si += ui7 * (notorious_fft_real)-4.572423233046384872e-01;
        tr += vi7 * (notorious_fft_real)8.893421488825189458e-01; ti += vr7 * (notorious_fft_real)8.893421488825189458e-01;
        sr += ur8 * (notorious_fft_real)-6.940741952206339604e-01; si += ui8 * (notorious_fft_real)-6.940741952206339604e-01;
        tr += vi8 * (notorious_fft_real)7.199034737579957932e-01; ti += vr8 * (notorious_fft_real)7.199034737579957932e-01;
        sr += ur9 * (notorious_fft_real)-8.720494081438077139e-01; si += ui9 * (notorious_fft_real)-8.720494081438077139e-01;
        tr += vi9 * (notorious_fft_real)4.894178478110854380e-01; ti += vr9 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur10 * (notorious_fft_real)-9.760758775559270628e-01; si += ui10 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi10 * (notorious_fft_real)2.174301755815572501e-01; ti += vr10 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur11 * (notorious_fft_real)-9.973322836635517552e-01; si += ui11 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi11 * (notorious_fft_real)-7.299531466090686260e-02; ti += vr11 * (notorious_fft_real)-7.299531466090686260e-02;
        sr += ur12 * (notorious_fft_real)-9.340161087325480338e-01; si += ui12 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi12 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr12 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur13 * (notorious_fft_real)-7.914964884292539082e-01; si += ui13 * (notorious_fft_real)-7.914964884292539082e-01;
        tr += vi13 * (notorious_fft_real)-6.111737140978495963e-01; ti += vr13 * (notorious_fft_real)-6.111737140978495963e-01;
        sr += ur14 * (notorious_fft_real)-5.818589155579528827e-01; si += ui14 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi14 * (notorious_fft_real)-8.132897407355653696e-01; ti += vr14 * (notorious_fft_real)-8.132897407355653696e-01;
        sr += ur15 * (notorious_fft_real)-3.228804047714466385e-01; si += ui15 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi15 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr15 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur16 * (notorious_fft_real)-3.652202305765874651e-02; si += ui16 * (notorious_fft_real)-3.652202305765874651e-02;
        tr += vi16 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr16 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur17 * (notorious_fft_real)2.529333823916804502e-01; si += ui17 * (notorious_fft_real)2.529333823916804502e-01;
        tr += vi17 * (notorious_fft_real)-9.674836970574253181e-01; ti += vr17 * (notorious_fft_real)-9.674836970574253181e-01;
        sr += ur18 * (notorious_fft_real)5.209403404879304755e-01; si += ui18 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi18 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr18 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur19 * (notorious_fft_real)7.447721827437817632e-01; si += ui19 * (notorious_fft_real)7.447721827437817632e-01;
        tr += vi19 * (notorious_fft_real)-6.673188112222395718e-01; ti += vr19 * (notorious_fft_real)-6.673188112222395718e-01;
        sr += ur20 * (notorious_fft_real)9.054482374931464017e-01; si += ui20 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi20 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr20 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur21 * (notorious_fft_real)9.893433680751101411e-01; si += ui21 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi21 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr21 * (notorious_fft_real)-1.456011677350049216e-01;
        x[2*st]         = xr0 + sr - sg * tr;
        x[2*st + 1]     = xi0 + si + sg * ti;
        x[41*st]       = xr0 + sr + sg * tr;
        x[41*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)9.054482374931466238e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)9.054482374931466238e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)4.244566988758150528e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)4.244566988758150528e-01;
        sr += ur2 * (notorious_fft_real)6.396730215588912749e-01; si += ui2 * (notorious_fft_real)6.396730215588912749e-01;
        tr += vi2 * (notorious_fft_real)7.686471397785320514e-01; ti += vr2 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur3 * (notorious_fft_real)2.529333823916806723e-01; si += ui3 * (notorious_fft_real)2.529333823916806723e-01;
        tr += vi3 * (notorious_fft_real)9.674836970574253181e-01; ti += vr3 * (notorious_fft_real)9.674836970574253181e-01;
        sr += ur4 * (notorious_fft_real)-1.816368509794363539e-01; si += ui4 * (notorious_fft_real)-1.816368509794363539e-01;
        tr += vi4 * (notorious_fft_real)9.833656768294661843e-01; ti += vr4 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur5 * (notorious_fft_real)-5.818589155579526606e-01; si += ui5 * (notorious_fft_real)-5.818589155579526606e-01;
        tr += vi5 * (notorious_fft_real)8.132897407355654806e-01; ti += vr5 * (notorious_fft_real)8.132897407355654806e-01;
        sr += ur6 * (notorious_fft_real)-8.720494081438077139e-01; si += ui6 * (notorious_fft_real)-8.720494081438077139e-01;
        tr += vi6 * (notorious_fft_real)4.894178478110854380e-01; ti += vr6 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur7 * (notorious_fft_real)-9.973322836635516442e-01; si += ui7 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi7 * (notorious_fft_real)7.299531466090755649e-02; ti += vr7 * (notorious_fft_real)7.299531466090755649e-02;
        sr += ur8 * (notorious_fft_real)-9.340161087325480338e-01; si += ui8 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi8 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr8 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur9 * (notorious_fft_real)-6.940741952206340715e-01; si += ui9 * (notorious_fft_real)-6.940741952206340715e-01;
        tr += vi9 * (notorious_fft_real)-7.199034737579956822e-01; ti += vr9 * (notorious_fft_real)-7.199034737579956822e-01;
        sr += ur10 * (notorious_fft_real)-3.228804047714466385e-01; si += ui10 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi10 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr10 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur11 * (notorious_fft_real)1.093712083778742850e-01; si += ui11 * (notorious_fft_real)1.093712083778742850e-01;
        tr += vi11 * (notorious_fft_real)-9.940009752399459053e-01; ti += vr11 * (notorious_fft_real)-9.940009752399459053e-01;
        sr += ur12 * (notorious_fft_real)5.209403404879304755e-01; si += ui12 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi12 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr12 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur13 * (notorious_fft_real)8.339978178898777017e-01; si += ui13 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi13 * (notorious_fft_real)-5.517677407704462134e-01; ti += vr13 * (notorious_fft_real)-5.517677407704462134e-01;
        sr += ur14 * (notorious_fft_real)9.893433680751101411e-01; si += ui14 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi14 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr14 * (notorious_fft_real)-1.456011677350049216e-01;
        sr += ur15 * (notorious_fft_real)9.576005999084058473e-01; si += ui15 * (notorious_fft_real)9.576005999084058473e-01;
        tr += vi15 * (notorious_fft_real)2.880990993652378096e-01; ti += vr15 * (notorious_fft_real)2.880990993652378096e-01;
        sr += ur16 * (notorious_fft_real)7.447721827437820963e-01; si += ui16 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi16 * (notorious_fft_real)6.673188112222391277e-01; ti += vr16 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur17 * (notorious_fft_real)3.911047204901568208e-01; si += ui17 * (notorious_fft_real)3.911047204901568208e-01;
        tr += vi17 * (notorious_fft_real)9.203461835691590931e-01; ti += vr17 * (notorious_fft_real)9.203461835691590931e-01;
        sr += ur18 * (notorious_fft_real)-3.652202305765826079e-02; si += ui18 * (notorious_fft_real)-3.652202305765826079e-02;
        tr += vi18 * (notorious_fft_real)9.993328483702393905e-01; ti += vr18 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur19 * (notorious_fft_real)-4.572423233046374880e-01; si += ui19 * (notorious_fft_real)-4.572423233046374880e-01;
        tr += vi19 * (notorious_fft_real)8.893421488825193899e-01; ti += vr19 * (notorious_fft_real)8.893421488825193899e-01;
        sr += ur20 * (notorious_fft_real)-7.914964884292535752e-01; si += ui20 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi20 * (notorious_fft_real)6.111737140978499294e-01; ti += vr20 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur21 * (notorious_fft_real)-9.760758775559270628e-01; si += ui21 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi21 * (notorious_fft_real)2.174301755815574722e-01; ti += vr21 * (notorious_fft_real)2.174301755815574722e-01;
        x[3*st]         = xr0 + sr - sg * tr;
        x[3*st + 1]     = xi0 + si + sg * ti;
        x[40*st]       = xr0 + sr + sg * tr;
        x[40*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)8.339978178898779237e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)8.339978178898779237e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)5.517677407704458803e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)5.517677407704458803e-01;
        sr += ur2 * (notorious_fft_real)3.911047204901559882e-01; si += ui2 * (notorious_fft_real)3.911047204901559882e-01;
        tr += vi2 * (notorious_fft_real)9.203461835691594262e-01; ti += vr2 * (notorious_fft_real)9.203461835691594262e-01;
        sr += ur3 * (notorious_fft_real)-1.816368509794363539e-01; si += ui3 * (notorious_fft_real)-1.816368509794363539e-01;
        tr += vi3 * (notorious_fft_real)9.833656768294661843e-01; ti += vr3 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur4 * (notorious_fft_real)-6.940741952206339604e-01; si += ui4 * (notorious_fft_real)-6.940741952206339604e-01;
        tr += vi4 * (notorious_fft_real)7.199034737579957932e-01; ti += vr4 * (notorious_fft_real)7.199034737579957932e-01;
        sr += ur5 * (notorious_fft_real)-9.760758775559270628e-01; si += ui5 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi5 * (notorious_fft_real)2.174301755815572501e-01; ti += vr5 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur6 * (notorious_fft_real)-9.340161087325480338e-01; si += ui6 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi6 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr6 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur7 * (notorious_fft_real)-5.818589155579528827e-01; si += ui7 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi7 * (notorious_fft_real)-8.132897407355653696e-01; ti += vr7 * (notorious_fft_real)-8.132897407355653696e-01;
        sr += ur8 * (notorious_fft_real)-3.652202305765874651e-02; si += ui8 * (notorious_fft_real)-3.652202305765874651e-02;
        tr += vi8 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr8 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur9 * (notorious_fft_real)5.209403404879304755e-01; si += ui9 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi9 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr9 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur10 * (notorious_fft_real)9.054482374931464017e-01; si += ui10 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi10 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr10 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur11 * (notorious_fft_real)9.893433680751103632e-01; si += ui11 * (notorious_fft_real)9.893433680751103632e-01;
        tr += vi11 * (notorious_fft_real)1.456011677350035616e-01; ti += vr11 * (notorious_fft_real)1.456011677350035616e-01;
        sr += ur12 * (notorious_fft_real)7.447721827437820963e-01; si += ui12 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi12 * (notorious_fft_real)6.673188112222391277e-01; ti += vr12 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur13 * (notorious_fft_real)2.529333823916800617e-01; si += ui13 * (notorious_fft_real)2.529333823916800617e-01;
        tr += vi13 * (notorious_fft_real)9.674836970574254291e-01; ti += vr13 * (notorious_fft_real)9.674836970574254291e-01;
        sr += ur14 * (notorious_fft_real)-3.228804047714461389e-01; si += ui14 * (notorious_fft_real)-3.228804047714461389e-01;
        tr += vi14 * (notorious_fft_real)9.464397731576094097e-01; ti += vr14 * (notorious_fft_real)9.464397731576094097e-01;
        sr += ur15 * (notorious_fft_real)-7.914964884292535752e-01; si += ui15 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi15 * (notorious_fft_real)6.111737140978499294e-01; ti += vr15 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur16 * (notorious_fft_real)-9.973322836635516442e-01; si += ui16 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi16 * (notorious_fft_real)7.299531466090734833e-02; ti += vr16 * (notorious_fft_real)7.299531466090734833e-02;
        sr += ur17 * (notorious_fft_real)-8.720494081438079359e-01; si += ui17 * (notorious_fft_real)-8.720494081438079359e-01;
        tr += vi17 * (notorious_fft_real)-4.894178478110849939e-01; ti += vr17 * (notorious_fft_real)-4.894178478110849939e-01;
        sr += ur18 * (notorious_fft_real)-4.572423233046381541e-01; si += ui18 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi18 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr18 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur19 * (notorious_fft_real)1.093712083778740352e-01; si += ui19 * (notorious_fft_real)1.093712083778740352e-01;
        tr += vi19 * (notorious_fft_real)-9.940009752399459053e-01; ti += vr19 * (notorious_fft_real)-9.940009752399459053e-01;
        sr += ur20 * (notorious_fft_real)6.396730215588903867e-01; si += ui20 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi20 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr20 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur21 * (notorious_fft_real)9.576005999084059583e-01; si += ui21 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi21 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr21 * (notorious_fft_real)-2.880990993652376986e-01;
        x[4*st]         = xr0 + sr - sg * tr;
        x[4*st + 1]     = xi0 + si + sg * ti;
        x[39*st]       = xr0 + sr + sg * tr;
        x[39*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)7.447721827437818742e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)7.447721827437818742e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)6.673188112222393498e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)6.673188112222393498e-01;
        sr += ur2 * (notorious_fft_real)1.093712083778745209e-01; si += ui2 * (notorious_fft_real)1.093712083778745209e-01;
        tr += vi2 * (notorious_fft_real)9.940009752399459053e-01; ti += vr2 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur3 * (notorious_fft_real)-5.818589155579526606e-01; si += ui3 * (notorious_fft_real)-5.818589155579526606e-01;
        tr += vi3 * (notorious_fft_real)8.132897407355654806e-01; ti += vr3 * (notorious_fft_real)8.132897407355654806e-01;
        sr += ur4 * (notorious_fft_real)-9.760758775559270628e-01; si += ui4 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi4 * (notorious_fft_real)2.174301755815572501e-01; ti += vr4 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur5 * (notorious_fft_real)-8.720494081438076028e-01; si += ui5 * (notorious_fft_real)-8.720494081438076028e-01;
        tr += vi5 * (notorious_fft_real)-4.894178478110856045e-01; ti += vr5 * (notorious_fft_real)-4.894178478110856045e-01;
        sr += ur6 * (notorious_fft_real)-3.228804047714466385e-01; si += ui6 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi6 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr6 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur7 * (notorious_fft_real)3.911047204901555441e-01; si += ui7 * (notorious_fft_real)3.911047204901555441e-01;
        tr += vi7 * (notorious_fft_real)-9.203461835691596482e-01; ti += vr7 * (notorious_fft_real)-9.203461835691596482e-01;
        sr += ur8 * (notorious_fft_real)9.054482374931464017e-01; si += ui8 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi8 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr8 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur9 * (notorious_fft_real)9.576005999084058473e-01; si += ui9 * (notorious_fft_real)9.576005999084058473e-01;
        tr += vi9 * (notorious_fft_real)2.880990993652378096e-01; ti += vr9 * (notorious_fft_real)2.880990993652378096e-01;
        sr += ur10 * (notorious_fft_real)5.209403404879301425e-01; si += ui10 * (notorious_fft_real)5.209403404879301425e-01;
        tr += vi10 * (notorious_fft_real)8.535930890373465418e-01; ti += vr10 * (notorious_fft_real)8.535930890373465418e-01;
        sr += ur11 * (notorious_fft_real)-1.816368509794374364e-01; si += ui11 * (notorious_fft_real)-1.816368509794374364e-01;
        tr += vi11 * (notorious_fft_real)9.833656768294659622e-01; ti += vr11 * (notorious_fft_real)9.833656768294659622e-01;
        sr += ur12 * (notorious_fft_real)-7.914964884292535752e-01; si += ui12 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi12 * (notorious_fft_real)6.111737140978499294e-01; ti += vr12 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur13 * (notorious_fft_real)-9.973322836635517552e-01; si += ui13 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi13 * (notorious_fft_real)-7.299531466090662668e-02; ti += vr13 * (notorious_fft_real)-7.299531466090662668e-02;
        sr += ur14 * (notorious_fft_real)-6.940741952206346266e-01; si += ui14 * (notorious_fft_real)-6.940741952206346266e-01;
        tr += vi14 * (notorious_fft_real)-7.199034737579951271e-01; ti += vr14 * (notorious_fft_real)-7.199034737579951271e-01;
        sr += ur15 * (notorious_fft_real)-3.652202305765988449e-02; si += ui15 * (notorious_fft_real)-3.652202305765988449e-02;
        tr += vi15 * (notorious_fft_real)-9.993328483702392795e-01; ti += vr15 * (notorious_fft_real)-9.993328483702392795e-01;
        sr += ur16 * (notorious_fft_real)6.396730215588903867e-01; si += ui16 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi16 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr16 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur17 * (notorious_fft_real)9.893433680751100301e-01; si += ui17 * (notorious_fft_real)9.893433680751100301e-01;
        tr += vi17 * (notorious_fft_real)-1.456011677350060596e-01; ti += vr17 * (notorious_fft_real)-1.456011677350060596e-01;
        sr += ur18 * (notorious_fft_real)8.339978178898777017e-01; si += ui18 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi18 * (notorious_fft_real)5.517677407704463244e-01; ti += vr18 * (notorious_fft_real)5.517677407704463244e-01;
        sr += ur19 * (notorious_fft_real)2.529333823916802837e-01; si += ui19 * (notorious_fft_real)2.529333823916802837e-01;
        tr += vi19 * (notorious_fft_real)9.674836970574253181e-01; ti += vr19 * (notorious_fft_real)9.674836970574253181e-01;
        sr += ur20 * (notorious_fft_real)-4.572423233046388757e-01; si += ui20 * (notorious_fft_real)-4.572423233046388757e-01;
        tr += vi20 * (notorious_fft_real)8.893421488825187238e-01; ti += vr20 * (notorious_fft_real)8.893421488825187238e-01;
        sr += ur21 * (notorious_fft_real)-9.340161087325481448e-01; si += ui21 * (notorious_fft_real)-9.340161087325481448e-01;
        tr += vi21 * (notorious_fft_real)3.572308898011324896e-01; ti += vr21 * (notorious_fft_real)3.572308898011324896e-01;
        x[5*st]         = xr0 + sr - sg * tr;
        x[5*st + 1]     = xi0 + si + sg * ti;
        x[38*st]       = xr0 + sr + sg * tr;
        x[38*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)6.396730215588912749e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)6.396730215588912749e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.686471397785320514e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur2 * (notorious_fft_real)-1.816368509794363539e-01; si += ui2 * (notorious_fft_real)-1.816368509794363539e-01;
        tr += vi2 * (notorious_fft_real)9.833656768294661843e-01; ti += vr2 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur3 * (notorious_fft_real)-8.720494081438077139e-01; si += ui3 * (notorious_fft_real)-8.720494081438077139e-01;
        tr += vi3 * (notorious_fft_real)4.894178478110854380e-01; ti += vr3 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur4 * (notorious_fft_real)-9.340161087325480338e-01; si += ui4 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi4 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr4 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur5 * (notorious_fft_real)-3.228804047714466385e-01; si += ui5 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi5 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr5 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur6 * (notorious_fft_real)5.209403404879304755e-01; si += ui6 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi6 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr6 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur7 * (notorious_fft_real)9.893433680751101411e-01; si += ui7 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi7 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr7 * (notorious_fft_real)-1.456011677350049216e-01;
        sr += ur8 * (notorious_fft_real)7.447721827437820963e-01; si += ui8 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi8 * (notorious_fft_real)6.673188112222391277e-01; ti += vr8 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur9 * (notorious_fft_real)-3.652202305765826079e-02; si += ui9 * (notorious_fft_real)-3.652202305765826079e-02;
        tr += vi9 * (notorious_fft_real)9.993328483702393905e-01; ti += vr9 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur10 * (notorious_fft_real)-7.914964884292535752e-01; si += ui10 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi10 * (notorious_fft_real)6.111737140978499294e-01; ti += vr10 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur11 * (notorious_fft_real)-9.760758775559271738e-01; si += ui11 * (notorious_fft_real)-9.760758775559271738e-01;
        tr += vi11 * (notorious_fft_real)-2.174301755815567505e-01; ti += vr11 * (notorious_fft_real)-2.174301755815567505e-01;
        sr += ur12 * (notorious_fft_real)-4.572423233046381541e-01; si += ui12 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi12 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr12 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur13 * (notorious_fft_real)3.911047204901553220e-01; si += ui13 * (notorious_fft_real)3.911047204901553220e-01;
        tr += vi13 * (notorious_fft_real)-9.203461835691597592e-01; ti += vr13 * (notorious_fft_real)-9.203461835691597592e-01;
        sr += ur14 * (notorious_fft_real)9.576005999084059583e-01; si += ui14 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi14 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr14 * (notorious_fft_real)-2.880990993652376986e-01;
        sr += ur15 * (notorious_fft_real)8.339978178898777017e-01; si += ui15 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi15 * (notorious_fft_real)5.517677407704463244e-01; ti += vr15 * (notorious_fft_real)5.517677407704463244e-01;
        sr += ur16 * (notorious_fft_real)1.093712083778750066e-01; si += ui16 * (notorious_fft_real)1.093712083778750066e-01;
        tr += vi16 * (notorious_fft_real)9.940009752399459053e-01; ti += vr16 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur17 * (notorious_fft_real)-6.940741952206326282e-01; si += ui17 * (notorious_fft_real)-6.940741952206326282e-01;
        tr += vi17 * (notorious_fft_real)7.199034737579971255e-01; ti += vr17 * (notorious_fft_real)7.199034737579971255e-01;
        sr += ur18 * (notorious_fft_real)-9.973322836635517552e-01; si += ui18 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi18 * (notorious_fft_real)-7.299531466090637688e-02; ti += vr18 * (notorious_fft_real)-7.299531466090637688e-02;
        sr += ur19 * (notorious_fft_real)-5.818589155579546590e-01; si += ui19 * (notorious_fft_real)-5.818589155579546590e-01;
        tr += vi19 * (notorious_fft_real)-8.132897407355640373e-01; ti += vr19 * (notorious_fft_real)-8.132897407355640373e-01;
        sr += ur20 * (notorious_fft_real)2.529333823916791180e-01; si += ui20 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi20 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr20 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur21 * (notorious_fft_real)9.054482374931461797e-01; si += ui21 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi21 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr21 * (notorious_fft_real)-4.244566988758160520e-01;
        x[6*st]         = xr0 + sr - sg * tr;
        x[6*st + 1]     = xi0 + si + sg * ti;
        x[37*st]       = xr0 + sr + sg * tr;
        x[37*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)5.209403404879302535e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)5.209403404879302535e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.535930890373464308e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.535930890373464308e-01;
        sr += ur2 * (notorious_fft_real)-4.572423233046384872e-01; si += ui2 * (notorious_fft_real)-4.572423233046384872e-01;
        tr += vi2 * (notorious_fft_real)8.893421488825189458e-01; ti += vr2 * (notorious_fft_real)8.893421488825189458e-01;
        sr += ur3 * (notorious_fft_real)-9.973322836635516442e-01; si += ui3 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi3 * (notorious_fft_real)7.299531466090755649e-02; ti += vr3 * (notorious_fft_real)7.299531466090755649e-02;
        sr += ur4 * (notorious_fft_real)-5.818589155579528827e-01; si += ui4 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi4 * (notorious_fft_real)-8.132897407355653696e-01; ti += vr4 * (notorious_fft_real)-8.132897407355653696e-01;
        sr += ur5 * (notorious_fft_real)3.911047204901555441e-01; si += ui5 * (notorious_fft_real)3.911047204901555441e-01;
        tr += vi5 * (notorious_fft_real)-9.203461835691596482e-01; ti += vr5 * (notorious_fft_real)-9.203461835691596482e-01;
        sr += ur6 * (notorious_fft_real)9.893433680751101411e-01; si += ui6 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi6 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr6 * (notorious_fft_real)-1.456011677350049216e-01;
        sr += ur7 * (notorious_fft_real)6.396730215588916080e-01; si += ui7 * (notorious_fft_real)6.396730215588916080e-01;
        tr += vi7 * (notorious_fft_real)7.686471397785317183e-01; ti += vr7 * (notorious_fft_real)7.686471397785317183e-01;
        sr += ur8 * (notorious_fft_real)-3.228804047714461389e-01; si += ui8 * (notorious_fft_real)-3.228804047714461389e-01;
        tr += vi8 * (notorious_fft_real)9.464397731576094097e-01; ti += vr8 * (notorious_fft_real)9.464397731576094097e-01;
        sr += ur9 * (notorious_fft_real)-9.760758775559270628e-01; si += ui9 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi9 * (notorious_fft_real)2.174301755815574722e-01; ti += vr9 * (notorious_fft_real)2.174301755815574722e-01;
        sr += ur10 * (notorious_fft_real)-6.940741952206346266e-01; si += ui10 * (notorious_fft_real)-6.940741952206346266e-01;
        tr += vi10 * (notorious_fft_real)-7.199034737579951271e-01; ti += vr10 * (notorious_fft_real)-7.199034737579951271e-01;
        sr += ur11 * (notorious_fft_real)2.529333823916810609e-01; si += ui11 * (notorious_fft_real)2.529333823916810609e-01;
        tr += vi11 * (notorious_fft_real)-9.674836970574252071e-01; ti += vr11 * (notorious_fft_real)-9.674836970574252071e-01;
        sr += ur12 * (notorious_fft_real)9.576005999084059583e-01; si += ui12 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi12 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr12 * (notorious_fft_real)-2.880990993652376986e-01;
        sr += ur13 * (notorious_fft_real)7.447721827437822073e-01; si += ui13 * (notorious_fft_real)7.447721827437822073e-01;
        tr += vi13 * (notorious_fft_real)6.673188112222390167e-01; ti += vr13 * (notorious_fft_real)6.673188112222390167e-01;
        sr += ur14 * (notorious_fft_real)-1.816368509794354380e-01; si += ui14 * (notorious_fft_real)-1.816368509794354380e-01;
        tr += vi14 * (notorious_fft_real)9.833656768294662953e-01; ti += vr14 * (notorious_fft_real)9.833656768294662953e-01;
        sr += ur15 * (notorious_fft_real)-9.340161087325481448e-01; si += ui15 * (notorious_fft_real)-9.340161087325481448e-01;
        tr += vi15 * (notorious_fft_real)3.572308898011324896e-01; ti += vr15 * (notorious_fft_real)3.572308898011324896e-01;
        sr += ur16 * (notorious_fft_real)-7.914964884292542413e-01; si += ui16 * (notorious_fft_real)-7.914964884292542413e-01;
        tr += vi16 * (notorious_fft_real)-6.111737140978491523e-01; ti += vr16 * (notorious_fft_real)-6.111737140978491523e-01;
        sr += ur17 * (notorious_fft_real)1.093712083778737992e-01; si += ui17 * (notorious_fft_real)1.093712083778737992e-01;
        tr += vi17 * (notorious_fft_real)-9.940009752399460163e-01; ti += vr17 * (notorious_fft_real)-9.940009752399460163e-01;
        sr += ur18 * (notorious_fft_real)9.054482374931461797e-01; si += ui18 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi18 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr18 * (notorious_fft_real)-4.244566988758160520e-01;
        sr += ur19 * (notorious_fft_real)8.339978178898788119e-01; si += ui19 * (notorious_fft_real)8.339978178898788119e-01;
        tr += vi19 * (notorious_fft_real)5.517677407704446590e-01; ti += vr19 * (notorious_fft_real)5.517677407704446590e-01;
        sr += ur20 * (notorious_fft_real)-3.652202305765687995e-02; si += ui20 * (notorious_fft_real)-3.652202305765687995e-02;
        tr += vi20 * (notorious_fft_real)9.993328483702393905e-01; ti += vr20 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur21 * (notorious_fft_real)-8.720494081438081579e-01; si += ui21 * (notorious_fft_real)-8.720494081438081579e-01;
        tr += vi21 * (notorious_fft_real)4.894178478110844943e-01; ti += vr21 * (notorious_fft_real)4.894178478110844943e-01;
        x[7*st]         = xr0 + sr - sg * tr;
        x[7*st + 1]     = xi0 + si + sg * ti;
        x[36*st]       = xr0 + sr + sg * tr;
        x[36*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)3.911047204901559882e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)3.911047204901559882e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.203461835691594262e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.203461835691594262e-01;
        sr += ur2 * (notorious_fft_real)-6.940741952206339604e-01; si += ui2 * (notorious_fft_real)-6.940741952206339604e-01;
        tr += vi2 * (notorious_fft_real)7.199034737579957932e-01; ti += vr2 * (notorious_fft_real)7.199034737579957932e-01;
        sr += ur3 * (notorious_fft_real)-9.340161087325480338e-01; si += ui3 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi3 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr3 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur4 * (notorious_fft_real)-3.652202305765874651e-02; si += ui4 * (notorious_fft_real)-3.652202305765874651e-02;
        tr += vi4 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr4 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur5 * (notorious_fft_real)9.054482374931464017e-01; si += ui5 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi5 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr5 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur6 * (notorious_fft_real)7.447721827437820963e-01; si += ui6 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi6 * (notorious_fft_real)6.673188112222391277e-01; ti += vr6 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur7 * (notorious_fft_real)-3.228804047714461389e-01; si += ui7 * (notorious_fft_real)-3.228804047714461389e-01;
        tr += vi7 * (notorious_fft_real)9.464397731576094097e-01; ti += vr7 * (notorious_fft_real)9.464397731576094097e-01;
        sr += ur8 * (notorious_fft_real)-9.973322836635516442e-01; si += ui8 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi8 * (notorious_fft_real)7.299531466090734833e-02; ti += vr8 * (notorious_fft_real)7.299531466090734833e-02;
        sr += ur9 * (notorious_fft_real)-4.572423233046381541e-01; si += ui9 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi9 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr9 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur10 * (notorious_fft_real)6.396730215588903867e-01; si += ui10 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi10 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr10 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur11 * (notorious_fft_real)9.576005999084067355e-01; si += ui11 * (notorious_fft_real)9.576005999084067355e-01;
        tr += vi11 * (notorious_fft_real)2.880990993652350340e-01; ti += vr11 * (notorious_fft_real)2.880990993652350340e-01;
        sr += ur12 * (notorious_fft_real)1.093712083778750066e-01; si += ui12 * (notorious_fft_real)1.093712083778750066e-01;
        tr += vi12 * (notorious_fft_real)9.940009752399459053e-01; ti += vr12 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur13 * (notorious_fft_real)-8.720494081438082690e-01; si += ui13 * (notorious_fft_real)-8.720494081438082690e-01;
        tr += vi13 * (notorious_fft_real)4.894178478110842723e-01; ti += vr13 * (notorious_fft_real)4.894178478110842723e-01;
        sr += ur14 * (notorious_fft_real)-7.914964884292542413e-01; si += ui14 * (notorious_fft_real)-7.914964884292542413e-01;
        tr += vi14 * (notorious_fft_real)-6.111737140978491523e-01; ti += vr14 * (notorious_fft_real)-6.111737140978491523e-01;
        sr += ur15 * (notorious_fft_real)2.529333823916791180e-01; si += ui15 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi15 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr15 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur16 * (notorious_fft_real)9.893433680751102521e-01; si += ui16 * (notorious_fft_real)9.893433680751102521e-01;
        tr += vi16 * (notorious_fft_real)-1.456011677350045330e-01; ti += vr16 * (notorious_fft_real)-1.456011677350045330e-01;
        sr += ur17 * (notorious_fft_real)5.209403404879312527e-01; si += ui17 * (notorious_fft_real)5.209403404879312527e-01;
        tr += vi17 * (notorious_fft_real)8.535930890373458757e-01; ti += vr17 * (notorious_fft_real)8.535930890373458757e-01;
        sr += ur18 * (notorious_fft_real)-5.818589155579535488e-01; si += ui18 * (notorious_fft_real)-5.818589155579535488e-01;
        tr += vi18 * (notorious_fft_real)8.132897407355648145e-01; ti += vr18 * (notorious_fft_real)8.132897407355648145e-01;
        sr += ur19 * (notorious_fft_real)-9.760758775559272848e-01; si += ui19 * (notorious_fft_real)-9.760758775559272848e-01;
        tr += vi19 * (notorious_fft_real)-2.174301755815562787e-01; ti += vr19 * (notorious_fft_real)-2.174301755815562787e-01;
        sr += ur20 * (notorious_fft_real)-1.816368509794386299e-01; si += ui20 * (notorious_fft_real)-1.816368509794386299e-01;
        tr += vi20 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr20 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur21 * (notorious_fft_real)8.339978178898778127e-01; si += ui21 * (notorious_fft_real)8.339978178898778127e-01;
        tr += vi21 * (notorious_fft_real)-5.517677407704461023e-01; ti += vr21 * (notorious_fft_real)-5.517677407704461023e-01;
        x[8*st]         = xr0 + sr - sg * tr;
        x[8*st + 1]     = xi0 + si + sg * ti;
        x[35*st]       = xr0 + sr + sg * tr;
        x[35*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)2.529333823916806723e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)2.529333823916806723e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.674836970574253181e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.674836970574253181e-01;
        sr += ur2 * (notorious_fft_real)-8.720494081438077139e-01; si += ui2 * (notorious_fft_real)-8.720494081438077139e-01;
        tr += vi2 * (notorious_fft_real)4.894178478110854380e-01; ti += vr2 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur3 * (notorious_fft_real)-6.940741952206340715e-01; si += ui3 * (notorious_fft_real)-6.940741952206340715e-01;
        tr += vi3 * (notorious_fft_real)-7.199034737579956822e-01; ti += vr3 * (notorious_fft_real)-7.199034737579956822e-01;
        sr += ur4 * (notorious_fft_real)5.209403404879304755e-01; si += ui4 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi4 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr4 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur5 * (notorious_fft_real)9.576005999084058473e-01; si += ui5 * (notorious_fft_real)9.576005999084058473e-01;
        tr += vi5 * (notorious_fft_real)2.880990993652378096e-01; ti += vr5 * (notorious_fft_real)2.880990993652378096e-01;
        sr += ur6 * (notorious_fft_real)-3.652202305765826079e-02; si += ui6 * (notorious_fft_real)-3.652202305765826079e-02;
        tr += vi6 * (notorious_fft_real)9.993328483702393905e-01; ti += vr6 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur7 * (notorious_fft_real)-9.760758775559270628e-01; si += ui7 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi7 * (notorious_fft_real)2.174301755815574722e-01; ti += vr7 * (notorious_fft_real)2.174301755815574722e-01;
        sr += ur8 * (notorious_fft_real)-4.572423233046381541e-01; si += ui8 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi8 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr8 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur9 * (notorious_fft_real)7.447721827437815412e-01; si += ui9 * (notorious_fft_real)7.447721827437815412e-01;
        tr += vi9 * (notorious_fft_real)-6.673188112222396828e-01; ti += vr9 * (notorious_fft_real)-6.673188112222396828e-01;
        sr += ur10 * (notorious_fft_real)8.339978178898777017e-01; si += ui10 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi10 * (notorious_fft_real)5.517677407704463244e-01; ti += vr10 * (notorious_fft_real)5.517677407704463244e-01;
        sr += ur11 * (notorious_fft_real)-3.228804047714475822e-01; si += ui11 * (notorious_fft_real)-3.228804047714475822e-01;
        tr += vi11 * (notorious_fft_real)9.464397731576088546e-01; ti += vr11 * (notorious_fft_real)9.464397731576088546e-01;
        sr += ur12 * (notorious_fft_real)-9.973322836635517552e-01; si += ui12 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi12 * (notorious_fft_real)-7.299531466090637688e-02; ti += vr12 * (notorious_fft_real)-7.299531466090637688e-02;
        sr += ur13 * (notorious_fft_real)-1.816368509794384078e-01; si += ui13 * (notorious_fft_real)-1.816368509794384078e-01;
        tr += vi13 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr13 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur14 * (notorious_fft_real)9.054482374931461797e-01; si += ui14 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi14 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr14 * (notorious_fft_real)-4.244566988758160520e-01;
        sr += ur15 * (notorious_fft_real)6.396730215588913859e-01; si += ui15 * (notorious_fft_real)6.396730215588913859e-01;
        tr += vi15 * (notorious_fft_real)7.686471397785320514e-01; ti += vr15 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur16 * (notorious_fft_real)-5.818589155579535488e-01; si += ui16 * (notorious_fft_real)-5.818589155579535488e-01;
        tr += vi16 * (notorious_fft_real)8.132897407355648145e-01; ti += vr16 * (notorious_fft_real)8.132897407355648145e-01;
        sr += ur17 * (notorious_fft_real)-9.340161087325485889e-01; si += ui17 * (notorious_fft_real)-9.340161087325485889e-01;
        tr += vi17 * (notorious_fft_real)-3.572308898011311018e-01; ti += vr17 * (notorious_fft_real)-3.572308898011311018e-01;
        sr += ur18 * (notorious_fft_real)1.093712083778735494e-01; si += ui18 * (notorious_fft_real)1.093712083778735494e-01;
        tr += vi18 * (notorious_fft_real)-9.940009752399460163e-01; ti += vr18 * (notorious_fft_real)-9.940009752399460163e-01;
        sr += ur19 * (notorious_fft_real)9.893433680751102521e-01; si += ui19 * (notorious_fft_real)9.893433680751102521e-01;
        tr += vi19 * (notorious_fft_real)-1.456011677350047828e-01; ti += vr19 * (notorious_fft_real)-1.456011677350047828e-01;
        sr += ur20 * (notorious_fft_real)3.911047204901550445e-01; si += ui20 * (notorious_fft_real)3.911047204901550445e-01;
        tr += vi20 * (notorious_fft_real)9.203461835691598703e-01; ti += vr20 * (notorious_fft_real)9.203461835691598703e-01;
        sr += ur21 * (notorious_fft_real)-7.914964884292553515e-01; si += ui21 * (notorious_fft_real)-7.914964884292553515e-01;
        tr += vi21 * (notorious_fft_real)6.111737140978477090e-01; ti += vr21 * (notorious_fft_real)6.111737140978477090e-01;
        x[9*st]         = xr0 + sr - sg * tr;
        x[9*st + 1]     = xi0 + si + sg * ti;
        x[34*st]       = xr0 + sr + sg * tr;
        x[34*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)1.093712083778745209e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)1.093712083778745209e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.940009752399459053e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur2 * (notorious_fft_real)-9.760758775559270628e-01; si += ui2 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi2 * (notorious_fft_real)2.174301755815572501e-01; ti += vr2 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur3 * (notorious_fft_real)-3.228804047714466385e-01; si += ui3 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi3 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr3 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur4 * (notorious_fft_real)9.054482374931464017e-01; si += ui4 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi4 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr4 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur5 * (notorious_fft_real)5.209403404879301425e-01; si += ui5 * (notorious_fft_real)5.209403404879301425e-01;
        tr += vi5 * (notorious_fft_real)8.535930890373465418e-01; ti += vr5 * (notorious_fft_real)8.535930890373465418e-01;
        sr += ur6 * (notorious_fft_real)-7.914964884292535752e-01; si += ui6 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi6 * (notorious_fft_real)6.111737140978499294e-01; ti += vr6 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur7 * (notorious_fft_real)-6.940741952206346266e-01; si += ui7 * (notorious_fft_real)-6.940741952206346266e-01;
        tr += vi7 * (notorious_fft_real)-7.199034737579951271e-01; ti += vr7 * (notorious_fft_real)-7.199034737579951271e-01;
        sr += ur8 * (notorious_fft_real)6.396730215588903867e-01; si += ui8 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi8 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr8 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur9 * (notorious_fft_real)8.339978178898777017e-01; si += ui9 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi9 * (notorious_fft_real)5.517677407704463244e-01; ti += vr9 * (notorious_fft_real)5.517677407704463244e-01;
        sr += ur10 * (notorious_fft_real)-4.572423233046388757e-01; si += ui10 * (notorious_fft_real)-4.572423233046388757e-01;
        tr += vi10 * (notorious_fft_real)8.893421488825187238e-01; ti += vr10 * (notorious_fft_real)8.893421488825187238e-01;
        sr += ur11 * (notorious_fft_real)-9.340161087325472566e-01; si += ui11 * (notorious_fft_real)-9.340161087325472566e-01;
        tr += vi11 * (notorious_fft_real)-3.572308898011346545e-01; ti += vr11 * (notorious_fft_real)-3.572308898011346545e-01;
        sr += ur12 * (notorious_fft_real)2.529333823916791180e-01; si += ui12 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi12 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr12 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur13 * (notorious_fft_real)9.893433680751104742e-01; si += ui13 * (notorious_fft_real)9.893433680751104742e-01;
        tr += vi13 * (notorious_fft_real)1.456011677350030897e-01; ti += vr13 * (notorious_fft_real)1.456011677350030897e-01;
        sr += ur14 * (notorious_fft_real)-3.652202305765687995e-02; si += ui14 * (notorious_fft_real)-3.652202305765687995e-02;
        tr += vi14 * (notorious_fft_real)9.993328483702393905e-01; ti += vr14 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur15 * (notorious_fft_real)-9.973322836635515332e-01; si += ui15 * (notorious_fft_real)-9.973322836635515332e-01;
        tr += vi15 * (notorious_fft_real)7.299531466090961040e-02; ti += vr15 * (notorious_fft_real)7.299531466090961040e-02;
        sr += ur16 * (notorious_fft_real)-1.816368509794386299e-01; si += ui16 * (notorious_fft_real)-1.816368509794386299e-01;
        tr += vi16 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr16 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur17 * (notorious_fft_real)9.576005999084052922e-01; si += ui17 * (notorious_fft_real)9.576005999084052922e-01;
        tr += vi17 * (notorious_fft_real)-2.880990993652398635e-01; ti += vr17 * (notorious_fft_real)-2.880990993652398635e-01;
        sr += ur18 * (notorious_fft_real)3.911047204901550445e-01; si += ui18 * (notorious_fft_real)3.911047204901550445e-01;
        tr += vi18 * (notorious_fft_real)9.203461835691598703e-01; ti += vr18 * (notorious_fft_real)9.203461835691598703e-01;
        sr += ur19 * (notorious_fft_real)-8.720494081438080469e-01; si += ui19 * (notorious_fft_real)-8.720494081438080469e-01;
        tr += vi19 * (notorious_fft_real)4.894178478110847164e-01; ti += vr19 * (notorious_fft_real)4.894178478110847164e-01;
        sr += ur20 * (notorious_fft_real)-5.818589155579522165e-01; si += ui20 * (notorious_fft_real)-5.818589155579522165e-01;
        tr += vi20 * (notorious_fft_real)-8.132897407355658137e-01; ti += vr20 * (notorious_fft_real)-8.132897407355658137e-01;
        sr += ur21 * (notorious_fft_real)7.447721827437823183e-01; si += ui21 * (notorious_fft_real)7.447721827437823183e-01;
        tr += vi21 * (notorious_fft_real)-6.673188112222389057e-01; ti += vr21 * (notorious_fft_real)-6.673188112222389057e-01;
        x[10*st]         = xr0 + sr - sg * tr;
        x[10*st + 1]     = xi0 + si + sg * ti;
        x[33*st]       = xr0 + sr + sg * tr;
        x[33*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-3.652202305765850365e-02;
        notorious_fft_real si = ui1 * (notorious_fft_real)-3.652202305765850365e-02;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.993328483702393905e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur2 * (notorious_fft_real)-9.973322836635517552e-01; si += ui2 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi2 * (notorious_fft_real)-7.299531466090686260e-02; ti += vr2 * (notorious_fft_real)-7.299531466090686260e-02;
        sr += ur3 * (notorious_fft_real)1.093712083778733968e-01; si += ui3 * (notorious_fft_real)1.093712083778733968e-01;
        tr += vi3 * (notorious_fft_real)-9.940009752399460163e-01; ti += vr3 * (notorious_fft_real)-9.940009752399460163e-01;
        sr += ur4 * (notorious_fft_real)9.893433680751103632e-01; si += ui4 * (notorious_fft_real)9.893433680751103632e-01;
        tr += vi4 * (notorious_fft_real)1.456011677350035616e-01; ti += vr4 * (notorious_fft_real)1.456011677350035616e-01;
        sr += ur5 * (notorious_fft_real)-1.816368509794356878e-01; si += ui5 * (notorious_fft_real)-1.816368509794356878e-01;
        tr += vi5 * (notorious_fft_real)9.833656768294662953e-01; ti += vr5 * (notorious_fft_real)9.833656768294662953e-01;
        sr += ur6 * (notorious_fft_real)-9.760758775559276179e-01; si += ui6 * (notorious_fft_real)-9.760758775559276179e-01;
        tr += vi6 * (notorious_fft_real)-2.174301755815550297e-01; ti += vr6 * (notorious_fft_real)-2.174301755815550297e-01;
        sr += ur7 * (notorious_fft_real)2.529333823916793400e-01; si += ui7 * (notorious_fft_real)2.529333823916793400e-01;
        tr += vi7 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr7 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur8 * (notorious_fft_real)9.576005999084067355e-01; si += ui8 * (notorious_fft_real)9.576005999084067355e-01;
        tr += vi8 * (notorious_fft_real)2.880990993652350340e-01; ti += vr8 * (notorious_fft_real)2.880990993652350340e-01;
        sr += ur9 * (notorious_fft_real)-3.228804047714442516e-01; si += ui9 * (notorious_fft_real)-3.228804047714442516e-01;
        tr += vi9 * (notorious_fft_real)9.464397731576100758e-01; ti += vr9 * (notorious_fft_real)9.464397731576100758e-01;
        sr += ur10 * (notorious_fft_real)-9.340161087325485889e-01; si += ui10 * (notorious_fft_real)-9.340161087325485889e-01;
        tr += vi10 * (notorious_fft_real)-3.572308898011313238e-01; ti += vr10 * (notorious_fft_real)-3.572308898011313238e-01;
        sr += ur11 * (notorious_fft_real)3.911047204901534902e-01; si += ui11 * (notorious_fft_real)3.911047204901534902e-01;
        tr += vi11 * (notorious_fft_real)-9.203461835691605364e-01; ti += vr11 * (notorious_fft_real)-9.203461835691605364e-01;
        sr += ur12 * (notorious_fft_real)9.054482374931482891e-01; si += ui12 * (notorious_fft_real)9.054482374931482891e-01;
        tr += vi12 * (notorious_fft_real)4.244566988758115000e-01; ti += vr12 * (notorious_fft_real)4.244566988758115000e-01;
        sr += ur13 * (notorious_fft_real)-4.572423233046370439e-01; si += ui13 * (notorious_fft_real)-4.572423233046370439e-01;
        tr += vi13 * (notorious_fft_real)8.893421488825196120e-01; ti += vr13 * (notorious_fft_real)8.893421488825196120e-01;
        sr += ur14 * (notorious_fft_real)-8.720494081438090461e-01; si += ui14 * (notorious_fft_real)-8.720494081438090461e-01;
        tr += vi14 * (notorious_fft_real)-4.894178478110829955e-01; ti += vr14 * (notorious_fft_real)-4.894178478110829955e-01;
        sr += ur15 * (notorious_fft_real)5.209403404879268118e-01; si += ui15 * (notorious_fft_real)5.209403404879268118e-01;
        tr += vi15 * (notorious_fft_real)-8.535930890373485402e-01; ti += vr15 * (notorious_fft_real)-8.535930890373485402e-01;
        sr += ur16 * (notorious_fft_real)8.339978178898808103e-01; si += ui16 * (notorious_fft_real)8.339978178898808103e-01;
        tr += vi16 * (notorious_fft_real)5.517677407704414394e-01; ti += vr16 * (notorious_fft_real)5.517677407704414394e-01;
        sr += ur17 * (notorious_fft_real)-5.818589155579504402e-01; si += ui17 * (notorious_fft_real)-5.818589155579504402e-01;
        tr += vi17 * (notorious_fft_real)8.132897407355670349e-01; ti += vr17 * (notorious_fft_real)8.132897407355670349e-01;
        sr += ur18 * (notorious_fft_real)-7.914964884292566838e-01; si += ui18 * (notorious_fft_real)-7.914964884292566838e-01;
        tr += vi18 * (notorious_fft_real)-6.111737140978459326e-01; ti += vr18 * (notorious_fft_real)-6.111737140978459326e-01;
        sr += ur19 * (notorious_fft_real)6.396730215588871671e-01; si += ui19 * (notorious_fft_real)6.396730215588871671e-01;
        tr += vi19 * (notorious_fft_real)-7.686471397785354931e-01; ti += vr19 * (notorious_fft_real)-7.686471397785354931e-01;
        sr += ur20 * (notorious_fft_real)7.447721827437838726e-01; si += ui20 * (notorious_fft_real)7.447721827437838726e-01;
        tr += vi20 * (notorious_fft_real)6.673188112222371293e-01; ti += vr20 * (notorious_fft_real)6.673188112222371293e-01;
        sr += ur21 * (notorious_fft_real)-6.940741952206307408e-01; si += ui21 * (notorious_fft_real)-6.940741952206307408e-01;
        tr += vi21 * (notorious_fft_real)7.199034737579987908e-01; ti += vr21 * (notorious_fft_real)7.199034737579987908e-01;
        x[11*st]         = xr0 + sr - sg * tr;
        x[11*st + 1]     = xi0 + si + sg * ti;
        x[32*st]       = xr0 + sr + sg * tr;
        x[32*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-1.816368509794363539e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-1.816368509794363539e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.833656768294661843e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur2 * (notorious_fft_real)-9.340161087325480338e-01; si += ui2 * (notorious_fft_real)-9.340161087325480338e-01;
        tr += vi2 * (notorious_fft_real)-3.572308898011326561e-01; ti += vr2 * (notorious_fft_real)-3.572308898011326561e-01;
        sr += ur3 * (notorious_fft_real)5.209403404879304755e-01; si += ui3 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi3 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr3 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur4 * (notorious_fft_real)7.447721827437820963e-01; si += ui4 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi4 * (notorious_fft_real)6.673188112222391277e-01; ti += vr4 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur5 * (notorious_fft_real)-7.914964884292535752e-01; si += ui5 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi5 * (notorious_fft_real)6.111737140978499294e-01; ti += vr5 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur6 * (notorious_fft_real)-4.572423233046381541e-01; si += ui6 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi6 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr6 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur7 * (notorious_fft_real)9.576005999084059583e-01; si += ui7 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi7 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr7 * (notorious_fft_real)-2.880990993652376986e-01;
        sr += ur8 * (notorious_fft_real)1.093712083778750066e-01; si += ui8 * (notorious_fft_real)1.093712083778750066e-01;
        tr += vi8 * (notorious_fft_real)9.940009752399459053e-01; ti += vr8 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur9 * (notorious_fft_real)-9.973322836635517552e-01; si += ui9 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi9 * (notorious_fft_real)-7.299531466090637688e-02; ti += vr9 * (notorious_fft_real)-7.299531466090637688e-02;
        sr += ur10 * (notorious_fft_real)2.529333823916791180e-01; si += ui10 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi10 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr10 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur11 * (notorious_fft_real)9.054482374931468458e-01; si += ui11 * (notorious_fft_real)9.054482374931468458e-01;
        tr += vi11 * (notorious_fft_real)4.244566988758147197e-01; ti += vr11 * (notorious_fft_real)4.244566988758147197e-01;
        sr += ur12 * (notorious_fft_real)-5.818589155579535488e-01; si += ui12 * (notorious_fft_real)-5.818589155579535488e-01;
        tr += vi12 * (notorious_fft_real)8.132897407355648145e-01; ti += vr12 * (notorious_fft_real)8.132897407355648145e-01;
        sr += ur13 * (notorious_fft_real)-6.940741952206349596e-01; si += ui13 * (notorious_fft_real)-6.940741952206349596e-01;
        tr += vi13 * (notorious_fft_real)-7.199034737579947940e-01; ti += vr13 * (notorious_fft_real)-7.199034737579947940e-01;
        sr += ur14 * (notorious_fft_real)8.339978178898778127e-01; si += ui14 * (notorious_fft_real)8.339978178898778127e-01;
        tr += vi14 * (notorious_fft_real)-5.517677407704461023e-01; ti += vr14 * (notorious_fft_real)-5.517677407704461023e-01;
        sr += ur15 * (notorious_fft_real)3.911047204901550445e-01; si += ui15 * (notorious_fft_real)3.911047204901550445e-01;
        tr += vi15 * (notorious_fft_real)9.203461835691598703e-01; ti += vr15 * (notorious_fft_real)9.203461835691598703e-01;
        sr += ur16 * (notorious_fft_real)-9.760758775559268408e-01; si += ui16 * (notorious_fft_real)-9.760758775559268408e-01;
        tr += vi16 * (notorious_fft_real)2.174301755815581938e-01; ti += vr16 * (notorious_fft_real)2.174301755815581938e-01;
        sr += ur17 * (notorious_fft_real)-3.652202305766238943e-02; si += ui17 * (notorious_fft_real)-3.652202305766238943e-02;
        tr += vi17 * (notorious_fft_real)-9.993328483702392795e-01; ti += vr17 * (notorious_fft_real)-9.993328483702392795e-01;
        sr += ur18 * (notorious_fft_real)9.893433680751105852e-01; si += ui18 * (notorious_fft_real)9.893433680751105852e-01;
        tr += vi18 * (notorious_fft_real)1.456011677350025901e-01; ti += vr18 * (notorious_fft_real)1.456011677350025901e-01;
        sr += ur19 * (notorious_fft_real)-3.228804047714418646e-01; si += ui19 * (notorious_fft_real)-3.228804047714418646e-01;
        tr += vi19 * (notorious_fft_real)9.464397731576108530e-01; ti += vr19 * (notorious_fft_real)9.464397731576108530e-01;
        sr += ur20 * (notorious_fft_real)-8.720494081438092682e-01; si += ui20 * (notorious_fft_real)-8.720494081438092682e-01;
        tr += vi20 * (notorious_fft_real)-4.894178478110826069e-01; ti += vr20 * (notorious_fft_real)-4.894178478110826069e-01;
        sr += ur21 * (notorious_fft_real)6.396730215588896096e-01; si += ui21 * (notorious_fft_real)6.396730215588896096e-01;
        tr += vi21 * (notorious_fft_real)-7.686471397785333837e-01; ti += vr21 * (notorious_fft_real)-7.686471397785333837e-01;
        x[12*st]         = xr0 + sr - sg * tr;
        x[12*st + 1]     = xi0 + si + sg * ti;
        x[31*st]       = xr0 + sr + sg * tr;
        x[31*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-3.228804047714463610e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-3.228804047714463610e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)9.464397731576092987e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)9.464397731576092987e-01;
        sr += ur2 * (notorious_fft_real)-7.914964884292539082e-01; si += ui2 * (notorious_fft_real)-7.914964884292539082e-01;
        tr += vi2 * (notorious_fft_real)-6.111737140978495963e-01; ti += vr2 * (notorious_fft_real)-6.111737140978495963e-01;
        sr += ur3 * (notorious_fft_real)8.339978178898782568e-01; si += ui3 * (notorious_fft_real)8.339978178898782568e-01;
        tr += vi3 * (notorious_fft_real)-5.517677407704454362e-01; ti += vr3 * (notorious_fft_real)-5.517677407704454362e-01;
        sr += ur4 * (notorious_fft_real)2.529333823916800617e-01; si += ui4 * (notorious_fft_real)2.529333823916800617e-01;
        tr += vi4 * (notorious_fft_real)9.674836970574254291e-01; ti += vr4 * (notorious_fft_real)9.674836970574254291e-01;
        sr += ur5 * (notorious_fft_real)-9.973322836635516442e-01; si += ui5 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi5 * (notorious_fft_real)-7.299531466090838916e-02; ti += vr5 * (notorious_fft_real)-7.299531466090838916e-02;
        sr += ur6 * (notorious_fft_real)3.911047204901569874e-01; si += ui6 * (notorious_fft_real)3.911047204901569874e-01;
        tr += vi6 * (notorious_fft_real)-9.203461835691590931e-01; ti += vr6 * (notorious_fft_real)-9.203461835691590931e-01;
        sr += ur7 * (notorious_fft_real)7.447721827437810971e-01; si += ui7 * (notorious_fft_real)7.447721827437810971e-01;
        tr += vi7 * (notorious_fft_real)6.673188112222403490e-01; ti += vr7 * (notorious_fft_real)6.673188112222403490e-01;
        sr += ur8 * (notorious_fft_real)-8.720494081438082690e-01; si += ui8 * (notorious_fft_real)-8.720494081438082690e-01;
        tr += vi8 * (notorious_fft_real)4.894178478110842723e-01; ti += vr8 * (notorious_fft_real)4.894178478110842723e-01;
        sr += ur9 * (notorious_fft_real)-1.816368509794384078e-01; si += ui9 * (notorious_fft_real)-1.816368509794384078e-01;
        tr += vi9 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr9 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur10 * (notorious_fft_real)9.893433680751099191e-01; si += ui10 * (notorious_fft_real)9.893433680751099191e-01;
        tr += vi10 * (notorious_fft_real)1.456011677350065869e-01; ti += vr10 * (notorious_fft_real)1.456011677350065869e-01;
        sr += ur11 * (notorious_fft_real)-4.572423233046402080e-01; si += ui11 * (notorious_fft_real)-4.572423233046402080e-01;
        tr += vi11 * (notorious_fft_real)8.893421488825180576e-01; ti += vr11 * (notorious_fft_real)8.893421488825180576e-01;
        sr += ur12 * (notorious_fft_real)-6.940741952206324061e-01; si += ui12 * (notorious_fft_real)-6.940741952206324061e-01;
        tr += vi12 * (notorious_fft_real)-7.199034737579973475e-01; ti += vr12 * (notorious_fft_real)-7.199034737579973475e-01;
        sr += ur13 * (notorious_fft_real)9.054482374931460686e-01; si += ui13 * (notorious_fft_real)9.054482374931460686e-01;
        tr += vi13 * (notorious_fft_real)-4.244566988758162740e-01; ti += vr13 * (notorious_fft_real)-4.244566988758162740e-01;
        sr += ur14 * (notorious_fft_real)1.093712083778719674e-01; si += ui14 * (notorious_fft_real)1.093712083778719674e-01;
        tr += vi14 * (notorious_fft_real)9.940009752399462384e-01; ti += vr14 * (notorious_fft_real)9.940009752399462384e-01;
        sr += ur15 * (notorious_fft_real)-9.760758775559266187e-01; si += ui15 * (notorious_fft_real)-9.760758775559266187e-01;
        tr += vi15 * (notorious_fft_real)-2.174301755815595261e-01; ti += vr15 * (notorious_fft_real)-2.174301755815595261e-01;
        sr += ur16 * (notorious_fft_real)5.209403404879326960e-01; si += ui16 * (notorious_fft_real)5.209403404879326960e-01;
        tr += vi16 * (notorious_fft_real)-8.535930890373449875e-01; ti += vr16 * (notorious_fft_real)-8.535930890373449875e-01;
        sr += ur17 * (notorious_fft_real)6.396730215588917190e-01; si += ui17 * (notorious_fft_real)6.396730215588917190e-01;
        tr += vi17 * (notorious_fft_real)7.686471397785317183e-01; ti += vr17 * (notorious_fft_real)7.686471397785317183e-01;
        sr += ur18 * (notorious_fft_real)-9.340161087325465905e-01; si += ui18 * (notorious_fft_real)-9.340161087325465905e-01;
        tr += vi18 * (notorious_fft_real)3.572308898011364864e-01; ti += vr18 * (notorious_fft_real)3.572308898011364864e-01;
        sr += ur19 * (notorious_fft_real)-3.652202305765908652e-02; si += ui19 * (notorious_fft_real)-3.652202305765908652e-02;
        tr += vi19 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr19 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur20 * (notorious_fft_real)9.576005999084049591e-01; si += ui20 * (notorious_fft_real)9.576005999084049591e-01;
        tr += vi20 * (notorious_fft_real)2.880990993652409182e-01; ti += vr20 * (notorious_fft_real)2.880990993652409182e-01;
        sr += ur21 * (notorious_fft_real)-5.818589155579528827e-01; si += ui21 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi21 * (notorious_fft_real)8.132897407355652586e-01; ti += vr21 * (notorious_fft_real)8.132897407355652586e-01;
        x[13*st]         = xr0 + sr - sg * tr;
        x[13*st + 1]     = xi0 + si + sg * ti;
        x[30*st]       = xr0 + sr + sg * tr;
        x[30*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-4.572423233046384872e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-4.572423233046384872e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.893421488825189458e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.893421488825189458e-01;
        sr += ur2 * (notorious_fft_real)-5.818589155579528827e-01; si += ui2 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi2 * (notorious_fft_real)-8.132897407355653696e-01; ti += vr2 * (notorious_fft_real)-8.132897407355653696e-01;
        sr += ur3 * (notorious_fft_real)9.893433680751101411e-01; si += ui3 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi3 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr3 * (notorious_fft_real)-1.456011677350049216e-01;
        sr += ur4 * (notorious_fft_real)-3.228804047714461389e-01; si += ui4 * (notorious_fft_real)-3.228804047714461389e-01;
        tr += vi4 * (notorious_fft_real)9.464397731576094097e-01; ti += vr4 * (notorious_fft_real)9.464397731576094097e-01;
        sr += ur5 * (notorious_fft_real)-6.940741952206346266e-01; si += ui5 * (notorious_fft_real)-6.940741952206346266e-01;
        tr += vi5 * (notorious_fft_real)-7.199034737579951271e-01; ti += vr5 * (notorious_fft_real)-7.199034737579951271e-01;
        sr += ur6 * (notorious_fft_real)9.576005999084059583e-01; si += ui6 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi6 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr6 * (notorious_fft_real)-2.880990993652376986e-01;
        sr += ur7 * (notorious_fft_real)-1.816368509794354380e-01; si += ui7 * (notorious_fft_real)-1.816368509794354380e-01;
        tr += vi7 * (notorious_fft_real)9.833656768294662953e-01; ti += vr7 * (notorious_fft_real)9.833656768294662953e-01;
        sr += ur8 * (notorious_fft_real)-7.914964884292542413e-01; si += ui8 * (notorious_fft_real)-7.914964884292542413e-01;
        tr += vi8 * (notorious_fft_real)-6.111737140978491523e-01; ti += vr8 * (notorious_fft_real)-6.111737140978491523e-01;
        sr += ur9 * (notorious_fft_real)9.054482374931461797e-01; si += ui9 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi9 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr9 * (notorious_fft_real)-4.244566988758160520e-01;
        sr += ur10 * (notorious_fft_real)-3.652202305765687995e-02; si += ui10 * (notorious_fft_real)-3.652202305765687995e-02;
        tr += vi10 * (notorious_fft_real)9.993328483702393905e-01; ti += vr10 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur11 * (notorious_fft_real)-8.720494081438072698e-01; si += ui11 * (notorious_fft_real)-8.720494081438072698e-01;
        tr += vi11 * (notorious_fft_real)-4.894178478110861041e-01; ti += vr11 * (notorious_fft_real)-4.894178478110861041e-01;
        sr += ur12 * (notorious_fft_real)8.339978178898778127e-01; si += ui12 * (notorious_fft_real)8.339978178898778127e-01;
        tr += vi12 * (notorious_fft_real)-5.517677407704461023e-01; ti += vr12 * (notorious_fft_real)-5.517677407704461023e-01;
        sr += ur13 * (notorious_fft_real)1.093712083778755062e-01; si += ui13 * (notorious_fft_real)1.093712083778755062e-01;
        tr += vi13 * (notorious_fft_real)9.940009752399457943e-01; ti += vr13 * (notorious_fft_real)9.940009752399457943e-01;
        sr += ur14 * (notorious_fft_real)-9.340161087325486999e-01; si += ui14 * (notorious_fft_real)-9.340161087325486999e-01;
        tr += vi14 * (notorious_fft_real)-3.572308898011308798e-01; ti += vr14 * (notorious_fft_real)-3.572308898011308798e-01;
        sr += ur15 * (notorious_fft_real)7.447721827437823183e-01; si += ui15 * (notorious_fft_real)7.447721827437823183e-01;
        tr += vi15 * (notorious_fft_real)-6.673188112222389057e-01; ti += vr15 * (notorious_fft_real)-6.673188112222389057e-01;
        sr += ur16 * (notorious_fft_real)2.529333823916810053e-01; si += ui16 * (notorious_fft_real)2.529333823916810053e-01;
        tr += vi16 * (notorious_fft_real)9.674836970574252071e-01; ti += vr16 * (notorious_fft_real)9.674836970574252071e-01;
        sr += ur17 * (notorious_fft_real)-9.760758775559273959e-01; si += ui17 * (notorious_fft_real)-9.760758775559273959e-01;
        tr += vi17 * (notorious_fft_real)-2.174301755815558068e-01; ti += vr17 * (notorious_fft_real)-2.174301755815558068e-01;
        sr += ur18 * (notorious_fft_real)6.396730215588896096e-01; si += ui18 * (notorious_fft_real)6.396730215588896096e-01;
        tr += vi18 * (notorious_fft_real)-7.686471397785333837e-01; ti += vr18 * (notorious_fft_real)-7.686471397785333837e-01;
        sr += ur19 * (notorious_fft_real)3.911047204901587637e-01; si += ui19 * (notorious_fft_real)3.911047204901587637e-01;
        tr += vi19 * (notorious_fft_real)9.203461835691583159e-01; ti += vr19 * (notorious_fft_real)9.203461835691583159e-01;
        sr += ur20 * (notorious_fft_real)-9.973322836635519772e-01; si += ui20 * (notorious_fft_real)-9.973322836635519772e-01;
        tr += vi20 * (notorious_fft_real)-7.299531466090362908e-02; ti += vr20 * (notorious_fft_real)-7.299531466090362908e-02;
        sr += ur21 * (notorious_fft_real)5.209403404879322519e-01; si += ui21 * (notorious_fft_real)5.209403404879322519e-01;
        tr += vi21 * (notorious_fft_real)-8.535930890373452096e-01; ti += vr21 * (notorious_fft_real)-8.535930890373452096e-01;
        x[14*st]         = xr0 + sr - sg * tr;
        x[14*st + 1]     = xi0 + si + sg * ti;
        x[29*st]       = xr0 + sr + sg * tr;
        x[29*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-5.818589155579526606e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-5.818589155579526606e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)8.132897407355654806e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)8.132897407355654806e-01;
        sr += ur2 * (notorious_fft_real)-3.228804047714466385e-01; si += ui2 * (notorious_fft_real)-3.228804047714466385e-01;
        tr += vi2 * (notorious_fft_real)-9.464397731576091877e-01; ti += vr2 * (notorious_fft_real)-9.464397731576091877e-01;
        sr += ur3 * (notorious_fft_real)9.576005999084064024e-01; si += ui3 * (notorious_fft_real)9.576005999084064024e-01;
        tr += vi3 * (notorious_fft_real)2.880990993652361443e-01; ti += vr3 * (notorious_fft_real)2.880990993652361443e-01;
        sr += ur4 * (notorious_fft_real)-7.914964884292535752e-01; si += ui4 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi4 * (notorious_fft_real)6.111737140978499294e-01; ti += vr4 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur5 * (notorious_fft_real)-3.652202305765988449e-02; si += ui5 * (notorious_fft_real)-3.652202305765988449e-02;
        tr += vi5 * (notorious_fft_real)-9.993328483702392795e-01; ti += vr5 * (notorious_fft_real)-9.993328483702392795e-01;
        sr += ur6 * (notorious_fft_real)8.339978178898795891e-01; si += ui6 * (notorious_fft_real)8.339978178898795891e-01;
        tr += vi6 * (notorious_fft_real)5.517677407704433268e-01; ti += vr6 * (notorious_fft_real)5.517677407704433268e-01;
        sr += ur7 * (notorious_fft_real)-9.340161087325468126e-01; si += ui7 * (notorious_fft_real)-9.340161087325468126e-01;
        tr += vi7 * (notorious_fft_real)3.572308898011358202e-01; ti += vr7 * (notorious_fft_real)3.572308898011358202e-01;
        sr += ur8 * (notorious_fft_real)2.529333823916791180e-01; si += ui8 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi8 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr8 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur9 * (notorious_fft_real)6.396730215588913859e-01; si += ui9 * (notorious_fft_real)6.396730215588913859e-01;
        tr += vi9 * (notorious_fft_real)7.686471397785320514e-01; ti += vr9 * (notorious_fft_real)7.686471397785320514e-01;
        sr += ur10 * (notorious_fft_real)-9.973322836635515332e-01; si += ui10 * (notorious_fft_real)-9.973322836635515332e-01;
        tr += vi10 * (notorious_fft_real)7.299531466090961040e-02; ti += vr10 * (notorious_fft_real)7.299531466090961040e-02;
        sr += ur11 * (notorious_fft_real)5.209403404879268118e-01; si += ui11 * (notorious_fft_real)5.209403404879268118e-01;
        tr += vi11 * (notorious_fft_real)-8.535930890373485402e-01; ti += vr11 * (notorious_fft_real)-8.535930890373485402e-01;
        sr += ur12 * (notorious_fft_real)3.911047204901615948e-01; si += ui12 * (notorious_fft_real)3.911047204901615948e-01;
        tr += vi12 * (notorious_fft_real)9.203461835691570947e-01; ti += vr12 * (notorious_fft_real)9.203461835691570947e-01;
        sr += ur13 * (notorious_fft_real)-9.760758775559273959e-01; si += ui13 * (notorious_fft_real)-9.760758775559273959e-01;
        tr += vi13 * (notorious_fft_real)-2.174301755815560566e-01; ti += vr13 * (notorious_fft_real)-2.174301755815560566e-01;
        sr += ur14 * (notorious_fft_real)7.447721827437775444e-01; si += ui14 * (notorious_fft_real)7.447721827437775444e-01;
        tr += vi14 * (notorious_fft_real)-6.673188112222442347e-01; ti += vr14 * (notorious_fft_real)-6.673188112222442347e-01;
        sr += ur15 * (notorious_fft_real)1.093712083778792810e-01; si += ui15 * (notorious_fft_real)1.093712083778792810e-01;
        tr += vi15 * (notorious_fft_real)9.940009752399453502e-01; ti += vr15 * (notorious_fft_real)9.940009752399453502e-01;
        sr += ur16 * (notorious_fft_real)-8.720494081438092682e-01; si += ui16 * (notorious_fft_real)-8.720494081438092682e-01;
        tr += vi16 * (notorious_fft_real)-4.894178478110826069e-01; ti += vr16 * (notorious_fft_real)-4.894178478110826069e-01;
        sr += ur17 * (notorious_fft_real)9.054482374931428490e-01; si += ui17 * (notorious_fft_real)9.054482374931428490e-01;
        tr += vi17 * (notorious_fft_real)-4.244566988758231574e-01; ti += vr17 * (notorious_fft_real)-4.244566988758231574e-01;
        sr += ur18 * (notorious_fft_real)-1.816368509794362152e-01; si += ui18 * (notorious_fft_real)-1.816368509794362152e-01;
        tr += vi18 * (notorious_fft_real)9.833656768294661843e-01; ti += vr18 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur19 * (notorious_fft_real)-6.940741952206380683e-01; si += ui19 * (notorious_fft_real)-6.940741952206380683e-01;
        tr += vi19 * (notorious_fft_real)-7.199034737579919074e-01; ti += vr19 * (notorious_fft_real)-7.199034737579919074e-01;
        sr += ur20 * (notorious_fft_real)9.893433680751095860e-01; si += ui20 * (notorious_fft_real)9.893433680751095860e-01;
        tr += vi20 * (notorious_fft_real)-1.456011677350090017e-01; ti += vr20 * (notorious_fft_real)-1.456011677350090017e-01;
        sr += ur21 * (notorious_fft_real)-4.572423233046362112e-01; si += ui21 * (notorious_fft_real)-4.572423233046362112e-01;
        tr += vi21 * (notorious_fft_real)8.893421488825200560e-01; ti += vr21 * (notorious_fft_real)8.893421488825200560e-01;
        x[15*st]         = xr0 + sr - sg * tr;
        x[15*st + 1]     = xi0 + si + sg * ti;
        x[28*st]       = xr0 + sr + sg * tr;
        x[28*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-6.940741952206339604e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-6.940741952206339604e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.199034737579957932e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.199034737579957932e-01;
        sr += ur2 * (notorious_fft_real)-3.652202305765874651e-02; si += ui2 * (notorious_fft_real)-3.652202305765874651e-02;
        tr += vi2 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr2 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur3 * (notorious_fft_real)7.447721827437820963e-01; si += ui3 * (notorious_fft_real)7.447721827437820963e-01;
        tr += vi3 * (notorious_fft_real)6.673188112222391277e-01; ti += vr3 * (notorious_fft_real)6.673188112222391277e-01;
        sr += ur4 * (notorious_fft_real)-9.973322836635516442e-01; si += ui4 * (notorious_fft_real)-9.973322836635516442e-01;
        tr += vi4 * (notorious_fft_real)7.299531466090734833e-02; ti += vr4 * (notorious_fft_real)7.299531466090734833e-02;
        sr += ur5 * (notorious_fft_real)6.396730215588903867e-01; si += ui5 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi5 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr5 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur6 * (notorious_fft_real)1.093712083778750066e-01; si += ui6 * (notorious_fft_real)1.093712083778750066e-01;
        tr += vi6 * (notorious_fft_real)9.940009752399459053e-01; ti += vr6 * (notorious_fft_real)9.940009752399459053e-01;
        sr += ur7 * (notorious_fft_real)-7.914964884292542413e-01; si += ui7 * (notorious_fft_real)-7.914964884292542413e-01;
        tr += vi7 * (notorious_fft_real)-6.111737140978491523e-01; ti += vr7 * (notorious_fft_real)-6.111737140978491523e-01;
        sr += ur8 * (notorious_fft_real)9.893433680751102521e-01; si += ui8 * (notorious_fft_real)9.893433680751102521e-01;
        tr += vi8 * (notorious_fft_real)-1.456011677350045330e-01; ti += vr8 * (notorious_fft_real)-1.456011677350045330e-01;
        sr += ur9 * (notorious_fft_real)-5.818589155579535488e-01; si += ui9 * (notorious_fft_real)-5.818589155579535488e-01;
        tr += vi9 * (notorious_fft_real)8.132897407355648145e-01; ti += vr9 * (notorious_fft_real)8.132897407355648145e-01;
        sr += ur10 * (notorious_fft_real)-1.816368509794386299e-01; si += ui10 * (notorious_fft_real)-1.816368509794386299e-01;
        tr += vi10 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr10 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur11 * (notorious_fft_real)8.339978178898808103e-01; si += ui11 * (notorious_fft_real)8.339978178898808103e-01;
        tr += vi11 * (notorious_fft_real)5.517677407704414394e-01; ti += vr11 * (notorious_fft_real)5.517677407704414394e-01;
        sr += ur12 * (notorious_fft_real)-9.760758775559268408e-01; si += ui12 * (notorious_fft_real)-9.760758775559268408e-01;
        tr += vi12 * (notorious_fft_real)2.174301755815581938e-01; ti += vr12 * (notorious_fft_real)2.174301755815581938e-01;
        sr += ur13 * (notorious_fft_real)5.209403404879326960e-01; si += ui13 * (notorious_fft_real)5.209403404879326960e-01;
        tr += vi13 * (notorious_fft_real)-8.535930890373449875e-01; ti += vr13 * (notorious_fft_real)-8.535930890373449875e-01;
        sr += ur14 * (notorious_fft_real)2.529333823916810053e-01; si += ui14 * (notorious_fft_real)2.529333823916810053e-01;
        tr += vi14 * (notorious_fft_real)9.674836970574252071e-01; ti += vr14 * (notorious_fft_real)9.674836970574252071e-01;
        sr += ur15 * (notorious_fft_real)-8.720494081438092682e-01; si += ui15 * (notorious_fft_real)-8.720494081438092682e-01;
        tr += vi15 * (notorious_fft_real)-4.894178478110826069e-01; ti += vr15 * (notorious_fft_real)-4.894178478110826069e-01;
        sr += ur16 * (notorious_fft_real)9.576005999084061804e-01; si += ui16 * (notorious_fft_real)9.576005999084061804e-01;
        tr += vi16 * (notorious_fft_real)-2.880990993652369214e-01; ti += vr16 * (notorious_fft_real)-2.880990993652369214e-01;
        sr += ur17 * (notorious_fft_real)-4.572423233046364333e-01; si += ui17 * (notorious_fft_real)-4.572423233046364333e-01;
        tr += vi17 * (notorious_fft_real)8.893421488825199450e-01; ti += vr17 * (notorious_fft_real)8.893421488825199450e-01;
        sr += ur18 * (notorious_fft_real)-3.228804047714446401e-01; si += ui18 * (notorious_fft_real)-3.228804047714446401e-01;
        tr += vi18 * (notorious_fft_real)-9.464397731576098538e-01; ti += vr18 * (notorious_fft_real)-9.464397731576098538e-01;
        sr += ur19 * (notorious_fft_real)9.054482374931471789e-01; si += ui19 * (notorious_fft_real)9.054482374931471789e-01;
        tr += vi19 * (notorious_fft_real)4.244566988758138315e-01; ti += vr19 * (notorious_fft_real)4.244566988758138315e-01;
        sr += ur20 * (notorious_fft_real)-9.340161087325463685e-01; si += ui20 * (notorious_fft_real)-9.340161087325463685e-01;
        tr += vi20 * (notorious_fft_real)3.572308898011369305e-01; ti += vr20 * (notorious_fft_real)3.572308898011369305e-01;
        sr += ur21 * (notorious_fft_real)3.911047204901555996e-01; si += ui21 * (notorious_fft_real)3.911047204901555996e-01;
        tr += vi21 * (notorious_fft_real)-9.203461835691596482e-01; ti += vr21 * (notorious_fft_real)-9.203461835691596482e-01;
        x[16*st]         = xr0 + sr - sg * tr;
        x[16*st + 1]     = xi0 + si + sg * ti;
        x[27*st]       = xr0 + sr + sg * tr;
        x[27*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-7.914964884292540193e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-7.914964884292540193e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)6.111737140978493743e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)6.111737140978493743e-01;
        sr += ur2 * (notorious_fft_real)2.529333823916804502e-01; si += ui2 * (notorious_fft_real)2.529333823916804502e-01;
        tr += vi2 * (notorious_fft_real)-9.674836970574253181e-01; ti += vr2 * (notorious_fft_real)-9.674836970574253181e-01;
        sr += ur3 * (notorious_fft_real)3.911047204901552110e-01; si += ui3 * (notorious_fft_real)3.911047204901552110e-01;
        tr += vi3 * (notorious_fft_real)9.203461835691597592e-01; ti += vr3 * (notorious_fft_real)9.203461835691597592e-01;
        sr += ur4 * (notorious_fft_real)-8.720494081438079359e-01; si += ui4 * (notorious_fft_real)-8.720494081438079359e-01;
        tr += vi4 * (notorious_fft_real)-4.894178478110849939e-01; ti += vr4 * (notorious_fft_real)-4.894178478110849939e-01;
        sr += ur5 * (notorious_fft_real)9.893433680751100301e-01; si += ui5 * (notorious_fft_real)9.893433680751100301e-01;
        tr += vi5 * (notorious_fft_real)-1.456011677350060596e-01; ti += vr5 * (notorious_fft_real)-1.456011677350060596e-01;
        sr += ur6 * (notorious_fft_real)-6.940741952206351817e-01; si += ui6 * (notorious_fft_real)-6.940741952206351817e-01;
        tr += vi6 * (notorious_fft_real)7.199034737579945720e-01; ti += vr6 * (notorious_fft_real)7.199034737579945720e-01;
        sr += ur7 * (notorious_fft_real)1.093712083778737992e-01; si += ui7 * (notorious_fft_real)1.093712083778737992e-01;
        tr += vi7 * (notorious_fft_real)-9.940009752399460163e-01; ti += vr7 * (notorious_fft_real)-9.940009752399460163e-01;
        sr += ur8 * (notorious_fft_real)5.209403404879312527e-01; si += ui8 * (notorious_fft_real)5.209403404879312527e-01;
        tr += vi8 * (notorious_fft_real)8.535930890373458757e-01; ti += vr8 * (notorious_fft_real)8.535930890373458757e-01;
        sr += ur9 * (notorious_fft_real)-9.340161087325485889e-01; si += ui9 * (notorious_fft_real)-9.340161087325485889e-01;
        tr += vi9 * (notorious_fft_real)-3.572308898011311018e-01; ti += vr9 * (notorious_fft_real)-3.572308898011311018e-01;
        sr += ur10 * (notorious_fft_real)9.576005999084052922e-01; si += ui10 * (notorious_fft_real)9.576005999084052922e-01;
        tr += vi10 * (notorious_fft_real)-2.880990993652398635e-01; ti += vr10 * (notorious_fft_real)-2.880990993652398635e-01;
        sr += ur11 * (notorious_fft_real)-5.818589155579533267e-01; si += ui11 * (notorious_fft_real)-5.818589155579533267e-01;
        tr += vi11 * (notorious_fft_real)8.132897407355650365e-01; ti += vr11 * (notorious_fft_real)8.132897407355650365e-01;
        sr += ur12 * (notorious_fft_real)-3.652202305765529095e-02; si += ui12 * (notorious_fft_real)-3.652202305765529095e-02;
        tr += vi12 * (notorious_fft_real)-9.993328483702395015e-01; ti += vr12 * (notorious_fft_real)-9.993328483702395015e-01;
        sr += ur13 * (notorious_fft_real)6.396730215588917190e-01; si += ui13 * (notorious_fft_real)6.396730215588917190e-01;
        tr += vi13 * (notorious_fft_real)7.686471397785317183e-01; ti += vr13 * (notorious_fft_real)7.686471397785317183e-01;
        sr += ur14 * (notorious_fft_real)-9.760758775559273959e-01; si += ui14 * (notorious_fft_real)-9.760758775559273959e-01;
        tr += vi14 * (notorious_fft_real)-2.174301755815558068e-01; ti += vr14 * (notorious_fft_real)-2.174301755815558068e-01;
        sr += ur15 * (notorious_fft_real)9.054482374931458466e-01; si += ui15 * (notorious_fft_real)9.054482374931458466e-01;
        tr += vi15 * (notorious_fft_real)-4.244566988758167181e-01; ti += vr15 * (notorious_fft_real)-4.244566988758167181e-01;
        sr += ur16 * (notorious_fft_real)-4.572423233046364333e-01; si += ui16 * (notorious_fft_real)-4.572423233046364333e-01;
        tr += vi16 * (notorious_fft_real)8.893421488825199450e-01; ti += vr16 * (notorious_fft_real)8.893421488825199450e-01;
        sr += ur17 * (notorious_fft_real)-1.816368509794323849e-01; si += ui17 * (notorious_fft_real)-1.816368509794323849e-01;
        tr += vi17 * (notorious_fft_real)-9.833656768294668504e-01; ti += vr17 * (notorious_fft_real)-9.833656768294668504e-01;
        sr += ur18 * (notorious_fft_real)7.447721827437842057e-01; si += ui18 * (notorious_fft_real)7.447721827437842057e-01;
        tr += vi18 * (notorious_fft_real)6.673188112222367963e-01; ti += vr18 * (notorious_fft_real)6.673188112222367963e-01;
        sr += ur19 * (notorious_fft_real)-9.973322836635514221e-01; si += ui19 * (notorious_fft_real)-9.973322836635514221e-01;
        tr += vi19 * (notorious_fft_real)-7.299531466091047083e-02; ti += vr19 * (notorious_fft_real)-7.299531466091047083e-02;
        sr += ur20 * (notorious_fft_real)8.339978178898753702e-01; si += ui20 * (notorious_fft_real)8.339978178898753702e-01;
        tr += vi20 * (notorious_fft_real)-5.517677407704498771e-01; ti += vr20 * (notorious_fft_real)-5.517677407704498771e-01;
        sr += ur21 * (notorious_fft_real)-3.228804047714546321e-01; si += ui21 * (notorious_fft_real)-3.228804047714546321e-01;
        tr += vi21 * (notorious_fft_real)9.464397731576065231e-01; ti += vr21 * (notorious_fft_real)9.464397731576065231e-01;
        x[17*st]         = xr0 + sr - sg * tr;
        x[17*st + 1]     = xi0 + si + sg * ti;
        x[26*st]       = xr0 + sr + sg * tr;
        x[26*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-8.720494081438077139e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-8.720494081438077139e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)4.894178478110854380e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)4.894178478110854380e-01;
        sr += ur2 * (notorious_fft_real)5.209403404879304755e-01; si += ui2 * (notorious_fft_real)5.209403404879304755e-01;
        tr += vi2 * (notorious_fft_real)-8.535930890373463198e-01; ti += vr2 * (notorious_fft_real)-8.535930890373463198e-01;
        sr += ur3 * (notorious_fft_real)-3.652202305765826079e-02; si += ui3 * (notorious_fft_real)-3.652202305765826079e-02;
        tr += vi3 * (notorious_fft_real)9.993328483702393905e-01; ti += vr3 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur4 * (notorious_fft_real)-4.572423233046381541e-01; si += ui4 * (notorious_fft_real)-4.572423233046381541e-01;
        tr += vi4 * (notorious_fft_real)-8.893421488825190568e-01; ti += vr4 * (notorious_fft_real)-8.893421488825190568e-01;
        sr += ur5 * (notorious_fft_real)8.339978178898777017e-01; si += ui5 * (notorious_fft_real)8.339978178898777017e-01;
        tr += vi5 * (notorious_fft_real)5.517677407704463244e-01; ti += vr5 * (notorious_fft_real)5.517677407704463244e-01;
        sr += ur6 * (notorious_fft_real)-9.973322836635517552e-01; si += ui6 * (notorious_fft_real)-9.973322836635517552e-01;
        tr += vi6 * (notorious_fft_real)-7.299531466090637688e-02; ti += vr6 * (notorious_fft_real)-7.299531466090637688e-02;
        sr += ur7 * (notorious_fft_real)9.054482374931461797e-01; si += ui7 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi7 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr7 * (notorious_fft_real)-4.244566988758160520e-01;
        sr += ur8 * (notorious_fft_real)-5.818589155579535488e-01; si += ui8 * (notorious_fft_real)-5.818589155579535488e-01;
        tr += vi8 * (notorious_fft_real)8.132897407355648145e-01; ti += vr8 * (notorious_fft_real)8.132897407355648145e-01;
        sr += ur9 * (notorious_fft_real)1.093712083778735494e-01; si += ui9 * (notorious_fft_real)1.093712083778735494e-01;
        tr += vi9 * (notorious_fft_real)-9.940009752399460163e-01; ti += vr9 * (notorious_fft_real)-9.940009752399460163e-01;
        sr += ur10 * (notorious_fft_real)3.911047204901550445e-01; si += ui10 * (notorious_fft_real)3.911047204901550445e-01;
        tr += vi10 * (notorious_fft_real)9.203461835691598703e-01; ti += vr10 * (notorious_fft_real)9.203461835691598703e-01;
        sr += ur11 * (notorious_fft_real)-7.914964884292523539e-01; si += ui11 * (notorious_fft_real)-7.914964884292523539e-01;
        tr += vi11 * (notorious_fft_real)-6.111737140978515948e-01; ti += vr11 * (notorious_fft_real)-6.111737140978515948e-01;
        sr += ur12 * (notorious_fft_real)9.893433680751105852e-01; si += ui12 * (notorious_fft_real)9.893433680751105852e-01;
        tr += vi12 * (notorious_fft_real)1.456011677350025901e-01; ti += vr12 * (notorious_fft_real)1.456011677350025901e-01;
        sr += ur13 * (notorious_fft_real)-9.340161087325465905e-01; si += ui13 * (notorious_fft_real)-9.340161087325465905e-01;
        tr += vi13 * (notorious_fft_real)3.572308898011364864e-01; ti += vr13 * (notorious_fft_real)3.572308898011364864e-01;
        sr += ur14 * (notorious_fft_real)6.396730215588896096e-01; si += ui14 * (notorious_fft_real)6.396730215588896096e-01;
        tr += vi14 * (notorious_fft_real)-7.686471397785333837e-01; ti += vr14 * (notorious_fft_real)-7.686471397785333837e-01;
        sr += ur15 * (notorious_fft_real)-1.816368509794362152e-01; si += ui15 * (notorious_fft_real)-1.816368509794362152e-01;
        tr += vi15 * (notorious_fft_real)9.833656768294661843e-01; ti += vr15 * (notorious_fft_real)9.833656768294661843e-01;
        sr += ur16 * (notorious_fft_real)-3.228804047714446401e-01; si += ui16 * (notorious_fft_real)-3.228804047714446401e-01;
        tr += vi16 * (notorious_fft_real)-9.464397731576098538e-01; ti += vr16 * (notorious_fft_real)-9.464397731576098538e-01;
        sr += ur17 * (notorious_fft_real)7.447721827437842057e-01; si += ui17 * (notorious_fft_real)7.447721827437842057e-01;
        tr += vi17 * (notorious_fft_real)6.673188112222367963e-01; ti += vr17 * (notorious_fft_real)6.673188112222367963e-01;
        sr += ur18 * (notorious_fft_real)-9.760758775559275069e-01; si += ui18 * (notorious_fft_real)-9.760758775559275069e-01;
        tr += vi18 * (notorious_fft_real)-2.174301755815553350e-01; ti += vr18 * (notorious_fft_real)-2.174301755815553350e-01;
        sr += ur19 * (notorious_fft_real)9.576005999084059583e-01; si += ui19 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi19 * (notorious_fft_real)-2.880990993652373655e-01; ti += vr19 * (notorious_fft_real)-2.880990993652373655e-01;
        sr += ur20 * (notorious_fft_real)-6.940741952206354037e-01; si += ui20 * (notorious_fft_real)-6.940741952206354037e-01;
        tr += vi20 * (notorious_fft_real)7.199034737579943499e-01; ti += vr20 * (notorious_fft_real)7.199034737579943499e-01;
        sr += ur21 * (notorious_fft_real)2.529333823916845581e-01; si += ui21 * (notorious_fft_real)2.529333823916845581e-01;
        tr += vi21 * (notorious_fft_real)-9.674836970574242079e-01; ti += vr21 * (notorious_fft_real)-9.674836970574242079e-01;
        x[18*st]         = xr0 + sr - sg * tr;
        x[18*st + 1]     = xi0 + si + sg * ti;
        x[25*st]       = xr0 + sr + sg * tr;
        x[25*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.340161087325479228e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.340161087325479228e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)3.572308898011328782e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)3.572308898011328782e-01;
        sr += ur2 * (notorious_fft_real)7.447721827437817632e-01; si += ui2 * (notorious_fft_real)7.447721827437817632e-01;
        tr += vi2 * (notorious_fft_real)-6.673188112222395718e-01; ti += vr2 * (notorious_fft_real)-6.673188112222395718e-01;
        sr += ur3 * (notorious_fft_real)-4.572423233046374880e-01; si += ui3 * (notorious_fft_real)-4.572423233046374880e-01;
        tr += vi3 * (notorious_fft_real)8.893421488825193899e-01; ti += vr3 * (notorious_fft_real)8.893421488825193899e-01;
        sr += ur4 * (notorious_fft_real)1.093712083778740352e-01; si += ui4 * (notorious_fft_real)1.093712083778740352e-01;
        tr += vi4 * (notorious_fft_real)-9.940009752399459053e-01; ti += vr4 * (notorious_fft_real)-9.940009752399459053e-01;
        sr += ur5 * (notorious_fft_real)2.529333823916802837e-01; si += ui5 * (notorious_fft_real)2.529333823916802837e-01;
        tr += vi5 * (notorious_fft_real)9.674836970574253181e-01; ti += vr5 * (notorious_fft_real)9.674836970574253181e-01;
        sr += ur6 * (notorious_fft_real)-5.818589155579546590e-01; si += ui6 * (notorious_fft_real)-5.818589155579546590e-01;
        tr += vi6 * (notorious_fft_real)-8.132897407355640373e-01; ti += vr6 * (notorious_fft_real)-8.132897407355640373e-01;
        sr += ur7 * (notorious_fft_real)8.339978178898788119e-01; si += ui7 * (notorious_fft_real)8.339978178898788119e-01;
        tr += vi7 * (notorious_fft_real)5.517677407704446590e-01; ti += vr7 * (notorious_fft_real)5.517677407704446590e-01;
        sr += ur8 * (notorious_fft_real)-9.760758775559272848e-01; si += ui8 * (notorious_fft_real)-9.760758775559272848e-01;
        tr += vi8 * (notorious_fft_real)-2.174301755815562787e-01; ti += vr8 * (notorious_fft_real)-2.174301755815562787e-01;
        sr += ur9 * (notorious_fft_real)9.893433680751102521e-01; si += ui9 * (notorious_fft_real)9.893433680751102521e-01;
        tr += vi9 * (notorious_fft_real)-1.456011677350047828e-01; ti += vr9 * (notorious_fft_real)-1.456011677350047828e-01;
        sr += ur10 * (notorious_fft_real)-8.720494081438080469e-01; si += ui10 * (notorious_fft_real)-8.720494081438080469e-01;
        tr += vi10 * (notorious_fft_real)4.894178478110847164e-01; ti += vr10 * (notorious_fft_real)4.894178478110847164e-01;
        sr += ur11 * (notorious_fft_real)6.396730215588871671e-01; si += ui11 * (notorious_fft_real)6.396730215588871671e-01;
        tr += vi11 * (notorious_fft_real)-7.686471397785354931e-01; ti += vr11 * (notorious_fft_real)-7.686471397785354931e-01;
        sr += ur12 * (notorious_fft_real)-3.228804047714418646e-01; si += ui12 * (notorious_fft_real)-3.228804047714418646e-01;
        tr += vi12 * (notorious_fft_real)9.464397731576108530e-01; ti += vr12 * (notorious_fft_real)9.464397731576108530e-01;
        sr += ur13 * (notorious_fft_real)-3.652202305765908652e-02; si += ui13 * (notorious_fft_real)-3.652202305765908652e-02;
        tr += vi13 * (notorious_fft_real)-9.993328483702393905e-01; ti += vr13 * (notorious_fft_real)-9.993328483702393905e-01;
        sr += ur14 * (notorious_fft_real)3.911047204901587637e-01; si += ui14 * (notorious_fft_real)3.911047204901587637e-01;
        tr += vi14 * (notorious_fft_real)9.203461835691583159e-01; ti += vr14 * (notorious_fft_real)9.203461835691583159e-01;
        sr += ur15 * (notorious_fft_real)-6.940741952206380683e-01; si += ui15 * (notorious_fft_real)-6.940741952206380683e-01;
        tr += vi15 * (notorious_fft_real)-7.199034737579919074e-01; ti += vr15 * (notorious_fft_real)-7.199034737579919074e-01;
        sr += ur16 * (notorious_fft_real)9.054482374931471789e-01; si += ui16 * (notorious_fft_real)9.054482374931471789e-01;
        tr += vi16 * (notorious_fft_real)4.244566988758138315e-01; ti += vr16 * (notorious_fft_real)4.244566988758138315e-01;
        sr += ur17 * (notorious_fft_real)-9.973322836635519772e-01; si += ui17 * (notorious_fft_real)-9.973322836635519772e-01;
        tr += vi17 * (notorious_fft_real)-7.299531466090337928e-02; ti += vr17 * (notorious_fft_real)-7.299531466090337928e-02;
        sr += ur18 * (notorious_fft_real)9.576005999084059583e-01; si += ui18 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi18 * (notorious_fft_real)-2.880990993652373655e-01; ti += vr18 * (notorious_fft_real)-2.880990993652373655e-01;
        sr += ur19 * (notorious_fft_real)-7.914964884292482461e-01; si += ui19 * (notorious_fft_real)-7.914964884292482461e-01;
        tr += vi19 * (notorious_fft_real)6.111737140978569238e-01; ti += vr19 * (notorious_fft_real)6.111737140978569238e-01;
        sr += ur20 * (notorious_fft_real)5.209403404879318078e-01; si += ui20 * (notorious_fft_real)5.209403404879318078e-01;
        tr += vi20 * (notorious_fft_real)-8.535930890373455426e-01; ti += vr20 * (notorious_fft_real)-8.535930890373455426e-01;
        sr += ur21 * (notorious_fft_real)-1.816368509794354935e-01; si += ui21 * (notorious_fft_real)-1.816368509794354935e-01;
        tr += vi21 * (notorious_fft_real)9.833656768294662953e-01; ti += vr21 * (notorious_fft_real)9.833656768294662953e-01;
        x[19*st]         = xr0 + sr - sg * tr;
        x[19*st + 1]     = xi0 + si + sg * ti;
        x[24*st]       = xr0 + sr + sg * tr;
        x[24*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.760758775559270628e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.760758775559270628e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)2.174301755815572501e-01;
        notorious_fft_real ti = vr1 * (notorious_fft_real)2.174301755815572501e-01;
        sr += ur2 * (notorious_fft_real)9.054482374931464017e-01; si += ui2 * (notorious_fft_real)9.054482374931464017e-01;
        tr += vi2 * (notorious_fft_real)-4.244566988758156079e-01; ti += vr2 * (notorious_fft_real)-4.244566988758156079e-01;
        sr += ur3 * (notorious_fft_real)-7.914964884292535752e-01; si += ui3 * (notorious_fft_real)-7.914964884292535752e-01;
        tr += vi3 * (notorious_fft_real)6.111737140978499294e-01; ti += vr3 * (notorious_fft_real)6.111737140978499294e-01;
        sr += ur4 * (notorious_fft_real)6.396730215588903867e-01; si += ui4 * (notorious_fft_real)6.396730215588903867e-01;
        tr += vi4 * (notorious_fft_real)-7.686471397785328286e-01; ti += vr4 * (notorious_fft_real)-7.686471397785328286e-01;
        sr += ur5 * (notorious_fft_real)-4.572423233046388757e-01; si += ui5 * (notorious_fft_real)-4.572423233046388757e-01;
        tr += vi5 * (notorious_fft_real)8.893421488825187238e-01; ti += vr5 * (notorious_fft_real)8.893421488825187238e-01;
        sr += ur6 * (notorious_fft_real)2.529333823916791180e-01; si += ui6 * (notorious_fft_real)2.529333823916791180e-01;
        tr += vi6 * (notorious_fft_real)-9.674836970574256512e-01; ti += vr6 * (notorious_fft_real)-9.674836970574256512e-01;
        sr += ur7 * (notorious_fft_real)-3.652202305765687995e-02; si += ui7 * (notorious_fft_real)-3.652202305765687995e-02;
        tr += vi7 * (notorious_fft_real)9.993328483702393905e-01; ti += vr7 * (notorious_fft_real)9.993328483702393905e-01;
        sr += ur8 * (notorious_fft_real)-1.816368509794386299e-01; si += ui8 * (notorious_fft_real)-1.816368509794386299e-01;
        tr += vi8 * (notorious_fft_real)-9.833656768294657402e-01; ti += vr8 * (notorious_fft_real)-9.833656768294657402e-01;
        sr += ur9 * (notorious_fft_real)3.911047204901550445e-01; si += ui9 * (notorious_fft_real)3.911047204901550445e-01;
        tr += vi9 * (notorious_fft_real)9.203461835691598703e-01; ti += vr9 * (notorious_fft_real)9.203461835691598703e-01;
        sr += ur10 * (notorious_fft_real)-5.818589155579522165e-01; si += ui10 * (notorious_fft_real)-5.818589155579522165e-01;
        tr += vi10 * (notorious_fft_real)-8.132897407355658137e-01; ti += vr10 * (notorious_fft_real)-8.132897407355658137e-01;
        sr += ur11 * (notorious_fft_real)7.447721827437792097e-01; si += ui11 * (notorious_fft_real)7.447721827437792097e-01;
        tr += vi11 * (notorious_fft_real)6.673188112222424584e-01; ti += vr11 * (notorious_fft_real)6.673188112222424584e-01;
        sr += ur12 * (notorious_fft_real)-8.720494081438092682e-01; si += ui12 * (notorious_fft_real)-8.720494081438092682e-01;
        tr += vi12 * (notorious_fft_real)-4.894178478110826069e-01; ti += vr12 * (notorious_fft_real)-4.894178478110826069e-01;
        sr += ur13 * (notorious_fft_real)9.576005999084069575e-01; si += ui13 * (notorious_fft_real)9.576005999084069575e-01;
        tr += vi13 * (notorious_fft_real)2.880990993652340904e-01; ti += vr13 * (notorious_fft_real)2.880990993652340904e-01;
        sr += ur14 * (notorious_fft_real)-9.973322836635519772e-01; si += ui14 * (notorious_fft_real)-9.973322836635519772e-01;
        tr += vi14 * (notorious_fft_real)-7.299531466090362908e-02; ti += vr14 * (notorious_fft_real)-7.299531466090362908e-02;
        sr += ur15 * (notorious_fft_real)9.893433680751095860e-01; si += ui15 * (notorious_fft_real)9.893433680751095860e-01;
        tr += vi15 * (notorious_fft_real)-1.456011677350090017e-01; ti += vr15 * (notorious_fft_real)-1.456011677350090017e-01;
        sr += ur16 * (notorious_fft_real)-9.340161087325463685e-01; si += ui16 * (notorious_fft_real)-9.340161087325463685e-01;
        tr += vi16 * (notorious_fft_real)3.572308898011369305e-01; ti += vr16 * (notorious_fft_real)3.572308898011369305e-01;
        sr += ur17 * (notorious_fft_real)8.339978178898753702e-01; si += ui17 * (notorious_fft_real)8.339978178898753702e-01;
        tr += vi17 * (notorious_fft_real)-5.517677407704498771e-01; ti += vr17 * (notorious_fft_real)-5.517677407704498771e-01;
        sr += ur18 * (notorious_fft_real)-6.940741952206354037e-01; si += ui18 * (notorious_fft_real)-6.940741952206354037e-01;
        tr += vi18 * (notorious_fft_real)7.199034737579943499e-01; ti += vr18 * (notorious_fft_real)7.199034737579943499e-01;
        sr += ur19 * (notorious_fft_real)5.209403404879318078e-01; si += ui19 * (notorious_fft_real)5.209403404879318078e-01;
        tr += vi19 * (notorious_fft_real)-8.535930890373455426e-01; ti += vr19 * (notorious_fft_real)-8.535930890373455426e-01;
        sr += ur20 * (notorious_fft_real)-3.228804047714476377e-01; si += ui20 * (notorious_fft_real)-3.228804047714476377e-01;
        tr += vi20 * (notorious_fft_real)9.464397731576088546e-01; ti += vr20 * (notorious_fft_real)9.464397731576088546e-01;
        sr += ur21 * (notorious_fft_real)1.093712083778756172e-01; si += ui21 * (notorious_fft_real)1.093712083778756172e-01;
        tr += vi21 * (notorious_fft_real)-9.940009752399457943e-01; ti += vr21 * (notorious_fft_real)-9.940009752399457943e-01;
        x[20*st]         = xr0 + sr - sg * tr;
        x[20*st + 1]     = xi0 + si + sg * ti;
        x[23*st]       = xr0 + sr + sg * tr;
        x[23*st + 1]   = xi0 + si - sg * ti;
    }
    {
        notorious_fft_real sr = ur1 * (notorious_fft_real)-9.973322836635516442e-01;
        notorious_fft_real si = ui1 * (notorious_fft_real)-9.973322836635516442e-01;
        notorious_fft_real tr = vi1 * (notorious_fft_real)7.299531466090755649e-02;
        notorious_fft_real ti = vr1 * (notorious_fft_real)7.299531466090755649e-02;
        sr += ur2 * (notorious_fft_real)9.893433680751101411e-01; si += ui2 * (notorious_fft_real)9.893433680751101411e-01;
        tr += vi2 * (notorious_fft_real)-1.456011677350049216e-01; ti += vr2 * (notorious_fft_real)-1.456011677350049216e-01;
        sr += ur3 * (notorious_fft_real)-9.760758775559270628e-01; si += ui3 * (notorious_fft_real)-9.760758775559270628e-01;
        tr += vi3 * (notorious_fft_real)2.174301755815574722e-01; ti += vr3 * (notorious_fft_real)2.174301755815574722e-01;
        sr += ur4 * (notorious_fft_real)9.576005999084059583e-01; si += ui4 * (notorious_fft_real)9.576005999084059583e-01;
        tr += vi4 * (notorious_fft_real)-2.880990993652376986e-01; ti += vr4 * (notorious_fft_real)-2.880990993652376986e-01;
        sr += ur5 * (notorious_fft_real)-9.340161087325481448e-01; si += ui5 * (notorious_fft_real)-9.340161087325481448e-01;
        tr += vi5 * (notorious_fft_real)3.572308898011324896e-01; ti += vr5 * (notorious_fft_real)3.572308898011324896e-01;
        sr += ur6 * (notorious_fft_real)9.054482374931461797e-01; si += ui6 * (notorious_fft_real)9.054482374931461797e-01;
        tr += vi6 * (notorious_fft_real)-4.244566988758160520e-01; ti += vr6 * (notorious_fft_real)-4.244566988758160520e-01;
        sr += ur7 * (notorious_fft_real)-8.720494081438081579e-01; si += ui7 * (notorious_fft_real)-8.720494081438081579e-01;
        tr += vi7 * (notorious_fft_real)4.894178478110844943e-01; ti += vr7 * (notorious_fft_real)4.894178478110844943e-01;
        sr += ur8 * (notorious_fft_real)8.339978178898778127e-01; si += ui8 * (notorious_fft_real)8.339978178898778127e-01;
        tr += vi8 * (notorious_fft_real)-5.517677407704461023e-01; ti += vr8 * (notorious_fft_real)-5.517677407704461023e-01;
        sr += ur9 * (notorious_fft_real)-7.914964884292553515e-01; si += ui9 * (notorious_fft_real)-7.914964884292553515e-01;
        tr += vi9 * (notorious_fft_real)6.111737140978477090e-01; ti += vr9 * (notorious_fft_real)6.111737140978477090e-01;
        sr += ur10 * (notorious_fft_real)7.447721827437823183e-01; si += ui10 * (notorious_fft_real)7.447721827437823183e-01;
        tr += vi10 * (notorious_fft_real)-6.673188112222389057e-01; ti += vr10 * (notorious_fft_real)-6.673188112222389057e-01;
        sr += ur11 * (notorious_fft_real)-6.940741952206358478e-01; si += ui11 * (notorious_fft_real)-6.940741952206358478e-01;
        tr += vi11 * (notorious_fft_real)7.199034737579939058e-01; ti += vr11 * (notorious_fft_real)7.199034737579939058e-01;
        sr += ur12 * (notorious_fft_real)6.396730215588896096e-01; si += ui12 * (notorious_fft_real)6.396730215588896096e-01;
        tr += vi12 * (notorious_fft_real)-7.686471397785333837e-01; ti += vr12 * (notorious_fft_real)-7.686471397785333837e-01;
        sr += ur13 * (notorious_fft_real)-5.818589155579528827e-01; si += ui13 * (notorious_fft_real)-5.818589155579528827e-01;
        tr += vi13 * (notorious_fft_real)8.132897407355652586e-01; ti += vr13 * (notorious_fft_real)8.132897407355652586e-01;
        sr += ur14 * (notorious_fft_real)5.209403404879322519e-01; si += ui14 * (notorious_fft_real)5.209403404879322519e-01;
        tr += vi14 * (notorious_fft_real)-8.535930890373452096e-01; ti += vr14 * (notorious_fft_real)-8.535930890373452096e-01;
        sr += ur15 * (notorious_fft_real)-4.572423233046362112e-01; si += ui15 * (notorious_fft_real)-4.572423233046362112e-01;
        tr += vi15 * (notorious_fft_real)8.893421488825200560e-01; ti += vr15 * (notorious_fft_real)8.893421488825200560e-01;
        sr += ur16 * (notorious_fft_real)3.911047204901555996e-01; si += ui16 * (notorious_fft_real)3.911047204901555996e-01;
        tr += vi16 * (notorious_fft_real)-9.203461835691596482e-01; ti += vr16 * (notorious_fft_real)-9.203461835691596482e-01;
        sr += ur17 * (notorious_fft_real)-3.228804047714546321e-01; si += ui17 * (notorious_fft_real)-3.228804047714546321e-01;
        tr += vi17 * (notorious_fft_real)9.464397731576065231e-01; ti += vr17 * (notorious_fft_real)9.464397731576065231e-01;
        sr += ur18 * (notorious_fft_real)2.529333823916845581e-01; si += ui18 * (notorious_fft_real)2.529333823916845581e-01;
        tr += vi18 * (notorious_fft_real)-9.674836970574242079e-01; ti += vr18 * (notorious_fft_real)-9.674836970574242079e-01;
        sr += ur19 * (notorious_fft_real)-1.816368509794354935e-01; si += ui19 * (notorious_fft_real)-1.816368509794354935e-01;
        tr += vi19 * (notorious_fft_real)9.833656768294662953e-01; ti += vr19 * (notorious_fft_real)9.833656768294662953e-01;
        sr += ur20 * (notorious_fft_real)1.093712083778756172e-01; si += ui20 * (notorious_fft_real)1.093712083778756172e-01;
        tr += vi20 * (notorious_fft_real)-9.940009752399457943e-01; ti += vr20 * (notorious_fft_real)-9.940009752399457943e-01;
        sr += ur21 * (notorious_fft_real)-3.652202305766936996e-02; si += ui21 * (notorious_fft_real)-3.652202305766936996e-02;
        tr += vi21 * (notorious_fft_real)9.993328483702389464e-01; ti += vr21 * (notorious_fft_real)9.993328483702389464e-01;
        x[21*st]         = xr0 + sr - sg * tr;
        x[21*st + 1]     = xi0 + si + sg * ti;
        x[22*st]       = xr0 + sr + sg * tr;
        x[22*st + 1]   = xi0 + si - sg * ti;
    }
}

static void notorious_fft_codelet_stride(notorious_fft_real* x, int n, int s, int inverse) {
    switch (n) {
    case 3: notorious_fft_radix3_stride(x, s, inverse); break;
    case 5: notorious_fft_radix5_stride(x, s, inverse); break;
    case 7: notorious_fft_radix7_stride(x, s, inverse); break;
    case 9: notorious_fft_fft9_stride(x, s, inverse); break;
    case 11: notorious_fft_fft11_stride(x, s, inverse); break;
    case 13: notorious_fft_fft13_stride(x, s, inverse); break;
    case 15: notorious_fft_fft15_stride(x, s, inverse); break;
    case 17: notorious_fft_fft17_stride(x, s, inverse); break;
    case 19: notorious_fft_fft19_stride(x, s, inverse); break;
    case 21: notorious_fft_fft21_stride(x, s, inverse); break;
    case 27: notorious_fft_fft27_stride(x, s, inverse); break;
    case 43: notorious_fft_fft43_stride(x, s, inverse); break;
    default: break;
    }
}

/* ============================================================================
 * SIMD mixed-radix: vectorize twiddles and radix-r across consecutive k2.
 * Layout is interleaved complexes; tw_re/tw_im are split-complex (contiguous).
 * ============================================================================ */

static void notorious_fft_mixed_twiddle_scalar(
    notorious_fft_real* tmp, const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im, int r, int m, int inverse, int k2_begin)
{
    for (int n1 = 1; n1 < r; n1++) {
        for (int k2 = k2_begin; k2 < m; k2++) {
            size_t ti = (size_t)(n1 - 1) * (size_t)m + (size_t)k2;
            size_t oi = ((size_t)n1 * (size_t)m + (size_t)k2) * 2;
            notorious_fft_real wr = tw_re[ti];
            notorious_fft_real wi = inverse ? -tw_im[ti] : tw_im[ti];
            notorious_fft_real ur = tmp[oi], ui = tmp[oi + 1];
            tmp[oi]     = ur * wr - ui * wi;
            tmp[oi + 1] = ur * wi + ui * wr;
        }
    }
}

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)

static NOTORIOUS_FFT_INLINE void notorious_fft_load4_cx_avx2(
    const double* p, __m256d* re, __m256d* im)
{
    __m256d a = _mm256_loadu_pd(p);
    __m256d b = _mm256_loadu_pd(p + 4);
    __m256d ap = _mm256_permute4x64_pd(a, 0xD8); /* r0 r1 i0 i1 */
    __m256d bp = _mm256_permute4x64_pd(b, 0xD8); /* r2 r3 i2 i3 */
    *re = _mm256_permute2f128_pd(ap, bp, 0x20);
    *im = _mm256_permute2f128_pd(ap, bp, 0x31);
}

static NOTORIOUS_FFT_INLINE void notorious_fft_store4_cx_avx2(
    double* p, __m256d re, __m256d im)
{
    __m256d lo = _mm256_unpacklo_pd(re, im);
    __m256d hi = _mm256_unpackhi_pd(re, im);
    _mm256_storeu_pd(p,     _mm256_permute2f128_pd(lo, hi, 0x20));
    _mm256_storeu_pd(p + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
}

static NOTORIOUS_FFT_INLINE void notorious_fft_cmac4_avx2(
    __m256d* sr, __m256d* si, __m256d xr, __m256d xi, __m256d wr, __m256d wi)
{
    *sr = _mm256_fmadd_pd(xr, wr, _mm256_fnmadd_pd(xi, wi, *sr));
    *si = _mm256_fmadd_pd(xi, wr, _mm256_fmadd_pd(xr, wi, *si));
}

static void notorious_fft_mixed_twiddle_avx2(
    notorious_fft_real* tmp, const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im, int r, int m, int inverse)
{
    int k2 = 0;
    for (int n1 = 1; n1 < r; n1++) {
        notorious_fft_real* row = tmp + (size_t)n1 * (size_t)m * 2;
        const notorious_fft_real* wrp = tw_re + (size_t)(n1 - 1) * (size_t)m;
        const notorious_fft_real* wip = tw_im + (size_t)(n1 - 1) * (size_t)m;
        k2 = 0;
        for (; k2 + 4 <= m; k2 += 4) {
            __m256d xr, xi;
            notorious_fft_load4_cx_avx2(row + 2 * k2, &xr, &xi);
            __m256d wr = _mm256_loadu_pd(wrp + k2);
            __m256d wi = _mm256_loadu_pd(wip + k2);
            if (inverse) wi = _mm256_xor_pd(wi, _mm256_set1_pd(-0.0));
            __m256d nr = _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi));
            __m256d ni = _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr));
            notorious_fft_store4_cx_avx2(row + 2 * k2, nr, ni);
        }
    }
    if (k2 < m)
        notorious_fft_mixed_twiddle_scalar(tmp, tw_re, tw_im, r, m, inverse, k2);
}

static void notorious_fft_radix3_cols_avx2(notorious_fft_real* tmp, int m, int inverse)
{
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vsig = _mm256_set1_pd(inverse
        ? 0.866025403784438646763723170752936183
        : -0.866025403784438646763723170752936183);
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 4 <= m; k2 += 4) {
        double* p0 = tmp + 2 * k2;
        __m256d x0r, x0i, x1r, x1i, x2r, x2i;
        notorious_fft_load4_cx_avx2(p0, &x0r, &x0i);
        notorious_fft_load4_cx_avx2(p0 + st, &x1r, &x1i);
        notorious_fft_load4_cx_avx2(p0 + 2 * st, &x2r, &x2i);
        __m256d ur = _mm256_add_pd(x1r, x2r);
        __m256d ui = _mm256_add_pd(x1i, x2i);
        __m256d vr = _mm256_sub_pd(x1r, x2r);
        __m256d vi = _mm256_sub_pd(x1i, x2i);
        notorious_fft_store4_cx_avx2(p0, _mm256_add_pd(x0r, ur), _mm256_add_pd(x0i, ui));
        __m256d t1r = _mm256_fnmadd_pd(ur, vhalf, x0r);
        __m256d t1i = _mm256_fnmadd_pd(ui, vhalf, x0i);
        notorious_fft_store4_cx_avx2(p0 + st,
            _mm256_fnmadd_pd(vi, vsig, t1r),
            _mm256_fmadd_pd(vr, vsig, t1i));
        notorious_fft_store4_cx_avx2(p0 + 2 * st,
            _mm256_fmadd_pd(vi, vsig, t1r),
            _mm256_fnmadd_pd(vr, vsig, t1i));
    }
    for (; k2 < m; k2++)
        notorious_fft_radix3_stride(tmp + 2 * k2, m, inverse);
}

static void notorious_fft_radixn_cols_avx2(
    notorious_fft_real* tmp, int r, int m, int inverse,
    const notorious_fft_real* wr, const notorious_fft_real* wi)
{
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 4 <= m; k2 += 4) {
        double* p0 = tmp + 2 * k2;
        __m256d xr[7], xi[7];
        for (int n = 0; n < r; n++)
            notorious_fft_load4_cx_avx2(p0 + n * st, &xr[n], &xi[n]);
        for (int k = 0; k < r; k++) {
            __m256d sr = xr[0], si = xi[0];
            for (int n = 1; n < r; n++) {
                int p = (k * n) % r;
                __m256d wR = _mm256_set1_pd(wr[p]);
                __m256d wI = _mm256_set1_pd(wi[p]);
                notorious_fft_cmac4_avx2(&sr, &si, xr[n], xi[n], wR, wI);
            }
            notorious_fft_store4_cx_avx2(p0 + k * st, sr, si);
        }
    }
    for (; k2 < m; k2++) {
        if (r == 5)
            notorious_fft_radix5_stride(tmp + 2 * k2, m, inverse);
        else
            notorious_fft_radix7_stride(tmp + 2 * k2, m, inverse);
    }
}

#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)

static NOTORIOUS_FFT_INLINE void notorious_fft_load2_cx_neon(
    const double* p, float64x2_t* re, float64x2_t* im)
{
    float64x2x2_t v = vld2q_f64(p);
    *re = v.val[0];
    *im = v.val[1];
}

static NOTORIOUS_FFT_INLINE void notorious_fft_store2_cx_neon(
    double* p, float64x2_t re, float64x2_t im)
{
    float64x2x2_t v;
    v.val[0] = re;
    v.val[1] = im;
    vst2q_f64(p, v);
}

static NOTORIOUS_FFT_INLINE void notorious_fft_cmac2_neon(
    float64x2_t* sr, float64x2_t* si, float64x2_t xr, float64x2_t xi,
    float64x2_t wr, float64x2_t wi)
{
    *sr = vfmaq_f64(vmlsq_f64(*sr, xi, wi), xr, wr);
    *si = vfmaq_f64(vfmaq_f64(*si, xr, wi), xi, wr);
}

static void notorious_fft_mixed_twiddle_neon(
    notorious_fft_real* tmp, const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im, int r, int m, int inverse)
{
    int k2 = 0;
    for (int n1 = 1; n1 < r; n1++) {
        notorious_fft_real* row = tmp + (size_t)n1 * (size_t)m * 2;
        const notorious_fft_real* wrp = tw_re + (size_t)(n1 - 1) * (size_t)m;
        const notorious_fft_real* wip = tw_im + (size_t)(n1 - 1) * (size_t)m;
        k2 = 0;
        for (; k2 + 2 <= m; k2 += 2) {
            float64x2_t xr, xi;
            notorious_fft_load2_cx_neon(row + 2 * k2, &xr, &xi);
            float64x2_t wr = vld1q_f64(wrp + k2);
            float64x2_t wi = vld1q_f64(wip + k2);
            if (inverse) wi = vnegq_f64(wi);
            float64x2_t nr = vmlsq_f64(vmulq_f64(xr, wr), xi, wi);
            float64x2_t ni = vfmaq_f64(vmulq_f64(xr, wi), xi, wr);
            notorious_fft_store2_cx_neon(row + 2 * k2, nr, ni);
        }
    }
    if (k2 < m)
        notorious_fft_mixed_twiddle_scalar(tmp, tw_re, tw_im, r, m, inverse, k2);
}

static void notorious_fft_radix3_cols_neon(notorious_fft_real* tmp, int m, int inverse)
{
    const float64x2_t vhalf = vdupq_n_f64(0.5);
    const float64x2_t vsig = vdupq_n_f64(inverse
        ? 0.866025403784438646763723170752936183
        : -0.866025403784438646763723170752936183);
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 2 <= m; k2 += 2) {
        double* p0 = tmp + 2 * k2;
        float64x2_t x0r, x0i, x1r, x1i, x2r, x2i;
        notorious_fft_load2_cx_neon(p0, &x0r, &x0i);
        notorious_fft_load2_cx_neon(p0 + st, &x1r, &x1i);
        notorious_fft_load2_cx_neon(p0 + 2 * st, &x2r, &x2i);
        float64x2_t ur = vaddq_f64(x1r, x2r);
        float64x2_t ui = vaddq_f64(x1i, x2i);
        float64x2_t vr = vsubq_f64(x1r, x2r);
        float64x2_t vi = vsubq_f64(x1i, x2i);
        notorious_fft_store2_cx_neon(p0, vaddq_f64(x0r, ur), vaddq_f64(x0i, ui));
        float64x2_t t1r = vmlsq_f64(x0r, ur, vhalf);
        float64x2_t t1i = vmlsq_f64(x0i, ui, vhalf);
        notorious_fft_store2_cx_neon(p0 + st,
            vmlsq_f64(t1r, vi, vsig),
            vfmaq_f64(t1i, vr, vsig));
        notorious_fft_store2_cx_neon(p0 + 2 * st,
            vfmaq_f64(t1r, vi, vsig),
            vmlsq_f64(t1i, vr, vsig));
    }
    for (; k2 < m; k2++)
        notorious_fft_radix3_stride(tmp + 2 * k2, m, inverse);
}

static void notorious_fft_radixn_cols_neon(
    notorious_fft_real* tmp, int r, int m, int inverse,
    const notorious_fft_real* wr, const notorious_fft_real* wi)
{
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 2 <= m; k2 += 2) {
        double* p0 = tmp + 2 * k2;
        float64x2_t xr[7], xi[7];
        for (int n = 0; n < r; n++)
            notorious_fft_load2_cx_neon(p0 + n * st, &xr[n], &xi[n]);
        for (int k = 0; k < r; k++) {
            float64x2_t sr = xr[0], si = xi[0];
            for (int n = 1; n < r; n++) {
                int p = (k * n) % r;
                float64x2_t wR = vdupq_n_f64(wr[p]);
                float64x2_t wI = vdupq_n_f64(wi[p]);
                notorious_fft_cmac2_neon(&sr, &si, xr[n], xi[n], wR, wI);
            }
            notorious_fft_store2_cx_neon(p0 + k * st, sr, si);
        }
    }
    for (; k2 < m; k2++) {
        if (r == 5)
            notorious_fft_radix5_stride(tmp + 2 * k2, m, inverse);
        else
            notorious_fft_radix7_stride(tmp + 2 * k2, m, inverse);
    }
}

#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)

static NOTORIOUS_FFT_INLINE void notorious_fft_load4_cx_neon_f32(
    const float* p, float32x4_t* re, float32x4_t* im)
{
    float32x4x2_t v = vld2q_f32(p);
    *re = v.val[0];
    *im = v.val[1];
}

static NOTORIOUS_FFT_INLINE void notorious_fft_store4_cx_neon_f32(
    float* p, float32x4_t re, float32x4_t im)
{
    float32x4x2_t v;
    v.val[0] = re;
    v.val[1] = im;
    vst2q_f32(p, v);
}

static NOTORIOUS_FFT_INLINE void notorious_fft_cmac4_neon_f32(
    float32x4_t* sr, float32x4_t* si, float32x4_t xr, float32x4_t xi,
    float32x4_t wr, float32x4_t wi)
{
    *sr = vfmaq_f32(vmlsq_f32(*sr, xi, wi), xr, wr);
    *si = vfmaq_f32(vfmaq_f32(*si, xr, wi), xi, wr);
}

static void notorious_fft_mixed_twiddle_neon_f32(
    notorious_fft_real* tmp, const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im, int r, int m, int inverse)
{
    int k2 = 0;
    for (int n1 = 1; n1 < r; n1++) {
        notorious_fft_real* row = tmp + (size_t)n1 * (size_t)m * 2;
        const notorious_fft_real* wrp = tw_re + (size_t)(n1 - 1) * (size_t)m;
        const notorious_fft_real* wip = tw_im + (size_t)(n1 - 1) * (size_t)m;
        k2 = 0;
        for (; k2 + 4 <= m; k2 += 4) {
            float32x4_t xr, xi;
            notorious_fft_load4_cx_neon_f32(row + 2 * k2, &xr, &xi);
            float32x4_t wr = vld1q_f32(wrp + k2);
            float32x4_t wi = vld1q_f32(wip + k2);
            if (inverse) wi = vnegq_f32(wi);
            float32x4_t nr = vmlsq_f32(vmulq_f32(xr, wr), xi, wi);
            float32x4_t ni = vfmaq_f32(vmulq_f32(xr, wi), xi, wr);
            notorious_fft_store4_cx_neon_f32(row + 2 * k2, nr, ni);
        }
    }
    if (k2 < m)
        notorious_fft_mixed_twiddle_scalar(tmp, tw_re, tw_im, r, m, inverse, k2);
}

static void notorious_fft_radix3_cols_neon_f32(notorious_fft_real* tmp, int m, int inverse)
{
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t vsig = vdupq_n_f32(inverse
        ? 0.866025403784438646763723170752936183f
        : -0.866025403784438646763723170752936183f);
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 4 <= m; k2 += 4) {
        float* p0 = tmp + 2 * k2;
        float32x4_t x0r, x0i, x1r, x1i, x2r, x2i;
        notorious_fft_load4_cx_neon_f32(p0, &x0r, &x0i);
        notorious_fft_load4_cx_neon_f32(p0 + st, &x1r, &x1i);
        notorious_fft_load4_cx_neon_f32(p0 + 2 * st, &x2r, &x2i);
        float32x4_t ur = vaddq_f32(x1r, x2r);
        float32x4_t ui = vaddq_f32(x1i, x2i);
        float32x4_t vr = vsubq_f32(x1r, x2r);
        float32x4_t vi = vsubq_f32(x1i, x2i);
        notorious_fft_store4_cx_neon_f32(p0, vaddq_f32(x0r, ur), vaddq_f32(x0i, ui));
        float32x4_t t1r = vmlsq_f32(x0r, ur, vhalf);
        float32x4_t t1i = vmlsq_f32(x0i, ui, vhalf);
        notorious_fft_store4_cx_neon_f32(p0 + st,
            vmlsq_f32(t1r, vi, vsig),
            vfmaq_f32(t1i, vr, vsig));
        notorious_fft_store4_cx_neon_f32(p0 + 2 * st,
            vfmaq_f32(t1r, vi, vsig),
            vmlsq_f32(t1i, vr, vsig));
    }
    for (; k2 < m; k2++)
        notorious_fft_radix3_stride(tmp + 2 * k2, m, inverse);
}

static void notorious_fft_radixn_cols_neon_f32(
    notorious_fft_real* tmp, int r, int m, int inverse,
    const notorious_fft_real* wr, const notorious_fft_real* wi)
{
    const int st = 2 * m;
    int k2 = 0;
    for (; k2 + 4 <= m; k2 += 4) {
        float* p0 = tmp + 2 * k2;
        float32x4_t xr[7], xi[7];
        for (int n = 0; n < r; n++)
            notorious_fft_load4_cx_neon_f32(p0 + n * st, &xr[n], &xi[n]);
        for (int k = 0; k < r; k++) {
            float32x4_t sr = xr[0], si = xi[0];
            for (int n = 1; n < r; n++) {
                int p = (k * n) % r;
                float32x4_t wR = vdupq_n_f32(wr[p]);
                float32x4_t wI = vdupq_n_f32(wi[p]);
                notorious_fft_cmac4_neon_f32(&sr, &si, xr[n], xi[n], wR, wI);
            }
            notorious_fft_store4_cx_neon_f32(p0 + k * st, sr, si);
        }
    }
    for (; k2 < m; k2++) {
        if (r == 5)
            notorious_fft_radix5_stride(tmp + 2 * k2, m, inverse);
        else
            notorious_fft_radix7_stride(tmp + 2 * k2, m, inverse);
    }
}

#endif /* mixed-radix SIMD ISAs */

static void notorious_fft_mixed_apply_twiddles(
    notorious_fft_real* tmp, const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im, int r, int m, int inverse)
{
    if (!tw_re || !tw_im || r < 2 || m < 1) return;
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    notorious_fft_mixed_twiddle_avx2(tmp, tw_re, tw_im, r, m, inverse);
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    notorious_fft_mixed_twiddle_neon(tmp, tw_re, tw_im, r, m, inverse);
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
    notorious_fft_mixed_twiddle_neon_f32(tmp, tw_re, tw_im, r, m, inverse);
#else
    notorious_fft_mixed_twiddle_scalar(tmp, tw_re, tw_im, r, m, inverse, 0);
#endif
}

static void notorious_fft_mixed_radix_across(
    notorious_fft_real* tmp, int r, int m, int inverse)
{
    if (r == 3) {
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radix3_cols_avx2(tmp, m, inverse);
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radix3_cols_neon(tmp, m, inverse);
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radix3_cols_neon_f32(tmp, m, inverse);
#else
        for (int k2 = 0; k2 < m; k2++)
            notorious_fft_radix3_stride(tmp + 2 * k2, m, inverse);
#endif
        return;
    }

    notorious_fft_real wr[7], wi[7];
    const notorious_fft_real sg = inverse ? (notorious_fft_real)1.0 : (notorious_fft_real)-1.0;
    wr[0] = (notorious_fft_real)1.0;
    wi[0] = (notorious_fft_real)0.0;
    if (r == 5) {
        wr[1] = (notorious_fft_real)0.309016994374947424102293417182819059;
        wi[1] = sg * (notorious_fft_real)0.951056516295153572116439333379382143;
        wr[2] = (notorious_fft_real)-0.809016994374947424102293417182819059;
        wi[2] = sg * (notorious_fft_real)0.587785252292473129168705954639072768;
        wr[3] = wr[2]; wi[3] = -wi[2];
        wr[4] = wr[1]; wi[4] = -wi[1];
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_avx2(tmp, 5, m, inverse, wr, wi);
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_neon(tmp, 5, m, inverse, wr, wi);
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_neon_f32(tmp, 5, m, inverse, wr, wi);
#else
        for (int k2 = 0; k2 < m; k2++)
            notorious_fft_radix5_stride(tmp + 2 * k2, m, inverse);
#endif
    } else {
        wr[1] = (notorious_fft_real)0.623489801858733530525004884004239737;
        wi[1] = sg * (notorious_fft_real)0.781831482468029808708444526674057751;
        wr[2] = (notorious_fft_real)-0.222520933956314404288902564496794972;
        wi[2] = sg * (notorious_fft_real)0.974927912181823607018131682993903878;
        wr[3] = (notorious_fft_real)-0.900968867902419126236102319507445351;
        wi[3] = sg * (notorious_fft_real)0.433883739117558120475768332848358754;
        wr[4] = wr[3]; wi[4] = -wi[3];
        wr[5] = wr[2]; wi[5] = -wi[2];
        wr[6] = wr[1]; wi[6] = -wi[1];
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_avx2(tmp, 7, m, inverse, wr, wi);
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_neon(tmp, 7, m, inverse, wr, wi);
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
        notorious_fft_radixn_cols_neon_f32(tmp, 7, m, inverse, wr, wi);
#else
        for (int k2 = 0; k2 < m; k2++)
            notorious_fft_radix7_stride(tmp + 2 * k2, m, inverse);
#endif
    }
}

#endif /* NOTORIOUS_FFT_KERNELS_H */
