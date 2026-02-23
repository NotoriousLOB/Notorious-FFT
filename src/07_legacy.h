/*
 * Notorious FFT - Legacy API
 *
 * All twiddles precomputed at plan time,
 * all scratch buffers preallocated — zero malloc at transform time.
 */

#ifndef NOTORIOUS_FFT_LEGACY_H
#define NOTORIOUS_FFT_LEGACY_H

#include "06_plan.h"

/* ============================================================================
 * Complex Number Type for API Compatibility
 * ============================================================================ */

/* Interleaved (re, im) pair
 * In C++ we use a plain struct so no C99 _Complex dependency is needed. */
#ifdef __cplusplus
    typedef struct { notorious_fft_real re, im; } notorious_fft_cmpl;
#else
    typedef notorious_fft_real notorious_fft_cmpl[2];
#endif

/* ============================================================================
 * Transform Function Prototypes
 * ============================================================================ */

void notorious_fft_dft(notorious_fft_cmpl* x, notorious_fft_cmpl* y, const notorious_fft_aux* a);
void notorious_fft_invdft(notorious_fft_cmpl* x, notorious_fft_cmpl* y, const notorious_fft_aux* a);
void notorious_fft_realdft(notorious_fft_real* x, notorious_fft_cmpl* z, const notorious_fft_aux* a);
void notorious_fft_invrealdft(notorious_fft_cmpl* z, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dct2(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dst2(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dct3(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dst3(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dct4(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);
void notorious_fft_dst4(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a);

notorious_fft_aux* notorious_fft_mkaux_dft_1d(int N);
notorious_fft_aux* notorious_fft_mkaux_dft_2d(int N1, int N2);
notorious_fft_aux* notorious_fft_mkaux_dft_3d(int N1, int N2, int N3);
notorious_fft_aux* notorious_fft_mkaux_dft(int d, int* Ns);
notorious_fft_aux* notorious_fft_mkaux_realdft_1d(int N);
notorious_fft_aux* notorious_fft_mkaux_realdft_2d(int N1, int N2);
notorious_fft_aux* notorious_fft_mkaux_realdft_3d(int N1, int N2, int N3);
notorious_fft_aux* notorious_fft_mkaux_realdft(int d, int* Ns);
notorious_fft_aux* notorious_fft_mkaux_t2t3_1d(int N);
notorious_fft_aux* notorious_fft_mkaux_t2t3_2d(int N1, int N2);
notorious_fft_aux* notorious_fft_mkaux_t2t3_3d(int N1, int N2, int N3);
notorious_fft_aux* notorious_fft_mkaux_t2t3(int d, int* Ns);
notorious_fft_aux* notorious_fft_mkaux_t4_1d(int N);
notorious_fft_aux* notorious_fft_mkaux_t4_2d(int N1, int N2);
notorious_fft_aux* notorious_fft_mkaux_t4_3d(int N1, int N2, int N3);
notorious_fft_aux* notorious_fft_mkaux_t4(int d, int* Ns);

void notorious_fft_free_aux(notorious_fft_aux* a);

/* ============================================================================
 * Implementation
 * ============================================================================ */

#ifdef NOTORIOUS_FFT_IMPLEMENTATION

/* C++ compatible cast */
#ifdef __cplusplus
    #define NOTORIOUS_FFT_CAST(T, x) static_cast<T>(x)
#else
    #define NOTORIOUS_FFT_CAST(T, x) (x)
#endif

/* 1D DFT operating directly on interleaved complex — no deinterleave round-trip */
static void notorious_fft_dft_1d_cx(notorious_fft_cmpl* x, notorious_fft_cmpl* y, int N,
                              const notorious_fft_plan* plan, int inverse) {
    notorious_fft_execute_cx(plan, (notorious_fft_real*)x, (notorious_fft_real*)y, inverse);
}

/* Strided 1D DFT: gather → contiguous FFT → scatter.
 * Matches minfft's s_dft_1d(x, y, sy, a) interface.
 * All pointers are in units of notorious_fft_cmpl (2 reals). */
static void notorious_fft_s_dft_1d(notorious_fft_cmpl* x, notorious_fft_cmpl* y, int sy,
                              const notorious_fft_aux* a, int inverse) {
    int N = a->N;
    notorious_fft_plan* plan = a->plan;
    if (!plan) return;

    if (sy == 1) {
        notorious_fft_dft_1d_cx(x, y, N, plan, inverse);
    } else {
        notorious_fft_cmpl* buf = (notorious_fft_cmpl*)malloc((size_t)N * sizeof(notorious_fft_cmpl));
        if (!buf) return;
        notorious_fft_dft_1d_cx(x, buf, N, plan, inverse);
        notorious_fft_real* br = (notorious_fft_real*)buf;
        notorious_fft_real* yr = (notorious_fft_real*)y;
        for (int i = 0; i < N; i++) {
            yr[2 * i * sy]     = br[2 * i];
            yr[2 * i * sy + 1] = br[2 * i + 1];
        }
        free(buf);
    }
}

/* Recursive strided multi-dimensional DFT — mirrors minfft's mkcx().
 *
 * Aux tree structure (from notorious_fft_make_aux):
 *   d==1 wrapper: plan=NULL, sub1=NULL, sub2=1D_aux (has plan)
 *   d>1 node:     plan=NULL, sub1=make_aux(d-1), sub2=1D_aux (has plan), t=temp buf
 *   direct 1D:    plan!=NULL (from mkaux_dft_1d called directly) */
static void notorious_fft_mkcx(notorious_fft_cmpl* x, notorious_fft_cmpl* y, int sy,
                          const notorious_fft_aux* a, int inverse) {
    /* Direct 1D aux (plan set, no sub-structures) */
    if (a->plan) {
        notorious_fft_s_dft_1d(x, y, sy, a, inverse);
        return;
    }

    if (a->sub1 == NULL) {
        /* d==1 wrapper from make_aux: sub2 is the actual 1D aux */
        if (a->sub2)
            notorious_fft_s_dft_1d(x, y, sy, a->sub2, inverse);
        return;
    }

    /* Recursive case: d > 1
     * sub1 = (d-1)-dimensional aux for inner dimensions
     * sub2 = 1D aux for outermost dimension
     * N1 = product of inner dims, N2 = outermost dim size */
    int N1 = a->sub1->N;
    int N2 = a->sub2->N;
    notorious_fft_cmpl* t = (notorious_fft_cmpl*)a->t;

    /* Pass 1: transform each hyperplane (inner dims), writing transposed into t */
    /* No OpenMP — shared plan buffers (sr_t, work_re, t) are not thread-safe */
    for (int n = 0; n < N2; n++)
        notorious_fft_mkcx(x + n * N1, t + n, N2, a->sub1, inverse);

    /* Pass 2: transform outermost dimension (now contiguous rows in t) */
    for (int n = 0; n < N1; n++)
        notorious_fft_s_dft_1d(t + n * N2, y + sy * n, sy * N1, a->sub2, inverse);
}

/* ============================================================================
 * Complex DFT
 * ============================================================================ */

void notorious_fft_dft(notorious_fft_cmpl* x, notorious_fft_cmpl* y, const notorious_fft_aux* a) {
    notorious_fft_mkcx(x, y, 1, a, 0);
}

void notorious_fft_invdft(notorious_fft_cmpl* x, notorious_fft_cmpl* y, const notorious_fft_aux* a) {
    notorious_fft_mkcx(x, y, 1, a, 1);
}

/* ============================================================================
 * Real DFT — Precomputed twiddles, split-radix DIF inner FFT
 *
 * aux->e  = precomputed unpack twiddles: pairs (tw_r, tw_i) for k=0..N/4-1
 *           where tw_r = cos(-2πk/N), tw_i = sin(-2πk/N)
 * aux->t  = preallocated scratch buffer (N reals = N/2 interleaved complex)
 * ============================================================================ */

void notorious_fft_realdft(notorious_fft_real* x, notorious_fft_cmpl* z, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) {
        ((notorious_fft_real*)z)[0] = x[0];
        ((notorious_fft_real*)z)[1] = 0;
        return;
    }
    if (N == 2) {
        notorious_fft_real t0 = x[0] + x[1];
        notorious_fft_real t1 = x[0] - x[1];
        ((notorious_fft_real*)z)[0] = t0;
        ((notorious_fft_real*)z)[1] = 0;
        ((notorious_fft_real*)z)[2] = t1;
        ((notorious_fft_real*)z)[3] = 0;
        return;
    }

    int M = N / 2;
    notorious_fft_real* zr = (notorious_fft_real*)z;

    /* Use scratch buffer for the N/2-point complex FFT.
     * We need separate input/output because notorious_fft_execute_cx may not be in-place. */
    notorious_fft_real* scratch = (notorious_fft_real*)a->t;  /* N reals = M interleaved complex */

    /* Pack real input as interleaved complex: {x[0],x[1]}, {x[2],x[3]}, ... */
    memcpy(scratch, x, (size_t)N * sizeof(notorious_fft_real));

    /* N/2-point complex FFT via split-radix DIF */
    notorious_fft_execute_cx(a->plan, scratch, zr, 0 /* forward */);

    /* Unpack using precomputed twiddles from aux->e */
    const notorious_fft_real* tw = (const notorious_fft_real*)a->e;  /* pairs: (cos, sin) for k=0..N/4-1 */

    /* Save z[0] before overwriting */
    notorious_fft_real t0r = zr[0], t0i = zr[1];

    /* DC (k=0) */
    zr[0] = t0r + t0i;
    zr[1] = 0;

    /* Nyquist (k=N/2) */
    zr[2*M] = t0r - t0i;
    zr[2*M+1] = 0;

    /* Bins k = 1 to N/4 - 1 — precomputed twiddles, no trig calls */
    for (int k = 1; k < N/4; k++) {
        notorious_fft_real tkr = zr[2*k], tki = zr[2*k+1];
        notorious_fft_real t_n_2_k_r = zr[2*(M-k)], t_n_2_k_i = zr[2*(M-k)+1];

        notorious_fft_real ur = (tkr + t_n_2_k_r) * (notorious_fft_real)0.5;
        notorious_fft_real ui = (tki - t_n_2_k_i) * (notorious_fft_real)0.5;

        notorious_fft_real diff_r = tkr - t_n_2_k_r;
        notorious_fft_real diff_i = tki + t_n_2_k_i;

        notorious_fft_real tw_r = tw[2*k];
        notorious_fft_real tw_i = tw[2*k+1];

        notorious_fft_real prod_r = diff_r * tw_r - diff_i * tw_i;
        notorious_fft_real prod_i = diff_r * tw_i + diff_i * tw_r;

        notorious_fft_real vr = prod_i * (notorious_fft_real)0.5;
        notorious_fft_real vi = -prod_r * (notorious_fft_real)0.5;

        /* Write high bins first to avoid clobbering low bins */
        zr[2*(M-k)]   = ur - vr;
        zr[2*(M-k)+1] = -(ui - vi);
        zr[2*k]   = ur + vr;
        zr[2*k+1] = ui + vi;
    }

    /* z[N/4]: imaginary negated (conjugate of t[N/4]) */
    if (N >= 4) {
        zr[2*(N/4)+1] = -zr[2*(N/4)+1];
    }
}

void notorious_fft_invrealdft(notorious_fft_cmpl* z, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) {
        y[0] = ((notorious_fft_real*)z)[0];
        return;
    }
    if (N == 2) {
        notorious_fft_real t0 = ((notorious_fft_real*)z)[0];
        notorious_fft_real t1 = ((notorious_fft_real*)z)[2];
        y[0] = t0 + t1;
        y[1] = t0 - t1;
        return;
    }

    int M = N / 2;
    /* Use preallocated scratch buffer — no malloc */
    notorious_fft_real* ibuf = (notorious_fft_real*)a->t;  /* N reals = M interleaved complex */

    notorious_fft_real* zr = (notorious_fft_real*)z;

    /* Precomputed twiddles — same table as forward, but conjugated usage */
    const notorious_fft_real* tw = (const notorious_fft_real*)a->e;

    /* Unpack spectrum z[0..M] → interleaved complex for N/2-point IFFT */
    ibuf[0] = zr[0] + zr[2*M];
    ibuf[1] = zr[0] - zr[2*M];

    for (int k = 1; k < N/4; k++) {
        notorious_fft_real z_k_r = zr[2*k],     z_k_i = zr[2*k+1];
        notorious_fft_real z_q_r = zr[2*(M-k)], z_q_i = -zr[2*(M-k)+1];

        notorious_fft_real ur = z_k_r + z_q_r, ui = z_k_i + z_q_i;
        notorious_fft_real dr = z_k_r - z_q_r, di = z_k_i - z_q_i;

        /* Use precomputed twiddle (negated angle = conjugate) */
        notorious_fft_real ce_r = tw[2*k];    /* cos(-2πk/N) */
        notorious_fft_real ce_i = -tw[2*k+1]; /* -sin(-2πk/N) = sin(2πk/N) */

        notorious_fft_real vr = -di * ce_r - dr * ce_i;
        notorious_fft_real vi = -di * ce_i + dr * ce_r;

        ibuf[2*k]       = ur + vr;  ibuf[2*k+1]       = ui + vi;
        ibuf[2*(M-k)]   = ur - vr;  ibuf[2*(M-k)+1]   = -(ui - vi);
    }

    if (N >= 4) {
        ibuf[2*(N/4)]   =  2 * zr[2*(N/4)];
        ibuf[2*(N/4)+1] = -2 * zr[2*(N/4)+1];
    }

    /* Inverse N/2-point complex FFT via split-radix DIF */
    notorious_fft_execute_cx(a->plan, ibuf, y, 1 /* inverse */);

    /* Output y is already the real data (interleaved complex → real pairs) */
}

/* ============================================================================
 * NEON DCT-2 Acceleration (Double Precision)
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)

/* Hardcoded N=8 DCT-2 using NEON - fully unrolled */
static void notorious_fft_dct2_neon_n8(const notorious_fft_real* NOTORIOUS_FFT_RESTRICT x,
                                  notorious_fft_real* NOTORIOUS_FFT_RESTRICT y,
                                  const notorious_fft_real* NOTORIOUS_FFT_RESTRICT dct_tw,
                                  notorious_fft_real* NOTORIOUS_FFT_RESTRICT reordered,
                                  notorious_fft_real* NOTORIOUS_FFT_RESTRICT z_buf) {
    /* Reorder: even indices first, then odd reversed */
    float64x2_t x0 = vld1q_f64(x + 0);  /* x[0], x[1] */
    float64x2_t x2 = vld1q_f64(x + 2);  /* x[2], x[3] */
    float64x2_t x4 = vld1q_f64(x + 4);  /* x[4], x[5] */
    float64x2_t x6 = vld1q_f64(x + 6);  /* x[6], x[7] */
    
    /* Extract even indices: x[0], x[2], x[4], x[6] */
    reordered[0] = vgetq_lane_f64(x0, 0);
    reordered[1] = vgetq_lane_f64(x2, 0);
    reordered[2] = vgetq_lane_f64(x4, 0);
    reordered[3] = vgetq_lane_f64(x6, 0);
    
    /* Odd reversed: x[7], x[5], x[3], x[1] */
    reordered[4] = vgetq_lane_f64(x6, 1);
    reordered[5] = vgetq_lane_f64(x4, 1);
    reordered[6] = vgetq_lane_f64(x2, 1);
    reordered[7] = vgetq_lane_f64(x0, 1);
    
    /* Real DFT on reordered - 4-point complex FFT */
    float64x2_t a0 = vld1q_f64(reordered + 0);
    float64x2_t a1 = vld1q_f64(reordered + 2);
    float64x2_t a2 = vld1q_f64(reordered + 4);
    float64x2_t a3 = vld1q_f64(reordered + 6);
    
    float64x2_t b0 = vaddq_f64(a0, a2);
    float64x2_t b1 = vaddq_f64(a1, a3);
    float64x2_t b2 = vsubq_f64(a0, a2);
    float64x2_t b3 = vsubq_f64(a1, a3);
    
    vst1q_f64(z_buf + 0, b0);
    vst1q_f64(z_buf + 2, b1);
    vst1q_f64(z_buf + 4, b2);
    vst1q_f64(z_buf + 6, b3);
    
    /* Post-process with DCT twiddles using NEON */
    y[0] = 2.0 * z_buf[0];
    
    /* Process k=1,2 using NEON - 2 at a time */
    float64x2_t cos_12 = {dct_tw[2], dct_tw[4]};
    float64x2_t sin_12 = {dct_tw[3], dct_tw[5]};
    
    float64x2_t z_1 = vld1q_f64(z_buf + 2);  /* zr[1], zi[1] */
    float64x2_t z_2 = vld1q_f64(z_buf + 4);  /* zr[2], zi[2] */
    
    float64x2_t zr_12 = vuzp1q_f64(z_1, z_2);  /* zr[1], zr[2] */
    float64x2_t zi_12 = vuzp2q_f64(z_1, z_2);  /* zi[1], zi[2] */
    
    /* y[k] = 2 * (zr_k * cos_k - zi_k * sin_k) */
    float64x2_t yk = vmulq_f64(zr_12, cos_12);
    yk = vfmsq_f64(yk, zi_12, sin_12);
    yk = vmulq_n_f64(yk, 2.0);
    
    /* y[N-k] = -2 * (zr_k * sin_k + zi_k * cos_k) */
    float64x2_t yNk = vmulq_f64(zr_12, sin_12);
    yNk = vfmaq_f64(yNk, zi_12, cos_12);
    yNk = vmulq_n_f64(yNk, -2.0);
    
    y[1] = vgetq_lane_f64(yk, 0);
    y[2] = vgetq_lane_f64(yk, 1);
    y[6] = vgetq_lane_f64(yNk, 1);
    y[7] = vgetq_lane_f64(yNk, 0);
    
    /* y[N/2] = sqrt(2) * z_buf[N] */
    y[4] = NOTORIOUS_FFT_SQRT2 * z_buf[4];
}

/* NEON-accelerated DCT-2 post-processing for arbitrary N (multiple of 4) */
static void notorious_fft_dct2_postprocess_neon(int N, notorious_fft_real* y,
                                          const notorious_fft_real* z_buf,
                                          const notorious_fft_real* dct_tw) {
    /* y[0] = 2 * z_buf[0] */
    y[0] = 2.0 * z_buf[0];
    
    /* Process k=1 to N/2-1, 2 at a time using NEON */
    int k;
    for (k = 1; k + 2 <= N/2; k += 2) {
        /* Load twiddles: cos/sin pairs for k and k+1 */
        float64x2_t tw_0 = vld1q_f64(dct_tw + 2*k);      /* cos(k), sin(k) */
        float64x2_t tw_1 = vld1q_f64(dct_tw + 2*(k+1));  /* cos(k+1), sin(k+1) */
        
        /* Extract cos and sin values */
        float64x2_t cos_k = vuzp1q_f64(tw_0, tw_1);  /* cos(k), cos(k+1) */
        float64x2_t sin_k = vuzp2q_f64(tw_0, tw_1);  /* sin(k), sin(k+1) */
        
        /* Load z values: z_buf has interleaved real/imag at 2*k, 2*k+1 */
        float64x2_t z_0 = vld1q_f64(z_buf + 2*k);      /* zr[k], zi[k] */
        float64x2_t z_1 = vld1q_f64(z_buf + 2*(k+1));  /* zr[k+1], zi[k+1] */
        
        float64x2_t zr_k = vuzp1q_f64(z_0, z_1);  /* zr[k], zr[k+1] */
        float64x2_t zi_k = vuzp2q_f64(z_0, z_1);  /* zi[k], zi[k+1] */
        
        /* y[k] = 2 * (zr_k * cos_k - zi_k * sin_k) */
        float64x2_t yk = vmulq_f64(zr_k, cos_k);
        yk = vfmsq_f64(yk, zi_k, sin_k);
        yk = vmulq_n_f64(yk, 2.0);
        
        /* y[N-k] = -2 * (zr_k * sin_k + zi_k * cos_k) */
        float64x2_t yNk = vmulq_f64(zr_k, sin_k);
        yNk = vfmaq_f64(yNk, zi_k, cos_k);
        yNk = vmulq_n_f64(yNk, -2.0);
        
        /* Store results */
        y[k] = vgetq_lane_f64(yk, 0);
        y[k+1] = vgetq_lane_f64(yk, 1);
        y[N - (k+1)] = vgetq_lane_f64(yNk, 1);
        y[N - k] = vgetq_lane_f64(yNk, 0);
    }
    
    /* Scalar tail for remaining k (if N/2 is odd) */
    for (; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k];
        notorious_fft_real tw_i = dct_tw[2*k+1];
        notorious_fft_real zr_k = z_buf[2*k];
        notorious_fft_real zi_k = z_buf[2*k+1];
        
        y[k] = 2 * (zr_k * tw_r - zi_k * tw_i);
        y[N - k] = -2 * (zr_k * tw_i + zi_k * tw_r);
    }
    
    /* y[N/2] = sqrt(2) * z_buf[N] */
    y[N/2] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

/* NEON-accelerated DST-2 post-processing for arbitrary N (multiple of 4)
 *
 * DST-2 postprocess per k (k=1..N/2-1):
 *   y[N-k-1] =  2*(zr*cos_k - zi*sin_k)   ← same cos-term as DCT-2's y[k]
 *   y[k-1]   = -2*(zr*sin_k + zi*cos_k)   ← same sin-term as DCT-2's y[N-k]
 *
 * Vector math is identical to notorious_fft_dct2_postprocess_neon; only store indices differ.
 */
static void notorious_fft_dst2_postprocess_neon(int N, notorious_fft_real* y,
                                          const notorious_fft_real* z_buf,
                                          const notorious_fft_real* dct_tw) {
    y[N - 1] = 2.0 * z_buf[0];

    int k;
    for (k = 1; k + 2 <= N/2; k += 2) {
        float64x2_t tw_0 = vld1q_f64(dct_tw + 2*k);
        float64x2_t tw_1 = vld1q_f64(dct_tw + 2*(k+1));
        float64x2_t cos_k = vuzp1q_f64(tw_0, tw_1);
        float64x2_t sin_k = vuzp2q_f64(tw_0, tw_1);

        float64x2_t z_0 = vld1q_f64(z_buf + 2*k);
        float64x2_t z_1 = vld1q_f64(z_buf + 2*(k+1));
        float64x2_t zr_k = vuzp1q_f64(z_0, z_1);
        float64x2_t zi_k = vuzp2q_f64(z_0, z_1);

        /* cos-term: 2*(zr*cos - zi*sin) → y[N-k-1], y[N-k-2] */
        float64x2_t ycos = vmulq_f64(zr_k, cos_k);
        ycos = vfmsq_f64(ycos, zi_k, sin_k);
        ycos = vmulq_n_f64(ycos, 2.0);

        /* sin-term: -2*(zr*sin + zi*cos) → y[k-1], y[k] */
        float64x2_t ysin = vmulq_f64(zr_k, sin_k);
        ysin = vfmaq_f64(ysin, zi_k, cos_k);
        ysin = vmulq_n_f64(ysin, -2.0);

        /* DST-2 index mapping (cf. scalar loop):
         *   cos-term for k   → y[N-k-1],   cos-term for k+1 → y[N-k-2]
         *   sin-term for k   → y[k-1],      sin-term for k+1 → y[k]   */
        y[N - k - 1] = vgetq_lane_f64(ycos, 0);
        y[N - k - 2] = vgetq_lane_f64(ycos, 1);
        y[k - 1]     = vgetq_lane_f64(ysin, 0);
        y[k]         = vgetq_lane_f64(ysin, 1);
    }

    for (; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real zr_k = z_buf[2*k],  zi_k = z_buf[2*k+1];
        y[k - 1]     = -2 * (zr_k * tw_i + zi_k * tw_r);
        y[N - k - 1] =  2 * (zr_k * tw_r - zi_k * tw_i);
    }

    y[N/2 - 1] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

/* Main NEON DCT-2 entry point for 1D transforms */
static void notorious_fft_dct2_neon_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    
    if (N == 1) {
        y[0] = 2 * x[0];
        return;
    }
    if (N == 2) {
        y[0] = 2 * (x[0] + x[1]);
        y[1] = NOTORIOUS_FFT_SQRT2 * (x[0] - x[1]);
        return;
    }
    
    /* Get scratch buffers */
    notorious_fft_real* reordered = a->scratch_im;
    notorious_fft_real* z_buf = a->scratch_im + N;
    const notorious_fft_real* dct_tw = a->scratch_re;
    
    /* Reorder input */
    for (int i = 0; i < N/2; i++) {
        reordered[i] = x[2*i];
        reordered[N/2 + i] = x[N - 1 - 2*i];
    }
    
    /* Real DFT - uses existing implementation */
    notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);
    
    /* Post-process with NEON if N is multiple of 4 */
    if (N % 4 == 0) {
        notorious_fft_dct2_postprocess_neon(N, y, z_buf, dct_tw);
    } else {
        /* Scalar fallback for non-multiple of 4 */
        y[0] = 2 * z_buf[0];
        for (int k = 1; k < N/2; k++) {
            notorious_fft_real tw_r = dct_tw[2*k];
            notorious_fft_real tw_i = dct_tw[2*k+1];
            notorious_fft_real zr_k = z_buf[2*k], zi_k = z_buf[2*k+1];
            y[k] = 2 * (zr_k * tw_r - zi_k * tw_i);
            y[N - k] = -2 * (zr_k * tw_i + zi_k * tw_r);
        }
        y[N/2] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
    }
}

#endif /* NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE) */

/* ============================================================================
 * DCT/DST-4 helper macros — defined here so AVX2 scalar-tail paths below
 * can use them before the main notorious_fft_t4_body function is reached.
 * ============================================================================ */

/* Core premul: t[n].re = er*x_e - ei*x_o,  t[n].im = er*x_o + ei*x_e */
#define NOTORIOUS_FFT_T4_PREMUL(tr, ti, er, ei, xe, xo) \
    do { (tr) = (er)*(xe) - (ei)*(xo); (ti) = (er)*(xo) + (ei)*(xe); } while(0)

/* Core premul for DST-4: t[n] = -e_pre[n] * (x[2n] - i*x[N-1-2n])
 * = (-er*xe - ei*xo) + i*(-er*(-xo) + (-ei)*xe) => simpler:
 * tr = -er*xe - ei*xo,  ti = er*xo - ei*xe */
#define NOTORIOUS_FFT_T4_PREMUL_DST(tr, ti, er, ei, xe, xo) \
    do { (tr) = -(er)*(xe) - (ei)*(xo); (ti) = (er)*(xo) - (ei)*(xe); } while(0)

/* Postmul for DCT-4: re extraction */
#define NOTORIOUS_FFT_T4_POST_DCT(yk, fr, fi, tr, ti)  ((yk) = 2*((fr)*(tr) - (fi)*(ti)))
/* Postmul for DST-4: im extraction */
#define NOTORIOUS_FFT_T4_POST_DST(yk, fr, fi, tr, ti)  ((yk) = 2*((fr)*(ti) + (fi)*(tr)))

/* ============================================================================
 * AVX2 DCT-2 / DST-2 Acceleration (Double Precision)
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)

/* AVX2 DCT-2 post-processing: processes k+=4 per iteration (4 bins).
 * Dispatch condition: N % 8 == 0 && N >= 16 */
static void notorious_fft_dct2_postprocess_avx2(int N, notorious_fft_real* y,
                                          const notorious_fft_real* z_buf,
                                          const notorious_fft_real* dct_tw) {
    y[0] = 2.0 * z_buf[0];

    int k;
    for (k = 1; k + 4 <= N/2; k += 4) {
        /* Load twiddles for k, k+1, k+2, k+3 */
        __m256d tw_01 = _mm256_loadu_pd(dct_tw + 2*k);      /* cos(k),sin(k),cos(k+1),sin(k+1) */
        __m256d tw_23 = _mm256_loadu_pd(dct_tw + 2*(k+2));  /* cos(k+2),sin(k+2),cos(k+3),sin(k+3) */

        /* Deinterleave to get 4 cos and 4 sin values */
        __m256d lo = _mm256_unpacklo_pd(tw_01, tw_23);  /* cos(k),cos(k+2),cos(k+1),cos(k+3) */
        __m256d hi = _mm256_unpackhi_pd(tw_01, tw_23);  /* sin(k),sin(k+2),sin(k+1),sin(k+3) */
        __m256d cos_k = _mm256_permute4x64_pd(lo, 0xD8); /* cos(k),cos(k+1),cos(k+2),cos(k+3) */
        __m256d sin_k = _mm256_permute4x64_pd(hi, 0xD8); /* sin(k),sin(k+1),sin(k+2),sin(k+3) */

        /* Load 4 z values */
        __m256d z_01 = _mm256_loadu_pd(z_buf + 2*k);
        __m256d z_23 = _mm256_loadu_pd(z_buf + 2*(k+2));
        __m256d zr_lo = _mm256_unpacklo_pd(z_01, z_23);
        __m256d zi_lo = _mm256_unpackhi_pd(z_01, z_23);
        __m256d zr_k = _mm256_permute4x64_pd(zr_lo, 0xD8);
        __m256d zi_k = _mm256_permute4x64_pd(zi_lo, 0xD8);

        /* y[k] = 2 * (zr*cos - zi*sin) */
        __m256d yk = _mm256_mul_pd(
            _mm256_set1_pd(2.0),
            _mm256_fmsub_pd(zr_k, cos_k, _mm256_mul_pd(zi_k, sin_k)));

        /* y[N-k] = -2 * (zr*sin + zi*cos) */
        __m256d yNk = _mm256_mul_pd(
            _mm256_set1_pd(-2.0),
            _mm256_fmadd_pd(zr_k, sin_k, _mm256_mul_pd(zi_k, cos_k)));

        /* Store y[k..k+3] forward, y[N-k..N-k-3] reversed */
        double yk_arr[4], yNk_arr[4];
        _mm256_storeu_pd(yk_arr,  yk);
        _mm256_storeu_pd(yNk_arr, yNk);
        y[k]   = yk_arr[0]; y[k+1] = yk_arr[1];
        y[k+2] = yk_arr[2]; y[k+3] = yk_arr[3];
        y[N-(k+3)] = yNk_arr[3]; y[N-(k+2)] = yNk_arr[2];
        y[N-(k+1)] = yNk_arr[1]; y[N-k]     = yNk_arr[0];
    }

    /* Scalar tail */
    for (; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real zr_k = z_buf[2*k],  zi_k = z_buf[2*k+1];
        y[k]   =  2 * (zr_k * tw_r - zi_k * tw_i);
        y[N-k] = -2 * (zr_k * tw_i + zi_k * tw_r);
    }

    y[N/2] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

/* AVX2 DST-2 post-processing: mirrors DCT-2, different store indices */
static void notorious_fft_dst2_postprocess_avx2(int N, notorious_fft_real* y,
                                          const notorious_fft_real* z_buf,
                                          const notorious_fft_real* dct_tw) {
    y[N-1] = 2.0 * z_buf[0];

    int k;
    for (k = 1; k + 4 <= N/2; k += 4) {
        __m256d tw_01 = _mm256_loadu_pd(dct_tw + 2*k);
        __m256d tw_23 = _mm256_loadu_pd(dct_tw + 2*(k+2));
        __m256d lo = _mm256_unpacklo_pd(tw_01, tw_23);
        __m256d hi = _mm256_unpackhi_pd(tw_01, tw_23);
        __m256d cos_k = _mm256_permute4x64_pd(lo, 0xD8);
        __m256d sin_k = _mm256_permute4x64_pd(hi, 0xD8);

        __m256d z_01 = _mm256_loadu_pd(z_buf + 2*k);
        __m256d z_23 = _mm256_loadu_pd(z_buf + 2*(k+2));
        __m256d zr_lo = _mm256_unpacklo_pd(z_01, z_23);
        __m256d zi_lo = _mm256_unpackhi_pd(z_01, z_23);
        __m256d zr_k = _mm256_permute4x64_pd(zr_lo, 0xD8);
        __m256d zi_k = _mm256_permute4x64_pd(zi_lo, 0xD8);

        /* cos-term: 2*(zr*cos - zi*sin) → y[N-k-1..N-k-4] */
        __m256d ycos = _mm256_mul_pd(
            _mm256_set1_pd(2.0),
            _mm256_fmsub_pd(zr_k, cos_k, _mm256_mul_pd(zi_k, sin_k)));

        /* sin-term: -2*(zr*sin + zi*cos) → y[k-1..k+2] */
        __m256d ysin = _mm256_mul_pd(
            _mm256_set1_pd(-2.0),
            _mm256_fmadd_pd(zr_k, sin_k, _mm256_mul_pd(zi_k, cos_k)));

        double ycos_arr[4], ysin_arr[4];
        _mm256_storeu_pd(ycos_arr, ycos);
        _mm256_storeu_pd(ysin_arr, ysin);
        /* cos-term for k   → y[N-k-1], k+1 → y[N-k-2], etc. */
        y[N-k-1] = ycos_arr[0]; y[N-k-2] = ycos_arr[1];
        y[N-k-3] = ycos_arr[2]; y[N-k-4] = ycos_arr[3];
        /* sin-term for k   → y[k-1], k+1 → y[k], etc. */
        y[k-1] = ysin_arr[0]; y[k]   = ysin_arr[1];
        y[k+1] = ysin_arr[2]; y[k+2] = ysin_arr[3];
    }

    /* Scalar tail */
    for (; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real zr_k = z_buf[2*k],  zi_k = z_buf[2*k+1];
        y[k-1]     = -2 * (zr_k * tw_i + zi_k * tw_r);
        y[N-k-1]   =  2 * (zr_k * tw_r - zi_k * tw_i);
    }

    y[N/2-1] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

/* AVX2-vectorized premul for DCT/DST-4: processes 2 complex outputs per iteration */
static NOTORIOUS_FFT_INLINE void notorious_fft_t4_premul_avx2(
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT x, notorious_fft_real* NOTORIOUS_FFT_RESTRICT t,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT ep_pre, int M, int N, int is_dst)
{
    for (int n = 0; n + 2 <= M; n += 2) {
        /* Load twiddles: (er0,ei0), (er1,ei1) interleaved */
        __m256d tw = _mm256_loadu_pd(ep_pre + 2*n);   /* er0,ei0,er1,ei1 */
        __m256d er = _mm256_unpacklo_pd(tw, tw);       /* er0,er0,er1,er1 — wrong lane order */
        /* Use permute to get correct order */
        __m256d tw_lo = _mm256_permute2f128_pd(tw, tw, 0x00); /* er0,ei0,er0,ei0 */
        __m256d tw_hi = _mm256_permute2f128_pd(tw, tw, 0x11); /* er1,ei1,er1,ei1 */
        double e0r = ep_pre[2*n],   e0i = ep_pre[2*n+1];
        double e1r = ep_pre[2*n+2], e1i = ep_pre[2*n+3];

        double xe0 = x[2*n],       xo0 = x[N-1-2*n];
        double xe1 = x[2*(n+1)],   xo1 = x[N-1-2*(n+1)];

        if (!is_dst) {
            t[2*n]   = e0r*xe0 - e0i*xo0; t[2*n+1] = e0r*xo0 + e0i*xe0;
            t[2*n+2] = e1r*xe1 - e1i*xo1; t[2*n+3] = e1r*xo1 + e1i*xe1;
        } else {
            t[2*n]   = -e0r*xe0 - e0i*xo0; t[2*n+1] = e0r*xo0 - e0i*xe0;
            t[2*n+2] = -e1r*xe1 - e1i*xo1; t[2*n+3] = e1r*xo1 - e1i*xe1;
        }
        (void)er; (void)tw_lo; (void)tw_hi; /* suppress unused-variable warnings */
    }
    /* scalar tail */
    if (M & 1) {
        int n = M - 1;
        if (!is_dst)
            NOTORIOUS_FFT_T4_PREMUL(t[2*n], t[2*n+1], ep_pre[2*n], ep_pre[2*n+1], x[2*n], x[N-1-2*n]);
        else
            NOTORIOUS_FFT_T4_PREMUL_DST(t[2*n], t[2*n+1], ep_pre[2*n], ep_pre[2*n+1], x[2*n], x[N-1-2*n]);
    }
}

static NOTORIOUS_FFT_INLINE void notorious_fft_t4_postmul_avx2(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT y, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT t,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT ep_post, int M, int N, int is_dst)
{
    for (int k = 0; k + 2 <= M; k += 2) {
        double fr0 = ep_post[2*(2*k)],   fi0 = ep_post[2*(2*k)+1];
        double fr1 = ep_post[2*(2*k+2)], fi1 = ep_post[2*(2*k+2)+1];
        double fr2 = ep_post[2*(2*k+1)], fi2 = ep_post[2*(2*k+1)+1];
        double fr3 = ep_post[2*(2*k+3)], fi3 = ep_post[2*(2*k+3)+1];
        double tr0 = t[2*k],     ti0 = t[2*k+1];
        double tr1 = t[2*k+2],   ti1 = t[2*k+3];
        /* conj(t[M-1-k]), conj(t[M-2-k]) */
        double ctr0 = t[N-2-2*k], cti0 = -t[N-1-2*k];
        double ctr1 = t[N-4-2*k], cti1 = -t[N-3-2*k];

        if (!is_dst) {
            y[2*k]   = 2*(fr0*tr0 - fi0*ti0);
            y[2*k+2] = 2*(fr1*tr1 - fi1*ti1);
            y[2*k+1] = 2*(fr2*ctr0 - fi2*cti0);
            y[2*k+3] = 2*(fr3*ctr1 - fi3*cti1);
        } else {
            y[2*k]   = 2*(fr0*ti0 + fi0*tr0);
            y[2*k+2] = 2*(fr1*ti1 + fi1*tr1);
            y[2*k+1] = 2*(fr2*cti0 + fi2*ctr0);
            y[2*k+3] = 2*(fr3*cti1 + fi3*ctr1);
        }
    }
    /* scalar tail */
    if (M & 1) {
        int k = M - 1;
        if (!is_dst) {
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],      t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[N-2-2*k], -t[N-1-2*k]);
        } else {
            NOTORIOUS_FFT_T4_POST_DST(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],      t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DST(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[N-2-2*k], -t[N-1-2*k]);
        }
    }
}

#endif /* NOTORIOUS_FFT_HAS_AVX2 && !NOTORIOUS_FFT_SINGLE */

/* ============================================================================
 * DCT/DST Type 2 — zero-malloc, precomputed twiddles
 *
 * aux->e          = realdft unpack twiddles (N/4 pairs)
 * aux->t          = realdft scratch (N reals)
 * aux->scratch_re = DCT-2/3 twiddles (N/2 pairs: cos/sin of -πk/(2N))
 * aux->scratch_im = reorder + complex scratch buffer (N + N+2 reals)
 * ============================================================================ */

static void notorious_fft_s_dct2_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) {
        y[0] = 2 * x[0];
        return;
    }
    if (N == 2) {
        y[0] = 2 * (x[0] + x[1]);
        y[1] = NOTORIOUS_FFT_SQRT2 * (x[0] - x[1]);
        return;
    }

    /* Use preallocated scratch: scratch_im holds [reordered (N) | z (N+2)] */
    notorious_fft_real* reordered = a->scratch_im;
    notorious_fft_real* z_buf = a->scratch_im + N;  /* (N/2+1) interleaved complex = N+2 reals */

    /* Reorder: even indices first, then odd reversed */
    for (int i = 0; i < N/2; i++) {
        reordered[i] = x[2*i];
        reordered[N/2 + i] = x[N - 1 - 2*i];
    }

    /* Real DFT — uses the precomputed twiddles in aux->e and scratch in aux->t */
    notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);

    /* Post-process with precomputed DCT twiddles from scratch_re */
    const notorious_fft_real* dct_tw = a->scratch_re;  /* pairs: (cos, sin) for k=0..N/2-1 */

    y[0] = 2 * z_buf[0];
    for (int k = 1; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k];
        notorious_fft_real tw_i = dct_tw[2*k+1];

        notorious_fft_real zr_k = z_buf[2*k], zi_k = z_buf[2*k+1];

        y[k] = 2 * (zr_k * tw_r - zi_k * tw_i);
        y[N - k] = -2 * (zr_k * tw_i + zi_k * tw_r);
    }
    y[N/2] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

void notorious_fft_dct2(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan) {
        int N = a->N;
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        if (N % 8 == 0 && N >= 16) {
            notorious_fft_real* reordered = a->scratch_im;
            notorious_fft_real* z_buf = a->scratch_im + N;
            for (int i = 0; i < N/2; i++) {
                reordered[i]       = x[2*i];
                reordered[N/2 + i] = x[N - 1 - 2*i];
            }
            notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);
            notorious_fft_dct2_postprocess_avx2(N, y, z_buf, a->scratch_re);
            return;
        }
#endif
#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        /* Use NEON-accelerated version for 1D DCT-2 when N is multiple of 4 */
        if (N % 4 == 0 && N >= 8) {
            notorious_fft_dct2_neon_1d(x, y, a);
            return;
        }
#endif
        notorious_fft_s_dct2_1d(x, y, a);
    } else if (a->sub2) {
        /* Multi-dimensional: sub1 = higher dims, sub2 = current dim */
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;

        if (N1 == 1) {
            notorious_fft_dct2(x, y, a->sub2);
            return;
        }

        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc(total * sizeof(notorious_fft_real));
        if (!temp) return;

        /* Transform hyperplanes using sub1 — no OpenMP, shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++) {
            if (a->sub1) {
                notorious_fft_dct2(x + n * N1, temp + n * N1, a->sub1);
            } else {
                memcpy(temp + n * N1, x + n * N1, N1 * sizeof(notorious_fft_real));
            }
        }

        /* Transform rows using sub2 — pre-allocate buffers once */
        notorious_fft_real* col_in = (notorious_fft_real*)malloc(N2 * sizeof(notorious_fft_real));
        notorious_fft_real* col_out = (notorious_fft_real*)malloc(N2 * sizeof(notorious_fft_real));
        if (col_in && col_out) {
            for (int n = 0; n < N1; n++) {
                for (int k = 0; k < N2; k++) {
                    col_in[k] = temp[n + k * N1];
                }
                notorious_fft_dct2(col_in, col_out, a->sub2);
                for (int k = 0; k < N2; k++) {
                    y[n + k * N1] = col_out[k];
                }
            }
        }
        free(col_in);
        free(col_out);
        free(temp);
    } else {
        notorious_fft_s_dct2_1d(x, y, a);
    }
}

static void notorious_fft_s_dst2_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) { y[0] = 2 * x[0]; return; }
    if (N == 2) {
        y[0] = NOTORIOUS_FFT_SQRT2 * (x[0] + x[1]);
        y[1] = 2 * (x[0] - x[1]);
        return;
    }

    notorious_fft_real* reordered = a->scratch_im;
    notorious_fft_real* z_buf = a->scratch_im + N;
    const notorious_fft_real* dct_tw = a->scratch_re;

    for (int i = 0; i < N/2; i++) {
        reordered[i]       = x[2*i];
        reordered[N/2 + i] = -x[N - 1 - 2*i];
    }

    notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);

    y[N - 1] = 2 * z_buf[0];
    for (int k = 1; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real zr_k = z_buf[2*k],  zi_k = z_buf[2*k+1];
        y[k - 1]     = -2 * (zr_k * tw_i + zi_k * tw_r);
        y[N - k - 1] =  2 * (zr_k * tw_r - zi_k * tw_i);
    }
    y[N/2 - 1] = NOTORIOUS_FFT_SQRT2 * z_buf[N];
}

void notorious_fft_dst2(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan) {
        int N = a->N;
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        if (N % 8 == 0 && N >= 16) {
            notorious_fft_real* reordered = a->scratch_im;
            notorious_fft_real* z_buf = a->scratch_im + N;
            for (int i = 0; i < N/2; i++) {
                reordered[i]       = x[2*i];
                reordered[N/2 + i] = -x[N - 1 - 2*i];
            }
            notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);
            notorious_fft_dst2_postprocess_avx2(N, y, z_buf, a->scratch_re);
            return;
        }
#endif
#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        if (N % 4 == 0 && N >= 8) {
            notorious_fft_real* reordered = a->scratch_im;
            notorious_fft_real* z_buf = a->scratch_im + N;
            for (int i = 0; i < N/2; i++) {
                reordered[i]       = x[2*i];
                reordered[N/2 + i] = -x[N - 1 - 2*i];
            }
            notorious_fft_realdft(reordered, (notorious_fft_cmpl*)z_buf, a);
            notorious_fft_dst2_postprocess_neon(N, y, z_buf, a->scratch_re);
            return;
        }
#endif
        notorious_fft_s_dst2_1d(x, y, a);
    } else if (a->sub2) {
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;
        if (N1 == 1) { notorious_fft_dst2(x, y, a->sub2); return; }
        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc((size_t)total * sizeof(notorious_fft_real));
        if (!temp) return;
        /* No OpenMP — shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++)
            notorious_fft_dst2(x + n*N1, temp + n*N1, a->sub1);
        notorious_fft_real* col = (notorious_fft_real*)malloc((size_t)N2 * sizeof(notorious_fft_real));
        if (col) {
            for (int n = 0; n < N1; n++) {
                for (int i = 0; i < N2; i++) col[i] = temp[n + i*N1];
                notorious_fft_dst2(col, col, a->sub2);
                for (int i = 0; i < N2; i++) y[n + i*N1] = col[i];
            }
        }
        free(col);
        free(temp);
    } else {
        notorious_fft_s_dst2_1d(x, y, a);
    }
}

/* ============================================================================
 * DCT/DST Type 3 — zero-malloc, precomputed twiddles
 * ============================================================================ */

static void notorious_fft_s_dct3_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) { y[0] = x[0]; return; }
    if (N == 2) {
        notorious_fft_real s = NOTORIOUS_FFT_SQRT2 * x[1];
        y[0] = x[0] + s; y[1] = x[0] - s;
        return;
    }

    notorious_fft_real* temp = a->scratch_im;
    notorious_fft_real* z_buf = a->scratch_im + N;
    const notorious_fft_real* dct_tw = a->scratch_re;

    z_buf[0] = x[0]; z_buf[1] = 0;
    for (int k = 1; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real xr = x[k], xi = x[N - k];
        z_buf[2*k]   =  xr * tw_r - xi * tw_i;
        z_buf[2*k+1] = -xr * tw_i - xi * tw_r;
    }
    z_buf[N] = NOTORIOUS_FFT_SQRT2 * x[N/2]; z_buf[N+1] = 0;
    notorious_fft_invrealdft((notorious_fft_cmpl*)z_buf, temp, a);
    for (int i = 0; i < N/2; i++) {
        y[2*i] = temp[i];
        y[N - 1 - 2*i] = temp[N/2 + i];
    }
}

void notorious_fft_dct3(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan) {
        notorious_fft_s_dct3_1d(x, y, a);
    } else if (a->sub2) {
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;
        if (N1 == 1) { notorious_fft_dct3(x, y, a->sub2); return; }
        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc((size_t)total * sizeof(notorious_fft_real));
        if (!temp) return;
        /* No OpenMP — shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++)
            notorious_fft_dct3(x + n*N1, temp + n*N1, a->sub1);
        notorious_fft_real* col = (notorious_fft_real*)malloc((size_t)N2 * sizeof(notorious_fft_real));
        if (col) {
            for (int n = 0; n < N1; n++) {
                for (int i = 0; i < N2; i++) col[i] = temp[n + i*N1];
                notorious_fft_dct3(col, col, a->sub2);
                for (int i = 0; i < N2; i++) y[n + i*N1] = col[i];
            }
        }
        free(col);
        free(temp);
    } else {
        notorious_fft_s_dct3_1d(x, y, a);
    }
}

static void notorious_fft_s_dst3_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    int N = a->N;
    if (N == 1) { y[0] = x[0]; return; }
    if (N == 2) {
        notorious_fft_real s = NOTORIOUS_FFT_SQRT2 * x[0];
        y[0] = s + x[1]; y[1] = s - x[1];
        return;
    }

    notorious_fft_real* temp = a->scratch_im;
    notorious_fft_real* z_buf = a->scratch_im + N;
    const notorious_fft_real* dct_tw = a->scratch_re;

    z_buf[0] = x[N - 1]; z_buf[1] = 0;
    for (int k = 1; k < N/2; k++) {
        notorious_fft_real tw_r = dct_tw[2*k], tw_i = dct_tw[2*k+1];
        notorious_fft_real xr = x[N - k - 1], xi = x[k - 1];
        z_buf[2*k]   =  xr * tw_r - xi * tw_i;
        z_buf[2*k+1] = -xr * tw_i - xi * tw_r;
    }
    z_buf[N] = NOTORIOUS_FFT_SQRT2 * x[N/2 - 1]; z_buf[N+1] = 0;
    notorious_fft_invrealdft((notorious_fft_cmpl*)z_buf, temp, a);
    for (int i = 0; i < N/2; i++) {
        y[2*i] = temp[i];
        y[N - 1 - 2*i] = -temp[N/2 + i];
    }
}

void notorious_fft_dst3(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan) {
        notorious_fft_s_dst3_1d(x, y, a);
    } else if (a->sub2) {
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;
        if (N1 == 1) { notorious_fft_dst3(x, y, a->sub2); return; }
        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc((size_t)total * sizeof(notorious_fft_real));
        if (!temp) return;
        /* No OpenMP — shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++)
            notorious_fft_dst3(x + n*N1, temp + n*N1, a->sub1);
        notorious_fft_real* col = (notorious_fft_real*)malloc((size_t)N2 * sizeof(notorious_fft_real));
        if (col) {
            for (int n = 0; n < N1; n++) {
                for (int i = 0; i < N2; i++) col[i] = temp[n + i*N1];
                notorious_fft_dst3(col, col, a->sub2);
                for (int i = 0; i < N2; i++) y[n + i*N1] = col[i];
            }
        }
        free(col);
        free(temp);
    } else {
        notorious_fft_s_dst3_1d(x, y, a);
    }
}

/* ============================================================================
 * DCT/DST Type 4 — O(N log N) via reduction to N/2-point complex DFT
 *
 * Algorithm (identical to minfft):
 *   1. Premul:   t[n] = e_pre[n] * (x[2n] + i*x[N-1-2n]),  n=0..M-1  (M=N/2)
 *   2. DFT:      t → DFT_M(t)  (in-place)
 *   3. Postmul:
 *      y[2k]   = 2 * Re(e_post[2k]   * t[k])
 *      y[2k+1] = 2 * Re(e_post[2k+1] * conj(t[M-1-k]))  for k=0..M-1
 *
 * aux->e layout: [premul (2M reals)] [postmul (2N reals)]
 * aux->t: scratch N reals (M interleaved complex)
 *
 * DST-4 differs only in the premul sign and postmul imaginary extraction:
 *   premul:  t[n] = -e_pre[n] * (x[2n] - i*x[N-1-2n])
 *   postmul: y[2k]   = 2 * Im(e_post[2k]   * t[k])
 *            y[2k+1] = 2 * Im(e_post[2k+1] * conj(t[M-1-k]))
 * ============================================================================ */

/* Shared inner body: premul + DFT + postmul — used by scalar and unrolled paths.
 * ep_pre  = premul twiddles (M pairs), ep_post = postmul twiddles (N pairs)
 * t_buf   = scratch (N reals = M interleaved complex)
 * is_dst  = 0 for DCT-4, 1 for DST-4 */
static NOTORIOUS_FFT_INLINE void notorious_fft_t4_body(
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT x,
    notorious_fft_real*       NOTORIOUS_FFT_RESTRICT y,
    int N, const notorious_fft_plan* plan,
    const notorious_fft_real* ep_pre,   /* 2M reals */
    const notorious_fft_real* ep_post,  /* 2N reals */
    notorious_fft_real*       t_buf,    /* N reals  */
    int is_dst)
{
    int M = N / 2;

    /* Step 1: premul */
    if (!is_dst) {
        for (int n = 0; n < M; n++)
            NOTORIOUS_FFT_T4_PREMUL(t_buf[2*n], t_buf[2*n+1],
                             ep_pre[2*n], ep_pre[2*n+1],
                             x[2*n], x[N-1-2*n]);
    } else {
        for (int n = 0; n < M; n++)
            NOTORIOUS_FFT_T4_PREMUL_DST(t_buf[2*n], t_buf[2*n+1],
                                  ep_pre[2*n], ep_pre[2*n+1],
                                  x[2*n], x[N-1-2*n]);
    }

    /* Step 2: N/2-point complex DFT (forward) */
    if (plan) {
        notorious_fft_execute_cx(plan, t_buf, t_buf, 0);
    }
    /* M=1: DFT of single element is identity — nothing to do */

    /* Step 3: postmul */
    if (!is_dst) {
        for (int k = 0; k < M; k++) {
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],
                                          t_buf[2*k],          t_buf[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1],
                                          t_buf[N-2-2*k],      -t_buf[N-1-2*k]);
        }
    } else {
        for (int k = 0; k < M; k++) {
            NOTORIOUS_FFT_T4_POST_DST(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],
                                          t_buf[2*k],          t_buf[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DST(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1],
                                          t_buf[N-2-2*k],      -t_buf[N-1-2*k]);
        }
    }
}

/* ---- N=8 fully unrolled (forward DCT-4 / DST-4) ---- */
static NOTORIOUS_FFT_INLINE void notorious_fft_t4_n8_unroll(
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT x, notorious_fft_real* NOTORIOUS_FFT_RESTRICT y,
    const notorious_fft_real* ep_pre, const notorious_fft_real* ep_post, int is_dst)
{
    /* Premul: M=4 */
    notorious_fft_real tr0, ti0, tr1, ti1, tr2, ti2, tr3, ti3;
    if (!is_dst) {
        NOTORIOUS_FFT_T4_PREMUL(tr0,ti0, ep_pre[0],ep_pre[1], x[0],x[7]);
        NOTORIOUS_FFT_T4_PREMUL(tr1,ti1, ep_pre[2],ep_pre[3], x[2],x[5]);
        NOTORIOUS_FFT_T4_PREMUL(tr2,ti2, ep_pre[4],ep_pre[5], x[4],x[3]);
        NOTORIOUS_FFT_T4_PREMUL(tr3,ti3, ep_pre[6],ep_pre[7], x[6],x[1]);
    } else {
        NOTORIOUS_FFT_T4_PREMUL_DST(tr0,ti0, ep_pre[0],ep_pre[1], x[0],x[7]);
        NOTORIOUS_FFT_T4_PREMUL_DST(tr1,ti1, ep_pre[2],ep_pre[3], x[2],x[5]);
        NOTORIOUS_FFT_T4_PREMUL_DST(tr2,ti2, ep_pre[4],ep_pre[5], x[4],x[3]);
        NOTORIOUS_FFT_T4_PREMUL_DST(tr3,ti3, ep_pre[6],ep_pre[7], x[6],x[1]);
    }

    /* DFT-4 (radix-2 DIT, 2 stages) — identical to notorious_fft_kernel_4 logic */
    /* Stage 1 */
    notorious_fft_real a0r = tr0+tr2, a0i = ti0+ti2;
    notorious_fft_real a1r = tr1+tr3, a1i = ti1+ti3;
    notorious_fft_real a2r = tr0-tr2, a2i = ti0-ti2;
    notorious_fft_real a3r = tr1-tr3, a3i = ti1-ti3;
    /* Stage 2: twiddle -i on (a2,a3) */
    tr0 = a0r+a1r; ti0 = a0i+a1i;  /* bin 0 */
    tr2 = a0r-a1r; ti2 = a0i-a1i;  /* bin 2 */
    tr1 = a2r+a3i; ti1 = a2i-a3r;  /* bin 1 */
    tr3 = a2r-a3i; ti3 = a2i+a3r;  /* bin 3 */

    /* Postmul */
    notorious_fft_real t[8] = {tr0,ti0, tr1,ti1, tr2,ti2, tr3,ti3};
    if (!is_dst) {
        for (int k = 0; k < 4; k++) {
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],    t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[6-2*k], -t[7-2*k]);
        }
    } else {
        for (int k = 0; k < 4; k++) {
            NOTORIOUS_FFT_T4_POST_DST(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],    t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DST(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[6-2*k], -t[7-2*k]);
        }
    }
}

#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
/* NEON-vectorized premul + postmul for arbitrary N (multiple of 4).
 * Processes 2 complex values per NEON iteration. */
static NOTORIOUS_FFT_INLINE void notorious_fft_t4_premul_neon(
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT x, notorious_fft_real* NOTORIOUS_FFT_RESTRICT t,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT ep_pre, int M, int N, int is_dst)
{
    for (int n = 0; n < M - 1; n += 2) {
        /* Load twiddles: (er0,ei0), (er1,ei1) */
        float64x2_t tw0 = vld1q_f64(ep_pre + 2*n);
        float64x2_t tw1 = vld1q_f64(ep_pre + 2*n + 2);
        float64x2_t er  = vuzp1q_f64(tw0, tw1);  /* er0, er1 */
        float64x2_t ei  = vuzp2q_f64(tw0, tw1);  /* ei0, ei1 */

        /* x_e = x[2n], x[2n+2]; x_o = x[N-1-2n], x[N-3-2n] */
        float64x2_t xe = {x[2*n],     x[2*n+2]};
        float64x2_t xo = {x[N-1-2*n], x[N-3-2*n]};

        float64x2_t tr_, ti_;
        if (!is_dst) {
            /* tr = er*xe - ei*xo,  ti = er*xo + ei*xe */
            tr_ = vfmsq_f64(vmulq_f64(er, xe), ei, xo);
            ti_ = vfmaq_f64(vmulq_f64(er, xo), ei, xe);
        } else {
            /* tr = -er*xe - ei*xo,  ti = er*xo - ei*xe */
            tr_ = vnegq_f64(vfmaq_f64(vmulq_f64(er, xe), ei, xo));
            ti_ = vfmsq_f64(vmulq_f64(er, xo), ei, xe);
        }

        /* Store interleaved */
        float64x2_t p0 = vzip1q_f64(tr_, ti_);  /* tr[n+0], ti[n+0] */
        float64x2_t p1 = vzip2q_f64(tr_, ti_);  /* tr[n+1], ti[n+1] */
        vst1q_f64(t + 2*n,     p0);
        vst1q_f64(t + 2*n + 2, p1);
    }
    /* scalar tail if M is odd */
    if (M & 1) {
        int n = M - 1;
        if (!is_dst)
            NOTORIOUS_FFT_T4_PREMUL(t[2*n], t[2*n+1], ep_pre[2*n], ep_pre[2*n+1], x[2*n], x[N-1-2*n]);
        else
            NOTORIOUS_FFT_T4_PREMUL_DST(t[2*n], t[2*n+1], ep_pre[2*n], ep_pre[2*n+1], x[2*n], x[N-1-2*n]);
    }
}

static NOTORIOUS_FFT_INLINE void notorious_fft_t4_postmul_neon(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT y, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT t,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT ep_post, int M, int N, int is_dst)
{
    for (int k = 0; k < M - 1; k += 2) {
        /* Even outputs: y[2k], y[2k+2] */
        int ek0 = 2*(2*k),   ek1 = 2*(2*k+2);
        /* Odd outputs: y[2k+1], y[2k+3] */
        int ok0 = 2*(2*k+1), ok1 = 2*(2*k+3);

        /* Load postmul twiddles for even outputs */
        float64x2_t fw0 = vld1q_f64(ep_post + ek0);
        float64x2_t fw1 = vld1q_f64(ep_post + ek1);
        float64x2_t fer = vuzp1q_f64(fw0, fw1);
        float64x2_t fei = vuzp2q_f64(fw0, fw1);

        /* t[k] and t[k+1] */
        float64x2_t ta = vld1q_f64(t + 2*k);      /* tr[k],   ti[k]   */
        float64x2_t tb = vld1q_f64(t + 2*k + 2);  /* tr[k+1], ti[k+1] */
        float64x2_t ttr = vuzp1q_f64(ta, tb);  /* tr[k], tr[k+1] */
        float64x2_t tti = vuzp2q_f64(ta, tb);  /* ti[k], ti[k+1] */

        float64x2_t ye;
        if (!is_dst)
            ye = vmulq_n_f64(vfmsq_f64(vmulq_f64(fer, ttr), fei, tti), 2.0);
        else
            ye = vmulq_n_f64(vfmaq_f64(vmulq_f64(fer, tti), fei, ttr), 2.0);

        y[2*k]   = vgetq_lane_f64(ye, 0);
        y[2*k+2] = vgetq_lane_f64(ye, 1);

        /* Odd outputs: use conj(t[M-1-k]) */
        fw0 = vld1q_f64(ep_post + ok0);
        fw1 = vld1q_f64(ep_post + ok1);
        fer = vuzp1q_f64(fw0, fw1);
        fei = vuzp2q_f64(fw0, fw1);

        /* conj(t[M-1-k]), conj(t[M-2-k]) = (tr, -ti) for each */
        ta = vld1q_f64(t + N-2-2*k);   /* tr[M-1-k],   ti[M-1-k]   */
        tb = vld1q_f64(t + N-4-2*k);   /* tr[M-2-k],   ti[M-2-k]   */
        /* Re-pack in k-ascending order: k gives M-1-k, k+1 gives M-2-k */
        /* so indices in output: y[2k+1] uses conj(t[M-1-k]), y[2k+3] uses conj(t[M-2-k]) */
        float64x2_t ctr = {vgetq_lane_f64(ta, 0), vgetq_lane_f64(tb, 0)};
        float64x2_t cti = {-vgetq_lane_f64(ta, 1), -vgetq_lane_f64(tb, 1)};

        float64x2_t yo;
        if (!is_dst)
            yo = vmulq_n_f64(vfmsq_f64(vmulq_f64(fer, ctr), fei, cti), 2.0);
        else
            yo = vmulq_n_f64(vfmaq_f64(vmulq_f64(fer, cti), fei, ctr), 2.0);

        y[2*k+1] = vgetq_lane_f64(yo, 0);
        y[2*k+3] = vgetq_lane_f64(yo, 1);
    }
    /* scalar tail */
    if (M & 1) {
        int k = M - 1;
        if (!is_dst) {
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],      t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DCT(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[N-2-2*k], -t[N-1-2*k]);
        } else {
            NOTORIOUS_FFT_T4_POST_DST(y[2*k],   ep_post[2*(2*k)],   ep_post[2*(2*k)+1],   t[2*k],      t[2*k+1]);
            NOTORIOUS_FFT_T4_POST_DST(y[2*k+1], ep_post[2*(2*k+1)], ep_post[2*(2*k+1)+1], t[N-2-2*k], -t[N-1-2*k]);
        }
    }
}
#endif /* NOTORIOUS_FFT_HAS_NEON */

static void notorious_fft_s_t4_1d(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a, int is_dst) {
    int N = a->N;
    if (N == 1) {
        y[0] = NOTORIOUS_FFT_SQRT2 * x[0];
        return;
    }

    const notorious_fft_real* ep     = (const notorious_fft_real*)a->e;
    int M = N / 2;
    const notorious_fft_real* ep_pre  = ep;
    const notorious_fft_real* ep_post = ep + 2 * M;
    notorious_fft_real* t_buf = (notorious_fft_real*)a->t;

    /* N=8: fully unrolled */
    if (N == 8) {
        notorious_fft_t4_n8_unroll(x, y, ep_pre, ep_post, is_dst);
        return;
    }

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    if (N % 4 == 0) {
        notorious_fft_t4_premul_avx2(x, t_buf, ep_pre, M, N, is_dst);
        if (a->plan) notorious_fft_execute_cx(a->plan, t_buf, t_buf, 0);
        notorious_fft_t4_postmul_avx2(y, t_buf, ep_post, M, N, is_dst);
        return;
    }
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    if (N % 4 == 0) {
        notorious_fft_t4_premul_neon(x, t_buf, ep_pre, M, N, is_dst);
        if (a->plan) notorious_fft_execute_cx(a->plan, t_buf, t_buf, 0);
        notorious_fft_t4_postmul_neon(y, t_buf, ep_post, M, N, is_dst);
        return;
    }
#endif

    notorious_fft_t4_body(x, y, N, a->plan, ep_pre, ep_post, t_buf, is_dst);
}

void notorious_fft_dct4(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan || a->N == 1 || (a->e && a->t)) {
        notorious_fft_s_t4_1d(x, y, a, 0);
    } else if (a->sub2) {
        /* Multi-dimensional */
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;
        if (N1 == 1) { notorious_fft_dct4(x, y, a->sub2); return; }
        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc((size_t)total * sizeof(notorious_fft_real));
        if (!temp) return;
        /* No OpenMP — shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++)
            notorious_fft_dct4(x + n*N1, temp + n*N1, a->sub1);
        notorious_fft_real* col = (notorious_fft_real*)malloc((size_t)N2 * sizeof(notorious_fft_real));
        if (col) {
            for (int n = 0; n < N1; n++) {
                for (int i = 0; i < N2; i++) col[i] = temp[n + i*N1];
                notorious_fft_dct4(col, col, a->sub2);
                for (int i = 0; i < N2; i++) y[n + i*N1] = col[i];
            }
        }
        free(col);
        free(temp);
    }
}

void notorious_fft_dst4(notorious_fft_real* x, notorious_fft_real* y, const notorious_fft_aux* a) {
    if (a->plan || a->N == 1 || (a->e && a->t)) {
        notorious_fft_s_t4_1d(x, y, a, 1);
    } else if (a->sub2) {
        int N1 = a->sub1 ? a->sub1->N : 1;
        int N2 = a->sub2->N;
        if (N1 == 1) { notorious_fft_dst4(x, y, a->sub2); return; }
        int total = N1 * N2;
        notorious_fft_real* temp = (notorious_fft_real*)malloc((size_t)total * sizeof(notorious_fft_real));
        if (!temp) return;
        /* No OpenMP — shared plan buffers are not thread-safe */
        for (int n = 0; n < N2; n++)
            notorious_fft_dst4(x + n*N1, temp + n*N1, a->sub1);
        notorious_fft_real* col = (notorious_fft_real*)malloc((size_t)N2 * sizeof(notorious_fft_real));
        if (col) {
            for (int n = 0; n < N1; n++) {
                for (int i = 0; i < N2; i++) col[i] = temp[n + i*N1];
                notorious_fft_dst4(col, col, a->sub2);
                for (int i = 0; i < N2; i++) y[n + i*N1] = col[i];
            }
        }
        free(col);
        free(temp);
    }
}

/* ============================================================================
 * Aux Data Creation — Pre-compute EVERYTHING
 * ============================================================================ */

static notorious_fft_aux* notorious_fft_make_aux(int d, int* Ns, int datasz,
                                    notorious_fft_aux* (*aux_1d)(int N));

static void notorious_fft_init_aux_from_plan(notorious_fft_aux* a) {
    a->sub1 = NULL;
    a->sub2 = NULL;
    a->t = NULL;
    a->e = NULL;
    a->scratch_re = NULL;
    a->scratch_im = NULL;
}

notorious_fft_aux* notorious_fft_mkaux_dft_1d(int N) {
    if (N <= 0) return NULL;

    notorious_fft_aux* a = (notorious_fft_aux*)malloc(sizeof(notorious_fft_aux));
    if (!a) return NULL;

    a->N = N;
    a->plan = notorious_fft_create_plan(N, 0);
    if (!a->plan) {
        free(a);
        return NULL;
    }

    notorious_fft_init_aux_from_plan(a);
    return a;
}

static notorious_fft_aux* notorious_fft_mkaux_dft_1d_internal(int N) {
    return notorious_fft_mkaux_dft_1d(N);
}

notorious_fft_aux* notorious_fft_mkaux_dft(int d, int* Ns) {
    return notorious_fft_make_aux(d, Ns, sizeof(notorious_fft_cmpl), notorious_fft_mkaux_dft_1d_internal);
}

notorious_fft_aux* notorious_fft_mkaux_dft_2d(int N1, int N2) {
    int Ns[2] = {N1, N2};
    return notorious_fft_mkaux_dft(2, Ns);
}

notorious_fft_aux* notorious_fft_mkaux_dft_3d(int N1, int N2, int N3) {
    int Ns[3] = {N1, N2, N3};
    return notorious_fft_mkaux_dft(3, Ns);
}

/* Precompute realdft twiddles and scratch buffers */
notorious_fft_aux* notorious_fft_mkaux_realdft_1d(int N) {
    if (N <= 0 || (N & (N - 1))) return NULL;

    notorious_fft_aux* a = (notorious_fft_aux*)malloc(sizeof(notorious_fft_aux));
    if (!a) return NULL;

    a->N = N;
    notorious_fft_init_aux_from_plan(a);

    if (N >= 4) {
        a->plan = notorious_fft_create_plan(N / 2, 0);
        if (!a->plan) {
            free(a);
            return NULL;
        }

        /* Precompute unpack twiddles: N/4 pairs (cos(-2πk/N), sin(-2πk/N)) */
        a->e = notorious_fft_malloc((size_t)(N / 2) * sizeof(notorious_fft_real));
        if (!a->e) {
            notorious_fft_destroy_plan(a->plan);
            free(a);
            return NULL;
        }
        {
            notorious_fft_real* tw = (notorious_fft_real*)a->e;
            for (int k = 0; k < N/4; k++) {
                notorious_fft_real angle = -NOTORIOUS_FFT_2PI * (notorious_fft_real)k / (notorious_fft_real)N;
                tw[2*k]   = notorious_fft_cos(angle);
                tw[2*k+1] = notorious_fft_sin(angle);
            }
        }

        /* Preallocate scratch: N reals (M interleaved complex for inner FFT) */
        a->t = notorious_fft_malloc((size_t)N * sizeof(notorious_fft_real));
        if (!a->t) {
            notorious_fft_free(a->e);
            notorious_fft_destroy_plan(a->plan);
            free(a);
            return NULL;
        }
    } else {
        a->plan = NULL;
    }

    return a;
}

static notorious_fft_aux* notorious_fft_mkaux_realdft_1d_internal(int N) {
    return notorious_fft_mkaux_realdft_1d(N);
}

notorious_fft_aux* notorious_fft_mkaux_realdft(int d, int* Ns) {
    if (d == 1) {
        return notorious_fft_mkaux_realdft_1d(Ns[0]);
    } else {
        int p = 1;
        for (int i = 0; i < d - 1; i++) p *= Ns[i];

        notorious_fft_aux* a = (notorious_fft_aux*)malloc(sizeof(notorious_fft_aux));
        if (!a) return NULL;

        a->N = Ns[d-1] * p;
        a->plan = NULL;
        a->sub1 = notorious_fft_mkaux_realdft_1d(Ns[d-1]);
        a->sub2 = notorious_fft_mkaux_dft(d - 1, Ns);

        if (!a->sub1 || !a->sub2) {
            notorious_fft_free_aux(a);
            return NULL;
        }

        return a;
    }
}

notorious_fft_aux* notorious_fft_mkaux_realdft_2d(int N1, int N2) {
    int Ns[2] = {N1, N2};
    return notorious_fft_mkaux_realdft(2, Ns);
}

notorious_fft_aux* notorious_fft_mkaux_realdft_3d(int N1, int N2, int N3) {
    int Ns[3] = {N1, N2, N3};
    return notorious_fft_mkaux_realdft(3, Ns);
}

/* DCT/DST Type 2/3: builds on realdft, adds precomputed DCT twiddles + scratch */
notorious_fft_aux* notorious_fft_mkaux_t2t3_1d(int N) {
    /* Start with realdft aux (gives us plan, unpack twiddles, realdft scratch) */
    notorious_fft_aux* a = notorious_fft_mkaux_realdft_1d(N);
    if (!a) return NULL;

    if (N >= 4) {
        /* Precompute DCT-2/3 post-processing twiddles:
         * N/2 pairs: (cos(-πk/(2N)), sin(-πk/(2N))) for k=0..N/2-1 */
        a->scratch_re = (notorious_fft_real*)notorious_fft_malloc((size_t)N * sizeof(notorious_fft_real));
        if (!a->scratch_re) {
            notorious_fft_free_aux(a);
            return NULL;
        }
        {
            notorious_fft_real* dct_tw = a->scratch_re;
            for (int k = 0; k < N/2; k++) {
                notorious_fft_real angle = -NOTORIOUS_FFT_PI * (notorious_fft_real)k / (notorious_fft_real)(2 * N);
                dct_tw[2*k]   = notorious_fft_cos(angle);
                dct_tw[2*k+1] = notorious_fft_sin(angle);
            }
        }

        /* Preallocate DCT scratch: reordered(N) + z_buf(N+2) */
        a->scratch_im = (notorious_fft_real*)notorious_fft_malloc((size_t)(2 * N + 2) * sizeof(notorious_fft_real));
        if (!a->scratch_im) {
            notorious_fft_free_aux(a);
            return NULL;
        }
    }

    return a;
}

static notorious_fft_aux* notorious_fft_mkaux_t2t3_1d_internal(int N) {
    return notorious_fft_mkaux_t2t3_1d(N);
}

notorious_fft_aux* notorious_fft_mkaux_t2t3(int d, int* Ns) {
    return notorious_fft_make_aux(d, Ns, sizeof(notorious_fft_real), notorious_fft_mkaux_t2t3_1d_internal);
}

notorious_fft_aux* notorious_fft_mkaux_t2t3_2d(int N1, int N2) {
    int Ns[2] = {N1, N2};
    return notorious_fft_mkaux_t2t3(2, Ns);
}

notorious_fft_aux* notorious_fft_mkaux_t2t3_3d(int N1, int N2, int N3) {
    int Ns[3] = {N1, N2, N3};
    return notorious_fft_mkaux_t2t3(3, Ns);
}

notorious_fft_aux* notorious_fft_mkaux_t4_1d(int N) {
    if (N <= 0 || (N & (N - 1))) return NULL;

    notorious_fft_aux* a = (notorious_fft_aux*)malloc(sizeof(notorious_fft_aux));
    if (!a) return NULL;

    a->N = N;
    a->plan = NULL;
    notorious_fft_init_aux_from_plan(a);

    if (N >= 2) {
        int M = N / 2;  /* inner DFT size */

        /* For N=2, M=1 — no plan needed, DFT is trivial identity */
        if (M >= 2) {
            a->plan = notorious_fft_create_plan(M, 0);
            if (!a->plan) { free(a); return NULL; }
        }

        /* e: premul twiddles exp(-iπn/N) for n=0..M-1  (M pairs = 2M reals)
         *    postmul twiddles exp(-iπ(2n+1)/(4N)) for n=0..N-1 (N pairs = 2N reals)
         * Total: 2*(M+N) reals = 2*(3N/2) = 3N reals */
        a->e = notorious_fft_malloc((size_t)(2 * M + 2 * N) * sizeof(notorious_fft_real));
        if (!a->e) {
            if (a->plan) notorious_fft_destroy_plan(a->plan);
            free(a);
            return NULL;
        }
        {
            notorious_fft_real* ep = (notorious_fft_real*)a->e;
            /* premul: exp(-iπn/N) for n=0..M-1 */
            for (int n = 0; n < M; n++) {
                notorious_fft_real ang = -NOTORIOUS_FFT_PI * (notorious_fft_real)n / (notorious_fft_real)N;
                ep[2*n]   = notorious_fft_cos(ang);
                ep[2*n+1] = notorious_fft_sin(ang);
            }
            /* postmul: exp(-iπ(2n+1)/(4N)) for n=0..N-1 */
            ep += 2 * M;
            for (int n = 0; n < N; n++) {
                notorious_fft_real ang = -NOTORIOUS_FFT_PI * (notorious_fft_real)(2*n + 1) / (notorious_fft_real)(4 * N);
                ep[2*n]   = notorious_fft_cos(ang);
                ep[2*n+1] = notorious_fft_sin(ang);
            }
        }

        /* t: scratch for M interleaved complex = 2M = N reals */
        a->t = notorious_fft_malloc((size_t)N * sizeof(notorious_fft_real));
        if (!a->t) {
            notorious_fft_free(a->e);
            if (a->plan) notorious_fft_destroy_plan(a->plan);
            free(a);
            return NULL;
        }
    }

    return a;
}

static notorious_fft_aux* notorious_fft_mkaux_t4_1d_internal(int N) {
    return notorious_fft_mkaux_t4_1d(N);
}

notorious_fft_aux* notorious_fft_mkaux_t4(int d, int* Ns) {
    return notorious_fft_make_aux(d, Ns, sizeof(notorious_fft_real), notorious_fft_mkaux_t4_1d_internal);
}

notorious_fft_aux* notorious_fft_mkaux_t4_2d(int N1, int N2) {
    int Ns[2] = {N1, N2};
    return notorious_fft_mkaux_t4(2, Ns);
}

notorious_fft_aux* notorious_fft_mkaux_t4_3d(int N1, int N2, int N3) {
    int Ns[3] = {N1, N2, N3};
    return notorious_fft_mkaux_t4(3, Ns);
}

static notorious_fft_aux* notorious_fft_make_aux(int d, int* Ns, int datasz,
                                    notorious_fft_aux* (*aux_1d)(int N)) {
    int p = 1;
    for (int i = 0; i < d; i++) p *= Ns[i];

    notorious_fft_aux* a = (notorious_fft_aux*)malloc(sizeof(notorious_fft_aux));
    if (!a) return NULL;

    a->N = p;
    a->plan = NULL;
    a->t = NULL;
    a->e = NULL;
    a->scratch_re = NULL;
    a->scratch_im = NULL;

    if (d == 1) {
        a->sub1 = NULL;
        a->sub2 = (*aux_1d)(Ns[0]);
    } else {
        a->t = notorious_fft_malloc((size_t)p * datasz);
        if (!a->t) {
            free(a);
            return NULL;
        }
        a->sub1 = notorious_fft_make_aux(d - 1, Ns + 1, datasz, aux_1d);
        a->sub2 = (*aux_1d)(Ns[0]);
    }

    if ((d > 1 && !a->sub1) || !a->sub2) {
        notorious_fft_free_aux(a);
        return NULL;
    }

    return a;
}

void notorious_fft_free_aux(notorious_fft_aux* a) {
    if (!a) return;
    if (a->plan) notorious_fft_destroy_plan(a->plan);
    notorious_fft_free(a->t);
    notorious_fft_free(a->e);
    notorious_fft_free(a->scratch_re);
    notorious_fft_free(a->scratch_im);
    notorious_fft_free_aux(a->sub1);
    notorious_fft_free_aux(a->sub2);
    free(a);
}

#endif /* NOTORIOUS_FFT_IMPLEMENTATION */

#endif /* NOTORIOUS_FFT_LEGACY_H */
