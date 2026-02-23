/*
 * Notorious FFT - Core FFT Algorithms
 * Iterative, cache-oblivious recursive, and Bluestein
 */

#ifndef NOTORIOUS_FFT_ALGORITHMS_H
#define NOTORIOUS_FFT_ALGORITHMS_H

#include "04_kernels.h"

/* Forward declarations */
static void notorious_fft_execute_sr_dif(const notorious_fft_plan* plan, const notorious_fft_real* x_in,
                                   notorious_fft_real* y_out, int inverse);
static void notorious_fft_sr_dif_cx(int N, notorious_fft_real* x, notorious_fft_real* t,
                               notorious_fft_real* y, int sy, const notorious_fft_real* e);
static void notorious_fft_sr_inv_dif_cx(int N, notorious_fft_real* x, notorious_fft_real* t,
                                   notorious_fft_real* y, int sy, const notorious_fft_real* e);
static void notorious_fft_execute_bluestein(
    const notorious_fft_plan* plan,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_in, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_in,
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_out, notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_out);

/* ============================================================================
 * Iterative Cooley-Tukey FFT
 * ============================================================================ */

/* Internal iterative FFT with direction control (0=forward, 1=inverse) */
static void notorious_fft_iterative_body_internal(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT wr, notorious_fft_real* NOTORIOUS_FFT_RESTRICT wi,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_in, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_in,
    const int* bitrev,
    const notorious_fft_real* tw_re, const notorious_fft_real* tw_im,
    size_t n,
    int inverse)
{
    /* Bit-reversal permutation */
    for (size_t i = 0; i < n; i++) {
        size_t j = bitrev[i];
        wr[i] = xr_in[j];
        wi[i] = xi_in[j];
    }
    
    /* Iterative butterfly stages */
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        
        #if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n >= 4096 && len >= 32)
        #endif
        for (size_t i = 0; i < n; i += len) {
            size_t j = 0;
            
            #if NOTORIOUS_FFT_HAS_AVX512
            {
                int32_t indices[8];
                for (; j + 8 <= half; j += 8) {
                    for (int k = 0; k < 8; k++) indices[k] = (int32_t)((j + k) * step);
                    if (inverse) {
                        /* For inverse, negate twiddle imag parts after gather */
                        __m256i idx = _mm256_loadu_si256((__m256i*)indices);
                        __m512d t_wr = _mm512_i32gather_pd(idx, tw_re, 8);
                        __m512d t_wi = _mm512_i32gather_pd(idx, tw_im, 8);
                        t_wi = _mm512_sub_pd(_mm512_setzero_pd(), t_wi); /* negate */
                        notorious_fft_butterfly8_avx512_inverse(wr, wi, i + j, i + j + half, t_wr, t_wi);
                    } else {
                        notorious_fft_butterfly8_avx512(wr, wi, tw_re, tw_im, i + j, i + j + half, indices);
                    }
                }
            }
            #elif NOTORIOUS_FFT_HAS_AVX2
            {
                int32_t idx[4];
                for (; j + 4 <= half; j += 4) {
                    for (int k = 0; k < 4; k++) idx[k] = (int32_t)((j + k) * step);
                    __m128i indices = _mm_loadu_si128((__m128i*)idx);
                    if (inverse) {
                        /* For inverse, negate twiddle imag parts after gather */
                        __m256d t_wr = _mm256_i32gather_pd(tw_re, indices, 8);
                        __m256d t_wi = _mm256_i32gather_pd(tw_im, indices, 8);
                        t_wi = _mm256_sub_pd(_mm256_setzero_pd(), t_wi); /* negate */
                        notorious_fft_butterfly4_avx2_inverse(wr, wi, i + j, i + j + half, t_wr, t_wi);
                    } else {
                        notorious_fft_butterfly4_avx2(wr, wi, tw_re, tw_im, i + j, i + j + half, indices);
                    }
                }
            }
            #elif NOTORIOUS_FFT_HAS_NEON
            {
                #ifdef NOTORIOUS_FFT_SINGLE
                for (; j + 4 <= half; j += 4) {
                    float wr0 = tw_re[(j+0)*step], wi0 = inverse ? -tw_im[(j+0)*step] : tw_im[(j+0)*step];
                    float wr1 = tw_re[(j+1)*step], wi1 = inverse ? -tw_im[(j+1)*step] : tw_im[(j+1)*step];
                    float wr2 = tw_re[(j+2)*step], wi2 = inverse ? -tw_im[(j+2)*step] : tw_im[(j+2)*step];
                    float wr3 = tw_re[(j+3)*step], wi3 = inverse ? -tw_im[(j+3)*step] : tw_im[(j+3)*step];
                    notorious_fft_butterfly4_neon_f32(wr, wi, wr0, wi0, wr1, wi1, wr2, wi2, wr3, wi3,
                                                i + j, i + j + half);
                }
                #else
                for (; j + 2 <= half; j += 2) {
                    double wr0 = tw_re[(j+0)*step], wi0 = inverse ? -tw_im[(j+0)*step] : tw_im[(j+0)*step];
                    double wr1 = tw_re[(j+1)*step], wi1 = inverse ? -tw_im[(j+1)*step] : tw_im[(j+1)*step];
                    notorious_fft_butterfly2_neon_f64(wr, wi, wr0, wi0, wr1, wi1, i + j, i + j + half);
                }
                #endif
            }
            #endif
            
            /* Scalar cleanup */
            for (; j < half; j++) {
                notorious_fft_real t_wr = tw_re[j * step];
                notorious_fft_real t_wi = inverse ? -tw_im[j * step] : tw_im[j * step];
                notorious_fft_butterfly_scalar(wr, wi, i + j, i + j + half, t_wr, t_wi);
            }
        }
    }
}

/* Internal function with direction control (0=forward, 1=inverse) */
static void notorious_fft_execute_iterative_internal(
    const notorious_fft_plan* plan,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_in, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_in,
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_out, notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_out,
    int inverse)
{
    size_t n = plan->n;
    
    /* Use iterative algorithm - hardcoded kernels have bugs */
    notorious_fft_real* wr = plan->work_re;
    notorious_fft_real* wi = plan->work_im;
    
    notorious_fft_iterative_body_internal(wr, wi, xr_in, xi_in, plan->bitrev, plan->tw_re, plan->tw_im, n, inverse);
    
    memcpy(xr_out, wr, n * sizeof(notorious_fft_real));
    memcpy(xi_out, wi, n * sizeof(notorious_fft_real));
}

/* Public API - forward FFT only (keeps original signature) */
static void notorious_fft_execute_iterative(
    const notorious_fft_plan* plan,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_in, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_in,
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_out, notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_out)
{
    notorious_fft_execute_iterative_internal(plan, xr_in, xi_in, xr_out, xi_out, 0);
}

/* ============================================================================
 * In-place iterative FFT on interleaved complex (re,im pairs)
 *
 * Avoids the split-complex deinterleave/interleave round-trip in the
 * minfft-compatible API.  Operates directly on the notorious_fft_cmpl* (double[2])
 * array that the caller already has.
 * ============================================================================ */

static void notorious_fft_iterative_inplace_cx(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT data,  /* interleaved: data[2*i]=re, data[2*i+1]=im */
    const int*         bitrev,
    const notorious_fft_real* tw_re,
    const notorious_fft_real* tw_im,
    size_t n,
    int inverse)
{
    /* Fast path: use proven split-radix terminal cases for small N.
     * This avoids bit-reversal overhead which is the main cost for small FFTs.
     * The split-radix N=8 terminal case is extensively tested and correct. */
    if (n == 8) {
        /* Use split-radix N=8 terminal case via notorious_fft_sr_dif_cx.
         * Need temp buffer for the butterfly stage output. */
        notorious_fft_real t[16];  /* 8 complex = 16 reals */
        if (inverse) {
            notorious_fft_sr_inv_dif_cx(8, data, t, data, 1, NULL);
        } else {
            notorious_fft_sr_dif_cx(8, data, t, data, 1, NULL);
        }
        return;
    }

    /* Fast path: N=4 using split-radix terminal case */
    if (n == 4) {
        notorious_fft_real t[8];  /* 4 complex = 8 reals */
        if (inverse) {
            notorious_fft_sr_inv_dif_cx(4, data, t, data, 1, NULL);
        } else {
            notorious_fft_sr_dif_cx(4, data, t, data, 1, NULL);
        }
        return;
    }

    /* Fast path: N=2 using split-radix terminal case */
    if (n == 2) {
        notorious_fft_real t[4];  /* 2 complex = 4 reals */
        if (inverse) {
            notorious_fft_sr_inv_dif_cx(2, data, t, data, 1, NULL);
        } else {
            notorious_fft_sr_dif_cx(2, data, t, data, 1, NULL);
        }
        return;
    }

    /* Bit-reversal permutation (swap pairs) */
    for (size_t i = 0; i < n; i++) {
        size_t j = (size_t)bitrev[i];
        if (j > i) {
            notorious_fft_real tr = data[2*i],   ti = data[2*i+1];
            data[2*i]   = data[2*j];   data[2*i+1] = data[2*j+1];
            data[2*j]   = tr;          data[2*j+1] = ti;
        }
    }

    /* Iterative Cooley–Tukey stages */
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;

#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n >= 4096 && len >= 32)
#endif
        for (size_t i = 0; i < n; i += len) {
            size_t j = 0;

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
            /* AVX2 double: process 2 complex per iteration on interleaved data */
            for (; j + 2 <= half; j += 2) {
                __m256d tw_r, tw_i;
                {
                    double wr0 = tw_re[(j+0)*step], wr1 = tw_re[(j+1)*step];
                    double wi0 = tw_im[(j+0)*step], wi1 = tw_im[(j+1)*step];
                    if (inverse) { wi0 = -wi0; wi1 = -wi1; }
                    tw_r = _mm256_set_pd(wr1, wr1, wr0, wr0);
                    tw_i = _mm256_set_pd(wi1, wi1, wi0, wi0);
                }

                __m256d ab = _mm256_loadu_pd(&data[2*(i+j)]);
                __m256d cd = _mm256_loadu_pd(&data[2*(i+j+half)]);

                /* Complex multiply cd * tw on interleaved data */
                __m256d cd_swap = _mm256_shuffle_pd(cd, cd, 0x5);
                __m256d cmul_sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
                __m256d p1 = _mm256_mul_pd(cd, tw_r);
                __m256d p2 = _mm256_mul_pd(cd_swap, tw_i);
                __m256d prod = _mm256_add_pd(p1, _mm256_mul_pd(p2, cmul_sign));

                _mm256_storeu_pd(&data[2*(i+j)],      _mm256_add_pd(ab, prod));
                _mm256_storeu_pd(&data[2*(i+j+half)],  _mm256_sub_pd(ab, prod));
            }
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
            /* NEON double: vld2q_f64 deinterleaves in hardware */
            for (; j + 2 <= half; j += 2) {
                float64x2_t tw_r = (float64x2_t){tw_re[(j+0)*step], tw_re[(j+1)*step]};
                float64x2_t tw_i = inverse
                    ? (float64x2_t){-tw_im[(j+0)*step], -tw_im[(j+1)*step]}
                    : (float64x2_t){ tw_im[(j+0)*step],  tw_im[(j+1)*step]};

                float64x2x2_t ab = vld2q_f64(&data[2*(i+j)]);
                float64x2x2_t cd = vld2q_f64(&data[2*(i+j+half)]);

                float64x2_t vr = vsubq_f64(vmulq_f64(cd.val[0], tw_r), vmulq_f64(cd.val[1], tw_i));
                float64x2_t vi = vaddq_f64(vmulq_f64(cd.val[0], tw_i), vmulq_f64(cd.val[1], tw_r));

                float64x2x2_t out_p, out_q;
                out_p.val[0] = vaddq_f64(ab.val[0], vr);
                out_p.val[1] = vaddq_f64(ab.val[1], vi);
                out_q.val[0] = vsubq_f64(ab.val[0], vr);
                out_q.val[1] = vsubq_f64(ab.val[1], vi);

                vst2q_f64(&data[2*(i+j)],        out_p);
                vst2q_f64(&data[2*(i+j+half)],   out_q);
            }
#endif
            /* Scalar remainder */
            for (; j < half; j++) {
                notorious_fft_real wr = tw_re[j * step];
                notorious_fft_real wi = inverse ? -tw_im[j * step] : tw_im[j * step];

                size_t p = 2 * (i + j);
                size_t q = 2 * (i + j + half);

                notorious_fft_real ur = data[p],   ui = data[p+1];
                notorious_fft_real vr = data[q] * wr - data[q+1] * wi;
                notorious_fft_real vi = data[q] * wi + data[q+1] * wr;

                data[p]   = ur + vr;  data[p+1] = ui + vi;
                data[q]   = ur - vr;  data[q+1] = ui - vi;
            }
        }
    }
}

/* Wrapper that accepts separate in/out buffers.
 * Routes to split-radix DIF (no bit-reversal, better cache) for large N
 * where sr_e/sr_t are available, otherwise falls back to iterative DIT. */
static void notorious_fft_execute_cx(
    const notorious_fft_plan* plan,
    const notorious_fft_real* x_in,   /* interleaved input  */
    notorious_fft_real*       y_out,  /* interleaved output */
    int inverse)
{
    size_t n = plan->n;

    /* Bluestein (non-power-of-2): deinterleave → split execute → reinterleave.
     * work_re/im are not used by notorious_fft_execute_bluestein, safe for input.
     * bluestein_fft_buf_re/im are free after execute returns, safe for output. */
    if (plan->execute_func == notorious_fft_execute_bluestein) {
        notorious_fft_real* in_re = plan->work_re;
        notorious_fft_real* in_im = plan->work_im;
        notorious_fft_real* out_re = plan->bluestein_fft_buf_re;
        notorious_fft_real* out_im = plan->bluestein_fft_buf_im;

        /* Deinterleave input into split re/im */
        for (size_t i = 0; i < n; i++) {
            in_re[i] = x_in[2*i];
            in_im[i] = x_in[2*i+1];
        }

        /* Handle inverse via conjugate trick: IDFT(x) = (1/N)*conj(DFT(conj(x))) */
        if (inverse) {
            for (size_t i = 0; i < n; i++) in_im[i] = -in_im[i];
        }

        notorious_fft_execute_bluestein(plan, in_re, in_im, out_re, out_im);

        if (inverse) {
            notorious_fft_real scale = (notorious_fft_real)1.0 / (notorious_fft_real)n;
            for (size_t i = 0; i < n; i++) {
                y_out[2*i]   =  out_re[i] * scale;
                y_out[2*i+1] = -out_im[i] * scale;
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                y_out[2*i]   = out_re[i];
                y_out[2*i+1] = out_im[i];
            }
        }
        return;
    }

    if (plan->sr_e && plan->sr_t && n >= 16) {
        notorious_fft_execute_sr_dif(plan, x_in, y_out, inverse);
    } else {
        if (x_in != y_out)
            memcpy(y_out, x_in, 2 * n * sizeof(notorious_fft_real));
        notorious_fft_iterative_inplace_cx(y_out, plan->bitrev, plan->tw_re, plan->tw_im, n, inverse);
    }
}

/* ============================================================================
 * Split-Radix DIF (Decimation-In-Frequency) — minfft-compatible algorithm
 *
 * Works directly on interleaved complex data (re,im pairs), no bit-reversal
 * needed.  Same split-radix 2/4 structure as minfft's rs_dft_1d.
 *
 * Twiddle layout in sr_e (same as minfft):
 *   For each recursion level of size N (N≥16, processed from large to small):
 *     N/4 quads: { cos(-k·2π/N), sin(-k·2π/N),
 *                  cos(-3k·2π/N), sin(-3k·2π/N) }   k = 0..N/4-1
 * ============================================================================ */

static void notorious_fft_sr_dif_cx(int N, notorious_fft_real* x, notorious_fft_real* t,
                               notorious_fft_real* y, int sy,
                               const notorious_fft_real* e)
{
    notorious_fft_real* xr = x, *xi = x + 1;
    notorious_fft_real* tr = t, *ti = t + 1;
    notorious_fft_real* yr = y, *yi = y + 1;

    if (N == 1) {
        yr[0] = xr[0]; yi[0] = xi[0];
        return;
    }
    if (N == 2) {
        notorious_fft_real t0r = xr[0] + xr[2], t0i = xi[0] + xi[2];
        notorious_fft_real t1r = xr[0] - xr[2], t1i = xi[0] - xi[2];
        yr[0]      = t0r; yi[0]      = t0i;
        yr[2*sy]   = t1r; yi[2*sy]   = t1i;
        return;
    }
    if (N == 4) {
        notorious_fft_real t0r = xr[0] + xr[4], t0i = xi[0] + xi[4];
        notorious_fft_real t1r = xr[2] + xr[6], t1i = xi[2] + xi[6];
        notorious_fft_real t2r = xr[0] - xr[4], t2i = xi[0] - xi[4];
        /* t3 = i*(x[1]-x[3]) */
        notorious_fft_real t3r = -xi[2] + xi[6], t3i = xr[2] - xr[6];
        yr[0]      = t0r + t1r; yi[0]      = t0i + t1i;
        yr[2*sy]   = t2r - t3r; yi[2*sy]   = t2i - t3i;
        yr[4*sy]   = t0r - t1r; yi[4*sy]   = t0i - t1i;
        yr[6*sy]   = t2r + t3r; yi[6*sy]   = t2i + t3i;
        return;
    }
    if (N == 8) {
        /* Unrolled N=8 split-radix — identical to minfft terminal case */
        notorious_fft_real t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;
        notorious_fft_real t00r,t00i,t01r,t01i,t02r,t02i,t03r,t03i;
        notorious_fft_real t10r,t10i,t11r,t11i,t12r,t12i,t13r,t13i;
        notorious_fft_real ttr,tti;
        const notorious_fft_real invsqrt2 = NOTORIOUS_FFT_INV_SQRT2;

        t0r=xr[0]+xr[8];  t0i=xi[0]+xi[8];
        t1r=xr[4]+xr[12]; t1i=xi[4]+xi[12];
        t2r=xr[0]-xr[8];  t2i=xi[0]-xi[8];
        t3r=-xi[4]+xi[12]; t3i=xr[4]-xr[12];
        t00r=t0r+t1r; t00i=t0i+t1i;
        t01r=t2r-t3r; t01i=t2i-t3i;
        t02r=t0r-t1r; t02i=t0i-t1i;
        t03r=t2r+t3r; t03i=t2i+t3i;

        t0r=xr[2]+xr[10]; t0i=xi[2]+xi[10];
        t1r=xr[6]+xr[14]; t1i=xi[6]+xi[14];
        t2r=xr[2]-xr[10]; t2i=xi[2]-xi[10];
        t3r=-xi[6]+xi[14]; t3i=xr[6]-xr[14];

        t10r=t0r+t1r; t10i=t0i+t1i;
        ttr=t2r-t3r; tti=t2i-t3i;
        t11r=invsqrt2*(ttr+tti); t11i=invsqrt2*(tti-ttr);
        t12r=t0i-t1i; t12i=-t0r+t1r;
        ttr=t2r+t3r; tti=t2i+t3i;
        t13r=invsqrt2*(tti-ttr); t13i=-invsqrt2*(tti+ttr);

        yr[0]=t00r+t10r;    yi[0]=t00i+t10i;
        yr[2*sy]=t01r+t11r; yi[2*sy]=t01i+t11i;
        yr[4*sy]=t02r+t12r; yi[4*sy]=t02i+t12i;
        yr[6*sy]=t03r+t13r; yi[6*sy]=t03i+t13i;
        yr[8*sy]=t00r-t10r; yi[8*sy]=t00i-t10i;
        yr[10*sy]=t01r-t11r; yi[10*sy]=t01i-t11i;
        yr[12*sy]=t02r-t12r; yi[12*sy]=t02i-t12i;
        yr[14*sy]=t03r-t13r; yi[14*sy]=t03i-t13i;
        return;
    }

    /* General recursion: split-radix DIF butterfly stage then recurse */
    /* N >= 16 */
    int n4 = N / 4;
    const notorious_fft_real* ep = e;  /* points to current level's twiddles */

    {
        int k = 0;

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        /* AVX2 double: process 2 complex per iteration using 256-bit ops.
         * Load 4 doubles {re0,im0,re1,im1}, deinterleave with permute. */
        for (; k + 2 <= n4; k += 2) {
            /* Twiddles: gather from ep[4*k..] layout */
            __m256d w1r = _mm256_set_pd(ep[4*(k+1)],   ep[4*(k+1)],   ep[4*k],   ep[4*k]);
            __m256d w1i = _mm256_set_pd(ep[4*(k+1)+1], ep[4*(k+1)+1], ep[4*k+1], ep[4*k+1]);
            __m256d w3r = _mm256_set_pd(ep[4*(k+1)+2], ep[4*(k+1)+2], ep[4*k+2], ep[4*k+2]);
            __m256d w3i = _mm256_set_pd(ep[4*(k+1)+3], ep[4*(k+1)+3], ep[4*k+3], ep[4*k+3]);

            /* Load 2 complex values each: {re0,im0,re1,im1} */
            __m256d a = _mm256_loadu_pd(xr + 2*k);
            __m256d b = _mm256_loadu_pd(xr + 2*(k+N/2));
            __m256d c = _mm256_loadu_pd(xr + 2*(k+n4));
            __m256d d = _mm256_loadu_pd(xr + 2*(k+3*n4));

            __m256d t0 = _mm256_add_pd(a, b);  /* a+b interleaved */
            __m256d t1 = _mm256_add_pd(c, d);  /* c+d interleaved */
            __m256d t2 = _mm256_sub_pd(a, b);  /* a-b interleaved */
            __m256d cd_diff = _mm256_sub_pd(c, d);

            /* t3 = i*(c-d): swap re/im and negate new re
             * cd_diff = {re0,im0,re1,im1} → t3 = {-im0,re0,-im1,re1} */
            __m256d cd_swap = _mm256_shuffle_pd(cd_diff, cd_diff, 0x5); /* {im0,re0,im1,re1} */
            __m256d sign_mask = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
            __m256d t3 = _mm256_mul_pd(cd_swap, sign_mask);

            __m256d u = _mm256_sub_pd(t2, t3);
            __m256d v = _mm256_add_pd(t2, t3);

            _mm256_storeu_pd(tr + 2*k, t0);
            _mm256_storeu_pd(tr + 2*(k+n4), t1);

            /* u * w1: complex multiply on interleaved data
             * u = {ur0,ui0,ur1,ui1}, w1r = {wr0,wr0,wr1,wr1}, w1i = {wi0,wi0,wi1,wi1}
             * result_re = ur*wr - ui*wi, result_im = ur*wi + ui*wr */
            __m256d u_swap = _mm256_shuffle_pd(u, u, 0x5); /* {ui0,ur0,ui1,ur1} */
            __m256d p1 = _mm256_mul_pd(u, w1r);           /* {ur*wr, ui*wr, ...} */
            __m256d p2 = _mm256_mul_pd(u_swap, w1i);      /* {ui*wi, ur*wi, ...} */
            __m256d cmul_sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
            __m256d uw1 = _mm256_add_pd(p1, _mm256_mul_pd(p2, cmul_sign));

            _mm256_storeu_pd(tr + 2*(k+N/2), uw1);

            /* v * w3: same complex multiply pattern */
            __m256d v_swap = _mm256_shuffle_pd(v, v, 0x5);
            __m256d q1 = _mm256_mul_pd(v, w3r);
            __m256d q2 = _mm256_mul_pd(v_swap, w3i);
            __m256d vw3 = _mm256_add_pd(q1, _mm256_mul_pd(q2, cmul_sign));

            _mm256_storeu_pd(tr + 2*(k+3*n4), vw3);
        }

#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        /* NEON double: process 2 complex per iteration using vld2q_f64 deinterleave */
        for (; k + 2 <= n4; k += 2) {
            float64x2_t w1r = (float64x2_t){ep[4*k],   ep[4*(k+1)]};
            float64x2_t w1i = (float64x2_t){ep[4*k+1], ep[4*(k+1)+1]};
            float64x2_t w3r = (float64x2_t){ep[4*k+2], ep[4*(k+1)+2]};
            float64x2_t w3i = (float64x2_t){ep[4*k+3], ep[4*(k+1)+3]};

            float64x2x2_t xa = vld2q_f64(xr + 2*k);
            float64x2x2_t xb = vld2q_f64(xr + 2*(k+N/2));
            float64x2x2_t xc = vld2q_f64(xr + 2*(k+n4));
            float64x2x2_t xd = vld2q_f64(xr + 2*(k+3*n4));

            float64x2_t t0r_ = vaddq_f64(xa.val[0], xb.val[0]);
            float64x2_t t0i_ = vaddq_f64(xa.val[1], xb.val[1]);
            float64x2_t t1r_ = vaddq_f64(xc.val[0], xd.val[0]);
            float64x2_t t1i_ = vaddq_f64(xc.val[1], xd.val[1]);
            float64x2_t t2r_ = vsubq_f64(xa.val[0], xb.val[0]);
            float64x2_t t2i_ = vsubq_f64(xa.val[1], xb.val[1]);
            float64x2_t t3r_ = vsubq_f64(xd.val[1], xc.val[1]);
            float64x2_t t3i_ = vsubq_f64(xc.val[0], xd.val[0]);

            float64x2_t ur_ = vsubq_f64(t2r_, t3r_), ui_ = vsubq_f64(t2i_, t3i_);
            float64x2_t vr_ = vaddq_f64(t2r_, t3r_), vi_ = vaddq_f64(t2i_, t3i_);

            float64x2x2_t out0 = {{t0r_, t0i_}};
            vst2q_f64(tr + 2*k, out0);
            float64x2x2_t out1 = {{t1r_, t1i_}};
            vst2q_f64(tr + 2*(k+n4), out1);

            float64x2_t pr_ = vsubq_f64(vmulq_f64(ur_,w1r), vmulq_f64(ui_,w1i));
            float64x2_t pi_ = vaddq_f64(vmulq_f64(ur_,w1i), vmulq_f64(ui_,w1r));
            float64x2x2_t out2 = {{pr_, pi_}};
            vst2q_f64(tr + 2*(k+N/2), out2);

            float64x2_t qr_ = vsubq_f64(vmulq_f64(vr_,w3r), vmulq_f64(vi_,w3i));
            float64x2_t qi_ = vaddq_f64(vmulq_f64(vr_,w3i), vmulq_f64(vi_,w3r));
            float64x2x2_t out3 = {{qr_, qi_}};
            vst2q_f64(tr + 2*(k+3*n4), out3);
        }
#endif

        /* Scalar remainder / fallback */
        for (; k < n4; k++) {
            notorious_fft_real x0r=xr[2*k],x0i=xi[2*k];
            notorious_fft_real x1r=xr[2*(k+N/2)],x1i=xi[2*(k+N/2)];
            notorious_fft_real x2r=xr[2*(k+n4)],x2i=xi[2*(k+n4)];
            notorious_fft_real x3r=xr[2*(k+3*n4)],x3i=xi[2*(k+3*n4)];
            notorious_fft_real t0r_=x0r+x1r,t0i_=x0i+x1i;
            notorious_fft_real t1r_=x2r+x3r,t1i_=x2i+x3i;
            notorious_fft_real t2r_=x0r-x1r,t2i_=x0i-x1i;
            notorious_fft_real t3r_=x3i-x2i,t3i_=x2r-x3r;
            notorious_fft_real ur_=t2r_-t3r_,ui_=t2i_-t3i_;
            notorious_fft_real vr_=t2r_+t3r_,vi_=t2i_+t3i_;
            tr[2*k]=t0r_; ti[2*k]=t0i_;
            tr[2*(k+n4)]=t1r_; ti[2*(k+n4)]=t1i_;
            tr[2*(k+N/2)]=ur_*ep[4*k]-ui_*ep[4*k+1];
            ti[2*(k+N/2)]=ur_*ep[4*k+1]+ui_*ep[4*k];
            tr[2*(k+3*n4)]=vr_*ep[4*k+2]-vi_*ep[4*k+3];
            ti[2*(k+3*n4)]=vr_*ep[4*k+3]+vi_*ep[4*k+2];
        }
    }

    /* e pointer advances by N/2 pairs = N reals to skip to next level */
    /* e_next: skip current level's N/4 quads × 4 reals = N reals */
    const notorious_fft_real* e_next = e + N;

    /* t offsets are in reals (interleaved: each complex = 2 reals).
     * t[k+N/2] starts at t + 2*(N/2) = t + N reals.
     * t[k+3N/4] starts at t + 2*(3N/4) = t + 3N/2 reals. */
    notorious_fft_sr_dif_cx(N/2, t,          t,          y,       2*sy, e_next);
    notorious_fft_sr_dif_cx(N/4, t+N,        t+N,        y+2*sy,  4*sy, e_next + N/2);
    notorious_fft_sr_dif_cx(N/4, t+3*(N/2),  t+3*(N/2),  y+6*sy,  4*sy, e_next + N/2);
}

/* Inverse split-radix DIF — uses conj(e) twiddles and swapped +/- t3 */
static void notorious_fft_sr_inv_dif_cx(int N, notorious_fft_real* x, notorious_fft_real* t,
                                   notorious_fft_real* y, int sy,
                                   const notorious_fft_real* e)
{
    notorious_fft_real* xr = x, *xi = x + 1;
    notorious_fft_real* tr = t, *ti = t + 1;
    notorious_fft_real* yr = y, *yi = y + 1;

    if (N == 1) {
        yr[0] = xr[0]; yi[0] = xi[0];
        return;
    }
    if (N == 2) {
        notorious_fft_real t0r = xr[0] + xr[2], t0i = xi[0] + xi[2];
        notorious_fft_real t1r = xr[0] - xr[2], t1i = xi[0] - xi[2];
        yr[0]      = t0r; yi[0]      = t0i;
        yr[2*sy]   = t1r; yi[2*sy]   = t1i;
        return;
    }
    if (N == 4) {
        notorious_fft_real t0r = xr[0] + xr[4], t0i = xi[0] + xi[4];
        notorious_fft_real t1r = xr[2] + xr[6], t1i = xi[2] + xi[6];
        notorious_fft_real t2r = xr[0] - xr[4], t2i = xi[0] - xi[4];
        notorious_fft_real t3r = -xi[2] + xi[6], t3i = xr[2] - xr[6];
        yr[0]      = t0r + t1r; yi[0]      = t0i + t1i;
        yr[2*sy]   = t2r + t3r; yi[2*sy]   = t2i + t3i;  /* swapped vs forward */
        yr[4*sy]   = t0r - t1r; yi[4*sy]   = t0i - t1i;
        yr[6*sy]   = t2r - t3r; yi[6*sy]   = t2i - t3i;  /* swapped vs forward */
        return;
    }
    if (N == 8) {
        notorious_fft_real t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;
        notorious_fft_real t00r,t00i,t01r,t01i,t02r,t02i,t03r,t03i;
        notorious_fft_real t10r,t10i,t11r,t11i,t12r,t12i,t13r,t13i;
        notorious_fft_real ttr,tti;
        const notorious_fft_real invsqrt2 = NOTORIOUS_FFT_INV_SQRT2;

        t0r=xr[0]+xr[8];  t0i=xi[0]+xi[8];
        t1r=xr[4]+xr[12]; t1i=xi[4]+xi[12];
        t2r=xr[0]-xr[8];  t2i=xi[0]-xi[8];
        t3r=-xi[4]+xi[12]; t3i=xr[4]-xr[12];
        t00r=t0r+t1r; t00i=t0i+t1i;
        t01r=t2r+t3r; t01i=t2i+t3i;  /* swapped vs forward */
        t02r=t0r-t1r; t02i=t0i-t1i;
        t03r=t2r-t3r; t03i=t2i-t3i;  /* swapped vs forward */

        t0r=xr[2]+xr[10]; t0i=xi[2]+xi[10];
        t1r=xr[6]+xr[14]; t1i=xi[6]+xi[14];
        t2r=xr[2]-xr[10]; t2i=xi[2]-xi[10];
        t3r=-xi[6]+xi[14]; t3i=xr[6]-xr[14];

        t10r=t0r+t1r; t10i=t0i+t1i;
        /* t11=(t2+t3)*invsqrt2*(1+I) — conjugated vs forward */
        ttr=t2r+t3r; tti=t2i+t3i;
        t11r=invsqrt2*(ttr-tti); t11i=invsqrt2*(ttr+tti);
        /* t12=(t0-t1)*(-I) — same sign flip as forward but opposite */
        t12r=-t0i+t1i; t12i=t0r-t1r;
        /* t13=(t2-t3)*invsqrt2*(-1+I) — conjugated vs forward */
        ttr=t2r-t3r; tti=t2i-t3i;
        t13r=-invsqrt2*(ttr+tti); t13i=invsqrt2*(ttr-tti);

        yr[0]=t00r+t10r;    yi[0]=t00i+t10i;
        yr[2*sy]=t01r+t11r; yi[2*sy]=t01i+t11i;
        yr[4*sy]=t02r+t12r; yi[4*sy]=t02i+t12i;
        yr[6*sy]=t03r+t13r; yi[6*sy]=t03i+t13i;
        yr[8*sy]=t00r-t10r; yi[8*sy]=t00i-t10i;
        yr[10*sy]=t01r-t11r; yi[10*sy]=t01i-t11i;
        yr[12*sy]=t02r-t12r; yi[12*sy]=t02i-t12i;
        yr[14*sy]=t03r-t13r; yi[14*sy]=t03i-t13i;
        return;
    }

    /* General recursion: inverse split-radix DIF butterfly */
    /* N >= 16 */
    int n4 = N / 4;
    const notorious_fft_real* ep = e;

    {
        int k = 0;

#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
        for (; k + 2 <= n4; k += 2) {
            float64x2_t w1r = (float64x2_t){ep[4*k],   ep[4*(k+1)]};
            float64x2_t w1i = (float64x2_t){ep[4*k+1], ep[4*(k+1)+1]};
            float64x2_t w3r = (float64x2_t){ep[4*k+2], ep[4*(k+1)+2]};
            float64x2_t w3i = (float64x2_t){ep[4*k+3], ep[4*(k+1)+3]};

            float64x2x2_t xa = vld2q_f64(xr + 2*k);
            float64x2x2_t xb = vld2q_f64(xr + 2*(k+N/2));
            float64x2x2_t xc = vld2q_f64(xr + 2*(k+n4));
            float64x2x2_t xd = vld2q_f64(xr + 2*(k+3*n4));

            float64x2_t t0r_ = vaddq_f64(xa.val[0], xb.val[0]);
            float64x2_t t0i_ = vaddq_f64(xa.val[1], xb.val[1]);
            float64x2_t t1r_ = vaddq_f64(xc.val[0], xd.val[0]);
            float64x2_t t1i_ = vaddq_f64(xc.val[1], xd.val[1]);
            float64x2_t t2r_ = vsubq_f64(xa.val[0], xb.val[0]);
            float64x2_t t2i_ = vsubq_f64(xa.val[1], xb.val[1]);
            float64x2_t t3r_ = vsubq_f64(xd.val[1], xc.val[1]);
            float64x2_t t3i_ = vsubq_f64(xc.val[0], xd.val[0]);

            /* Inverse: u = t2+t3, v = t2-t3 (swapped vs forward) */
            float64x2_t ur_ = vaddq_f64(t2r_, t3r_), ui_ = vaddq_f64(t2i_, t3i_);
            float64x2_t vr_ = vsubq_f64(t2r_, t3r_), vi_ = vsubq_f64(t2i_, t3i_);

            float64x2x2_t out0 = {{t0r_, t0i_}};
            vst2q_f64(tr + 2*k, out0);
            float64x2x2_t out1 = {{t1r_, t1i_}};
            vst2q_f64(tr + 2*(k+n4), out1);

            /* conj(w1) multiply: re = ur*wr + ui*wi, im = -ur*wi + ui*wr */
            float64x2_t pr_ = vaddq_f64(vmulq_f64(ur_,w1r), vmulq_f64(ui_,w1i));
            float64x2_t pi_ = vsubq_f64(vmulq_f64(ui_,w1r), vmulq_f64(ur_,w1i));
            float64x2x2_t out2 = {{pr_, pi_}};
            vst2q_f64(tr + 2*(k+N/2), out2);

            /* conj(w3) multiply */
            float64x2_t qr_ = vaddq_f64(vmulq_f64(vr_,w3r), vmulq_f64(vi_,w3i));
            float64x2_t qi_ = vsubq_f64(vmulq_f64(vi_,w3r), vmulq_f64(vr_,w3i));
            float64x2x2_t out3 = {{qr_, qi_}};
            vst2q_f64(tr + 2*(k+3*n4), out3);
        }
#endif

        /* Scalar remainder / fallback */
        for (; k < n4; k++) {
            notorious_fft_real x0r=xr[2*k],x0i=xi[2*k];
            notorious_fft_real x1r=xr[2*(k+N/2)],x1i=xi[2*(k+N/2)];
            notorious_fft_real x2r=xr[2*(k+n4)],x2i=xi[2*(k+n4)];
            notorious_fft_real x3r=xr[2*(k+3*n4)],x3i=xi[2*(k+3*n4)];
            notorious_fft_real t0r_=x0r+x1r,t0i_=x0i+x1i;
            notorious_fft_real t1r_=x2r+x3r,t1i_=x2i+x3i;
            notorious_fft_real t2r_=x0r-x1r,t2i_=x0i-x1i;
            notorious_fft_real t3r_=x3i-x2i,t3i_=x2r-x3r;
            /* Inverse: u=t2+t3, v=t2-t3 (swapped vs forward) */
            notorious_fft_real ur_=t2r_+t3r_,ui_=t2i_+t3i_;
            notorious_fft_real vr_=t2r_-t3r_,vi_=t2i_-t3i_;
            tr[2*k]=t0r_; ti[2*k]=t0i_;
            tr[2*(k+n4)]=t1r_; ti[2*(k+n4)]=t1i_;
            /* conj(e) multiply: re = ur*er + ui*ei, im = -ur*ei + ui*er */
            tr[2*(k+N/2)]=ur_*ep[4*k]+ui_*ep[4*k+1];
            ti[2*(k+N/2)]=-ur_*ep[4*k+1]+ui_*ep[4*k];
            tr[2*(k+3*n4)]=vr_*ep[4*k+2]+vi_*ep[4*k+3];
            ti[2*(k+3*n4)]=-vr_*ep[4*k+3]+vi_*ep[4*k+2];
        }
    }

    const notorious_fft_real* e_next = e + N;
    notorious_fft_sr_inv_dif_cx(N/2, t,          t,          y,       2*sy, e_next);
    notorious_fft_sr_inv_dif_cx(N/4, t+N,        t+N,        y+2*sy,  4*sy, e_next + N/2);
    notorious_fft_sr_inv_dif_cx(N/4, t+3*(N/2),  t+3*(N/2),  y+6*sy,  4*sy, e_next + N/2);
}

/* Public wrapper: run split-radix DIF on interleaved complex data */
static void notorious_fft_execute_sr_dif(
    const notorious_fft_plan* plan,
    const notorious_fft_real* x_in,
    notorious_fft_real* y_out,
    int inverse)
{
    int N = (int)plan->n;
    notorious_fft_real* t = plan->sr_t;  /* temp buffer: 2*N reals */

    /* Copy input to temp (sr_dif writes to t during butterfly stage) */
    memcpy(t, x_in, (size_t)(2*N) * sizeof(notorious_fft_real));
    if (inverse) {
        notorious_fft_sr_inv_dif_cx(N, t, plan->work_re, y_out, 1, plan->sr_e);
    } else {
        notorious_fft_sr_dif_cx(N, t, plan->work_re, y_out, 1, plan->sr_e);
    }
}

/* ============================================================================
 * Bluestein's Algorithm for Arbitrary Size FFT
 * ============================================================================ */

static void notorious_fft_execute_bluestein(
    const notorious_fft_plan* plan,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_in, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_in,
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT xr_out, notorious_fft_real* NOTORIOUS_FFT_RESTRICT xi_out)
{
    if (!plan || !plan->bluestein_plan ||
        !plan->bluestein_chirp_re || !plan->bluestein_chirp_im ||
        !plan->bluestein_buf_re || !plan->bluestein_buf_im) {
        return;
    }

    size_t n = plan->n;
    size_t m = plan->bluestein_n;
    
    /* For inverse DFT, use the property: IDFT(X) = (1/N) * conj(DFT(conj(X)))
     * This lets us use the same forward Bluestein algorithm for inverse. */
    if (plan->is_inverse) {
        /* Use bluestein buffers for temporary storage */
        notorious_fft_real* x_conj_re = plan->bluestein_buf_re;
        notorious_fft_real* x_conj_im = plan->bluestein_buf_im;
        
        /* Compute conj(X) */
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n > 1024)
#endif
        for (size_t i = 0; i < n; i++) {
            x_conj_re[i] = xr_in[i];
            x_conj_im[i] = -xi_in[i];
        }
        
        /* Apply forward Bluestein to conj(X) */
        /* First save original is_inverse flag and set to forward */
        int saved_inverse = plan->is_inverse;
        ((notorious_fft_plan*)plan)->is_inverse = 0;
        
        notorious_fft_execute_bluestein(plan, x_conj_re, x_conj_im, xr_out, xi_out);
        
        /* Restore inverse flag */
        ((notorious_fft_plan*)plan)->is_inverse = saved_inverse;
        
        /* Take conj(y) and scale by 1/N: IDFT(X) = conj(y) / N */
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n > 1024)
#endif
        for (size_t i = 0; i < n; i++) {
            notorious_fft_real tmp_re = xr_out[i] / (notorious_fft_real)n;
            notorious_fft_real tmp_im = -xi_out[i] / (notorious_fft_real)n;
            xr_out[i] = tmp_re;
            xi_out[i] = tmp_im;
        }
        
        return;
    }
    
    /* Forward Bluestein algorithm */
    
    /* chirp_re/im contains the original chirp factors exp(-i*pi*k^2/n) for pre/post multiply */
    const notorious_fft_real* chirp_re = plan->bluestein_chirp_re;
    const notorious_fft_real* chirp_im = plan->bluestein_chirp_im;
    /* chirp_fft_re/im contains conj(FFT(chirp)) for frequency-domain convolution */
    const notorious_fft_real* chirp_fft_re = plan->bluestein_chirp_fft_re;
    const notorious_fft_real* chirp_fft_im = plan->bluestein_chirp_fft_im;
    notorious_fft_real* buf_re = plan->bluestein_buf_re;
    notorious_fft_real* buf_im = plan->bluestein_buf_im;
    
    /* Step 1: Multiply input by chirp: x[n] * exp(-i*pi*n^2/N) */
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    if (n >= 4) {
        notorious_fft_bluestein_premul_avx2(n, buf_re, buf_im, xr_in, xi_in, chirp_re, chirp_im);
        notorious_fft_bluestein_zeropad_avx2(n, m, buf_re, buf_im);
    } else
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    if (n >= 2) {
        notorious_fft_bluestein_premul_neon(n, buf_re, buf_im, xr_in, xi_in, chirp_re, chirp_im);
        notorious_fft_bluestein_zeropad_neon(n, m, buf_re, buf_im);
    } else
#endif
    {
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n > 1024)
#endif
        for (size_t i = 0; i < n; i++) {
            buf_re[i] = xr_in[i] * chirp_re[i] - xi_in[i] * chirp_im[i];
            buf_im[i] = xr_in[i] * chirp_im[i] + xi_in[i] * chirp_re[i];
        }
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(m - n > 1024)
#endif
        for (size_t i = n; i < m; i++) {
            buf_re[i] = 0;
            buf_im[i] = 0;
        }
    }
    
    /* Step 2: FFT of padded sequence
     * Use separate fft_buf for output since iterative FFT clobbers work buffers */
    notorious_fft_real* fft_re = plan->bluestein_fft_buf_re;
    notorious_fft_real* fft_im = plan->bluestein_fft_buf_im;
    notorious_fft_execute_iterative_internal(plan->bluestein_plan, buf_re, buf_im, fft_re, fft_im, 0);
    
    /* Step 3: Pointwise multiply with conj(FFT(chirp)) for convolution */
#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    if (m >= 4) {
        notorious_fft_bluestein_convolve_avx2(m, fft_re, fft_im, chirp_fft_re, chirp_fft_im);
    } else
#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    if (m >= 2) {
        notorious_fft_bluestein_convolve_neon(m, fft_re, fft_im, chirp_fft_re, chirp_fft_im);
    } else
#endif
    {
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(m > 1024)
#endif
        for (size_t i = 0; i < m; i++) {
            notorious_fft_real tr = fft_re[i] * chirp_fft_re[i] - fft_im[i] * chirp_fft_im[i];
            notorious_fft_real ti = fft_re[i] * chirp_fft_im[i] + fft_im[i] * chirp_fft_re[i];
            fft_re[i] = tr;
            fft_im[i] = ti;
        }
    }
    
    /* Step 4: IFFT using inverse FFT */
    notorious_fft_execute_iterative_internal(plan->bluestein_plan, fft_re, fft_im, buf_re, buf_im, 1);
    
    /* Step 5: Scale and multiply by original chirp to get final result
     * 
     * The convolution result is at indices [n-1, 2n-2] of the IFFT output.
     * We take the first n values from there, multiply by chirp, and scale.
     * Scale by 1/m from the IFFT (unscaled iterative IFFT).
     */
    notorious_fft_real scale = 1.0 / (notorious_fft_real)m;
    size_t conv_offset = n - 1;  /* Start of valid convolution results */
    
#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    if (n >= 2) {
        /* Use scalar loop with offset since NEON version doesn't support offset */
        for (size_t i = 0; i < n; i++) {
            size_t j = conv_offset + i;
            xr_out[i] = (buf_re[j] * chirp_re[i] - buf_im[j] * chirp_im[i]) * scale;
            xi_out[i] = (buf_re[j] * chirp_im[i] + buf_im[j] * chirp_re[i]) * scale;
        }
    } else
#endif
    {
#if NOTORIOUS_FFT_HAS_OPENMP
        #pragma omp parallel for schedule(static) if(n > 1024)
#endif
        for (size_t i = 0; i < n; i++) {
            size_t j = conv_offset + i;
            xr_out[i] = (buf_re[j] * chirp_re[i] - buf_im[j] * chirp_im[i]) * scale;
            xi_out[i] = (buf_re[j] * chirp_im[i] + buf_im[j] * chirp_re[i]) * scale;
        }
    }
}


#endif /* NOTORIOUS_FFT_ALGORITHMS_H */
