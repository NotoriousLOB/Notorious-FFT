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
static void notorious_fft_execute_mixed_cx(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out, int inverse);
static void notorious_fft_execute_four_step(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out, int inverse);
static void notorious_fft_execute_sr_dit(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out, int inverse);
static void notorious_fft_sr_dit_cx(int N, notorious_fft_real* z, const notorious_fft_real* e);
static void notorious_fft_sr_inv_dit_cx(int N, notorious_fft_real* z, const notorious_fft_real* e);
static void notorious_fft_execute_rader_cx(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out);

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
 * Power-of-2 default is split-radix DIF (no permute). MEASURE may pick
 * DIT (bit-reverse + unit-stride combine) or iterative Cooley–Tukey. */
static void notorious_fft_execute_cx(
    const notorious_fft_plan* plan,
    const notorious_fft_real* x_in,   /* interleaved input  */
    notorious_fft_real*       y_out,  /* interleaved output */
    int inverse)
{
    if (!plan || !x_in || !y_out) return;
    size_t n = plan->n;

    if (plan->mixed_radix) {
        notorious_fft_execute_mixed_cx(plan, x_in, y_out, inverse);
        return;
    }

    if (plan->rader_sub) {
        if (!inverse) {
            notorious_fft_execute_rader_cx(plan, x_in, y_out);
            return;
        }
        /* IDFT(x) = conj(DFT(conj(x))) — unnormalized */
        if (x_in != y_out) {
            for (size_t i = 0; i < n; i++) {
                y_out[2 * i]     =  x_in[2 * i];
                y_out[2 * i + 1] = -x_in[2 * i + 1];
            }
        } else {
            for (size_t i = 0; i < n; i++)
                y_out[2 * i + 1] = -y_out[2 * i + 1];
        }
        notorious_fft_execute_rader_cx(plan, y_out, y_out);
        for (size_t i = 0; i < n; i++)
            y_out[2 * i + 1] = -y_out[2 * i + 1];
        return;
    }

    if (plan->four_n1 && !plan->prefer_iterative) {
        notorious_fft_execute_four_step(plan, x_in, y_out, inverse);
        return;
    }

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

        /* Inverse via conjugate identity (unnormalized, FFTW/minfft semantics):
         *   IDFT(x) = conj(DFT(conj(x)))
         * No 1/N — the inner Bluestein 1/M factor is the convolution IFFT only. */
        if (inverse) {
            for (size_t i = 0; i < n; i++) in_im[i] = -in_im[i];
        }

        notorious_fft_execute_bluestein(plan, in_re, in_im, out_re, out_im);

        if (inverse) {
            for (size_t i = 0; i < n; i++) {
                y_out[2*i]   =  out_re[i];
                y_out[2*i+1] = -out_im[i];
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                y_out[2*i]   = out_re[i];
                y_out[2*i+1] = out_im[i];
            }
        }
        return;
    }

    if (plan->prefer_dit && plan->sr_e && plan->bitrev && n >= 64 && !plan->prefer_iterative) {
        notorious_fft_execute_sr_dit(plan, x_in, y_out, inverse);
    } else if (plan->sr_e && plan->sr_t && n >= 16 && !plan->prefer_iterative) {
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

    /* N=16: FFmpeg-style — hardcoded twiddles (cos table of n/4), no ep[] gather. */
    if (N == 16) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        const notorious_fft_real c1 = (notorious_fft_real)0.923879532511286756128183189396788287; /* cos(π/8) */
        const notorious_fft_real s1 = (notorious_fft_real)0.382683432365089771728459984030398867; /* sin(π/8) */
#define NOTORIOUS_FFT_SR16(k, wr, wi, w3r, w3i) do { \
            notorious_fft_real x0r=xr[2*(k)],x0i=xi[2*(k)]; \
            notorious_fft_real x1r=xr[2*((k)+8)],x1i=xi[2*((k)+8)]; \
            notorious_fft_real x2r=xr[2*((k)+4)],x2i=xi[2*((k)+4)]; \
            notorious_fft_real x3r=xr[2*((k)+12)],x3i=xi[2*((k)+12)]; \
            notorious_fft_real t0r=x0r+x1r,t0i=x0i+x1i; \
            notorious_fft_real t1r=x2r+x3r,t1i=x2i+x3i; \
            notorious_fft_real t2r=x0r-x1r,t2i=x0i-x1i; \
            notorious_fft_real t3r=x3i-x2i,t3i=x2r-x3r; \
            notorious_fft_real ur=t2r-t3r,ui=t2i-t3i; \
            notorious_fft_real vr=t2r+t3r,vi=t2i+t3i; \
            tr[2*(k)]=t0r; ti[2*(k)]=t0i; \
            tr[2*((k)+4)]=t1r; ti[2*((k)+4)]=t1i; \
            tr[2*((k)+8)]=ur*(wr)-ui*(wi); ti[2*((k)+8)]=ur*(wi)+ui*(wr); \
            tr[2*((k)+12)]=vr*(w3r)-vi*(w3i); ti[2*((k)+12)]=vr*(w3i)+vi*(w3r); \
        } while (0)
        NOTORIOUS_FFT_SR16(0, 1, 0, 1, 0);
        NOTORIOUS_FFT_SR16(1, c1, -s1, s1, -c1);
        NOTORIOUS_FFT_SR16(2, c8, -c8, -c8, -c8);
        NOTORIOUS_FFT_SR16(3, s1, -c1, -c1, s1);
#undef NOTORIOUS_FFT_SR16
        notorious_fft_sr_dif_cx(8, t,    t,    y,      2*sy, e);
        notorious_fft_sr_dif_cx(4, t+16, t+16, y+2*sy, 4*sy, e);
        notorious_fft_sr_dif_cx(4, t+24, t+24, y+6*sy, 4*sy, e);
        return;
    }

    /* N=32: one scalar stage then N=16 + 2×N=8 (FFmpeg DECL_SR_CODELET(32,16,8)). */
    if (N == 32) {
        int n4 = 8;
        const notorious_fft_real* ep = e;
        for (int k = 0; k < n4; k++) {
            notorious_fft_real x0r=xr[2*k],x0i=xi[2*k];
            notorious_fft_real x1r=xr[2*(k+16)],x1i=xi[2*(k+16)];
            notorious_fft_real x2r=xr[2*(k+8)],x2i=xi[2*(k+8)];
            notorious_fft_real x3r=xr[2*(k+24)],x3i=xi[2*(k+24)];
            notorious_fft_real t0r_=x0r+x1r,t0i_=x0i+x1i;
            notorious_fft_real t1r_=x2r+x3r,t1i_=x2i+x3i;
            notorious_fft_real t2r_=x0r-x1r,t2i_=x0i-x1i;
            notorious_fft_real t3r_=x3i-x2i,t3i_=x2r-x3r;
            notorious_fft_real ur_=t2r_-t3r_,ui_=t2i_-t3i_;
            notorious_fft_real vr_=t2r_+t3r_,vi_=t2i_+t3i_;
            tr[2*k]=t0r_; ti[2*k]=t0i_;
            tr[2*(k+8)]=t1r_; ti[2*(k+8)]=t1i_;
            tr[2*(k+16)]=ur_*ep[4*k]-ui_*ep[4*k+1];
            ti[2*(k+16)]=ur_*ep[4*k+1]+ui_*ep[4*k];
            tr[2*(k+24)]=vr_*ep[4*k+2]-vi_*ep[4*k+3];
            ti[2*(k+24)]=vr_*ep[4*k+3]+vi_*ep[4*k+2];
        }
        const notorious_fft_real* e_next = e + 32;
        notorious_fft_sr_dif_cx(16, t,    t,    y,      2*sy, e_next);
        notorious_fft_sr_dif_cx(8,  t+32, t+32, y+2*sy, 4*sy, e_next + 16);
        notorious_fft_sr_dif_cx(8,  t+48, t+48, y+6*sy, 4*sy, e_next + 16);
        return;
    }

    /* General recursion: split-radix DIF butterfly stage then recurse */
    /* N >= 64 */
    int n4 = N / 4;
    const notorious_fft_real* ep = e;  /* points to current level's twiddles */

    {
        int k = 0;

#if NOTORIOUS_FFT_HAS_AVX512 && !defined(NOTORIOUS_FFT_SINGLE)
        for (; k + 4 <= n4; k += 4) {
            __m512d w1r = _mm512_set_pd(ep[4*(k+3)], ep[4*(k+3)], ep[4*(k+2)], ep[4*(k+2)],
                                        ep[4*(k+1)], ep[4*(k+1)], ep[4*k],   ep[4*k]);
            __m512d w1i = _mm512_set_pd(ep[4*(k+3)+1], ep[4*(k+3)+1], ep[4*(k+2)+1], ep[4*(k+2)+1],
                                        ep[4*(k+1)+1], ep[4*(k+1)+1], ep[4*k+1], ep[4*k+1]);
            __m512d w3r = _mm512_set_pd(ep[4*(k+3)+2], ep[4*(k+3)+2], ep[4*(k+2)+2], ep[4*(k+2)+2],
                                        ep[4*(k+1)+2], ep[4*(k+1)+2], ep[4*k+2], ep[4*k+2]);
            __m512d w3i = _mm512_set_pd(ep[4*(k+3)+3], ep[4*(k+3)+3], ep[4*(k+2)+3], ep[4*(k+2)+3],
                                        ep[4*(k+1)+3], ep[4*(k+1)+3], ep[4*k+3], ep[4*k+3]);
            __m512d a = _mm512_loadu_pd(xr + 2*k);
            __m512d b = _mm512_loadu_pd(xr + 2*(k+N/2));
            __m512d c = _mm512_loadu_pd(xr + 2*(k+n4));
            __m512d d = _mm512_loadu_pd(xr + 2*(k+3*n4));
            __m512d t0 = _mm512_add_pd(a, b);
            __m512d t1 = _mm512_add_pd(c, d);
            __m512d t2 = _mm512_sub_pd(a, b);
            __m512d cd_diff = _mm512_sub_pd(c, d);
            __m512d cd_swap = _mm512_permute_pd(cd_diff, 0x55);
            __m512d sign_mask = _mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
            __m512d t3 = _mm512_mul_pd(cd_swap, sign_mask);
            __m512d u = _mm512_sub_pd(t2, t3);
            __m512d v = _mm512_add_pd(t2, t3);
            _mm512_storeu_pd(tr + 2*k, t0);
            _mm512_storeu_pd(tr + 2*(k+n4), t1);
            __m512d u_swap = _mm512_permute_pd(u, 0x55);
            __m512d p1 = _mm512_mul_pd(u, w1r);
            __m512d p2 = _mm512_mul_pd(u_swap, w1i);
            __m512d cmul_sign = _mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
            __m512d uw1 = _mm512_fmadd_pd(p2, cmul_sign, p1);
            _mm512_storeu_pd(tr + 2*(k+N/2), uw1);
            __m512d v_swap = _mm512_permute_pd(v, 0x55);
            __m512d q1 = _mm512_mul_pd(v, w3r);
            __m512d q2 = _mm512_mul_pd(v_swap, w3i);
            __m512d vw3 = _mm512_fmadd_pd(q2, cmul_sign, q1);
            _mm512_storeu_pd(tr + 2*(k+3*n4), vw3);
        }
#elif NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
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
            __m256d uw1 = _mm256_fmadd_pd(p2, cmul_sign, p1);

            _mm256_storeu_pd(tr + 2*(k+N/2), uw1);

            /* v * w3: same complex multiply pattern */
            __m256d v_swap = _mm256_shuffle_pd(v, v, 0x5);
            __m256d q1 = _mm256_mul_pd(v, w3r);
            __m256d q2 = _mm256_mul_pd(v_swap, w3i);
            __m256d vw3 = _mm256_fmadd_pd(q2, cmul_sign, q1);

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
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
        for (; k + 4 <= n4; k += 4) {
            float32x4_t w1r = (float32x4_t){ep[4*k],   ep[4*(k+1)], ep[4*(k+2)], ep[4*(k+3)]};
            float32x4_t w1i = (float32x4_t){ep[4*k+1], ep[4*(k+1)+1], ep[4*(k+2)+1], ep[4*(k+3)+1]};
            float32x4_t w3r = (float32x4_t){ep[4*k+2], ep[4*(k+1)+2], ep[4*(k+2)+2], ep[4*(k+3)+2]};
            float32x4_t w3i = (float32x4_t){ep[4*k+3], ep[4*(k+1)+3], ep[4*(k+2)+3], ep[4*(k+3)+3]};
            float32x4x2_t xa = vld2q_f32(xr + 2*k);
            float32x4x2_t xb = vld2q_f32(xr + 2*(k+N/2));
            float32x4x2_t xc = vld2q_f32(xr + 2*(k+n4));
            float32x4x2_t xd = vld2q_f32(xr + 2*(k+3*n4));
            float32x4_t t0r_ = vaddq_f32(xa.val[0], xb.val[0]);
            float32x4_t t0i_ = vaddq_f32(xa.val[1], xb.val[1]);
            float32x4_t t1r_ = vaddq_f32(xc.val[0], xd.val[0]);
            float32x4_t t1i_ = vaddq_f32(xc.val[1], xd.val[1]);
            float32x4_t t2r_ = vsubq_f32(xa.val[0], xb.val[0]);
            float32x4_t t2i_ = vsubq_f32(xa.val[1], xb.val[1]);
            float32x4_t t3r_ = vsubq_f32(xd.val[1], xc.val[1]);
            float32x4_t t3i_ = vsubq_f32(xc.val[0], xd.val[0]);
            float32x4_t ur_ = vsubq_f32(t2r_, t3r_), ui_ = vsubq_f32(t2i_, t3i_);
            float32x4_t vr_ = vaddq_f32(t2r_, t3r_), vi_ = vaddq_f32(t2i_, t3i_);
            float32x4x2_t out0 = {{t0r_, t0i_}};
            vst2q_f32(tr + 2*k, out0);
            float32x4x2_t out1 = {{t1r_, t1i_}};
            vst2q_f32(tr + 2*(k+n4), out1);
            float32x4_t pr_ = vsubq_f32(vmulq_f32(ur_,w1r), vmulq_f32(ui_,w1i));
            float32x4_t pi_ = vaddq_f32(vmulq_f32(ur_,w1i), vmulq_f32(ui_,w1r));
            float32x4x2_t out2 = {{pr_, pi_}};
            vst2q_f32(tr + 2*(k+N/2), out2);
            float32x4_t qr_ = vsubq_f32(vmulq_f32(vr_,w3r), vmulq_f32(vi_,w3i));
            float32x4_t qi_ = vaddq_f32(vmulq_f32(vr_,w3i), vmulq_f32(vi_,w3r));
            float32x4x2_t out3 = {{qr_, qi_}};
            vst2q_f32(tr + 2*(k+3*n4), out3);
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

    if (N == 16) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        const notorious_fft_real c1 = (notorious_fft_real)0.923879532511286756128183189396788287;
        const notorious_fft_real s1 = (notorious_fft_real)0.382683432365089771728459984030398867;
#define NOTORIOUS_FFT_SR16I(k, wr, wi, w3r, w3i) do { \
            notorious_fft_real x0r=xr[2*(k)],x0i=xi[2*(k)]; \
            notorious_fft_real x1r=xr[2*((k)+8)],x1i=xi[2*((k)+8)]; \
            notorious_fft_real x2r=xr[2*((k)+4)],x2i=xi[2*((k)+4)]; \
            notorious_fft_real x3r=xr[2*((k)+12)],x3i=xi[2*((k)+12)]; \
            notorious_fft_real t0r=x0r+x1r,t0i=x0i+x1i; \
            notorious_fft_real t1r=x2r+x3r,t1i=x2i+x3i; \
            notorious_fft_real t2r=x0r-x1r,t2i=x0i-x1i; \
            notorious_fft_real t3r=x3i-x2i,t3i=x2r-x3r; \
            notorious_fft_real ur=t2r+t3r,ui=t2i+t3i; \
            notorious_fft_real vr=t2r-t3r,vi=t2i-t3i; \
            tr[2*(k)]=t0r; ti[2*(k)]=t0i; \
            tr[2*((k)+4)]=t1r; ti[2*((k)+4)]=t1i; \
            tr[2*((k)+8)]=ur*(wr)+ui*(wi); ti[2*((k)+8)]=-ur*(wi)+ui*(wr); \
            tr[2*((k)+12)]=vr*(w3r)+vi*(w3i); ti[2*((k)+12)]=-vr*(w3i)+vi*(w3r); \
        } while (0)
        /* Same forward twiddles; formula uses conj(w). */
        NOTORIOUS_FFT_SR16I(0, 1, 0, 1, 0);
        NOTORIOUS_FFT_SR16I(1, c1, -s1, s1, -c1);
        NOTORIOUS_FFT_SR16I(2, c8, -c8, -c8, -c8);
        NOTORIOUS_FFT_SR16I(3, s1, -c1, -c1, s1);
#undef NOTORIOUS_FFT_SR16I
        notorious_fft_sr_inv_dif_cx(8, t,    t,    y,      2*sy, e);
        notorious_fft_sr_inv_dif_cx(4, t+16, t+16, y+2*sy, 4*sy, e);
        notorious_fft_sr_inv_dif_cx(4, t+24, t+24, y+6*sy, 4*sy, e);
        return;
    }

    if (N == 32) {
        int n4 = 8;
        const notorious_fft_real* ep = e;
        for (int k = 0; k < n4; k++) {
            notorious_fft_real x0r=xr[2*k],x0i=xi[2*k];
            notorious_fft_real x1r=xr[2*(k+16)],x1i=xi[2*(k+16)];
            notorious_fft_real x2r=xr[2*(k+8)],x2i=xi[2*(k+8)];
            notorious_fft_real x3r=xr[2*(k+24)],x3i=xi[2*(k+24)];
            notorious_fft_real t0r_=x0r+x1r,t0i_=x0i+x1i;
            notorious_fft_real t1r_=x2r+x3r,t1i_=x2i+x3i;
            notorious_fft_real t2r_=x0r-x1r,t2i_=x0i-x1i;
            notorious_fft_real t3r_=x3i-x2i,t3i_=x2r-x3r;
            notorious_fft_real ur_=t2r_+t3r_,ui_=t2i_+t3i_;
            notorious_fft_real vr_=t2r_-t3r_,vi_=t2i_-t3i_;
            tr[2*k]=t0r_; ti[2*k]=t0i_;
            tr[2*(k+8)]=t1r_; ti[2*(k+8)]=t1i_;
            tr[2*(k+16)]=ur_*ep[4*k]+ui_*ep[4*k+1];
            ti[2*(k+16)]=-ur_*ep[4*k+1]+ui_*ep[4*k];
            tr[2*(k+24)]=vr_*ep[4*k+2]+vi_*ep[4*k+3];
            ti[2*(k+24)]=-vr_*ep[4*k+3]+vi_*ep[4*k+2];
        }
        const notorious_fft_real* e_next = e + 32;
        notorious_fft_sr_inv_dif_cx(16, t,    t,    y,      2*sy, e_next);
        notorious_fft_sr_inv_dif_cx(8,  t+32, t+32, y+2*sy, 4*sy, e_next + 16);
        notorious_fft_sr_inv_dif_cx(8,  t+48, t+48, y+6*sy, 4*sy, e_next + 16);
        return;
    }

    /* General recursion: inverse split-radix DIF butterfly */
    /* N >= 64 */
    int n4 = N / 4;
    const notorious_fft_real* ep = e;

    {
        int k = 0;

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
        for (; k + 2 <= n4; k += 2) {
            __m256d w1r = _mm256_set_pd(ep[4*(k+1)],   ep[4*(k+1)],   ep[4*k],   ep[4*k]);
            __m256d w1i = _mm256_set_pd(ep[4*(k+1)+1], ep[4*(k+1)+1], ep[4*k+1], ep[4*k+1]);
            __m256d w3r = _mm256_set_pd(ep[4*(k+1)+2], ep[4*(k+1)+2], ep[4*k+2], ep[4*k+2]);
            __m256d w3i = _mm256_set_pd(ep[4*(k+1)+3], ep[4*(k+1)+3], ep[4*k+3], ep[4*k+3]);

            __m256d a = _mm256_loadu_pd(xr + 2*k);
            __m256d b = _mm256_loadu_pd(xr + 2*(k+N/2));
            __m256d c = _mm256_loadu_pd(xr + 2*(k+n4));
            __m256d d = _mm256_loadu_pd(xr + 2*(k+3*n4));

            __m256d t0 = _mm256_add_pd(a, b);
            __m256d t1 = _mm256_add_pd(c, d);
            __m256d t2 = _mm256_sub_pd(a, b);
            __m256d cd_diff = _mm256_sub_pd(c, d);

            __m256d cd_swap = _mm256_shuffle_pd(cd_diff, cd_diff, 0x5);
            __m256d sign_mask = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
            __m256d t3 = _mm256_mul_pd(cd_swap, sign_mask);

            /* Inverse: u = t2+t3, v = t2-t3 */
            __m256d u = _mm256_add_pd(t2, t3);
            __m256d v = _mm256_sub_pd(t2, t3);

            _mm256_storeu_pd(tr + 2*k, t0);
            _mm256_storeu_pd(tr + 2*(k+n4), t1);

            /* u * conj(w1): re = ur*wr + ui*wi, im = ui*wr - ur*wi */
            __m256d u_swap = _mm256_shuffle_pd(u, u, 0x5);
            __m256d p1 = _mm256_mul_pd(u, w1r);
            __m256d p2 = _mm256_mul_pd(u_swap, w1i);
            __m256d cmul_sign = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0);
            __m256d uw1 = _mm256_fmadd_pd(p2, cmul_sign, p1);
            _mm256_storeu_pd(tr + 2*(k+N/2), uw1);

            __m256d v_swap = _mm256_shuffle_pd(v, v, 0x5);
            __m256d q1 = _mm256_mul_pd(v, w3r);
            __m256d q2 = _mm256_mul_pd(v_swap, w3i);
            __m256d vw3 = _mm256_fmadd_pd(q2, cmul_sign, q1);
            _mm256_storeu_pd(tr + 2*(k+3*n4), vw3);
        }

#elif NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
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
#elif NOTORIOUS_FFT_HAS_NEON && defined(NOTORIOUS_FFT_SINGLE)
        for (; k + 4 <= n4; k += 4) {
            float32x4_t w1r = (float32x4_t){ep[4*k],   ep[4*(k+1)], ep[4*(k+2)], ep[4*(k+3)]};
            float32x4_t w1i = (float32x4_t){ep[4*k+1], ep[4*(k+1)+1], ep[4*(k+2)+1], ep[4*(k+3)+1]};
            float32x4_t w3r = (float32x4_t){ep[4*k+2], ep[4*(k+1)+2], ep[4*(k+2)+2], ep[4*(k+3)+2]};
            float32x4_t w3i = (float32x4_t){ep[4*k+3], ep[4*(k+1)+3], ep[4*(k+2)+3], ep[4*(k+3)+3]};
            float32x4x2_t xa = vld2q_f32(xr + 2*k);
            float32x4x2_t xb = vld2q_f32(xr + 2*(k+N/2));
            float32x4x2_t xc = vld2q_f32(xr + 2*(k+n4));
            float32x4x2_t xd = vld2q_f32(xr + 2*(k+3*n4));
            float32x4_t t0r_ = vaddq_f32(xa.val[0], xb.val[0]);
            float32x4_t t0i_ = vaddq_f32(xa.val[1], xb.val[1]);
            float32x4_t t1r_ = vaddq_f32(xc.val[0], xd.val[0]);
            float32x4_t t1i_ = vaddq_f32(xc.val[1], xd.val[1]);
            float32x4_t t2r_ = vsubq_f32(xa.val[0], xb.val[0]);
            float32x4_t t2i_ = vsubq_f32(xa.val[1], xb.val[1]);
            float32x4_t t3r_ = vsubq_f32(xd.val[1], xc.val[1]);
            float32x4_t t3i_ = vsubq_f32(xc.val[0], xd.val[0]);
            float32x4_t ur_ = vaddq_f32(t2r_, t3r_), ui_ = vaddq_f32(t2i_, t3i_);
            float32x4_t vr_ = vsubq_f32(t2r_, t3r_), vi_ = vsubq_f32(t2i_, t3i_);
            float32x4x2_t out0 = {{t0r_, t0i_}};
            vst2q_f32(tr + 2*k, out0);
            float32x4x2_t out1 = {{t1r_, t1i_}};
            vst2q_f32(tr + 2*(k+n4), out1);
            float32x4_t pr_ = vaddq_f32(vmulq_f32(ur_,w1r), vmulq_f32(ui_,w1i));
            float32x4_t pi_ = vsubq_f32(vmulq_f32(ui_,w1r), vmulq_f32(ur_,w1i));
            float32x4x2_t out2 = {{pr_, pi_}};
            vst2q_f32(tr + 2*(k+N/2), out2);
            float32x4_t qr_ = vaddq_f32(vmulq_f32(vr_,w3r), vmulq_f32(vi_,w3i));
            float32x4_t qi_ = vsubq_f32(vmulq_f32(vi_,w3r), vmulq_f32(vr_,w3i));
            float32x4x2_t out3 = {{qr_, qi_}};
            vst2q_f32(tr + 2*(k+3*n4), out3);
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
    /* First stage only reads x; recursion lives in work_re. x_in may alias y_out. */
    if (inverse) {
        notorious_fft_sr_inv_dif_cx(N, (notorious_fft_real*)x_in, plan->work_re, y_out, 1, plan->sr_e);
    } else {
        notorious_fft_sr_dif_cx(N, (notorious_fft_real*)x_in, plan->work_re, y_out, 1, plan->sr_e);
    }
}

/* ============================================================================
 * Split-radix DIT — bit-reversed input, unit-stride combine
 *
 * Dual of the DIF recursion: sub-FFTs first, then the 2/4 butterfly.
 * After a single bit-reversal gather, every stage is contiguous (stride 1),
 * which vectorizes and caches better than DIF's growing output stride.
 *
 * Layout after the N/2 + N/4 + N/4 subtransforms:
 *   z[0 .. N/2)       even DFT U
 *   z[N/2 .. 3N/4)    DFT of x[4n+1]  (V)
 *   z[3N/4 .. N)      DFT of x[4n+3]  (W)
 *
 *   t2 = V[k] W_N^k ,  t3 = W[k] W_N^{3k} ,  u = t2+t3 ,  d = t2-t3
 *   X[k]      = U[k] + u
 *   X[k+N/2]  = U[k] - u
 *   X[k+N/4]  = U[k+N/4] - i d     (forward; +i d inverse)
 *   X[k+3N/4] = U[k+N/4] + i d
 * ============================================================================ */

static NOTORIOUS_FFT_INLINE void notorious_fft_dit_combine1(
    notorious_fft_real* z, int k, int n4, int n2,
    notorious_fft_real w1r, notorious_fft_real w1i,
    notorious_fft_real w3r, notorious_fft_real w3i,
    int inverse)
{
    notorious_fft_real ukr = z[2 * k],           uki = z[2 * k + 1];
    notorious_fft_real u2r = z[2 * (k + n4)],    u2i = z[2 * (k + n4) + 1];
    notorious_fft_real vr  = z[2 * (k + n2)],    vi  = z[2 * (k + n2) + 1];
    notorious_fft_real wr  = z[2 * (k + n2 + n4)], wi = z[2 * (k + n2 + n4) + 1];
    notorious_fft_real t2r, t2i, t3r, t3i;
    if (!inverse) {
        t2r = vr * w1r - vi * w1i; t2i = vr * w1i + vi * w1r;
        t3r = wr * w3r - wi * w3i; t3i = wr * w3i + wi * w3r;
    } else {
        t2r = vr * w1r + vi * w1i; t2i = vi * w1r - vr * w1i;
        t3r = wr * w3r + wi * w3i; t3i = wi * w3r - wr * w3i;
    }
    notorious_fft_real ur = t2r + t3r, ui = t2i + t3i;
    notorious_fft_real dr = t2r - t3r, di = t2i - t3i;
    z[2 * k]                = ukr + ur; z[2 * k + 1]                = uki + ui;
    z[2 * (k + n2)]         = ukr - ur; z[2 * (k + n2) + 1]         = uki - ui;
    if (!inverse) {
        z[2 * (k + n4)]         = u2r + di; z[2 * (k + n4) + 1]         = u2i - dr;
        z[2 * (k + n2 + n4)]    = u2r - di; z[2 * (k + n2 + n4) + 1]    = u2i + dr;
    } else {
        z[2 * (k + n4)]         = u2r - di; z[2 * (k + n4) + 1]         = u2i + dr;
        z[2 * (k + n2 + n4)]    = u2r + di; z[2 * (k + n2 + n4) + 1]    = u2i - dr;
    }
}

static void notorious_fft_sr_dit_cx(int N, notorious_fft_real* z, const notorious_fft_real* e)
{
    if (N == 1)
        return;
    if (N == 2) {
        notorious_fft_real t0r = z[0] + z[2], t0i = z[1] + z[3];
        notorious_fft_real t1r = z[0] - z[2], t1i = z[1] - z[3];
        z[0] = t0r; z[1] = t0i; z[2] = t1r; z[3] = t1i;
        return;
    }
    if (N == 4) {
        notorious_fft_real t0r = z[0] + z[2], t0i = z[1] + z[3];
        notorious_fft_real t1r = z[0] - z[2], t1i = z[1] - z[3];
        notorious_fft_real vr = z[4], vi = z[5], wr = z[6], wi = z[7];
        notorious_fft_real ur = vr + wr, ui = vi + wi;
        notorious_fft_real dr = vr - wr, di = vi - wi;
        z[0] = t0r + ur; z[1] = t0i + ui;
        z[4] = t0r - ur; z[5] = t0i - ui;
        z[2] = t1r + di; z[3] = t1i - dr; /* U1 − i d */
        z[6] = t1r - di; z[7] = t1i + dr; /* U1 + i d */
        return;
    }
    if (N == 8) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        notorious_fft_sr_dit_cx(4, z, e);
        notorious_fft_sr_dit_cx(2, z + 8, e);
        notorious_fft_sr_dit_cx(2, z + 12, e);
        notorious_fft_dit_combine1(z, 0, 2, 4, 1, 0, 1, 0, 0);
        notorious_fft_dit_combine1(z, 1, 2, 4, c8, -c8, -c8, -c8, 0);
        return;
    }
    if (N == 16) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        const notorious_fft_real c1 = (notorious_fft_real)0.923879532511286756128183189396788287;
        const notorious_fft_real s1 = (notorious_fft_real)0.382683432365089771728459984030398867;
        notorious_fft_sr_dit_cx(8, z, e);
        notorious_fft_sr_dit_cx(4, z + 16, e);
        notorious_fft_sr_dit_cx(4, z + 24, e);
        notorious_fft_dit_combine1(z, 0, 4, 8, 1, 0, 1, 0, 0);
        notorious_fft_dit_combine1(z, 1, 4, 8, c1, -s1, s1, -c1, 0);
        notorious_fft_dit_combine1(z, 2, 4, 8, c8, -c8, -c8, -c8, 0);
        notorious_fft_dit_combine1(z, 3, 4, 8, s1, -c1, -c1, s1, 0);
        return;
    }

    /* N ≥ 32 */
    const notorious_fft_real* e_next = e + N;
    notorious_fft_sr_dit_cx(N / 2, z, e_next);
    notorious_fft_sr_dit_cx(N / 4, z + N, e_next + N / 2);
    notorious_fft_sr_dit_cx(N / 4, z + 3 * (N / 2), e_next + N / 2);

    int n4 = N / 4, n2 = N / 2;
    const notorious_fft_real* ep = e;
    int k = 0;

#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    for (; k + 2 <= n4; k += 2) {
        float64x2_t w1r = (float64x2_t){ep[4 * k], ep[4 * (k + 1)]};
        float64x2_t w1i = (float64x2_t){ep[4 * k + 1], ep[4 * (k + 1) + 1]};
        float64x2_t w3r = (float64x2_t){ep[4 * k + 2], ep[4 * (k + 1) + 2]};
        float64x2_t w3i = (float64x2_t){ep[4 * k + 3], ep[4 * (k + 1) + 3]};
        float64x2x2_t uk = vld2q_f64(z + 2 * k);
        float64x2x2_t u2 = vld2q_f64(z + 2 * (k + n4));
        float64x2x2_t vv = vld2q_f64(z + 2 * (k + n2));
        float64x2x2_t ww = vld2q_f64(z + 2 * (k + n2 + n4));
        float64x2_t t2r = vsubq_f64(vmulq_f64(vv.val[0], w1r), vmulq_f64(vv.val[1], w1i));
        float64x2_t t2i = vaddq_f64(vmulq_f64(vv.val[0], w1i), vmulq_f64(vv.val[1], w1r));
        float64x2_t t3r = vsubq_f64(vmulq_f64(ww.val[0], w3r), vmulq_f64(ww.val[1], w3i));
        float64x2_t t3i = vaddq_f64(vmulq_f64(ww.val[0], w3i), vmulq_f64(ww.val[1], w3r));
        float64x2_t ur = vaddq_f64(t2r, t3r), ui = vaddq_f64(t2i, t3i);
        float64x2_t dr = vsubq_f64(t2r, t3r), di = vsubq_f64(t2i, t3i);
        float64x2x2_t x0 = {{vaddq_f64(uk.val[0], ur), vaddq_f64(uk.val[1], ui)}};
        float64x2x2_t x2 = {{vsubq_f64(uk.val[0], ur), vsubq_f64(uk.val[1], ui)}};
        float64x2x2_t x1 = {{vaddq_f64(u2.val[0], di), vsubq_f64(u2.val[1], dr)}};
        float64x2x2_t x3 = {{vsubq_f64(u2.val[0], di), vaddq_f64(u2.val[1], dr)}};
        vst2q_f64(z + 2 * k, x0);
        vst2q_f64(z + 2 * (k + n2), x2);
        vst2q_f64(z + 2 * (k + n4), x1);
        vst2q_f64(z + 2 * (k + n2 + n4), x3);
    }
#elif NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    for (; k + 2 <= n4; k += 2) {
        __m256d w1r = _mm256_set_pd(ep[4 * (k + 1)], ep[4 * (k + 1)], ep[4 * k], ep[4 * k]);
        __m256d w1i = _mm256_set_pd(ep[4 * (k + 1) + 1], ep[4 * (k + 1) + 1], ep[4 * k + 1], ep[4 * k + 1]);
        __m256d w3r = _mm256_set_pd(ep[4 * (k + 1) + 2], ep[4 * (k + 1) + 2], ep[4 * k + 2], ep[4 * k + 2]);
        __m256d w3i = _mm256_set_pd(ep[4 * (k + 1) + 3], ep[4 * (k + 1) + 3], ep[4 * k + 3], ep[4 * k + 3]);
        __m256d uk = _mm256_loadu_pd(z + 2 * k);
        __m256d u2 = _mm256_loadu_pd(z + 2 * (k + n4));
        __m256d vv = _mm256_loadu_pd(z + 2 * (k + n2));
        __m256d ww = _mm256_loadu_pd(z + 2 * (k + n2 + n4));
        __m256d cmul = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
        __m256d t2 = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_shuffle_pd(vv, vv, 0x5), w1i), cmul,
                                     _mm256_mul_pd(vv, w1r));
        __m256d t3 = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_shuffle_pd(ww, ww, 0x5), w3i), cmul,
                                     _mm256_mul_pd(ww, w3r));
        __m256d u = _mm256_add_pd(t2, t3);
        __m256d d = _mm256_sub_pd(t2, t3);
        _mm256_storeu_pd(z + 2 * k, _mm256_add_pd(uk, u));
        _mm256_storeu_pd(z + 2 * (k + n2), _mm256_sub_pd(uk, u));
        /* −i d on interleaved {re,im}: {di, −dr} */
        __m256d dswap = _mm256_shuffle_pd(d, d, 0x5);
        __m256d minus_i_d = _mm256_mul_pd(dswap, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));
        _mm256_storeu_pd(z + 2 * (k + n4), _mm256_add_pd(u2, minus_i_d));
        _mm256_storeu_pd(z + 2 * (k + n2 + n4), _mm256_sub_pd(u2, minus_i_d));
    }
#endif
    for (; k < n4; k++)
        notorious_fft_dit_combine1(z, k, n4, n2, ep[4 * k], ep[4 * k + 1],
                                   ep[4 * k + 2], ep[4 * k + 3], 0);
}

static void notorious_fft_sr_inv_dit_cx(int N, notorious_fft_real* z, const notorious_fft_real* e)
{
    if (N == 1)
        return;
    if (N == 2) {
        notorious_fft_real t0r = z[0] + z[2], t0i = z[1] + z[3];
        notorious_fft_real t1r = z[0] - z[2], t1i = z[1] - z[3];
        z[0] = t0r; z[1] = t0i; z[2] = t1r; z[3] = t1i;
        return;
    }
    if (N == 4) {
        notorious_fft_real t0r = z[0] + z[2], t0i = z[1] + z[3];
        notorious_fft_real t1r = z[0] - z[2], t1i = z[1] - z[3];
        notorious_fft_real vr = z[4], vi = z[5], wr = z[6], wi = z[7];
        notorious_fft_real ur = vr + wr, ui = vi + wi;
        notorious_fft_real dr = vr - wr, di = vi - wi;
        z[0] = t0r + ur; z[1] = t0i + ui;
        z[4] = t0r - ur; z[5] = t0i - ui;
        z[2] = t1r - di; z[3] = t1i + dr; /* U1 + i d */
        z[6] = t1r + di; z[7] = t1i - dr; /* U1 − i d */
        return;
    }
    if (N == 8) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        notorious_fft_sr_inv_dit_cx(4, z, e);
        notorious_fft_sr_inv_dit_cx(2, z + 8, e);
        notorious_fft_sr_inv_dit_cx(2, z + 12, e);
        notorious_fft_dit_combine1(z, 0, 2, 4, 1, 0, 1, 0, 1);
        notorious_fft_dit_combine1(z, 1, 2, 4, c8, -c8, -c8, -c8, 1);
        return;
    }
    if (N == 16) {
        const notorious_fft_real c8 = NOTORIOUS_FFT_INV_SQRT2;
        const notorious_fft_real c1 = (notorious_fft_real)0.923879532511286756128183189396788287;
        const notorious_fft_real s1 = (notorious_fft_real)0.382683432365089771728459984030398867;
        notorious_fft_sr_inv_dit_cx(8, z, e);
        notorious_fft_sr_inv_dit_cx(4, z + 16, e);
        notorious_fft_sr_inv_dit_cx(4, z + 24, e);
        notorious_fft_dit_combine1(z, 0, 4, 8, 1, 0, 1, 0, 1);
        notorious_fft_dit_combine1(z, 1, 4, 8, c1, -s1, s1, -c1, 1);
        notorious_fft_dit_combine1(z, 2, 4, 8, c8, -c8, -c8, -c8, 1);
        notorious_fft_dit_combine1(z, 3, 4, 8, s1, -c1, -c1, s1, 1);
        return;
    }

    const notorious_fft_real* e_next = e + N;
    notorious_fft_sr_inv_dit_cx(N / 2, z, e_next);
    notorious_fft_sr_inv_dit_cx(N / 4, z + N, e_next + N / 2);
    notorious_fft_sr_inv_dit_cx(N / 4, z + 3 * (N / 2), e_next + N / 2);

    int n4 = N / 4, n2 = N / 2;
    const notorious_fft_real* ep = e;
    int k = 0;

#if NOTORIOUS_FFT_HAS_NEON && !defined(NOTORIOUS_FFT_SINGLE)
    for (; k + 2 <= n4; k += 2) {
        float64x2_t w1r = (float64x2_t){ep[4 * k], ep[4 * (k + 1)]};
        float64x2_t w1i = (float64x2_t){ep[4 * k + 1], ep[4 * (k + 1) + 1]};
        float64x2_t w3r = (float64x2_t){ep[4 * k + 2], ep[4 * (k + 1) + 2]};
        float64x2_t w3i = (float64x2_t){ep[4 * k + 3], ep[4 * (k + 1) + 3]};
        float64x2x2_t uk = vld2q_f64(z + 2 * k);
        float64x2x2_t u2 = vld2q_f64(z + 2 * (k + n4));
        float64x2x2_t vv = vld2q_f64(z + 2 * (k + n2));
        float64x2x2_t ww = vld2q_f64(z + 2 * (k + n2 + n4));
        /* conj(w): re = vr*wr + vi*wi, im = vi*wr − vr*wi */
        float64x2_t t2r = vaddq_f64(vmulq_f64(vv.val[0], w1r), vmulq_f64(vv.val[1], w1i));
        float64x2_t t2i = vsubq_f64(vmulq_f64(vv.val[1], w1r), vmulq_f64(vv.val[0], w1i));
        float64x2_t t3r = vaddq_f64(vmulq_f64(ww.val[0], w3r), vmulq_f64(ww.val[1], w3i));
        float64x2_t t3i = vsubq_f64(vmulq_f64(ww.val[1], w3r), vmulq_f64(ww.val[0], w3i));
        float64x2_t ur = vaddq_f64(t2r, t3r), ui = vaddq_f64(t2i, t3i);
        float64x2_t dr = vsubq_f64(t2r, t3r), di = vsubq_f64(t2i, t3i);
        float64x2x2_t x0 = {{vaddq_f64(uk.val[0], ur), vaddq_f64(uk.val[1], ui)}};
        float64x2x2_t x2 = {{vsubq_f64(uk.val[0], ur), vsubq_f64(uk.val[1], ui)}};
        float64x2x2_t x1 = {{vsubq_f64(u2.val[0], di), vaddq_f64(u2.val[1], dr)}};
        float64x2x2_t x3 = {{vaddq_f64(u2.val[0], di), vsubq_f64(u2.val[1], dr)}};
        vst2q_f64(z + 2 * k, x0);
        vst2q_f64(z + 2 * (k + n2), x2);
        vst2q_f64(z + 2 * (k + n4), x1);
        vst2q_f64(z + 2 * (k + n2 + n4), x3);
    }
#elif NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)
    for (; k + 2 <= n4; k += 2) {
        __m256d w1r = _mm256_set_pd(ep[4 * (k + 1)], ep[4 * (k + 1)], ep[4 * k], ep[4 * k]);
        __m256d w1i = _mm256_set_pd(ep[4 * (k + 1) + 1], ep[4 * (k + 1) + 1], ep[4 * k + 1], ep[4 * k + 1]);
        __m256d w3r = _mm256_set_pd(ep[4 * (k + 1) + 2], ep[4 * (k + 1) + 2], ep[4 * k + 2], ep[4 * k + 2]);
        __m256d w3i = _mm256_set_pd(ep[4 * (k + 1) + 3], ep[4 * (k + 1) + 3], ep[4 * k + 3], ep[4 * k + 3]);
        __m256d uk = _mm256_loadu_pd(z + 2 * k);
        __m256d u2 = _mm256_loadu_pd(z + 2 * (k + n4));
        __m256d vv = _mm256_loadu_pd(z + 2 * (k + n2));
        __m256d ww = _mm256_loadu_pd(z + 2 * (k + n2 + n4));
        __m256d cmul = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0); /* conj multiply */
        __m256d t2 = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_shuffle_pd(vv, vv, 0x5), w1i), cmul,
                                     _mm256_mul_pd(vv, w1r));
        __m256d t3 = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_shuffle_pd(ww, ww, 0x5), w3i), cmul,
                                     _mm256_mul_pd(ww, w3r));
        __m256d u = _mm256_add_pd(t2, t3);
        __m256d d = _mm256_sub_pd(t2, t3);
        _mm256_storeu_pd(z + 2 * k, _mm256_add_pd(uk, u));
        _mm256_storeu_pd(z + 2 * (k + n2), _mm256_sub_pd(uk, u));
        /* +i d: {−di, dr} */
        __m256d dswap = _mm256_shuffle_pd(d, d, 0x5);
        __m256d plus_i_d = _mm256_mul_pd(dswap, _mm256_set_pd(1.0, -1.0, 1.0, -1.0));
        _mm256_storeu_pd(z + 2 * (k + n4), _mm256_add_pd(u2, plus_i_d));
        _mm256_storeu_pd(z + 2 * (k + n2 + n4), _mm256_sub_pd(u2, plus_i_d));
    }
#endif
    for (; k < n4; k++)
        notorious_fft_dit_combine1(z, k, n4, n2, ep[4 * k], ep[4 * k + 1],
                                   ep[4 * k + 2], ep[4 * k + 3], 1);
}

static void notorious_fft_execute_sr_dit(
    const notorious_fft_plan* plan,
    const notorious_fft_real* x_in,
    notorious_fft_real* y_out,
    int inverse)
{
    int N = (int)plan->n;
    const int* rev = plan->bitrev;
    notorious_fft_real* z = (x_in == y_out) ? plan->work_re : y_out;
    for (int i = 0; i < N; i++) {
        int j = rev[i];
        z[2 * i]     = x_in[2 * j];
        z[2 * i + 1] = x_in[2 * j + 1];
    }
    if (inverse)
        notorious_fft_sr_inv_dit_cx(N, z, plan->sr_e);
    else
        notorious_fft_sr_dit_cx(N, z, plan->sr_e);
    if (z != y_out)
        memcpy(y_out, z, 2 * (size_t)N * sizeof(notorious_fft_real));
}

/* Mixed-radix Cooley–Tukey, N = r × m, r ∈ {3,5}.
 * n = n1 + r n2,  k = m k1 + k2
 * 1) gather x[n1 + r n2] → tmp[n1 m + n2]
 * 2) m-point FFT of each n1-row
 * 3) twiddle tmp[n1 m + k2] *= W_N^{n1 k2}
 * 4) r-point DFT across n1 for each k2
 * 5) tmp[k1 m + k2] is X[m k1 + k2] */
static void notorious_fft_execute_mixed_cx(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out, int inverse)
{
    const int r = plan->mixed_radix;
    const int n = (int)plan->n;
    const int m = n / r;
    notorious_fft_plan* sub = plan->mixed_sub;
    if (!sub || (r != 3 && r != 5 && r != 7) || m < 1 || !plan->sr_t) return;

    notorious_fft_real* tmp = plan->sr_t;
    for (int n1 = 0; n1 < r; n1++) {
        for (int n2 = 0; n2 < m; n2++) {
            size_t src = (size_t)(n1 + r * n2) * 2;
            size_t dst = (size_t)(n1 * m + n2) * 2;
            tmp[dst]     = x_in[src];
            tmp[dst + 1] = x_in[src + 1];
        }
    }

    for (int n1 = 0; n1 < r; n1++)
        notorious_fft_execute_cx(sub, tmp + (size_t)n1 * m * 2,
                                 tmp + (size_t)n1 * m * 2, inverse);

    const notorious_fft_real* tw_re = plan->tw_re;
    const notorious_fft_real* tw_im = plan->tw_im;
    if (tw_re && tw_im) {
        for (int n1 = 1; n1 < r; n1++) {
            for (int k2 = 0; k2 < m; k2++) {
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

    for (int k2 = 0; k2 < m; k2++) {
        if (r == 3)
            notorious_fft_radix3_stride(tmp + 2 * k2, m, inverse);
        else if (r == 5)
            notorious_fft_radix5_stride(tmp + 2 * k2, m, inverse);
        else
            notorious_fft_radix7_stride(tmp + 2 * k2, m, inverse);
    }

    if (y_out != tmp)
        memcpy(y_out, tmp, (size_t)n * 2 * sizeof(notorious_fft_real));
}

/* Rader: prime-N DFT as cyclic convolution of length N−1.
 * a[p] = x[g^p], b[p] = ω^{g^{-p}}, y[g^{-q}] = x[0] + (a ∗ b)[q],
 * y[0] = Σ x. Inner FFT is unnormalized, so IFFT is scaled by 1/(N−1). */
static void notorious_fft_execute_rader_cx(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out)
{
    const size_t n = plan->n;
    const size_t m = n - 1;
    notorious_fft_real* a = plan->sr_t;
    const int* in_idx = plan->rader_in;
    const int* out_idx = plan->rader_out;
    if (!a || !in_idx || !out_idx || !plan->rader_sub || !plan->rader_b_re) return;

    notorious_fft_real x0r = x_in[0], x0i = x_in[1];
    notorious_fft_real dcr = 0, dci = 0;
    for (size_t i = 0; i < n; i++) {
        dcr += x_in[2 * i];
        dci += x_in[2 * i + 1];
    }
    for (size_t j = 0; j < m; j++) {
        size_t s = (size_t)in_idx[j] * 2;
        a[2 * j]     = x_in[s];
        a[2 * j + 1] = x_in[s + 1];
    }

    notorious_fft_execute_cx(plan->rader_sub, a, a, 0);

    const notorious_fft_real* br = plan->rader_b_re;
    const notorious_fft_real* bi = plan->rader_b_im;
    for (size_t j = 0; j < m; j++) {
        notorious_fft_real ar = a[2 * j], ai = a[2 * j + 1];
        a[2 * j]     = ar * br[j] - ai * bi[j];
        a[2 * j + 1] = ar * bi[j] + ai * br[j];
    }

    notorious_fft_execute_cx(plan->rader_sub, a, a, 1);

    notorious_fft_real scale = (notorious_fft_real)1.0 / (notorious_fft_real)m;
    y_out[0] = dcr;
    y_out[1] = dci;
    for (size_t j = 0; j < m; j++) {
        size_t d = (size_t)out_idx[j] * 2;
        y_out[d]     = x0r + a[2 * j] * scale;
        y_out[d + 1] = x0i + a[2 * j + 1] * scale;
    }
}

/* Row-major complex transpose: src[i*cols+j] -> dst[j*rows+i] */
static void notorious_fft_transpose_rm(
    const notorious_fft_real* src, notorious_fft_real* dst, int rows, int cols)
{
    const int TS = 16;
    for (int i0 = 0; i0 < rows; i0 += TS) {
        int i1 = i0 + TS < rows ? i0 + TS : rows;
        for (int j0 = 0; j0 < cols; j0 += TS) {
            int j1 = j0 + TS < cols ? j0 + TS : cols;
            for (int i = i0; i < i1; i++) {
                for (int j = j0; j < j1; j++) {
                    size_t s = ((size_t)i * (size_t)cols + (size_t)j) * 2;
                    size_t d = ((size_t)j * (size_t)rows + (size_t)i) * 2;
                    dst[d]     = src[s];
                    dst[d + 1] = src[s + 1];
                }
            }
        }
    }
}

/* Four-step / six-step Cooley–Tukey: N = n1 × n2.
 * n = n1_idx + n1 n2_idx,  k = n2 k1 + k2
 * tmp[p,q] = x[p + n1 q] → FFT_n2 along q → twiddle W^{p q} → FFT_n1 along p. */
static void notorious_fft_execute_four_step(const notorious_fft_plan* plan,
    const notorious_fft_real* x_in, notorious_fft_real* y_out, int inverse)
{
    const int n1 = plan->four_n1;
    const int n2 = plan->four_n2;
    notorious_fft_plan* s1 = plan->four_sub1;
    notorious_fft_plan* s2 = plan->four_sub2 ? plan->four_sub2 : plan->four_sub1;
    if (!s1 || !s2 || n1 < 1 || n2 < 1 || !plan->sr_t || !plan->work_re) return;

    notorious_fft_real* tmp = plan->sr_t;
    notorious_fft_real* work = plan->work_re;

    /* x[q*n1+p] -> tmp[p*n2+q] */
    notorious_fft_transpose_rm(x_in, tmp, n2, n1);

    for (int p = 0; p < n1; p++)
        notorious_fft_execute_cx(s2, tmp + (size_t)p * n2 * 2,
                                 tmp + (size_t)p * n2 * 2, inverse);

    const notorious_fft_real* tw_re = plan->four_tw_re;
    const notorious_fft_real* tw_im = plan->four_tw_im;
    if (tw_re && tw_im) {
        for (int p = 0; p < n1; p++) {
            for (int q = 0; q < n2; q++) {
                size_t i = (size_t)p * (size_t)n2 + (size_t)q;
                /* stored at plan as q*n1+p; map to p*n2+q */
                size_t ti = (size_t)q * (size_t)n1 + (size_t)p;
                notorious_fft_real wr = tw_re[ti];
                notorious_fft_real wi = inverse ? -tw_im[ti] : tw_im[ti];
                notorious_fft_real ur = tmp[2 * i], ui = tmp[2 * i + 1];
                tmp[2 * i]     = ur * wr - ui * wi;
                tmp[2 * i + 1] = ur * wi + ui * wr;
            }
        }
    }

    /* tmp[p*n2+q] -> work[q*n1+p] */
    notorious_fft_transpose_rm(tmp, work, n1, n2);

    for (int q = 0; q < n2; q++)
        notorious_fft_execute_cx(s1, work + (size_t)q * n1 * 2,
                                 work + (size_t)q * n1 * 2, inverse);

    /* work[q*n1+p] -> y[p*n2+q] = X[k1*n2 + k2] */
    notorious_fft_transpose_rm(work, y_out, n2, n1);
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

    /* Forward Bluestein only. Inverse is the unnormalized conjugate identity
     * applied by notorious_fft_execute_cx; this function never mutates the plan. */
    
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
