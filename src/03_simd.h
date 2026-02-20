/*
 * Notorious FFT - SIMD Butterfly Operations
 */

#ifndef NOTORIOUS_FFT_SIMD_H
#define NOTORIOUS_FFT_SIMD_H

#include "02_math.h"

/* ============================================================================
 * AVX-512 Implementation
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_AVX512

/* Process 8 elements (4 complex) at a time with gather */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly8_avx512(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT wr, notorious_fft_real* NOTORIOUS_FFT_RESTRICT wi,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT tw_re, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT tw_im,
    size_t i1, size_t i2, const int32_t* indices
) {
    __m512d ur = _mm512_loadu_pd(&wr[i1]);
    __m512d ui = _mm512_loadu_pd(&wi[i1]);
    __m512d vr = _mm512_loadu_pd(&wr[i2]);
    __m512d vi = _mm512_loadu_pd(&wi[i2]);
    
    /* Gather twiddles using 32-bit indices */
    __m256i idx = _mm256_loadu_si256((__m256i*)indices);
    __m512d t_wr = _mm512_i32gather_pd(idx, tw_re, 8);
    __m512d t_wi = _mm512_i32gather_pd(idx, tw_im, 8);
    
    /* Complex multiply: vr*wr - vi*wi + i*(vr*wi + vi*wr) */
    __m512d vro = _mm512_fmsub_pd(vr, t_wr, _mm512_mul_pd(vi, t_wi));
    __m512d vio = _mm512_fmadd_pd(vr, t_wi, _mm512_mul_pd(vi, t_wr));
    
    _mm512_storeu_pd(&wr[i1], _mm512_add_pd(ur, vro));
    _mm512_storeu_pd(&wi[i1], _mm512_add_pd(ui, vio));
    _mm512_storeu_pd(&wr[i2], _mm512_sub_pd(ur, vro));
    _mm512_storeu_pd(&wi[i2], _mm512_sub_pd(ui, vio));
}

/* Process single butterfly with AVX-512 (8 complex = 4 butterflies) */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly16_avx512(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT wr, notorious_fft_real* NOTORIOUS_FFT_RESTRICT wi,
    size_t i1, size_t i2,
    __m512d wr_vec, __m512d wi_vec
) {
    __m512d ur = _mm512_loadu_pd(&wr[i1]);
    __m512d ui = _mm512_loadu_pd(&wi[i1]);
    __m512d vr = _mm512_loadu_pd(&wr[i2]);
    __m512d vi = _mm512_loadu_pd(&wi[i2]);
    
    __m512d vro = _mm512_fmsub_pd(vr, wr_vec, _mm512_mul_pd(vi, wi_vec));
    __m512d vio = _mm512_fmadd_pd(vr, wi_vec, _mm512_mul_pd(vi, wr_vec));
    
    _mm512_storeu_pd(&wr[i1], _mm512_add_pd(ur, vro));
    _mm512_storeu_pd(&wi[i1], _mm512_add_pd(ui, vio));
    _mm512_storeu_pd(&wr[i2], _mm512_sub_pd(ur, vro));
    _mm512_storeu_pd(&wi[i2], _mm512_sub_pd(ui, vio));
}

#endif /* NOTORIOUS_FFT_HAS_AVX512 */

/* ============================================================================
 * AVX2 Implementation
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_AVX2

/* Process 4 elements (2 complex) at a time with gather */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly4_avx2(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT wr, notorious_fft_real* NOTORIOUS_FFT_RESTRICT wi,
    const notorious_fft_real* NOTORIOUS_FFT_RESTRICT tw_re, const notorious_fft_real* NOTORIOUS_FFT_RESTRICT tw_im,
    size_t i1, size_t i2, __m128i indices
) {
    __m256d ur = _mm256_loadu_pd(&wr[i1]);
    __m256d ui = _mm256_loadu_pd(&wi[i1]);
    __m256d vr = _mm256_loadu_pd(&wr[i2]);
    __m256d vi = _mm256_loadu_pd(&wi[i2]);
    
    /* Gather twiddles */
    __m256d t_wr = _mm256_i32gather_pd(tw_re, indices, 8);
    __m256d t_wi = _mm256_i32gather_pd(tw_im, indices, 8);
    
    /* Complex multiply */
    __m256d vro = _mm256_fmsub_pd(vr, t_wr, _mm256_mul_pd(vi, t_wi));
    __m256d vio = _mm256_fmadd_pd(vr, t_wi, _mm256_mul_pd(vi, t_wr));
    
    _mm256_storeu_pd(&wr[i1], _mm256_add_pd(ur, vro));
    _mm256_storeu_pd(&wi[i1], _mm256_add_pd(ui, vio));
    _mm256_storeu_pd(&wr[i2], _mm256_sub_pd(ur, vro));
    _mm256_storeu_pd(&wi[i2], _mm256_sub_pd(ui, vio));
}

/* Process single butterfly with AVX2 (4 complex = 2 butterflies) */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly8_avx2(
    notorious_fft_real* NOTORIOUS_FFT_RESTRICT wr, notorious_fft_real* NOTORIOUS_FFT_RESTRICT wi,
    size_t i1, size_t i2,
    __m256d wr_vec, __m256d wi_vec
) {
    __m256d ur = _mm256_loadu_pd(&wr[i1]);
    __m256d ui = _mm256_loadu_pd(&wi[i1]);
    __m256d vr = _mm256_loadu_pd(&wr[i2]);
    __m256d vi = _mm256_loadu_pd(&wi[i2]);
    
    __m256d vro = _mm256_fmsub_pd(vr, wr_vec, _mm256_mul_pd(vi, wi_vec));
    __m256d vio = _mm256_fmadd_pd(vr, wi_vec, _mm256_mul_pd(vi, wr_vec));
    
    _mm256_storeu_pd(&wr[i1], _mm256_add_pd(ur, vro));
    _mm256_storeu_pd(&wi[i1], _mm256_add_pd(ui, vio));
    _mm256_storeu_pd(&wr[i2], _mm256_sub_pd(ur, vro));
    _mm256_storeu_pd(&wi[i2], _mm256_sub_pd(ui, vio));
}

#endif /* NOTORIOUS_FFT_HAS_AVX2 */

/* ============================================================================
 * NEON Implementation
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_NEON

#ifdef NOTORIOUS_FFT_SINGLE
/* Float32 version - 4 complex per operation */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly4_neon_f32(
    float* NOTORIOUS_FFT_RESTRICT wr, float* NOTORIOUS_FFT_RESTRICT wi,
    float wr0, float wi0, float wr1, float wi1, 
    float wr2, float wi2, float wr3, float wi3,
    size_t i1, size_t i2
) {
    float32x4_t ur = vld1q_f32(&wr[i1]);
    float32x4_t ui = vld1q_f32(&wi[i1]);
    float32x4_t vr = vld1q_f32(&wr[i2]);
    float32x4_t vi = vld1q_f32(&wi[i2]);
    
    float32x4_t t_wr = (float32x4_t){wr0, wr1, wr2, wr3};
    float32x4_t t_wi = (float32x4_t){wi0, wi1, wi2, wi3};
    
    float32x4_t vro = vsubq_f32(vmulq_f32(vr, t_wr), vmulq_f32(vi, t_wi));
    float32x4_t vio = vaddq_f32(vmulq_f32(vr, t_wi), vmulq_f32(vi, t_wr));
    
    vst1q_f32(&wr[i1], vaddq_f32(ur, vro));
    vst1q_f32(&wi[i1], vaddq_f32(ui, vio));
    vst1q_f32(&wr[i2], vsubq_f32(ur, vro));
    vst1q_f32(&wi[i2], vsubq_f32(ui, vio));
}
#else
/* Float64 version - 2 complex per operation */
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly2_neon_f64(
    double* NOTORIOUS_FFT_RESTRICT wr, double* NOTORIOUS_FFT_RESTRICT wi,
    double wr0, double wi0, double wr1, double wi1,
    size_t i1, size_t i2
) {
    float64x2_t ur = vld1q_f64(&wr[i1]);
    float64x2_t ui = vld1q_f64(&wi[i1]);
    float64x2_t vr = vld1q_f64(&wr[i2]);
    float64x2_t vi = vld1q_f64(&wi[i2]);
    
    float64x2_t t_wr = (float64x2_t){wr0, wr1};
    float64x2_t t_wi = (float64x2_t){wi0, wi1};
    
    float64x2_t vro = vsubq_f64(vmulq_f64(vr, t_wr), vmulq_f64(vi, t_wi));
    float64x2_t vio = vaddq_f64(vmulq_f64(vr, t_wi), vmulq_f64(vi, t_wr));
    
    vst1q_f64(&wr[i1], vaddq_f64(ur, vro));
    vst1q_f64(&wi[i1], vaddq_f64(ui, vio));
    vst1q_f64(&wr[i2], vsubq_f64(ur, vro));
    vst1q_f64(&wi[i2], vsubq_f64(ui, vio));
}
#endif /* NOTORIOUS_FFT_SINGLE */

/* ============================================================================
 * NEON Bluestein Algorithm Acceleration
 * ============================================================================ */

#ifndef NOTORIOUS_FFT_SINGLE

/* Vectorized complex multiply for pre-multiply by chirp (Step 1)
 * buf = x * chirp (complex pointwise multiplication)
 * N must be >= 2 for NEON path; scalar tail handles remainder */
static void notorious_fft_bluestein_premul_neon(size_t N,
                                          notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                          notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT x_re,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT x_im,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_re,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_im) {
    size_t i;
    /* Process 2 elements at a time using float64x2_t */
    for (i = 0; i + 2 <= N; i += 2) {
        float64x2_t xr = vld1q_f64(x_re + i);
        float64x2_t xi = vld1q_f64(x_im + i);
        float64x2_t cr = vld1q_f64(chirp_re + i);
        float64x2_t ci = vld1q_f64(chirp_im + i);

        /* buf_re = xr * cr - xi * ci */
        float64x2_t bre = vmlsq_f64(vmulq_f64(xr, cr), xi, ci);
        /* buf_im = xr * ci + xi * cr */
        float64x2_t bim = vfmaq_f64(vmulq_f64(xr, ci), xi, cr);

        vst1q_f64(buf_re + i, bre);
        vst1q_f64(buf_im + i, bim);
    }
    /* Scalar tail for remaining elements */
    for (; i < N; i++) {
        buf_re[i] = x_re[i] * chirp_re[i] - x_im[i] * chirp_im[i];
        buf_im[i] = x_re[i] * chirp_im[i] + x_im[i] * chirp_re[i];
    }
}

/* Vectorized complex multiply for frequency-domain convolution (Step 3)
 * buf = buf * chirp_fft (pointwise multiplication in freq domain)
 * M is the padded FFT size (power of 2, >= 2N-1) */
static void notorious_fft_bluestein_convolve_neon(size_t M,
                                            notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                            notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                            const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_fft_re,
                                            const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_fft_im) {
    size_t i;
    for (i = 0; i + 2 <= M; i += 2) {
        float64x2_t br = vld1q_f64(buf_re + i);
        float64x2_t bi = vld1q_f64(buf_im + i);
        float64x2_t cr = vld1q_f64(chirp_fft_re + i);
        float64x2_t ci = vld1q_f64(chirp_fft_im + i);

        /* new_re = br * cr - bi * ci */
        float64x2_t new_re = vmlsq_f64(vmulq_f64(br, cr), bi, ci);
        /* new_im = br * ci + bi * cr */
        float64x2_t new_im = vfmaq_f64(vmulq_f64(br, ci), bi, cr);

        vst1q_f64(buf_re + i, new_re);
        vst1q_f64(buf_im + i, new_im);
    }
    /* Scalar tail */
    for (; i < M; i++) {
        notorious_fft_real temp_re = buf_re[i] * chirp_fft_re[i] - buf_im[i] * chirp_fft_im[i];
        notorious_fft_real temp_im = buf_re[i] * chirp_fft_im[i] + buf_im[i] * chirp_fft_re[i];
        buf_re[i] = temp_re;
        buf_im[i] = temp_im;
    }
}

/* Vectorized complex multiply with scaling for post-multiply (Step 5)
 * out = buf * chirp * scale */
static void notorious_fft_bluestein_postmul_neon(size_t N,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT out_re,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT out_im,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_re,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_im,
                                           notorious_fft_real scale) {
    size_t i;
    float64x2_t vscale = vdupq_n_f64(scale);
    
    for (i = 0; i + 2 <= N; i += 2) {
        float64x2_t br = vld1q_f64(buf_re + i);
        float64x2_t bi = vld1q_f64(buf_im + i);
        float64x2_t cr = vld1q_f64(chirp_re + i);
        float64x2_t ci = vld1q_f64(chirp_im + i);

        /* out_re = (br * cr - bi * ci) * scale */
        float64x2_t ore = vmlsq_f64(vmulq_f64(br, cr), bi, ci);
        ore = vmulq_f64(ore, vscale);
        /* out_im = (br * ci + bi * cr) * scale */
        float64x2_t oim = vfmaq_f64(vmulq_f64(br, ci), bi, cr);
        oim = vmulq_f64(oim, vscale);

        vst1q_f64(out_re + i, ore);
        vst1q_f64(out_im + i, oim);
    }
    /* Scalar tail */
    for (; i < N; i++) {
        out_re[i] = (buf_re[i] * chirp_re[i] - buf_im[i] * chirp_im[i]) * scale;
        out_im[i] = (buf_re[i] * chirp_im[i] + buf_im[i] * chirp_re[i]) * scale;
    }
}

/* Zero-pad using NEON (set buf[n..m-1] to 0) */
static void notorious_fft_bluestein_zeropad_neon(size_t n, size_t m,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im) {
    size_t i;
    float64x2_t zero = vdupq_n_f64(0.0);
    
    for (i = n; i + 2 <= m; i += 2) {
        vst1q_f64(buf_re + i, zero);
        vst1q_f64(buf_im + i, zero);
    }
    for (; i < m; i++) {
        buf_re[i] = 0;
        buf_im[i] = 0;
    }
}

#endif /* !NOTORIOUS_FFT_SINGLE */

#endif /* NOTORIOUS_FFT_HAS_NEON */

/* ============================================================================
 * AVX2 Bluestein Algorithm Acceleration
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_AVX2 && !defined(NOTORIOUS_FFT_SINGLE)

/* Vectorized complex multiply for pre-multiply by chirp (Step 1)
 * buf = x * chirp (complex pointwise multiplication)
 * Processes 4 elements per iteration using 256-bit AVX2 vectors. */
static void notorious_fft_bluestein_premul_avx2(size_t N,
                                          notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                          notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT x_re,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT x_im,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_re,
                                          const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_im) {
    size_t i;
    for (i = 0; i + 4 <= N; i += 4) {
        __m256d xr = _mm256_loadu_pd(x_re + i);
        __m256d xi = _mm256_loadu_pd(x_im + i);
        __m256d cr = _mm256_loadu_pd(chirp_re + i);
        __m256d ci = _mm256_loadu_pd(chirp_im + i);

        /* buf_re = xr * cr - xi * ci */
        __m256d bre = _mm256_fmsub_pd(xr, cr, _mm256_mul_pd(xi, ci));
        /* buf_im = xr * ci + xi * cr */
        __m256d bim = _mm256_fmadd_pd(xr, ci, _mm256_mul_pd(xi, cr));

        _mm256_storeu_pd(buf_re + i, bre);
        _mm256_storeu_pd(buf_im + i, bim);
    }
    /* Scalar tail */
    for (; i < N; i++) {
        buf_re[i] = x_re[i] * chirp_re[i] - x_im[i] * chirp_im[i];
        buf_im[i] = x_re[i] * chirp_im[i] + x_im[i] * chirp_re[i];
    }
}

/* Vectorized complex multiply for frequency-domain convolution (Step 3)
 * Processes 4 elements per iteration. */
static void notorious_fft_bluestein_convolve_avx2(size_t M,
                                            notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                            notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                            const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_fft_re,
                                            const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_fft_im) {
    size_t i;
    for (i = 0; i + 4 <= M; i += 4) {
        __m256d br = _mm256_loadu_pd(buf_re + i);
        __m256d bi = _mm256_loadu_pd(buf_im + i);
        __m256d cr = _mm256_loadu_pd(chirp_fft_re + i);
        __m256d ci = _mm256_loadu_pd(chirp_fft_im + i);

        __m256d new_re = _mm256_fmsub_pd(br, cr, _mm256_mul_pd(bi, ci));
        __m256d new_im = _mm256_fmadd_pd(br, ci, _mm256_mul_pd(bi, cr));

        _mm256_storeu_pd(buf_re + i, new_re);
        _mm256_storeu_pd(buf_im + i, new_im);
    }
    /* Scalar tail */
    for (; i < M; i++) {
        notorious_fft_real temp_re = buf_re[i] * chirp_fft_re[i] - buf_im[i] * chirp_fft_im[i];
        notorious_fft_real temp_im = buf_re[i] * chirp_fft_im[i] + buf_im[i] * chirp_fft_re[i];
        buf_re[i] = temp_re;
        buf_im[i] = temp_im;
    }
}

/* Vectorized complex multiply with scaling for post-multiply (Step 5) */
static void notorious_fft_bluestein_postmul_avx2(size_t N,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT out_re,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT out_im,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_re,
                                           const notorious_fft_real *NOTORIOUS_FFT_RESTRICT chirp_im,
                                           notorious_fft_real scale) {
    size_t i;
    __m256d vscale = _mm256_set1_pd(scale);

    for (i = 0; i + 4 <= N; i += 4) {
        __m256d br = _mm256_loadu_pd(buf_re + i);
        __m256d bi = _mm256_loadu_pd(buf_im + i);
        __m256d cr = _mm256_loadu_pd(chirp_re + i);
        __m256d ci = _mm256_loadu_pd(chirp_im + i);

        __m256d ore = _mm256_mul_pd(_mm256_fmsub_pd(br, cr, _mm256_mul_pd(bi, ci)), vscale);
        __m256d oim = _mm256_mul_pd(_mm256_fmadd_pd(br, ci, _mm256_mul_pd(bi, cr)), vscale);

        _mm256_storeu_pd(out_re + i, ore);
        _mm256_storeu_pd(out_im + i, oim);
    }
    /* Scalar tail */
    for (; i < N; i++) {
        out_re[i] = (buf_re[i] * chirp_re[i] - buf_im[i] * chirp_im[i]) * scale;
        out_im[i] = (buf_re[i] * chirp_im[i] + buf_im[i] * chirp_re[i]) * scale;
    }
}

/* Zero-pad using AVX2 */
static void notorious_fft_bluestein_zeropad_avx2(size_t n, size_t m,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_re,
                                           notorious_fft_real *NOTORIOUS_FFT_RESTRICT buf_im) {
    size_t i;
    __m256d zero = _mm256_setzero_pd();

    for (i = n; i + 4 <= m; i += 4) {
        _mm256_storeu_pd(buf_re + i, zero);
        _mm256_storeu_pd(buf_im + i, zero);
    }
    for (; i < m; i++) {
        buf_re[i] = 0;
        buf_im[i] = 0;
    }
}

#endif /* NOTORIOUS_FFT_HAS_AVX2 && !NOTORIOUS_FFT_SINGLE */

/* ============================================================================
 * SSE2 Implementation
 * ============================================================================ */

#if NOTORIOUS_FFT_HAS_SSE2

#ifdef NOTORIOUS_FFT_SINGLE
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly4_sse_f32(
    float* NOTORIOUS_FFT_RESTRICT wr, float* NOTORIOUS_FFT_RESTRICT wi,
    float wr0, float wi0, float wr1, float wi1,
    float wr2, float wi2, float wr3, float wi3,
    size_t i1, size_t i2
) {
    __m128 ur = _mm_loadu_ps(&wr[i1]);
    __m128 ui = _mm_loadu_ps(&wi[i1]);
    __m128 vr = _mm_loadu_ps(&wr[i2]);
    __m128 vi = _mm_loadu_ps(&wi[i2]);
    
    __m128 t_wr = _mm_set_ps(wr3, wr2, wr1, wr0);
    __m128 t_wi = _mm_set_ps(wi3, wi2, wi1, wi0);
    
    __m128 vro = _mm_sub_ps(_mm_mul_ps(vr, t_wr), _mm_mul_ps(vi, t_wi));
    __m128 vio = _mm_add_ps(_mm_mul_ps(vr, t_wi), _mm_mul_ps(vi, t_wr));
    
    _mm_storeu_ps(&wr[i1], _mm_add_ps(ur, vro));
    _mm_storeu_ps(&wi[i1], _mm_add_ps(ui, vio));
    _mm_storeu_ps(&wr[i2], _mm_sub_ps(ur, vro));
    _mm_storeu_ps(&wi[i2], _mm_sub_ps(ui, vio));
}
#else
static NOTORIOUS_FFT_INLINE void notorious_fft_butterfly2_sse_f64(
    double* NOTORIOUS_FFT_RESTRICT wr, double* NOTORIOUS_FFT_RESTRICT wi,
    double wr0, double wi0, double wr1, double wi1,
    size_t i1, size_t i2
) {
    __m128d ur = _mm_loadu_pd(&wr[i1]);
    __m128d ui = _mm_loadu_pd(&wi[i1]);
    __m128d vr = _mm_loadu_pd(&wr[i2]);
    __m128d vi = _mm_loadu_pd(&wi[i2]);
    
    __m128d t_wr = _mm_set_pd(wr1, wr0);
    __m128d t_wi = _mm_set_pd(wi1, wi0);
    
    __m128d vro = _mm_sub_pd(_mm_mul_pd(vr, t_wr), _mm_mul_pd(vi, t_wi));
    __m128d vio = _mm_add_pd(_mm_mul_pd(vr, t_wi), _mm_mul_pd(vi, t_wr));
    
    _mm_storeu_pd(&wr[i1], _mm_add_pd(ur, vro));
    _mm_storeu_pd(&wi[i1], _mm_add_pd(ui, vio));
    _mm_storeu_pd(&wr[i2], _mm_sub_pd(ur, vro));
    _mm_storeu_pd(&wi[i2], _mm_sub_pd(ui, vio));
}
#endif /* NOTORIOUS_FFT_SINGLE */

#endif /* NOTORIOUS_FFT_HAS_SSE2 */

#endif /* NOTORIOUS_FFT_SIMD_H */
