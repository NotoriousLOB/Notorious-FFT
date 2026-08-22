/*
 * notorious_fft_fftw.h — FFTW3 basic/advanced API aliases
 *
 * Include after notorious_fft.h. Does not include <fftw3.h> and must not be
 * mixed with linking libfftw3 in the same translation unit (symbol clash).
 *
 * Supported subset:
 *   fftw_plan_dft_1d/2d/3d, fftw_plan_dft
 *   fftw_plan_dft_r2c_1d/2d, fftw_plan_dft_c2r_1d/2d
 *   fftw_plan_r2r_1d (REDFT10/01/11, RODFT10/01/11)
 *   fftw_plan_many_dft (rank-1 fully; higher rank uses the stored aux)
 *   fftw_execute, fftw_execute_dft, _dft_r2c, _dft_c2r, _r2r
 *   fftw_destroy_plan, fftw_cleanup
 *   fftw_malloc / fftw_free / fftw_alloc_real / fftw_alloc_complex
 *
 * FFTW_MEASURE/PATIENT currently behave as ESTIMATE (heuristics only).
 * Inverse transforms are unnormalized, matching FFTW.
 */

#ifndef NOTORIOUS_FFT_FFTW_H
#define NOTORIOUS_FFT_FFTW_H

#include "notorious_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FFTW_FORWARD
#define FFTW_FORWARD           NOTORIOUS_FFT_FORWARD
#define FFTW_BACKWARD          NOTORIOUS_FFT_BACKWARD
#define FFTW_ESTIMATE          NOTORIOUS_FFT_ESTIMATE
#define FFTW_MEASURE           NOTORIOUS_FFT_MEASURE
#define FFTW_DESTROY_INPUT     NOTORIOUS_FFT_DESTROY_INPUT
#define FFTW_UNALIGNED         NOTORIOUS_FFT_UNALIGNED
#define FFTW_CONSERVE_MEMORY   NOTORIOUS_FFT_CONSERVE_MEMORY
#define FFTW_EXHAUSTIVE        NOTORIOUS_FFT_EXHAUSTIVE
#define FFTW_PRESERVE_INPUT    NOTORIOUS_FFT_PRESERVE_INPUT
#define FFTW_PATIENT           NOTORIOUS_FFT_PATIENT

#define FFTW_REDFT00           NOTORIOUS_FFT_REDFT00
#define FFTW_REDFT01           NOTORIOUS_FFT_REDFT01
#define FFTW_REDFT10           NOTORIOUS_FFT_REDFT10
#define FFTW_REDFT11           NOTORIOUS_FFT_REDFT11
#define FFTW_RODFT00           NOTORIOUS_FFT_RODFT00
#define FFTW_RODFT01           NOTORIOUS_FFT_RODFT01
#define FFTW_RODFT10           NOTORIOUS_FFT_RODFT10
#define FFTW_RODFT11           NOTORIOUS_FFT_RODFT11
#endif

typedef notorious_fft_cmpl notorious_fft_fftw_complex;
typedef notorious_fft_io_plan *fftw_plan;
typedef notorious_fft_r2r_kind fftw_r2r_kind;

#ifdef NOTORIOUS_FFT_SINGLE
typedef notorious_fft_cmpl fftwf_complex;
#else
typedef notorious_fft_cmpl fftw_complex;
#endif

#define fftw_malloc              notorious_fft_malloc
#define fftw_free                notorious_fft_free
#define fftw_plan_dft_1d         notorious_fft_plan_dft_1d
#define fftw_plan_dft_2d         notorious_fft_plan_dft_2d
#define fftw_plan_dft_3d         notorious_fft_plan_dft_3d
#define fftw_plan_dft            notorious_fft_plan_dft
#define fftw_plan_dft_r2c_1d     notorious_fft_plan_dft_r2c_1d
#define fftw_plan_dft_c2r_1d     notorious_fft_plan_dft_c2r_1d
#define fftw_plan_dft_r2c_2d     notorious_fft_plan_dft_r2c_2d
#define fftw_plan_dft_c2r_2d     notorious_fft_plan_dft_c2r_2d
#define fftw_plan_r2r_1d         notorious_fft_plan_r2r_1d
#define fftw_plan_many_dft       notorious_fft_plan_many_dft
#define fftw_execute             notorious_fft_execute
#define fftw_execute_dft         notorious_fft_execute_dft
#define fftw_execute_dft_r2c     notorious_fft_execute_dft_r2c
#define fftw_execute_dft_c2r     notorious_fft_execute_dft_c2r
#define fftw_execute_r2r         notorious_fft_execute_r2r
#define fftw_destroy_plan        notorious_fft_destroy_io_plan
#define fftw_cleanup             notorious_fft_cleanup

static NOTORIOUS_FFT_INLINE void *fftw_alloc_real(size_t n) {
    return notorious_fft_malloc(n * sizeof(notorious_fft_real));
}

static NOTORIOUS_FFT_INLINE void *fftw_alloc_complex(size_t n) {
    return notorious_fft_malloc(n * sizeof(notorious_fft_cmpl));
}

#ifdef NOTORIOUS_FFT_SINGLE
#define fftwf_malloc            fftw_malloc
#define fftwf_free              fftw_free
#define fftwf_plan_dft_1d       fftw_plan_dft_1d
#define fftwf_execute           fftw_execute
#define fftwf_destroy_plan      fftw_destroy_plan
#define fftwf_cleanup           fftw_cleanup
#define fftwf_alloc_real        fftw_alloc_real
#define fftwf_alloc_complex     fftw_alloc_complex
typedef fftw_plan fftwf_plan;
#endif

#ifdef __cplusplus
}
#endif

#endif /* NOTORIOUS_FFT_FFTW_H */
