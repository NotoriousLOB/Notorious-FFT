/*
 * Isolated DCT-II/III (REDFT10/REDFT01) microbenchmark.
 *
 * Times the public execute path only (packing, phase, child FFT included).
 * Optionally compares against FFTW3 when built with NOTORIOUS_TEST_FFTW.
 *
 * Usage: bench_r2r_dct
 */
#include "notorious_fft.h"
#ifdef NOTORIOUS_TEST_FFTW
#include <fftw3.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    return v[n / 2];
}

static const char *algo_name(int algo) {
    switch (algo) {
    case NOTORIOUS_R2R_ALGO_DIRECT: return "direct";
    case NOTORIOUS_R2R_ALGO_POW2_REALDFT: return "pow2-realdft";
    case NOTORIOUS_R2R_ALGO_DCT_FFT_N: return "fft-N";
    case NOTORIOUS_R2R_ALGO_DCT_FFT_2N: return "fft-2N";
    default: return "none";
    }
}

int main(void) {
    const int ns[] = {256, 257, 258, 511, 512, 513};
    const int n_sizes = (int)(sizeof(ns) / sizeof(*ns));
    const int warmup = 8, steady = 40;
    printf("# Isolated REDFT10/REDFT01 execute medians (µs), warmup=%d steady=%d\n", warmup, steady);
    printf("# N kind notorious_us fftw_us ratio algo direct plan\n");

    for (int si = 0; si < n_sizes; si++) {
        int n = ns[si];
        notorious_fft_r2r_kind kinds[2] = {NOTORIOUS_FFT_REDFT10, NOTORIOUS_FFT_REDFT01};
        const char *knames[2] = {"REDFT10", "REDFT01"};
        for (int ki = 0; ki < 2; ki++) {
            double *in = (double *)notorious_fft_malloc((size_t)n * sizeof(double));
            double *out = (double *)notorious_fft_malloc((size_t)n * sizeof(double));
            if (!in || !out) return 2;
            for (int i = 0; i < n; i++) in[i] = sin(0.1 * (i + 1));
            notorious_fft_io_plan *p = notorious_fft_plan_r2r_1d(n, in, out, kinds[ki], NOTORIOUS_FFT_ESTIMATE);
            if (!p || !p->axis_aux[0]) {
                fprintf(stderr, "plan failed N=%d\n", n);
                return 2;
            }
            double samples[64];
            for (int i = 0; i < warmup; i++) notorious_fft_execute(p);
            for (int i = 0; i < steady; i++) {
                double t0 = now_s();
                notorious_fft_execute(p);
                samples[i] = (now_s() - t0) * 1e6;
            }
            double nmed = median(samples, steady);
            double fmed = 0;
#ifdef NOTORIOUS_TEST_FFTW
            fftw_plan fp = fftw_plan_r2r_1d(n, in, out, (fftw_r2r_kind)kinds[ki], FFTW_ESTIMATE);
            if (!fp) return 2;
            for (int i = 0; i < warmup; i++) fftw_execute(fp);
            for (int i = 0; i < steady; i++) {
                double t0 = now_s();
                fftw_execute(fp);
                samples[i] = (now_s() - t0) * 1e6;
            }
            fmed = median(samples, steady);
            fftw_destroy_plan(fp);
#endif
            const notorious_fft_aux *a = p->axis_aux[0];
            printf("%d %s %.3f %.3f %.3f %s %d %p\n",
                   n, knames[ki], nmed, fmed,
                   (fmed > 0) ? nmed / fmed : 0.0,
                   algo_name(a->r2r_algo), a->direct_r2r, (void *)a->plan);
            notorious_fft_destroy_io_plan(p);
            notorious_fft_free(in);
            notorious_fft_free(out);
        }
    }
#ifdef NOTORIOUS_TEST_FFTW
    fftw_cleanup();
#endif
    return 0;
}
