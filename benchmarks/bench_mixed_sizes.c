/*
 * Enhanced benchmark for non-power-of-two, prime, and multi-dimensional FFTs
 * 
 * This benchmark tests NotoriousFFT (and optionally minfft) with:
 * - Non-power-of-two sizes (NPO2)
 * - Prime sizes
 * - Mixed sizes (composite with small factors)
 * - Multi-dimensional transforms (1D, 2D, 3D, 4D)
 * 
 * Usage:
 *   bench_mixed_sizes              Run all benchmarks
 *   bench_mixed_sizes --1d         Run 1D benchmarks only
 *   bench_mixed_sizes --2d         Run 2D benchmarks only
 *   bench_mixed_sizes --3d         Run 3D benchmarks only
 *   bench_mixed_sizes --4d         Run 4D benchmarks only
 *   bench_mixed_sizes --power2     Run power-of-2 benchmarks only
 *   bench_mixed_sizes --npo2       Run non-power-of-2 benchmarks only
 *   bench_mixed_sizes --prime      Run prime size benchmarks only
 *   bench_mixed_sizes --json       Output results in JSON format
 *   bench_mixed_sizes --help       Show help message
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "minfft.h"
#include "notorious_fft.h"

/* Version info for NotoriousFFT - check if Bluestein is available */
#ifndef NOTORIOUS_FFT_HAS_BLUESTEIN
#define NOTORIOUS_FFT_HAS_BLUESTEIN 0
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill(double *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = sin(i * 1.23456789 + 0.1 * i * i);
}

typedef void (*bench_fn)(void *ctx);

static double bench(bench_fn fn, void *ctx, int reps) {
    /* warmup */
    fn(ctx);
    fn(ctx);
    
    double t0 = get_time();
    for (int i = 0; i < reps; i++)
        fn(ctx);
    return (get_time() - t0) / reps;
}

static int choose_reps_1d(int N) {
    if (N <= 64) return 100000;
    if (N <= 256) return 50000;
    if (N <= 1024) return 10000;
    if (N <= 4096) return 5000;
    if (N <= 16384) return 1000;
    if (N <= 65536) return 500;
    if (N <= 262144) return 100;
    return 50;
}

static int choose_reps_nd(int total_n) {
    if (total_n <= 64) return 10000;
    if (total_n <= 256) return 5000;
    if (total_n <= 1024) return 2000;
    if (total_n <= 4096) return 1000;
    if (total_n <= 16384) return 500;
    if (total_n <= 65536) return 200;
    if (total_n <= 262144) return 100;
    return 50;
}

/* ============================================================================
   Test size definitions
   ============================================================================ */

/* Power-of-2 sizes for reference */
static int sizes_power2[] = {16, 64, 256, 1024, 4096, 16384, 65536};
static int nsizes_power2 = sizeof(sizes_power2) / sizeof(sizes_power2[0]);

/* Non-power-of-2 sizes (highly composite, small prime factors) */
static int sizes_npo2[] = {12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 
                           20, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 30720, 61440,
                           30, 72, 144, 288, 576, 1152, 2304, 4608, 9216, 18432, 36864, 73728,
                           36, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000};
static int nsizes_npo2 = sizeof(sizes_npo2) / sizeof(sizes_npo2[0]);

/* Prime sizes (challenging for FFT) */
static int sizes_prime[] = {17, 31, 61, 127, 257, 521, 1031, 2053, 4099, 8191, 16381, 32771, 65537,
                            101, 211, 503, 1009, 2003, 5003, 10007, 20011, 50021};
static int nsizes_prime = sizeof(sizes_prime) / sizeof(sizes_prime[0]);

/* Mixed sizes with medium prime factors (e.g., 2*3*5*7=210, 2*3*5*7*11=2310) */
static int sizes_mixed[] = {6, 12, 30, 60, 120, 210, 420, 840, 1680, 2310, 4620, 9240, 18480, 27720, 55440,
                            42, 90, 180, 360, 720, 1260, 2520, 5040, 7560, 15120, 30240, 60480};
static int nsizes_mixed = sizeof(sizes_mixed) / sizeof(sizes_mixed[0]);

/* 2D sizes: {N1, N2} pairs - all power-of-2 for minfft compatibility */
static int sizes_2d[][2] = {
    /* Power-of-2 squares */
    {16, 16}, {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512},
    /* Non-square power-of-2 */
    {32, 64}, {64, 128}, {128, 256}, {256, 512},
    {32, 128}, {64, 256}, {128, 512},
    {16, 64}, {32, 256}, {64, 512},
    /* Large rectangular */
    {1024, 64}, {64, 1024}, {512, 256}, {256, 512},
    {1024, 128}, {128, 1024}, {1024, 256}, {256, 1024},
};
static int nsizes_2d = sizeof(sizes_2d) / sizeof(sizes_2d[0]);

/* 3D sizes: {N1, N2, N3} triples - all power-of-2 for minfft compatibility */
static int sizes_3d[][3] = {
    /* Small power-of-2 cubes */
    {8, 8, 8}, {16, 16, 16}, {32, 32, 32},
    /* Small non-cubic */
    {8, 8, 16}, {16, 16, 32}, {8, 16, 16},
    {8, 16, 32}, {16, 32, 32}, {16, 32, 64},
    /* Medium sizes */
    {32, 32, 64}, {64, 64, 8}, {64, 64, 16},
    {32, 64, 64}, {64, 32, 32},
};
static int nsizes_3d = sizeof(sizes_3d) / sizeof(sizes_3d[0]);

/* 4D sizes: {N1, N2, N3, N4} quadruples - all power-of-2 for minfft compatibility */
static int sizes_4d[][4] = {
    /* Small hypercubes */
    {4, 4, 4, 4}, {8, 8, 8, 8}, {16, 16, 16, 16},
    /* Small non-hypercubic */
    {4, 4, 4, 8}, {4, 4, 8, 8}, {4, 8, 8, 8},
    {8, 8, 8, 16}, {8, 8, 16, 16}, {8, 16, 16, 16},
    /* Medium rectangular */
    {4, 8, 16, 16}, {8, 8, 8, 32}, {8, 16, 16, 32},
    {16, 16, 16, 8}, {16, 16, 8, 8},
};
static int nsizes_4d = sizeof(sizes_4d) / sizeof(sizes_4d[0]);

/* ============================================================================
   Output format
   ============================================================================ */

typedef struct {
    char library[16];
    char transform[16];
    char size_type[16];
    int dim;
    int n1, n2, n3, n4;
    double time_us;
    double mflops;
    int total_n;
} BenchResult;

#define MAX_RESULTS 1000
static BenchResult results[MAX_RESULTS];
static int nresults = 0;
static bool output_json = false;

static void add_result(const char *library, const char *transform, const char *size_type,
                       int dim, int n1, int n2, int n3, int n4,
                       double time_us, double mflops, int total_n) {
    if (nresults >= MAX_RESULTS) return;
    BenchResult *r = &results[nresults++];
    strncpy(r->library, library, 15); r->library[15] = '\0';
    strncpy(r->transform, transform, 15); r->transform[15] = '\0';
    strncpy(r->size_type, size_type, 15); r->size_type[15] = '\0';
    r->dim = dim;
    r->n1 = n1; r->n2 = n2; r->n3 = n3; r->n4 = n4;
    r->time_us = time_us;
    r->mflops = mflops;
    r->total_n = total_n;
}

static void print_json_results(void) {
    printf("[\n");
    for (int i = 0; i < nresults; i++) {
        BenchResult *r = &results[i];
        printf("  {\n");
        printf("    \"library\": \"%s\",\n", r->library);
        printf("    \"transform\": \"%s\",\n", r->transform);
        printf("    \"size_type\": \"%s\",\n", r->size_type);
        printf("    \"dim\": %d,\n", r->dim);
        printf("    \"n1\": %d,\n", r->n1);
        if (r->dim >= 2) printf("    \"n2\": %d,\n", r->n2);
        if (r->dim >= 3) printf("    \"n3\": %d,\n", r->n3);
        if (r->dim >= 4) printf("    \"n4\": %d,\n", r->n4);
        printf("    \"total_n\": %d,\n", r->total_n);
        printf("    \"time_us\": %.4f,\n", r->time_us);
        printf("    \"mflops\": %.2f\n", r->mflops);
        printf("  }%s\n", (i < nresults - 1) ? "," : "");
    }
    printf("]\n");
}

static void print_text_results(void) {
    printf("\n");
    printf("================================================================================\n");
    printf("                          BENCHMARK RESULTS SUMMARY\n");
    printf("================================================================================\n");
    printf("%-10s %-12s %-10s %-4s %-20s %-12s %-10s\n",
           "Library", "Transform", "SizeType", "Dim", "Dimensions", "Time(us)", "Mflop/s");
    printf("--------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < nresults; i++) {
        BenchResult *r = &results[i];
        char dims[32];
        if (r->dim == 1) snprintf(dims, sizeof(dims), "%d", r->n1);
        else if (r->dim == 2) snprintf(dims, sizeof(dims), "%dx%d", r->n1, r->n2);
        else if (r->dim == 3) snprintf(dims, sizeof(dims), "%dx%dx%d", r->n1, r->n2, r->n3);
        else snprintf(dims, sizeof(dims), "%dx%dx%dx%d", r->n1, r->n2, r->n3, r->n4);
        
        printf("%-10s %-12s %-10s %-4d %-20s %-12.2f %-10.1f\n",
               r->library, r->transform, r->size_type, r->dim, dims,
               r->time_us, r->mflops);
    }
    printf("================================================================================\n");
}

/* ============================================================================
   1D Complex DFT Benchmarks
   ============================================================================ */

struct dft1d_ctx {
    int N;
    minfft_cmpl *mx, *my;
    minfft_aux *ma;
    notorious_fft_cmpl *lx, *ly;
    notorious_fft_aux *la;
};

static void bench_minfft_dft(void *ctx) {
    struct dft1d_ctx *c = ctx;
    minfft_dft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dft(void *ctx) {
    struct dft1d_ctx *c = ctx;
    notorious_fft_dft(c->lx, c->ly, c->la);
}

static int benchmark_1d_complex_dft(int *sizes, int nsizes, const char *size_type) {
    if (!output_json) {
        printf("\n--- 1D Complex DFT (%s) ---\n", size_type);
        printf("%-10s  %-12s  %-12s  %-10s  %-10s\n", "N", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("--------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes; i++) {
        int N = sizes[i];
        int reps = choose_reps_1d(N);
        struct dft1d_ctx c;
        c.N = N;
        c.mx = malloc(N * sizeof(minfft_cmpl));
        c.my = malloc(N * sizeof(minfft_cmpl));
        c.lx = malloc(N * sizeof(notorious_fft_cmpl));
        c.ly = malloc(N * sizeof(notorious_fft_cmpl));
        
        if (!c.mx || !c.my || !c.lx || !c.ly) {
            if (!output_json) printf("%-10d  Memory allocation failed\n", N);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        fill((double*)c.mx, 2 * N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
        
        c.ma = minfft_mkaux_dft_1d(N);
        c.la = notorious_fft_mkaux_dft_1d(N);
        
        if (!c.ma) {
            if (!output_json) printf("%-10d  minfft: unsupported size\n", N);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            if (c.la) notorious_fft_free_aux(c.la);
            continue;
        }
        
        if (!c.la) {
            if (!output_json) printf("%-10d  notoriousfft: unsupported size\n", N);
            minfft_free_aux(c.ma);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        double tm = bench(bench_minfft_dft, &c, reps);
        double tl = bench(bench_notorious_fft_dft, &c, reps);
        double ratio = tm / tl;
        
        /* Mflops: 5*N*log2(N) flops per FFT */
        double mflops_m = 5.0 * N * log2((double)N) / (tm * 1e6);
        double mflops_l = 5.0 * N * log2((double)N) / (tl * 1e6);
        
        if (!output_json) {
            printf("%-10d  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   N, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "dft", size_type, 1, N, 0, 0, 0, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "dft", size_type, 1, N, 0, 0, 0, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.my); free(c.lx); free(c.ly);
    }
    return success_count;
}

/* ============================================================================
   1D Real DFT Benchmarks
   ============================================================================ */

struct rdft1d_ctx {
    int N;
    minfft_real *mx;
    minfft_cmpl *mz;
    minfft_aux *ma;
    notorious_fft_real *lx;
    notorious_fft_cmpl *lz;
    notorious_fft_aux *la;
};

static void bench_minfft_realdft(void *ctx) {
    struct rdft1d_ctx *c = ctx;
    minfft_realdft(c->mx, c->mz, c->ma);
}

static void bench_notorious_fft_realdft(void *ctx) {
    struct rdft1d_ctx *c = ctx;
    notorious_fft_realdft(c->lx, c->lz, c->la);
}

static int benchmark_1d_real_dft(int *sizes, int nsizes, const char *size_type) {
    if (!output_json) {
        printf("\n--- 1D Real DFT (%s) ---\n", size_type);
        printf("%-10s  %-12s  %-12s  %-10s  %-10s\n", "N", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("--------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes; i++) {
        int N = sizes[i];
        int reps = choose_reps_1d(N);
        struct rdft1d_ctx c;
        c.N = N;
        c.mx = malloc(N * sizeof(minfft_real));
        c.mz = malloc((N / 2 + 1) * sizeof(minfft_cmpl));
        c.lx = malloc(N * sizeof(notorious_fft_real));
        c.lz = malloc((N / 2 + 1) * sizeof(notorious_fft_cmpl));
        
        if (!c.mx || !c.mz || !c.lx || !c.lz) {
            free(c.mx); free(c.mz); free(c.lx); free(c.lz);
            continue;
        }
        
        fill(c.mx, N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_real));
        
        c.ma = minfft_mkaux_realdft_1d(N);
        c.la = notorious_fft_mkaux_realdft_1d(N);
        
        if (!c.ma || !c.la) {
            minfft_free_aux(c.ma);
            notorious_fft_free_aux(c.la);
            free(c.mx); free(c.mz); free(c.lx); free(c.lz);
            continue;
        }
        
        double tm = bench(bench_minfft_realdft, &c, reps);
        double tl = bench(bench_notorious_fft_realdft, &c, reps);
        double ratio = tm / tl;
        
        double mflops_m = 2.5 * N * log2((double)N) / (tm * 1e6);
        double mflops_l = 2.5 * N * log2((double)N) / (tl * 1e6);
        
        if (!output_json) {
            printf("%-10d  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   N, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "realdft", size_type, 1, N, 0, 0, 0, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "realdft", size_type, 1, N, 0, 0, 0, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.mz); free(c.lx); free(c.lz);
    }
    return success_count;
}

/* ============================================================================
   2D Complex DFT Benchmarks
   ============================================================================ */

struct dft2d_ctx {
    int N1, N2, N;
    minfft_cmpl *mx, *my;
    minfft_aux *ma;
    notorious_fft_cmpl *lx, *ly;
    notorious_fft_aux *la;
};

static void bench_minfft_dft_2d(void *ctx) {
    struct dft2d_ctx *c = ctx;
    minfft_dft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dft_2d(void *ctx) {
    struct dft2d_ctx *c = ctx;
    notorious_fft_dft(c->lx, c->ly, c->la);
}

static int benchmark_2d_complex_dft(void) {
    if (!output_json) {
        printf("\n--- 2D Complex DFT ---\n");
        printf("%-14s  %-12s  %-12s  %-10s  %-10s\n", "Dimensions", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("--------------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes_2d; i++) {
        int N1 = sizes_2d[i][0];
        int N2 = sizes_2d[i][1];
        int N = N1 * N2;
        int reps = choose_reps_nd(N);
        
        struct dft2d_ctx c;
        c.N1 = N1; c.N2 = N2; c.N = N;
        c.mx = malloc(N * sizeof(minfft_cmpl));
        c.my = malloc(N * sizeof(minfft_cmpl));
        c.lx = malloc(N * sizeof(notorious_fft_cmpl));
        c.ly = malloc(N * sizeof(notorious_fft_cmpl));
        
        if (!c.mx || !c.my || !c.lx || !c.ly) {
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        fill((double*)c.mx, 2 * N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
        
        c.ma = minfft_mkaux_dft_2d(N1, N2);
        c.la = notorious_fft_mkaux_dft_2d(N1, N2);
        
        if (!c.ma) {
            if (!output_json) printf("%-14s  minfft: unsupported\n", "");
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            if (c.la) notorious_fft_free_aux(c.la);
            continue;
        }
        
        if (!c.la) {
            if (!output_json) printf("%-14s  notoriousfft: unsupported\n", "");
            minfft_free_aux(c.ma);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        double tm = bench(bench_minfft_dft_2d, &c, reps);
        double tl = bench(bench_notorious_fft_dft_2d, &c, reps);
        double ratio = tm / tl;
        
        /* Flops: 5*N1*N2*(log2(N1) + log2(N2)) = 5*N*log2(N1*N2) */
        double flops = 5.0 * N * (log2((double)N1) + log2((double)N2));
        double mflops_m = flops / (tm * 1e6);
        double mflops_l = flops / (tl * 1e6);
        
        char dims[16];
        snprintf(dims, sizeof(dims), "%dx%d", N1, N2);
        
        /* Determine size type */
        const char *size_type = "mixed";
        bool is_power2 = (N1 & (N1-1)) == 0 && (N2 & (N2-1)) == 0;
        bool is_npo2 = !is_power2;
        bool has_prime = false;  /* Simplified check */
        if (is_power2) size_type = "power2";
        else if (is_npo2) size_type = "npo2";
        
        if (!output_json) {
            printf("%-14s  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   dims, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "dft", size_type, 2, N1, N2, 0, 0, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "dft", size_type, 2, N1, N2, 0, 0, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.my); free(c.lx); free(c.ly);
    }
    return success_count;
}

/* ============================================================================
   2D Real DCT-2 Benchmarks
   ============================================================================ */

struct dct2d_ctx {
    int N1, N2, N;
    minfft_real *mx, *my;
    minfft_aux *ma;
    notorious_fft_real *lx, *ly;
    notorious_fft_aux *la;
};

static void bench_minfft_dct2_2d(void *ctx) {
    struct dct2d_ctx *c = ctx;
    minfft_dct2(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dct2_2d(void *ctx) {
    struct dct2d_ctx *c = ctx;
    notorious_fft_dct2(c->lx, c->ly, c->la);
}

static int benchmark_2d_dct2(void) {
    if (!output_json) {
        printf("\n--- 2D DCT-2 ---\n");
        printf("%-14s  %-12s  %-12s  %-10s  %-10s\n", "Dimensions", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("--------------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes_2d; i++) {
        int N1 = sizes_2d[i][0];
        int N2 = sizes_2d[i][1];
        int N = N1 * N2;
        int reps = choose_reps_nd(N);
        
        struct dct2d_ctx c;
        c.N1 = N1; c.N2 = N2; c.N = N;
        c.mx = malloc(N * sizeof(minfft_real));
        c.my = malloc(N * sizeof(minfft_real));
        c.lx = malloc(N * sizeof(notorious_fft_real));
        c.ly = malloc(N * sizeof(notorious_fft_real));
        
        if (!c.mx || !c.my || !c.lx || !c.ly) {
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        fill(c.mx, N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_real));
        
        c.ma = minfft_mkaux_t2t3_2d(N1, N2);
        c.la = notorious_fft_mkaux_t2t3_2d(N1, N2);
        
        if (!c.ma || !c.la) {
            minfft_free_aux(c.ma);
            notorious_fft_free_aux(c.la);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        double tm = bench(bench_minfft_dct2_2d, &c, reps);
        double tl = bench(bench_notorious_fft_dct2_2d, &c, reps);
        double ratio = tm / tl;
        
        double flops = 5.0 * N * (log2((double)N1) + log2((double)N2));
        double mflops_m = flops / (tm * 1e6);
        double mflops_l = flops / (tl * 1e6);
        
        char dims[16];
        snprintf(dims, sizeof(dims), "%dx%d", N1, N2);
        
        const char *size_type = "mixed";
        bool is_power2 = (N1 & (N1-1)) == 0 && (N2 & (N2-1)) == 0;
        if (is_power2) size_type = "power2";
        else size_type = "npo2";
        
        if (!output_json) {
            printf("%-14s  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   dims, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "dct2", size_type, 2, N1, N2, 0, 0, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "dct2", size_type, 2, N1, N2, 0, 0, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.my); free(c.lx); free(c.ly);
    }
    return success_count;
}

/* ============================================================================
   3D Complex DFT Benchmarks
   ============================================================================ */

struct dft3d_ctx {
    int N1, N2, N3, N;
    minfft_cmpl *mx, *my;
    minfft_aux *ma;
    notorious_fft_cmpl *lx, *ly;
    notorious_fft_aux *la;
};

static void bench_minfft_dft_3d(void *ctx) {
    struct dft3d_ctx *c = ctx;
    minfft_dft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dft_3d(void *ctx) {
    struct dft3d_ctx *c = ctx;
    notorious_fft_dft(c->lx, c->ly, c->la);
}

static int benchmark_3d_complex_dft(void) {
    if (!output_json) {
        printf("\n--- 3D Complex DFT ---\n");
        printf("%-18s  %-12s  %-12s  %-10s  %-10s\n", "Dimensions", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("------------------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes_3d; i++) {
        int N1 = sizes_3d[i][0];
        int N2 = sizes_3d[i][1];
        int N3 = sizes_3d[i][2];
        int N = N1 * N2 * N3;
        int reps = choose_reps_nd(N);
        
        struct dft3d_ctx c;
        c.N1 = N1; c.N2 = N2; c.N3 = N3; c.N = N;
        c.mx = malloc(N * sizeof(minfft_cmpl));
        c.my = malloc(N * sizeof(minfft_cmpl));
        c.lx = malloc(N * sizeof(notorious_fft_cmpl));
        c.ly = malloc(N * sizeof(notorious_fft_cmpl));
        
        if (!c.mx || !c.my || !c.lx || !c.ly) {
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        fill((double*)c.mx, 2 * N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
        
        c.ma = minfft_mkaux_dft_3d(N1, N2, N3);
        c.la = notorious_fft_mkaux_dft_3d(N1, N2, N3);
        
        if (!c.ma) {
            if (!output_json) printf("%-18s  minfft: unsupported\n", "");
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            if (c.la) notorious_fft_free_aux(c.la);
            continue;
        }
        
        if (!c.la) {
            if (!output_json) printf("%-18s  notoriousfft: unsupported\n", "");
            minfft_free_aux(c.ma);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        double tm = bench(bench_minfft_dft_3d, &c, reps);
        double tl = bench(bench_notorious_fft_dft_3d, &c, reps);
        double ratio = tm / tl;
        
        double flops = 5.0 * N * (log2((double)N1) + log2((double)N2) + log2((double)N3));
        double mflops_m = flops / (tm * 1e6);
        double mflops_l = flops / (tl * 1e6);
        
        char dims[20];
        snprintf(dims, sizeof(dims), "%dx%dx%d", N1, N2, N3);
        
        const char *size_type = "mixed";
        bool is_power2 = (N1 & (N1-1)) == 0 && (N2 & (N2-1)) == 0 && (N3 & (N3-1)) == 0;
        if (is_power2) size_type = "power2";
        else size_type = "npo2";
        
        if (!output_json) {
            printf("%-18s  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   dims, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "dft", size_type, 3, N1, N2, N3, 0, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "dft", size_type, 3, N1, N2, N3, 0, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.my); free(c.lx); free(c.ly);
    }
    return success_count;
}

/* ============================================================================
   4D Complex DFT Benchmarks
   ============================================================================ */

struct dft4d_ctx {
    int N1, N2, N3, N4, N;
    minfft_cmpl *mx, *my;
    minfft_aux *ma;
    notorious_fft_cmpl *lx, *ly;
    notorious_fft_aux *la;
};

static void bench_minfft_dft_4d(void *ctx) {
    struct dft4d_ctx *c = ctx;
    minfft_dft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dft_4d(void *ctx) {
    struct dft4d_ctx *c = ctx;
    notorious_fft_dft(c->lx, c->ly, c->la);
}

static int benchmark_4d_complex_dft(void) {
    if (!output_json) {
        printf("\n--- 4D Complex DFT ---\n");
        printf("%-22s  %-12s  %-12s  %-10s  %-10s\n", "Dimensions", "minfft(us)", "notoriousfft(us)", "ratio", "Mflop/s");
        printf("----------------------------------------------------------------------------------\n");
    }
    
    int success_count = 0;
    for (int i = 0; i < nsizes_4d; i++) {
        int N1 = sizes_4d[i][0];
        int N2 = sizes_4d[i][1];
        int N3 = sizes_4d[i][2];
        int N4 = sizes_4d[i][3];
        int N = N1 * N2 * N3 * N4;
        int reps = choose_reps_nd(N);
        
        struct dft4d_ctx c;
        c.N1 = N1; c.N2 = N2; c.N3 = N3; c.N4 = N4; c.N = N;
        c.mx = malloc(N * sizeof(minfft_cmpl));
        c.my = malloc(N * sizeof(minfft_cmpl));
        c.lx = malloc(N * sizeof(notorious_fft_cmpl));
        c.ly = malloc(N * sizeof(notorious_fft_cmpl));
        
        if (!c.mx || !c.my || !c.lx || !c.ly) {
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        fill((double*)c.mx, 2 * N);
        memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
        
        /* Use generic N-dimensional API for 4D */
        int Ns[4] = {N1, N2, N3, N4};
        c.ma = minfft_mkaux_dft(4, Ns);
        c.la = notorious_fft_mkaux_dft(4, Ns);
        
        if (!c.ma) {
            if (!output_json) printf("%-22s  minfft: unsupported\n", "");
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            if (c.la) notorious_fft_free_aux(c.la);
            continue;
        }
        
        if (!c.la) {
            if (!output_json) printf("%-22s  notoriousfft: unsupported\n", "");
            minfft_free_aux(c.ma);
            free(c.mx); free(c.my); free(c.lx); free(c.ly);
            continue;
        }
        
        double tm = bench(bench_minfft_dft_4d, &c, reps);
        double tl = bench(bench_notorious_fft_dft_4d, &c, reps);
        double ratio = tm / tl;
        
        double flops = 5.0 * N * (log2((double)N1) + log2((double)N2) + log2((double)N3) + log2((double)N4));
        double mflops_m = flops / (tm * 1e6);
        double mflops_l = flops / (tl * 1e6);
        
        char dims[24];
        snprintf(dims, sizeof(dims), "%dx%dx%dx%d", N1, N2, N3, N4);
        
        const char *size_type = "mixed";
        bool is_power2 = (N1 & (N1-1)) == 0 && (N2 & (N2-1)) == 0 && 
                         (N3 & (N3-1)) == 0 && (N4 & (N4-1)) == 0;
        if (is_power2) size_type = "power2";
        else size_type = "npo2";
        
        if (!output_json) {
            printf("%-22s  %-12.2f  %-12.2f  %-10.2f  %.1f/%.1f\n",
                   dims, tm * 1e6, tl * 1e6, ratio, mflops_m, mflops_l);
        }
        
        add_result("minfft", "dft", size_type, 4, N1, N2, N3, N4, tm * 1e6, mflops_m, N);
        add_result("notoriousfft", "dft", size_type, 4, N1, N2, N3, N4, tl * 1e6, mflops_l, N);
        success_count++;
        
        minfft_free_aux(c.ma);
        notorious_fft_free_aux(c.la);
        free(c.mx); free(c.my); free(c.lx); free(c.ly);
    }
    return success_count;
}

/* ============================================================================
   Main
   ============================================================================ */

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  (no args)     Run all benchmarks\n");
    printf("  --1d          Run 1D benchmarks only\n");
    printf("  --2d          Run 2D benchmarks only\n");
    printf("  --3d          Run 3D benchmarks only\n");
    printf("  --4d          Run 4D benchmarks only\n");
    printf("  --power2      Run power-of-2 size benchmarks only\n");
    printf("  --npo2        Run non-power-of-2 size benchmarks only\n");
    printf("  --prime       Run prime size benchmarks only\n");
    printf("  --mixed       Run mixed (composite) size benchmarks only\n");
    printf("  --json        Output results in JSON format\n");
    printf("  --help        Show this help message\n");
    printf("\nBenchmark types:\n");
    printf("  - 1D Complex DFT (power-of-2, NPO2, prime, mixed)\n");
    printf("  - 1D Real DFT (power-of-2, NPO2)\n");
    printf("  - 2D Complex DFT (various dimensions)\n");
    printf("  - 2D DCT-2 (various dimensions)\n");
    printf("  - 3D Complex DFT (various dimensions)\n");
    printf("  - 4D Complex DFT (various dimensions)\n");
}

int main(int argc, char *argv[]) {
    bool run_1d = true, run_2d = true, run_3d = true, run_4d = true;
    bool run_power2 = true, run_npo2 = true, run_prime = true, run_mixed = true;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--1d") == 0) {
            run_1d = true; run_2d = false; run_3d = false; run_4d = false;
        } else if (strcmp(argv[i], "--2d") == 0) {
            run_1d = false; run_2d = true; run_3d = false; run_4d = false;
        } else if (strcmp(argv[i], "--3d") == 0) {
            run_1d = false; run_2d = false; run_3d = true; run_4d = false;
        } else if (strcmp(argv[i], "--4d") == 0) {
            run_1d = false; run_2d = false; run_3d = false; run_4d = true;
        } else if (strcmp(argv[i], "--power2") == 0) {
            run_power2 = true; run_npo2 = false; run_prime = false; run_mixed = false;
        } else if (strcmp(argv[i], "--npo2") == 0) {
            run_power2 = false; run_npo2 = true; run_prime = false; run_mixed = false;
        } else if (strcmp(argv[i], "--prime") == 0) {
            run_power2 = false; run_npo2 = false; run_prime = true; run_mixed = false;
        } else if (strcmp(argv[i], "--mixed") == 0) {
            run_power2 = false; run_npo2 = false; run_prime = false; run_mixed = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            output_json = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!output_json) {
        printf("================================================================================\n");
        printf("            Enhanced FFT Benchmark: NPO2, Prime, and Multi-Dimensional\n");
        printf("================================================================================\n");
        printf("\nThis benchmark compares NotoriousFFT vs minfft with:\n");
        printf("  - Non-power-of-2 sizes (NPO2)\n");
        printf("  - Prime sizes\n");
        printf("  - Mixed/composite sizes\n");
        printf("  - Multi-dimensional transforms (1D, 2D, 3D, 4D)\n");
        printf("\nNote: minfft only supports power-of-2 sizes.\n");
        printf("      NotoriousFFT supports arbitrary sizes via Bluestein's algorithm.\n");
    }
    
    /* Run benchmarks based on options */
    
    /* 1D benchmarks */
    if (run_1d) {
        if (run_power2) {
            benchmark_1d_complex_dft(sizes_power2, nsizes_power2, "power2");
            benchmark_1d_real_dft(sizes_power2, nsizes_power2, "power2");
        }
        if (run_npo2) {
            benchmark_1d_complex_dft(sizes_npo2, nsizes_npo2, "npo2");
            benchmark_1d_real_dft(sizes_npo2, nsizes_npo2, "npo2");
        }
        if (run_prime) {
            benchmark_1d_complex_dft(sizes_prime, nsizes_prime, "prime");
        }
        if (run_mixed) {
            benchmark_1d_complex_dft(sizes_mixed, nsizes_mixed, "mixed");
            benchmark_1d_real_dft(sizes_mixed, nsizes_mixed, "mixed");
        }
    }
    
    /* 2D benchmarks */
    if (run_2d && run_power2) {
        benchmark_2d_complex_dft();
        benchmark_2d_dct2();
    }
    
    /* 3D benchmarks */
    if (run_3d && run_power2) {
        benchmark_3d_complex_dft();
    }
    
    /* 4D benchmarks */
    if (run_4d && run_power2) {
        benchmark_4d_complex_dft();
    }
    
    /* Output results */
    if (output_json) {
        print_json_results();
    } else {
        print_text_results();
        
        printf("\n");
        printf("================================================================================\n");
        printf("Benchmark complete. Total results: %d\n", nresults);
        printf("================================================================================\n");
    }
    
    return 0;
}
