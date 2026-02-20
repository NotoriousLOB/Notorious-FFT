/*
	OpenMP benchmark: multi-dimensional transforms, comparing
	single-threaded vs OpenMP-parallelized NotoriousFFT
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "notorious_fft.h"

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
	fn(ctx); fn(ctx);
	double t0 = get_time();
	for (int i = 0; i < reps; i++)
		fn(ctx);
	return (get_time() - t0) / reps;
}

/* ---- 2D complex DFT ---- */
struct dft2d_ctx {
	notorious_fft_cmpl *x, *y;
	notorious_fft_aux *a;
};
static void run_dft2d(void *ctx) {
	struct dft2d_ctx *c = ctx;
	notorious_fft_dft(c->x, c->y, c->a);
}

/* ---- 2D DCT-2 ---- */
struct dct2d_ctx {
	notorious_fft_real *x, *y;
	notorious_fft_aux *a;
};
static void run_dct2d(void *ctx) {
	struct dct2d_ctx *c = ctx;
	notorious_fft_dct2(c->x, c->y, c->a);
}

int main(void) {
#ifdef _OPENMP
	int nthreads = omp_get_max_threads();
	printf("OpenMP enabled: %d threads\n\n", nthreads);
#else
	printf("OpenMP NOT enabled (single-threaded)\n\n");
#endif

	int sizes[][2] = {
		{64,  64},
		{128, 128},
		{256, 256},
		{512, 512},
		{1024, 64},
		{64,  1024},
		{512, 1024},
	};
	int nsizes = (int)(sizeof(sizes)/sizeof(sizes[0]));

	printf("--- 2D Complex DFT ---\n");
	printf("%-14s  %-12s  %-10s\n", "N1 x N2", "time (ms)", "Mflop/s");
	for (int i = 0; i < nsizes; i++) {
		int N1 = sizes[i][0], N2 = sizes[i][1], N = N1*N2;
		int reps = (N <= 16384) ? 500 : (N <= 262144) ? 100 : 20;
		struct dft2d_ctx c;
		c.x = malloc(N * sizeof(notorious_fft_cmpl));
		c.y = malloc(N * sizeof(notorious_fft_cmpl));
		fill((double*)c.x, 2*N);
		c.a = notorious_fft_mkaux_dft_2d(N1, N2);
		double t = bench(run_dft2d, &c, reps);
		double mflops = 5.0*N*log2(N) / t / 1e6;
		char name[32]; snprintf(name, sizeof(name), "%dx%d", N1, N2);
		printf("%-14s  %-12.3f  %.1f\n", name, t*1e3, mflops);
		notorious_fft_free_aux(c.a); free(c.x); free(c.y);
	}

	printf("\n--- 2D DCT-2 ---\n");
	printf("%-14s  %-12s  %-10s\n", "N1 x N2", "time (ms)", "Mflop/s");
	for (int i = 0; i < nsizes; i++) {
		int N1 = sizes[i][0], N2 = sizes[i][1], N = N1*N2;
		int reps = (N <= 16384) ? 500 : (N <= 262144) ? 100 : 20;
		struct dct2d_ctx c;
		c.x = malloc(N * sizeof(notorious_fft_real));
		c.y = malloc(N * sizeof(notorious_fft_real));
		fill(c.x, N);
		c.a = notorious_fft_mkaux_t2t3_2d(N1, N2);
		double t = bench(run_dct2d, &c, reps);
		double mflops = 5.0*N*log2(N) / t / 1e6;
		char name[32]; snprintf(name, sizeof(name), "%dx%d", N1, N2);
		printf("%-14s  %-12.3f  %.1f\n", name, t*1e3, mflops);
		notorious_fft_free_aux(c.a); free(c.x); free(c.y);
	}

	return 0;
}
