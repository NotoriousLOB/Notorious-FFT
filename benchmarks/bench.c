/*
	Benchmark: NotoriousFFT vs minfft performance comparison
	
	Usage:
	  bench              Run default benchmarks (dft, invdft, realdft, dct2)
	  bench --all        Run comprehensive benchmarks for all API functions
	  bench --help       Show this help message
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <complex.h>

#include "minfft.h"

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

/* returns elapsed time in seconds for 'reps' iterations */
typedef void (*bench_fn)(void *ctx);

static double bench(bench_fn fn, void *ctx, int reps) {
	/* warmup */
	fn(ctx);
	fn(ctx);

	double t0 = get_time();
	for (int i = 0; i < reps; i++)
		fn(ctx);
	double t1 = get_time();
	return (t1 - t0) / reps;
}

static int choose_reps(int N) {
	if (N <= 64) return 100000;
	if (N <= 256) return 50000;
	if (N <= 1024) return 10000;
	if (N <= 4096) return 5000;
	if (N <= 16384) return 1000;
	if (N <= 65536) return 500;
	return 100;
}

/* ============================================================================
   Complex DFT Benchmarks
   ============================================================================ */

struct dft_ctx {
	int N;
	minfft_cmpl *mx, *my;
	minfft_aux *ma;
	notorious_fft_cmpl *lx, *ly;
	notorious_fft_aux *la;
};

static void bench_minfft_dft(void *ctx) {
	struct dft_ctx *c = ctx;
	minfft_dft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dft(void *ctx) {
	struct dft_ctx *c = ctx;
	notorious_fft_dft(c->lx, c->ly, c->la);
}

static void bench_minfft_invdft(void *ctx) {
	struct dft_ctx *c = ctx;
	minfft_invdft(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_invdft(void *ctx) {
	struct dft_ctx *c = ctx;
	notorious_fft_invdft(c->lx, c->ly, c->la);
}

static void benchmark_dft(int *sizes, int nsizes) {
	printf("\n--- Complex DFT (Forward) ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dft_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_cmpl));
		c.my = malloc(N * sizeof(minfft_cmpl));
		c.lx = malloc(N * sizeof(notorious_fft_cmpl));
		c.ly = malloc(N * sizeof(notorious_fft_cmpl));
		fill((double*)c.mx, 2 * N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
		c.ma = minfft_mkaux_dft_1d(N);
		c.la = notorious_fft_mkaux_dft_1d(N);

		double tm = bench(bench_minfft_dft, &c, reps);
		double tl = bench(bench_notorious_fft_dft, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

static void benchmark_invdft(int *sizes, int nsizes) {
	printf("\n--- Complex DFT (Inverse) ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dft_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_cmpl));
		c.my = malloc(N * sizeof(minfft_cmpl));
		c.lx = malloc(N * sizeof(notorious_fft_cmpl));
		c.ly = malloc(N * sizeof(notorious_fft_cmpl));
		fill((double*)c.mx, 2 * N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_cmpl));
		c.ma = minfft_mkaux_dft_1d(N);
		c.la = notorious_fft_mkaux_dft_1d(N);

		double tm = bench(bench_minfft_invdft, &c, reps);
		double tl = bench(bench_notorious_fft_invdft, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

/* ============================================================================
   Real DFT Benchmarks
   ============================================================================ */

struct realdft_ctx {
	int N;
	minfft_real *mx;
	minfft_cmpl *mz;
	minfft_aux *ma;
	notorious_fft_real *lx;
	notorious_fft_cmpl *lz;
	notorious_fft_aux *la;
};

static void bench_minfft_realdft(void *ctx) {
	struct realdft_ctx *c = ctx;
	minfft_realdft(c->mx, c->mz, c->ma);
}

static void bench_notorious_fft_realdft(void *ctx) {
	struct realdft_ctx *c = ctx;
	notorious_fft_realdft(c->lx, c->lz, c->la);
}

static void bench_minfft_invrealdft(void *ctx) {
	struct realdft_ctx *c = ctx;
	minfft_invrealdft(c->mz, c->mx, c->ma);
}

static void bench_notorious_fft_invrealdft(void *ctx) {
	struct realdft_ctx *c = ctx;
	notorious_fft_invrealdft(c->lz, c->lx, c->la);
}

static void benchmark_realdft(int *sizes, int nsizes) {
	printf("\n--- Real DFT (Forward) ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct realdft_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.mz = malloc((N / 2 + 1) * sizeof(minfft_cmpl));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.lz = malloc((N / 2 + 1) * sizeof(notorious_fft_cmpl));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_realdft_1d(N);
		c.la = notorious_fft_mkaux_realdft_1d(N);

		double tm = bench(bench_minfft_realdft, &c, reps);
		double tl = bench(bench_notorious_fft_realdft, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.mz); free(c.lx); free(c.lz);
	}
}

static void benchmark_invrealdft(int *sizes, int nsizes) {
	printf("\n--- Real DFT (Inverse) ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct realdft_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.mz = malloc((N / 2 + 1) * sizeof(minfft_cmpl));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.lz = malloc((N / 2 + 1) * sizeof(notorious_fft_cmpl));
		/* Fill spectrum with data for inverse transform */
		fill((double*)c.mz, N + 2);
		memcpy(c.lz, c.mz, (N / 2 + 1) * sizeof(minfft_cmpl));
		c.ma = minfft_mkaux_realdft_1d(N);
		c.la = notorious_fft_mkaux_realdft_1d(N);

		double tm = bench(bench_minfft_invrealdft, &c, reps);
		double tl = bench(bench_notorious_fft_invrealdft, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.mz); free(c.lx); free(c.lz);
	}
}

/* ============================================================================
   DCT/DST Type 2 Benchmarks
   ============================================================================ */

struct dct_ctx {
	int N;
	minfft_real *mx, *my;
	minfft_aux *ma;
	notorious_fft_real *lx, *ly;
	notorious_fft_aux *la;
};

static void bench_minfft_dct2(void *ctx) {
	struct dct_ctx *c = ctx;
	minfft_dct2(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dct2(void *ctx) {
	struct dct_ctx *c = ctx;
	notorious_fft_dct2(c->lx, c->ly, c->la);
}

static void bench_minfft_dst2(void *ctx) {
	struct dct_ctx *c = ctx;
	minfft_dst2(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dst2(void *ctx) {
	struct dct_ctx *c = ctx;
	notorious_fft_dst2(c->lx, c->ly, c->la);
}

static void benchmark_dct2(int *sizes, int nsizes) {
	printf("\n--- DCT Type 2 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dct_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t2t3_1d(N);
		c.la = notorious_fft_mkaux_t2t3_1d(N);

		double tm = bench(bench_minfft_dct2, &c, reps);
		double tl = bench(bench_notorious_fft_dct2, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

static void benchmark_dst2(int *sizes, int nsizes) {
	printf("\n--- DST Type 2 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dct_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t2t3_1d(N);
		c.la = notorious_fft_mkaux_t2t3_1d(N);

		double tm = bench(bench_minfft_dst2, &c, reps);
		double tl = bench(bench_notorious_fft_dst2, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

/* ============================================================================
   DCT/DST Type 3 Benchmarks
   ============================================================================ */

static void bench_minfft_dct3(void *ctx) {
	struct dct_ctx *c = ctx;
	minfft_dct3(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dct3(void *ctx) {
	struct dct_ctx *c = ctx;
	notorious_fft_dct3(c->lx, c->ly, c->la);
}

static void bench_minfft_dst3(void *ctx) {
	struct dct_ctx *c = ctx;
	minfft_dst3(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dst3(void *ctx) {
	struct dct_ctx *c = ctx;
	notorious_fft_dst3(c->lx, c->ly, c->la);
}

static void benchmark_dct3(int *sizes, int nsizes) {
	printf("\n--- DCT Type 3 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dct_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t2t3_1d(N);
		c.la = notorious_fft_mkaux_t2t3_1d(N);

		double tm = bench(bench_minfft_dct3, &c, reps);
		double tl = bench(bench_notorious_fft_dct3, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

static void benchmark_dst3(int *sizes, int nsizes) {
	printf("\n--- DST Type 3 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct dct_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t2t3_1d(N);
		c.la = notorious_fft_mkaux_t2t3_1d(N);

		double tm = bench(bench_minfft_dst3, &c, reps);
		double tl = bench(bench_notorious_fft_dst3, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

/* ============================================================================
   DCT/DST Type 4 Benchmarks
   ============================================================================ */

struct t4_ctx {
	int N;
	minfft_real *mx, *my;
	minfft_aux *ma;
	notorious_fft_real *lx, *ly;
	notorious_fft_aux *la;
};

static void bench_minfft_dct4(void *ctx) {
	struct t4_ctx *c = ctx;
	minfft_dct4(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dct4(void *ctx) {
	struct t4_ctx *c = ctx;
	notorious_fft_dct4(c->lx, c->ly, c->la);
}

static void bench_minfft_dst4(void *ctx) {
	struct t4_ctx *c = ctx;
	minfft_dst4(c->mx, c->my, c->ma);
}

static void bench_notorious_fft_dst4(void *ctx) {
	struct t4_ctx *c = ctx;
	notorious_fft_dst4(c->lx, c->ly, c->la);
}

static void benchmark_dct4(int *sizes, int nsizes) {
	printf("\n--- DCT Type 4 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct t4_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t4_1d(N);
		c.la = notorious_fft_mkaux_t4_1d(N);

		if (!c.ma || !c.la) {
			printf("%-10d  skipped (aux creation failed)\n", N);
			free(c.mx); free(c.my); free(c.lx); free(c.ly);
			continue;
		}

		double tm = bench(bench_minfft_dct4, &c, reps);
		double tl = bench(bench_notorious_fft_dct4, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

static void benchmark_dst4(int *sizes, int nsizes) {
	printf("\n--- DST Type 4 ---\n");
	printf("%-10s  %-12s  %-12s  %-8s\n", "N", "minfft (us)", "notoriousfft (us)", "ratio");
	printf("------------------------------------------------------\n");
	for (int i = 0; i < nsizes; i++) {
		int N = sizes[i];
		int reps = choose_reps(N);
		struct t4_ctx c;
		c.N = N;
		c.mx = malloc(N * sizeof(minfft_real));
		c.my = malloc(N * sizeof(minfft_real));
		c.lx = malloc(N * sizeof(notorious_fft_real));
		c.ly = malloc(N * sizeof(notorious_fft_real));
		fill(c.mx, N);
		memcpy(c.lx, c.mx, N * sizeof(minfft_real));
		c.ma = minfft_mkaux_t4_1d(N);
		c.la = notorious_fft_mkaux_t4_1d(N);

		if (!c.ma || !c.la) {
			printf("%-10d  skipped (aux creation failed)\n", N);
			free(c.mx); free(c.my); free(c.lx); free(c.ly);
			continue;
		}

		double tm = bench(bench_minfft_dst4, &c, reps);
		double tl = bench(bench_notorious_fft_dst4, &c, reps);
		printf("%-10d  %-12.2f  %-12.2f  %.3fx\n", N, tm * 1e6, tl * 1e6, tm / tl);

		minfft_free_aux(c.ma);
		notorious_fft_free_aux(c.la);
		free(c.mx); free(c.my); free(c.lx); free(c.ly);
	}
}

/* ============================================================================
   Main
   ============================================================================ */

static void print_usage(const char *prog) {
	printf("Usage: %s [options]\n\n", prog);
	printf("Options:\n");
	printf("  --all       Run comprehensive benchmarks for all API functions\n");
	printf("  --help      Show this help message\n");
	printf("\nDefault benchmarks (without --all):\n");
	printf("  - Complex DFT (forward)\n");
	printf("  - Complex DFT (inverse)\n");
	printf("  - Real DFT (forward)\n");
	printf("  - DCT Type 2\n");
	printf("\nComprehensive benchmarks (--all):\n");
	printf("  All default benchmarks plus:\n");
	printf("  - Real DFT (inverse)\n");
	printf("  - DST Type 2\n");
	printf("  - DCT Type 3\n");
	printf("  - DST Type 3\n");
	printf("  - DCT Type 4\n");
	printf("  - DST Type 4\n");
}

int main(int argc, char *argv[]) {
	int run_all = 0;

	/* Parse command line arguments */
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--all") == 0) {
			run_all = 1;
		} else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			print_usage(argv[0]);
			return 0;
		} else {
			printf("Unknown option: %s\n", argv[i]);
			print_usage(argv[0]);
			return 1;
		}
	}

	int sizes[] = {16, 64, 256, 1024, 4096, 16384, 65536};
	int nsizes = sizeof(sizes) / sizeof(sizes[0]);

	printf("========================================\n");
	printf("  NotoriousFFT vs minfft Benchmark\n");
	if (run_all) {
		printf("  Mode: Comprehensive (--all)\n");
	} else {
		printf("  Mode: Default\n");
	}
	printf("========================================\n");

	/* Always run default benchmarks */
	benchmark_dft(sizes, nsizes);
	benchmark_invdft(sizes, nsizes);
	benchmark_realdft(sizes, nsizes);
	benchmark_dct2(sizes, nsizes);

	/* Additional benchmarks for --all mode */
	if (run_all) {
		benchmark_invrealdft(sizes, nsizes);
		benchmark_dst2(sizes, nsizes);
		benchmark_dct3(sizes, nsizes);
		benchmark_dst3(sizes, nsizes);
		benchmark_dct4(sizes, nsizes);
		benchmark_dst4(sizes, nsizes);
	}

	printf("\n========================================\n");
	printf("Benchmark complete.\n");
	printf("========================================\n");

	return 0;
}
