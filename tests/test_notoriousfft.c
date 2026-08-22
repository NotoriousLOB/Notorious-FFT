/*
	Tests for NotoriousFFT - compare results against reference minfft implementation
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

/* include minfft reference */
#include "minfft.h"

/* include NotoriousFFT implementation */
#include "notorious_fft.h"

#define TOL 1e-10
#define MAX_FAIL_PRINT 5

static int tests_passed = 0;
static int tests_failed = 0;

static double max_err(double *a, double *b, int n) {
	double m = 0;
	for (int i = 0; i < n; i++) {
		double d = fabs(a[i] - b[i]);
		if (d > m) m = d;
	}
	return m;
}

static void check(const char *name, double *ref, double *got, int n) {
	double e = max_err(ref, got, n);
	if (e < TOL) {
		printf("  PASS: %-30s (max_err=%.2e)\n", name, e);
		tests_passed++;
	} else {
		printf("  FAIL: %-30s (max_err=%.2e)\n", name, e);
		tests_failed++;
	}
}

/* fill array with deterministic pseudo-random values */
static void fill(double *x, int n) {
	for (int i = 0; i < n; i++)
		x[i] = sin(i * 1.23456789 + 0.1 * i * i);
}

/* ---- complex DFT tests ---- */

static void test_dft_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dft_1d N=%d", N);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_1d(N);

	minfft_dft(mx, my, ma);
	notorious_fft_dft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_invdft_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "invdft_1d N=%d", N);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_1d(N);

	minfft_invdft(mx, my, ma);
	notorious_fft_invdft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

/* ---- real DFT tests ---- */

static void test_realdft_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "realdft_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_cmpl *my = malloc((N / 2 + 1) * sizeof(minfft_cmpl));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_cmpl *ly = malloc((N / 2 + 1) * sizeof(notorious_fft_cmpl));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_realdft_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_realdft_1d(N);

	minfft_realdft(mx, my, ma);
	notorious_fft_realdft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * (N / 2 + 1));

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_invrealdft_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "invrealdft_1d N=%d", N);

	/* first do forward real DFT to get valid input */
	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_cmpl *mz = malloc((N / 2 + 1) * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lz = malloc((N / 2 + 1) * sizeof(notorious_fft_cmpl));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	minfft_aux *ma_r = minfft_mkaux_realdft_1d(N);
	minfft_realdft(mx, mz, ma_r);
	memcpy(lz, mz, (N / 2 + 1) * sizeof(minfft_cmpl));

	minfft_invrealdft(mz, my, ma_r);

	notorious_fft_aux *la_r = notorious_fft_mkaux_realdft_1d(N);
	notorious_fft_invrealdft(lz, ly, la_r);

	check(name, my, ly, N);

	minfft_free_aux(ma_r);
	notorious_fft_free_aux(la_r);
	free(mx); free(mz); free(lz); free(my); free(ly);
}

/* ---- DCT/DST tests ---- */

static void test_dct2_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dct2_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_1d(N);

	minfft_dct2(mx, my, ma);
	notorious_fft_dct2(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dst2_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dst2_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_1d(N);

	minfft_dst2(mx, my, ma);
	notorious_fft_dst2(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dct3_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dct3_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_1d(N);

	minfft_dct3(mx, my, ma);
	notorious_fft_dct3(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dst3_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dst3_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_1d(N);

	minfft_dst3(mx, my, ma);
	notorious_fft_dst3(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dct4_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dct4_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t4_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t4_1d(N);

	minfft_dct4(mx, my, ma);
	notorious_fft_dct4(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dst4_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dst4_1d N=%d", N);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t4_1d(N);
	notorious_fft_aux *la = notorious_fft_mkaux_t4_1d(N);

	minfft_dst4(mx, my, ma);
	notorious_fft_dst4(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

/* ---- 2D tests ---- */

static void test_dft_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dft_2d %dx%d", N1, N2);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_2d(N1, N2);

	minfft_dft(mx, my, ma);
	notorious_fft_dft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dct2_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dct2_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_2d(N1, N2);

	minfft_dct2(mx, my, ma);
	notorious_fft_dct2(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

/* ---- additional 2D tests ---- */

static void test_invdft_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "invdft_2d %dx%d", N1, N2);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_2d(N1, N2);

	minfft_invdft(mx, my, ma);
	notorious_fft_invdft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_realdft_2d(int N1, int N2) {
	char name[64];
	int nc = N2 / 2 + 1;
	int nout = N1 * nc;
	snprintf(name, sizeof(name), "realdft_2d %dx%d", N1, N2);

	minfft_real *mx = malloc((size_t)(N1 * N2) * sizeof(minfft_real));
	minfft_cmpl *my = malloc((size_t)nout * sizeof(minfft_cmpl));
	notorious_fft_real *lx = malloc((size_t)(N1 * N2) * sizeof(notorious_fft_real));
	notorious_fft_cmpl *ly = malloc((size_t)nout * sizeof(notorious_fft_cmpl));

	fill(mx, N1 * N2);
	memcpy(lx, mx, (size_t)(N1 * N2) * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_realdft_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_realdft_2d(N1, N2);

	minfft_realdft(mx, my, ma);
	notorious_fft_realdft(lx, ly, la);

	check(name, (double *)my, (double *)ly, 2 * nout);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_invrealdft_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "invrealdft_2d %dx%d", N1, N2);

	minfft_real *mx = malloc((size_t)N * sizeof(minfft_real));
	minfft_cmpl *mz = malloc((size_t)(N1 * (N2 / 2 + 1)) * sizeof(minfft_cmpl));
	minfft_real *my = malloc((size_t)N * sizeof(minfft_real));
	notorious_fft_cmpl *lz = malloc((size_t)(N1 * (N2 / 2 + 1)) * sizeof(notorious_fft_cmpl));
	notorious_fft_real *ly = malloc((size_t)N * sizeof(notorious_fft_real));

	fill(mx, N);
	minfft_aux *ma = minfft_mkaux_realdft_2d(N1, N2);
	minfft_realdft(mx, mz, ma);
	memcpy(lz, mz, (size_t)(N1 * (N2 / 2 + 1)) * sizeof(minfft_cmpl));
	minfft_invrealdft(mz, my, ma);

	notorious_fft_aux *la = notorious_fft_mkaux_realdft_2d(N1, N2);
	notorious_fft_invrealdft(lz, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(mz); free(my); free(lz); free(ly);
}

static void test_dst2_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dst2_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_2d(N1, N2);

	minfft_dst2(mx, my, ma);
	notorious_fft_dst2(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dct3_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dct3_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_2d(N1, N2);

	minfft_dct3(mx, my, ma);
	notorious_fft_dct3(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dst3_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dst3_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_2d(N1, N2);

	minfft_dst3(mx, my, ma);
	notorious_fft_dst3(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dct4_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dct4_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t4_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t4_2d(N1, N2);

	minfft_dct4(mx, my, ma);
	notorious_fft_dct4(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_dst4_2d(int N1, int N2) {
	char name[64];
	int N = N1 * N2;
	snprintf(name, sizeof(name), "dst4_2d %dx%d", N1, N2);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t4_2d(N1, N2);
	notorious_fft_aux *la = notorious_fft_mkaux_t4_2d(N1, N2);

	minfft_dst4(mx, my, ma);
	notorious_fft_dst4(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

/* ---- 3D tests ---- */

static void test_dft_3d(int N1, int N2, int N3) {
	char name[64];
	int N = N1 * N2 * N3;
	snprintf(name, sizeof(name), "dft_3d %dx%dx%d", N1, N2, N3);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_3d(N1, N2, N3);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_3d(N1, N2, N3);

	minfft_dft(mx, my, ma);
	notorious_fft_dft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

static void test_invdft_3d(int N1, int N2, int N3) {
	char name[64];
	int N = N1 * N2 * N3;
	snprintf(name, sizeof(name), "invdft_3d %dx%dx%d", N1, N2, N3);

	minfft_cmpl *mx = malloc(N * sizeof(minfft_cmpl));
	minfft_cmpl *my = malloc(N * sizeof(minfft_cmpl));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill((double*)mx, 2 * N);
	memcpy(lx, mx, N * sizeof(minfft_cmpl));

	minfft_aux *ma = minfft_mkaux_dft_3d(N1, N2, N3);
	notorious_fft_aux *la = notorious_fft_mkaux_dft_3d(N1, N2, N3);

	minfft_invdft(mx, my, ma);
	notorious_fft_invdft(lx, ly, la);

	check(name, (double*)my, (double*)ly, 2 * N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}


static void test_dct2_3d(int N1, int N2, int N3) {
	char name[64];
	int N = N1 * N2 * N3;
	snprintf(name, sizeof(name), "dct2_3d %dx%dx%d", N1, N2, N3);

	minfft_real *mx = malloc(N * sizeof(minfft_real));
	minfft_real *my = malloc(N * sizeof(minfft_real));
	notorious_fft_real *lx = malloc(N * sizeof(notorious_fft_real));
	notorious_fft_real *ly = malloc(N * sizeof(notorious_fft_real));

	fill(mx, N);
	memcpy(lx, mx, N * sizeof(minfft_real));

	minfft_aux *ma = minfft_mkaux_t2t3_3d(N1, N2, N3);
	notorious_fft_aux *la = notorious_fft_mkaux_t2t3_3d(N1, N2, N3);

	minfft_dct2(mx, my, ma);
	notorious_fft_dct2(lx, ly, la);

	check(name, my, ly, N);

	minfft_free_aux(ma);
	notorious_fft_free_aux(la);
	free(mx); free(my); free(lx); free(ly);
}

/* ---- Bluestein / non-power-of-2 tests ---- */

/* Reference O(N²) DFT for validation */
static void ref_dft(const double* xr, const double* xi, double* yr, double* yi, int N) {
	for (int k = 0; k < N; k++) {
		double sr = 0, si = 0;
		for (int n = 0; n < N; n++) {
			double angle = -2.0 * M_PI * k * n / N;
			double c = cos(angle), s = sin(angle);
			sr += xr[n] * c - xi[n] * s;
			si += xr[n] * s + xi[n] * c;
		}
		yr[k] = sr;
		yi[k] = si;
	}
}

static void test_dft_bluestein(int N) {
	char name[64];
	snprintf(name, sizeof(name), "dft_bluestein N=%d", N);

	double *xr = malloc(N * sizeof(double));
	double *xi = malloc(N * sizeof(double));
	double *ref_yr = malloc(N * sizeof(double));
	double *ref_yi = malloc(N * sizeof(double));
	notorious_fft_cmpl *lx = malloc(N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc(N * sizeof(notorious_fft_cmpl));

	fill(xr, N);
	fill(xi, N);
	/* Shift xi values to avoid overlap with xr pattern */
	for (int i = 0; i < N; i++) xi[i] = sin(i * 2.71828 + 0.5);

	/* Interleave for NotoriousFFT */
	for (int i = 0; i < N; i++) {
		((double*)lx)[2*i]   = xr[i];
		((double*)lx)[2*i+1] = xi[i];
	}

	ref_dft(xr, xi, ref_yr, ref_yi, N);

	notorious_fft_aux *la = notorious_fft_mkaux_dft_1d(N);
	notorious_fft_dft(lx, ly, la);

	/* Interleave reference result for comparison */
	double *ref_interleaved = malloc(2 * N * sizeof(double));
	for (int i = 0; i < N; i++) {
		ref_interleaved[2*i]   = ref_yr[i];
		ref_interleaved[2*i+1] = ref_yi[i];
	}

	/* Use higher tolerance for Bluestein (multi-step algorithm) */
	double e = max_err(ref_interleaved, (double*)ly, 2 * N);
	if (e < 1e-9) {
		printf("  PASS: %-30s (max_err=%.2e)\n", name, e);
		tests_passed++;
	} else {
		printf("  FAIL: %-30s (max_err=%.2e)\n", name, e);
		tests_failed++;
	}

	notorious_fft_free_aux(la);
	free(xr); free(xi); free(ref_yr); free(ref_yi);
	free(lx); free(ly); free(ref_interleaved);
}

static void ref_idft(const double* xr, const double* xi, double* yr, double* yi, int N) {
	/* Unnormalized inverse: +2π kn/N, no 1/N */
	for (int k = 0; k < N; k++) {
		double sr = 0, si = 0;
		for (int n = 0; n < N; n++) {
			double angle = 2.0 * M_PI * k * n / N;
			double c = cos(angle), s = sin(angle);
			sr += xr[n] * c - xi[n] * s;
			si += xr[n] * s + xi[n] * c;
		}
		yr[k] = sr;
		yi[k] = si;
	}
}

static void test_invdft_bluestein(int N) {
	char name[64];
	snprintf(name, sizeof(name), "invdft_bluestein N=%d", N);

	double *xr = malloc((size_t)N * sizeof(double));
	double *xi = malloc((size_t)N * sizeof(double));
	double *ref_yr = malloc((size_t)N * sizeof(double));
	double *ref_yi = malloc((size_t)N * sizeof(double));
	notorious_fft_cmpl *lx = malloc((size_t)N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *ly = malloc((size_t)N * sizeof(notorious_fft_cmpl));

	fill(xr, N);
	for (int i = 0; i < N; i++) xi[i] = sin(i * 2.71828 + 0.5);
	for (int i = 0; i < N; i++) {
		((double*)lx)[2*i]   = xr[i];
		((double*)lx)[2*i+1] = xi[i];
	}

	ref_idft(xr, xi, ref_yr, ref_yi, N);

	notorious_fft_aux *la = notorious_fft_mkaux_dft_1d(N);
	notorious_fft_invdft(lx, ly, la);

	double *ref_interleaved = malloc(2 * (size_t)N * sizeof(double));
	for (int i = 0; i < N; i++) {
		ref_interleaved[2*i]   = ref_yr[i];
		ref_interleaved[2*i+1] = ref_yi[i];
	}

	double e = max_err(ref_interleaved, (double*)ly, 2 * N);
	if (e < 1e-9) {
		printf("  PASS: %-30s (max_err=%.2e)\n", name, e);
		tests_passed++;
	} else {
		printf("  FAIL: %-30s (max_err=%.2e)\n", name, e);
		tests_failed++;
	}

	notorious_fft_free_aux(la);
	free(xr); free(xi); free(ref_yr); free(ref_yi);
	free(lx); free(ly); free(ref_interleaved);
}

static void test_roundtrip_1d(int N) {
	char name[64];
	snprintf(name, sizeof(name), "roundtrip_dft N=%d", N);

	notorious_fft_cmpl *x = malloc((size_t)N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *y = malloc((size_t)N * sizeof(notorious_fft_cmpl));
	notorious_fft_cmpl *z = malloc((size_t)N * sizeof(notorious_fft_cmpl));
	fill((double*)x, 2 * N);

	notorious_fft_aux *a = notorious_fft_mkaux_dft_1d(N);
	notorious_fft_dft(x, y, a);
	notorious_fft_invdft(y, z, a);

	double e = 0;
	for (int i = 0; i < N; i++) {
		double dr = ((double*)z)[2*i]     - ((double*)x)[2*i]     * N;
		double di = ((double*)z)[2*i + 1] - ((double*)x)[2*i + 1] * N;
		double d = fabs(dr);
		if (fabs(di) > d) d = fabs(di);
		if (d > e) e = d;
	}
	double tol = (N >= 64) ? 1e-8 : 1e-9;
	if (e < tol) {
		printf("  PASS: %-30s (max_err=%.2e)\n", name, e);
		tests_passed++;
	} else {
		printf("  FAIL: %-30s (max_err=%.2e)\n", name, e);
		tests_failed++;
	}
	notorious_fft_free_aux(a);
	free(x); free(y); free(z);
}

static void test_null_aux(void) {
	notorious_fft_cmpl x[4], y[4];
	memset(x, 0, sizeof x);
	memset(y, 0, sizeof y);
	notorious_fft_dft(x, y, NULL);
	notorious_fft_invdft(x, y, NULL);
	printf("  PASS: %-30s\n", "null_aux_noop");
	tests_passed++;
}

int main(void) {
	int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
	int nsizes = sizeof(sizes) / sizeof(sizes[0]);

	printf("=== Complex DFT ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dft_1d(sizes[i]);

	printf("\n=== Inverse Complex DFT ===\n");
	for (int i = 0; i < nsizes; i++)
		test_invdft_1d(sizes[i]);

	printf("\n=== Real DFT ===\n");
	for (int i = 0; i < nsizes; i++)
		test_realdft_1d(sizes[i]);

	printf("\n=== Inverse Real DFT ===\n");
	for (int i = 1; i < nsizes; i++) /* skip N=1 */
		test_invrealdft_1d(sizes[i]);

	printf("\n=== DCT-2 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dct2_1d(sizes[i]);

	printf("\n=== DST-2 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dst2_1d(sizes[i]);

	printf("\n=== DCT-3 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dct3_1d(sizes[i]);

	printf("\n=== DST-3 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dst3_1d(sizes[i]);

	printf("\n=== DCT-4 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dct4_1d(sizes[i]);

	printf("\n=== DST-4 ===\n");
	for (int i = 0; i < nsizes; i++)
		test_dst4_1d(sizes[i]);

	printf("\n=== 2D Complex DFT ===\n");
	test_dft_2d(4, 4);
	test_dft_2d(8, 16);
	test_dft_2d(32, 32);

	printf("\n=== 2D Real DFT ===\n");
	test_realdft_2d(4, 4);
	test_realdft_2d(8, 16);
	test_realdft_2d(32, 32);

	printf("\n=== 2D Inverse Real DFT ===\n");
	test_invrealdft_2d(4, 4);
	test_invrealdft_2d(8, 16);
	test_invrealdft_2d(32, 32);

	printf("\n=== 2D Inverse Complex DFT ===\n");
	test_invdft_2d(4, 4);
	test_invdft_2d(8, 16);
	test_invdft_2d(32, 32);

	printf("\n=== 2D DCT-2 ===\n");
	test_dct2_2d(4, 4);
	test_dct2_2d(8, 16);
	test_dct2_2d(32, 32);

	printf("\n=== 2D DST-2 ===\n");
	test_dst2_2d(4, 4);
	test_dst2_2d(8, 16);
	test_dst2_2d(32, 32);

	printf("\n=== 2D DCT-3 ===\n");
	test_dct3_2d(4, 4);
	test_dct3_2d(8, 16);
	test_dct3_2d(32, 32);

	printf("\n=== 2D DST-3 ===\n");
	test_dst3_2d(4, 4);
	test_dst3_2d(8, 16);
	test_dst3_2d(32, 32);

	printf("\n=== 2D DCT-4 ===\n");
	test_dct4_2d(4, 4);
	test_dct4_2d(8, 16);
	test_dct4_2d(32, 32);

	printf("\n=== 2D DST-4 ===\n");
	test_dst4_2d(4, 4);
	test_dst4_2d(8, 16);
	test_dst4_2d(32, 32);

	printf("\n=== 3D Complex DFT ===\n");
	test_dft_3d(4, 4, 4);
	test_dft_3d(4, 8, 8);

	printf("\n=== 3D Inverse Complex DFT ===\n");
	test_invdft_3d(4, 4, 4);
	test_invdft_3d(4, 8, 8);

	printf("\n=== Large power-of-2 ===\n");
	test_dft_1d(65536);
	test_invdft_1d(65536);

	printf("\n=== 3D DCT-2 ===\n");
	test_dct2_3d(4, 4, 4);
	test_dct2_3d(4, 8, 8);

	printf("\n=== Non-Power-of-2 / Bluestein DFT ===\n");
	{
		int bluestein_sizes[] = {3, 5, 6, 7, 9, 10, 12, 14, 15, 18, 20, 21, 25, 27, 35, 48, 49, 100};
		int nb = sizeof(bluestein_sizes) / sizeof(bluestein_sizes[0]);
		for (int i = 0; i < nb; i++)
			test_dft_bluestein(bluestein_sizes[i]);
	}

	printf("\n=== Inverse Bluestein (unnormalized) ===\n");
	{
		int bluestein_sizes[] = {3, 5, 6, 7, 9, 10, 12, 14, 15, 18, 20, 21, 25, 27, 35, 48, 49, 100};
		int nb = sizeof(bluestein_sizes) / sizeof(bluestein_sizes[0]);
		for (int i = 0; i < nb; i++)
			test_invdft_bluestein(bluestein_sizes[i]);
	}

	printf("\n=== Prime / Rader (incl. Mersenne 31,127 and Fermat 17,257) ===\n");
	{
		/* 11,13,17,19,31,37,61,127,257: N−1 is 2·3·5·7-smooth → Rader.
		 * 23,47,53,67: N−1 not smooth → Bluestein (covers AVX inverse IFFT).
		 * 255=3·5·17 mixed+Rader. */
		int prime_sizes[] = {11, 13, 17, 19, 23, 31, 37, 47, 53, 61, 67, 127, 255, 257};
		int np = sizeof(prime_sizes) / sizeof(prime_sizes[0]);
		for (int i = 0; i < np; i++)
			test_dft_bluestein(prime_sizes[i]);
		for (int i = 0; i < np; i++)
			test_invdft_bluestein(prime_sizes[i]);
	}

	printf("\n=== Mersenne identification (bit test, not a padded 2^p DFT) ===\n");
	{
		int ok = notorious_fft_is_mersenne_number(31)
		      && notorious_fft_is_mersenne_number(127)
		      && notorious_fft_is_mersenne_number(7)
		      && !notorious_fft_is_mersenne_number(32)
		      && !notorious_fft_is_mersenne_number(30)
		      && notorious_fft_is_mersenne_number(15) /* 2^4-1, composite */
		      && notorious_fft_is_prime(31)
		      && !notorious_fft_is_prime(15);
		if (ok) {
			printf("  PASS: %-30s\n", "mersenne_bits");
			tests_passed++;
		} else {
			printf("  FAIL: %-30s\n", "mersenne_bits");
			tests_failed++;
		}
	}

	printf("\n=== Round-trip DFT -> IDFT / N ===\n");
	{
		int rt[] = {8, 16, 64, 7, 12, 17, 31, 100, 127};
		int nrt = sizeof(rt) / sizeof(rt[0]);
		for (int i = 0; i < nrt; i++)
			test_roundtrip_1d(rt[i]);
	}

	printf("\n=== Error policy ===\n");
	test_null_aux();
	if (notorious_fft_mkaux_dft_1d(0) == NULL &&
	    notorious_fft_mkaux_dft_1d(-1) == NULL) {
		printf("  PASS: %-30s\n", "mkaux_dft_1d rejects N<=0");
		tests_passed++;
	} else {
		printf("  FAIL: %-30s\n", "mkaux_dft_1d rejects N<=0");
		tests_failed++;
	}

	printf("\n=== FFTW-shaped planner API ===\n");
	{
		int N = 64;
		notorious_fft_cmpl *in = (notorious_fft_cmpl *)notorious_fft_malloc((size_t)N * sizeof(notorious_fft_cmpl));
		notorious_fft_cmpl *out = (notorious_fft_cmpl *)notorious_fft_malloc((size_t)N * sizeof(notorious_fft_cmpl));
		notorious_fft_cmpl *ref = (notorious_fft_cmpl *)malloc((size_t)N * sizeof(notorious_fft_cmpl));
		fill((double *)in, 2 * N);
		memcpy(ref, in, (size_t)N * sizeof(notorious_fft_cmpl));

		notorious_fft_io_plan *p = notorious_fft_plan_dft_1d(N, in, out, NOTORIOUS_FFT_FORWARD, NOTORIOUS_FFT_ESTIMATE);
		notorious_fft_execute(p);

		notorious_fft_aux *a = notorious_fft_mkaux_dft_1d(N);
		notorious_fft_dft(ref, ref, a);

		check("plan_dft_1d execute", (double *)ref, (double *)out, 2 * N);

		notorious_fft_execute_dft(p, in, out); /* new-array, same buffers */
		check("execute_dft new-array", (double *)ref, (double *)out, 2 * N);

		notorious_fft_destroy_io_plan(p);
		notorious_fft_free_aux(a);
		notorious_fft_free(in);
		notorious_fft_free(out);
		free(ref);
	}

	printf("\n========================================\n");
	printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

	return tests_failed > 0 ? 1 : 0;
}
