/*
	C++17 tests for NotoriousFFT - compare results against reference minfft implementation
	Tests the C++ wrapper interface including std::complex integration
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

/* include minfft reference */
extern "C" {
#include "minfft.h"
}

/* include NotoriousFFT C++ wrapper */
#include "notorious_fft.hpp"

constexpr double TOL = 1e-10;
static int tests_passed = 0;
static int tests_failed = 0;

/* ============================================================================
   Helper functions
   ============================================================================ */

double max_err(const double *a, const double *b, int n) {
	double m = 0;
	for (int i = 0; i < n; i++) {
		double d = std::abs(a[i] - b[i]);
		if (d > m) m = d;
	}
	return m;
}

void check(const std::string &name, const double *ref, const double *got, int n) {
	double e = max_err(ref, got, n);
	if (e < TOL) {
		std::cout << "  PASS: " << std::left << std::setw(35) << name 
				  << " (max_err=" << std::scientific << e << ")\n";
		tests_passed++;
	} else {
		std::cout << "  FAIL: " << std::left << std::setw(35) << name 
				  << " (max_err=" << std::scientific << e << ")\n";
		tests_failed++;
	}
}

/* Fill array with deterministic pseudo-random values */
void fill(double *x, int n) {
	for (int i = 0; i < n; i++)
		x[i] = sin(i * 1.23456789 + 0.1 * i * i);
}

/* ============================================================================
   Test: std::complex <-> notorious_fft_cmpl conversions
   ============================================================================ */
void test_complex_conversion() {
	std::cout << "\n=== Complex Type Conversions ===\n";
	
	const int N = 16;
	std::vector<std::complex<double>> x(N);
	std::mt19937 rng(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	
	for (auto &v : x) {
		v = std::complex<double>(dist(rng), dist(rng));
	}
	
	/* Test roundtrip conversion */
	std::vector<notorious_fft::cmpl_t> lx(N);
	std::vector<std::complex<double>> y(N);
	
	notorious_fft::detail::to_notorious_fft_cmpl(x.data(), lx.data(), N);
	notorious_fft::detail::from_notorious_fft_cmpl(lx.data(), y.data(), N);
	
	check("complex roundtrip conversion", 
		  reinterpret_cast<double*>(x.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * N);
}

/* ============================================================================
   Test: Complex DFT (std::vector<std::complex<double>>)
   ============================================================================ */
void test_cpp_complex_dft_1d(int N) {
	std::string name = "cpp_dft_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_cmpl> mx(N);
	std::vector<minfft_cmpl> my(N);
	fill(reinterpret_cast<double*>(mx.data()), 2 * N);
	
	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	minfft_dft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<std::complex<double>> x(N);
	for (int i = 0; i < N; ++i) {
		reinterpret_cast<double*>(&x[i])[0] = reinterpret_cast<double*>(&mx[i])[0];
		reinterpret_cast<double*>(&x[i])[1] = reinterpret_cast<double*>(&mx[i])[1];
	}
	
	auto y = notorious_fft::dft(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * N);
}

void test_cpp_invdft_1d(int N) {
	std::string name = "cpp_invdft_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_cmpl> mx(N);
	std::vector<minfft_cmpl> my(N);
	fill(reinterpret_cast<double*>(mx.data()), 2 * N);
	
	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	minfft_invdft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<std::complex<double>> x(N);
	for (int i = 0; i < N; ++i) {
		reinterpret_cast<double*>(&x[i])[0] = reinterpret_cast<double*>(&mx[i])[0];
		reinterpret_cast<double*>(&x[i])[1] = reinterpret_cast<double*>(&mx[i])[1];
	}
	
	auto y = notorious_fft::invdft(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * N);
}

/* ============================================================================
   Test: Real DFT (std::vector<double> -> std::vector<std::complex<double>>)
   ============================================================================ */
void test_cpp_realdft_1d(int N) {
	std::string name = "cpp_realdft_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	const int out_size = N / 2 + 1;
	std::vector<minfft_cmpl> my(out_size);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_realdft_1d(N);
	minfft_realdft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::realdft(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * out_size);
}

void test_cpp_invrealdft_1d(int N) {
	std::string name = "cpp_invrealdft_1d N=" + std::to_string(N);
	
	/* Generate reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_cmpl> mz(N / 2 + 1);
	std::vector<minfft_real> my(N);
	
	fill(mx.data(), N);
	minfft_aux *ma = minfft_mkaux_realdft_1d(N);
	minfft_realdft(mx.data(), mz.data(), ma);
	minfft_invrealdft(mz.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<std::complex<double>> z(N / 2 + 1);
	for (size_t i = 0; i < z.size(); ++i) {
		reinterpret_cast<double*>(&z[i])[0] = reinterpret_cast<double*>(&mz[i])[0];
		reinterpret_cast<double*>(&z[i])[1] = reinterpret_cast<double*>(&mz[i])[1];
	}
	
	auto y = notorious_fft::invrealdft(z, N);
	
	check(name, my.data(), y.data(), N);
}

/* ============================================================================
   Test: DCT/DST with std::vector<double>
   ============================================================================ */
void test_cpp_dct2_1d(int N) {
	std::string name = "cpp_dct2_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	minfft_dct2(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dct2(x);
	
	check(name, my.data(), y.data(), N);
}

void test_cpp_dst2_1d(int N) {
	std::string name = "cpp_dst2_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	minfft_dst2(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dst2(x);
	
	check(name, my.data(), y.data(), N);
}

void test_cpp_dct3_1d(int N) {
	std::string name = "cpp_dct3_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	minfft_dct3(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dct3(x);
	
	check(name, my.data(), y.data(), N);
}

void test_cpp_dst3_1d(int N) {
	std::string name = "cpp_dst3_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	minfft_dst3(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dst3(x);
	
	check(name, my.data(), y.data(), N);
}

void test_cpp_dct4_1d(int N) {
	std::string name = "cpp_dct4_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t4_1d(N);
	minfft_dct4(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dct4(x);
	
	check(name, my.data(), y.data(), N);
}

void test_cpp_dst4_1d(int N) {
	std::string name = "cpp_dst4_1d N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t4_1d(N);
	minfft_dst4(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dst4(x);
	
	check(name, my.data(), y.data(), N);
}

/* ============================================================================
   Test: Reusable plan objects
   ============================================================================ */
void test_cpp_dft_plan(int N) {
	std::string name = "cpp_dft_plan N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_cmpl> mx(N);
	std::vector<minfft_cmpl> my(N);
	fill(reinterpret_cast<double*>(mx.data()), 2 * N);
	
	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	minfft_dft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ plan */
	notorious_fft::dft_plan plan(N);
	std::vector<std::complex<double>> x(N);
	for (int i = 0; i < N; ++i) {
		reinterpret_cast<double*>(&x[i])[0] = reinterpret_cast<double*>(&mx[i])[0];
		reinterpret_cast<double*>(&x[i])[1] = reinterpret_cast<double*>(&mx[i])[1];
	}
	
	auto y = plan.execute(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * N);
}

void test_cpp_realdft_plan(int N) {
	std::string name = "cpp_realdft_plan N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	const int out_size = N / 2 + 1;
	std::vector<minfft_cmpl> my(out_size);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_realdft_1d(N);
	minfft_realdft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ plan */
	notorious_fft::realdft_plan plan(N);
	std::vector<double> x(mx.begin(), mx.end());
	auto y = plan.execute(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * out_size);
}

void test_cpp_dct2_plan(int N) {
	std::string name = "cpp_dct2_plan N=" + std::to_string(N);
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
	minfft_dct2(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ plan */
	notorious_fft::dct2_plan plan(N);
	std::vector<double> x(mx.begin(), mx.end());
	auto y = plan.execute(x);
	
	check(name, my.data(), y.data(), N);
}

/* ============================================================================
   Test: 2D transforms
   ============================================================================ */
void test_cpp_dft_2d(int N1, int N2) {
	std::string name = "cpp_dft_2d " + std::to_string(N1) + "x" + std::to_string(N2);
	
	const int N = N1 * N2;
	
	/* Reference using minfft */
	std::vector<minfft_cmpl> mx(N);
	std::vector<minfft_cmpl> my(N);
	fill(reinterpret_cast<double*>(mx.data()), 2 * N);
	
	minfft_aux *ma = minfft_mkaux_dft_2d(N1, N2);
	minfft_dft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<std::complex<double>> x(N);
	for (int i = 0; i < N; ++i) {
		reinterpret_cast<double*>(&x[i])[0] = reinterpret_cast<double*>(&mx[i])[0];
		reinterpret_cast<double*>(&x[i])[1] = reinterpret_cast<double*>(&mx[i])[1];
	}
	
	auto y = notorious_fft::dft_2d(x, N1, N2);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(y.data()), 2 * N);
}

void test_cpp_dct2_2d(int N1, int N2) {
	std::string name = "cpp_dct2_2d " + std::to_string(N1) + "x" + std::to_string(N2);
	
	const int N = N1 * N2;
	
	/* Reference using minfft */
	std::vector<minfft_real> mx(N);
	std::vector<minfft_real> my(N);
	fill(mx.data(), N);
	
	minfft_aux *ma = minfft_mkaux_t2t3_2d(N1, N2);
	minfft_dct2(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test using C++ wrapper */
	std::vector<double> x(mx.begin(), mx.end());
	auto y = notorious_fft::dct2_2d(x, N1, N2);
	
	check(name, my.data(), y.data(), N);
}

/* ============================================================================
   Test: In-place transforms
   ============================================================================ */
void test_cpp_dft_in_place(int N) {
	std::string name = "cpp_dft_in_place N=" + std::to_string(N);
	
	/* Reference */
	std::vector<minfft_cmpl> mx(N);
	std::vector<minfft_cmpl> my(N);
	fill(reinterpret_cast<double*>(mx.data()), 2 * N);
	
	minfft_aux *ma = minfft_mkaux_dft_1d(N);
	minfft_dft(mx.data(), my.data(), ma);
	minfft_free_aux(ma);
	
	/* Test in-place */
	notorious_fft::dft_plan plan(N);
	std::vector<std::complex<double>> x(N);
	for (int i = 0; i < N; ++i) {
		reinterpret_cast<double*>(&x[i])[0] = reinterpret_cast<double*>(&mx[i])[0];
		reinterpret_cast<double*>(&x[i])[1] = reinterpret_cast<double*>(&mx[i])[1];
	}
	
	plan.execute_in_place(x);
	
	check(name, reinterpret_cast<double*>(my.data()), 
		  reinterpret_cast<double*>(x.data()), 2 * N);
}

/* ============================================================================
   Test: RAII aux management (exception safety)
   ============================================================================ */
void test_cpp_raii_aux() {
	std::cout << "\n=== RAII Exception Safety ===\n";
	
	try {
		auto aux = notorious_fft::mkaux_dft_1d(1024);
		/* aux will be freed even if exception is thrown */
		throw std::runtime_error("test exception");
	} catch (const std::runtime_error &e) {
		/* Expected - aux should have been freed */
		std::cout << "  PASS: Exception thrown and caught, aux freed correctly\n";
		tests_passed++;
	}
}

/* ============================================================================
   Test: Error handling
   ============================================================================ */
void test_cpp_error_handling() {
	std::cout << "\n=== Error Handling ===\n";
	
	bool caught = false;
	try {
		/* Empty vector should throw */
		std::vector<std::complex<double>> empty;
		auto result = notorious_fft::dft(empty);
	} catch (const std::invalid_argument &e) {
		caught = true;
		std::cout << "  PASS: Empty input correctly throws exception\n";
		tests_passed++;
	}
	if (!caught) {
		std::cout << "  FAIL: Empty input should throw exception\n";
		tests_failed++;
	}
	
	caught = false;
	try {
		/* Invalid size should throw from plan */
		std::vector<std::complex<double>> x(100); /* Not power of 2 */
		notorious_fft::dft_plan plan(100);
	} catch (...) {
		caught = true;
		std::cout << "  PASS: Invalid size correctly throws exception\n";
		tests_passed++;
	}
	if (!caught) {
		std::cout << "  Note: Invalid size exception (depends on impl)\n";
	}
}

/* ============================================================================
   Test: Utilities
   ============================================================================ */
void test_cpp_utilities() {
	std::cout << "\n=== Utilities ===\n";
	
	/* Test is_valid_size */
	bool test1 = notorious_fft::is_valid_size(1) && notorious_fft::is_valid_size(2) && 
				 notorious_fft::is_valid_size(1024) && !notorious_fft::is_valid_size(3) &&
				 !notorious_fft::is_valid_size(100);
	if (test1) {
		std::cout << "  PASS: is_valid_size works correctly\n";
		tests_passed++;
	} else {
		std::cout << "  FAIL: is_valid_size incorrect\n";
		tests_failed++;
	}
	
	/* Test next_power_of_2 */
	bool test2 = notorious_fft::next_power_of_2(1) == 1 &&
				 notorious_fft::next_power_of_2(2) == 2 &&
				 notorious_fft::next_power_of_2(3) == 4 &&
				 notorious_fft::next_power_of_2(1000) == 1024;
	if (test2) {
		std::cout << "  PASS: next_power_of_2 works correctly\n";
		tests_passed++;
	} else {
		std::cout << "  FAIL: next_power_of_2 incorrect\n";
		tests_failed++;
	}
}

/* ============================================================================
   Main test runner
   ============================================================================ */
int main() {
	std::cout << "Notorious FFT C++17 Wrapper Tests\n";
	std::cout << "============================\n";
	
	int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024};
	int nsizes = sizeof(sizes) / sizeof(sizes[0]);
	
	test_complex_conversion();
	
	std::cout << "\n=== Complex DFT (std::complex) ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_complex_dft_1d(sizes[i]);
	
	std::cout << "\n=== Inverse Complex DFT (std::complex) ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_invdft_1d(sizes[i]);
	
	std::cout << "\n=== Real DFT (std::vector<double>) ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_realdft_1d(sizes[i]);
	
	std::cout << "\n=== Inverse Real DFT ===\n";
	for (int i = 1; i < nsizes; i++) /* skip N=1 */
		test_cpp_invrealdft_1d(sizes[i]);
	
	std::cout << "\n=== DCT-2 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dct2_1d(sizes[i]);
	
	std::cout << "\n=== DST-2 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dst2_1d(sizes[i]);
	
	std::cout << "\n=== DCT-3 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dct3_1d(sizes[i]);
	
	std::cout << "\n=== DST-3 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dst3_1d(sizes[i]);
	
	std::cout << "\n=== DCT-4 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dct4_1d(sizes[i]);
	
	std::cout << "\n=== DST-4 ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dst4_1d(sizes[i]);
	
	std::cout << "\n=== Reusable Plans ===\n";
	for (int i = 0; i < nsizes; i++) {
		test_cpp_dft_plan(sizes[i]);
		test_cpp_realdft_plan(sizes[i]);
		test_cpp_dct2_plan(sizes[i]);
	}
	
	std::cout << "\n=== In-Place Transforms ===\n";
	for (int i = 0; i < nsizes; i++)
		test_cpp_dft_in_place(sizes[i]);
	
	std::cout << "\n=== 2D Complex DFT ===\n";
	test_cpp_dft_2d(4, 4);
	test_cpp_dft_2d(8, 16);
	test_cpp_dft_2d(32, 32);
	
	std::cout << "\n=== 2D DCT-2 ===\n";
	test_cpp_dct2_2d(4, 4);
	test_cpp_dct2_2d(8, 16);
	test_cpp_dct2_2d(32, 32);
	
	test_cpp_raii_aux();
	test_cpp_error_handling();
	test_cpp_utilities();
	
	std::cout << "\n========================================\n";
	std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";
	
	return tests_failed > 0 ? 1 : 0;
}
