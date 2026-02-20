/*
	Advanced C++17 usage example for NotoriousFFT
	
	Features demonstrated:
	- std::complex integration
	- RAII wrappers for aux data
	- Reusable transform plans
	- Vector-based high-level interface
	- Multi-dimensional transforms
	- Iterator-based interface
	- Template-based generic code
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>

#include "notorious_fft.hpp"

using namespace notorious_fft;
using namespace std::complex_literals;

/* ============================================================================
   Utility functions
   ============================================================================ */

template<typename T>
void print_vector(const std::vector<T> &v, const std::string &label, int max_print = 8) {
	std::cout << label << ": ";
	int n = std::min(static_cast<int>(v.size()), max_print);
	for (int i = 0; i < n; ++i) {
		std::cout << v[i];
		if (i < n - 1) std::cout << ", ";
	}
	if (v.size() > static_cast<size_t>(max_print))
		std::cout << "... (" << v.size() << " elements total)";
	std::cout << "\n";
}

/* Generate a cosine wave */
std::vector<double> generate_cosine(int N, int freq) {
	std::vector<double> x(N);
	for (int n = 0; n < N; ++n) {
		x[n] = std::cos(2.0 * M_PI * freq * n / N);
	}
	return x;
}

/* Generate complex exponential */
std::vector<std::complex<double>> generate_complex_exp(int N, int freq) {
	std::vector<std::complex<double>> x(N);
	for (int n = 0; n < N; ++n) {
		x[n] = std::exp(2i * M_PI * static_cast<double>(freq * n) / static_cast<double>(N));
	}
	return x;
}

/* ============================================================================
   Example 1: Basic complex DFT with std::complex
   ============================================================================ */
void example_complex_dft() {
	std::cout << "\n=== Example 1: Complex DFT with std::complex ===\n";
	
	const int N = 16;
	
	/* Generate a complex signal with two frequency components */
	auto x = generate_complex_exp(N, 1);
	for (int n = 0; n < N; ++n) {
		x[n] += 0.5 * std::exp(4i * M_PI * static_cast<double>(n) / static_cast<double>(N));
	}
	
	print_vector(x, "Input signal", 6);
	
	/* Compute DFT using high-level interface */
	auto X = dft(x);
	print_vector(X, "DFT result", 6);
	
	/* Inverse DFT to recover original */
	auto x_recovered = invdft(X);
	print_vector(x_recovered, "Recovered signal", 6);
	
	/* Check reconstruction error */
	double max_err = 0.0;
	for (int i = 0; i < N; ++i) {
		max_err = std::max(max_err, std::abs(x[i] - x_recovered[i]));
	}
	std::cout << "Reconstruction error: " << max_err << "\n";
}

/* ============================================================================
   Example 2: Real DFT - more efficient for real inputs
   ============================================================================ */
void example_real_dft() {
	std::cout << "\n=== Example 2: Real DFT (R2C) ===\n";
	
	const int N = 32;
	
	/* Generate real cosine signal */
	auto x = generate_cosine(N, 3);
	print_vector(x, "Real input", 8);
	
	/* Forward real DFT - output is N/2+1 complex values */
	auto X = realdft(x);
	print_vector(X, "DFT output (N/2+1 complex values)", 6);
	
	/* Inverse real DFT to recover signal */
	auto x_back = invrealdft(X, N);
	print_vector(x_back, "Recovered signal", 8);
	
	/* Verify reconstruction */
	double max_err = 0.0;
	for (int i = 0; i < N; ++i) {
		max_err = std::max(max_err, std::abs(x[i] - x_back[i]));
	}
	std::cout << "Reconstruction error: " << max_err << "\n";
}

/* ============================================================================
   Example 3: DCT/DST for real symmetric transforms
   ============================================================================ */
void example_dct_dst() {
	std::cout << "\n=== Example 3: DCT/DST transforms ===\n";
	
	const int N = 8;
	
	std::vector<double> x(N);
	for (int i = 0; i < N; ++i) x[i] = i + 1;
	
	print_vector(x, "Input");
	
	auto y_dct2 = dct2(x);
	print_vector(y_dct2, "DCT-2");
	
	auto y_dct3 = dct3(x);
	print_vector(y_dct3, "DCT-3");
	
	auto y_dst2 = dst2(x);
	print_vector(y_dst2, "DST-2");
	
	auto y_dst3 = dst3(x);
	print_vector(y_dst3, "DST-3");
	
	/* DCT-2 and DCT-3 are inverses (up to scaling) */
	auto x_recovered = dct3(y_dct2);
	/* Scale by 2/N */
	for (auto &v : x_recovered) v *= 2.0 / N;
	print_vector(x_recovered, "DCT-3(DCT-2(x)) scaled");
}

/* ============================================================================
   Example 4: Reusable transform plans (for repeated transforms)
   ============================================================================ */
void example_reusable_plans() {
	std::cout << "\n=== Example 4: Reusable transform plans ===\n";
	
	const int N = 1024;
	const int n_transforms = 1000;
	
	/* Create reusable plan - aux data is computed once */
	dft_plan plan(N);
	
	/* Generate random input */
	std::mt19937 rng(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	
	std::vector<std::complex<double>> x(N);
	for (auto &v : x) v = std::complex<double>(dist(rng), dist(rng));
	
	/* Warmup */
	volatile double sum = 0.0;
	auto result = plan.execute(x);
	
	/* Benchmark */
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n_transforms; ++i) {
		result = plan.execute(x);
		sum += std::abs(result[0]);
	}
	auto end = std::chrono::high_resolution_clock::now();
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Performed " << n_transforms << " DFTs of size " << N << "\n";
	std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
	std::cout << "Average time: " << duration.count() / static_cast<double>(n_transforms) << " us\n";
	std::cout << "Throughput: " << (5.0 * N * std::log2(N) * n_transforms) / duration.count() << " Mflop/s\n";
	(void)sum; /* prevent optimization */
}

/* ============================================================================
   Example 5: 2D transforms for image processing
   ============================================================================ */
void example_2d_transforms() {
	std::cout << "\n=== Example 5: 2D transforms ===\n";
	
	const int N1 = 8, N2 = 8;
	const int N = N1 * N2;
	
	/* Create 2D signal (e.g., image) */
	std::vector<std::complex<double>> img(N);
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N2; ++j) {
			/* Simple checkerboard pattern */
			double val = ((i + j) % 2 == 0) ? 1.0 : -1.0;
			img[i * N2 + j] = val;
		}
	}
	
	std::cout << "Input 8x8 pattern:\n";
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N2; ++j) {
			std::cout << std::setw(4) << static_cast<int>(img[i * N2 + j].real());
		}
		std::cout << "\n";
	}
	
	/* 2D DFT */
	auto dft_2d_result = dft_2d(img, N1, N2);
	
	std::cout << "\n2D DFT magnitude (8x8):\n";
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N2; ++j) {
			std::cout << std::setw(8) << std::fixed << std::setprecision(1) 
					  << std::abs(dft_2d_result[i * N2 + j]);
		}
		std::cout << "\n";
	}
}

/* ============================================================================
   Example 6: RAII aux management
   ============================================================================ */
void example_raii_aux() {
	std::cout << "\n=== Example 6: RAII aux management ===\n";
	
	{
		/* Aux data is automatically freed when aux_ptr goes out of scope */
		auto aux_dft = mkaux_dft_1d(256);
		auto aux_dct = mkaux_t2t3_1d(256);
		auto aux_real = mkaux_realdft_1d(256);
		
		std::cout << "Created 3 aux structures\n";
		std::cout << "They will be automatically freed at end of scope\n";
	} /* <-- automatic cleanup here */
	
	std::cout << "Aux data freed automatically\n";
}

/* ============================================================================
   Example 7: Generic programming with templates
   ============================================================================ */

template<typename Plan, typename Input>
void benchmark_transform(const std::string &name, int N, int n_iters) {
	Plan plan(N);
	
	Input x(N);
	std::mt19937 rng(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	
	for (auto &v : x) {
		if constexpr (std::is_same_v<typename Input::value_type, std::complex<double>>) {
			v = std::complex<double>(dist(rng), dist(rng));
		} else {
			v = dist(rng);
		}
	}
	
	/* Warmup */
	auto result = plan.execute(x);
	
	/* Benchmark */
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n_iters; ++i) {
		result = plan.execute(x);
	}
	auto end = std::chrono::high_resolution_clock::now();
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << name << ": " << duration.count() / static_cast<double>(n_iters) << " us/transform\n";
}

void example_generic() {
	std::cout << "\n=== Example 7: Generic programming ===\n";
	
	std::cout << "Benchmarking transforms (1000 iterations each):\n";
	benchmark_transform<dft_plan, std::vector<std::complex<double>>>(
		"Complex DFT", 1024, 1000);
	benchmark_transform<realdft_plan, std::vector<double>>(
		"Real DFT", 1024, 1000);
	benchmark_transform<dct2_plan, std::vector<double>>(
		"DCT-2", 1024, 1000);
}

/* ============================================================================
   Example 8: Multi-dimensional aux creation
   ============================================================================ */
void example_multidim_aux() {
	std::cout << "\n=== Example 8: Multi-dimensional aux creation ===\n";
	
	/* Using initializer lists for dimensions */
	auto aux_2d = mkaux_dft({16, 16});
	std::cout << "Created 2D DFT aux (16x16)\n";
	
	auto aux_3d = mkaux_dft({8, 8, 8});
	std::cout << "Created 3D DFT aux (8x8x8)\n";
	
	/* Real DFT in 2D */
	auto aux_real_2d = mkaux_realdft({32, 32});
	std::cout << "Created 2D real DFT aux (32x32)\n";
	
	/* DCT in 2D */
	auto aux_dct_2d = mkaux_t2t3({64, 64});
	std::cout << "Created 2D DCT aux (64x64)\n";
}

/* ============================================================================
   Main
   ============================================================================ */
int main() {
	std::cout << "Notorious FFT C++17 Advanced Examples\n";
	std::cout << "================================\n";
	
	try {
		example_complex_dft();
		example_real_dft();
		example_dct_dst();
		example_reusable_plans();
		example_2d_transforms();
		example_raii_aux();
		example_generic();
		example_multidim_aux();
		
		std::cout << "\n=== All examples completed successfully ===\n";
		
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
	
	return 0;
}
