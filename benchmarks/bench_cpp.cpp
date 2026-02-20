/*
	C++17 benchmark: NotoriousFFT C++ wrapper vs minfft performance comparison
	
	Demonstrates:
	- Reusable plan performance
	- Vector-based interface overhead
	- Memory allocation impact
*/

#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

extern "C" {
#include "minfft.h"
}

#include "notorious_fft.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/* ============================================================================
   Timer utilities
   ============================================================================ */

class Timer {
	Clock::time_point start_;
public:
	Timer() : start_(Clock::now()) {}
	
	void reset() { start_ = Clock::now(); }
	
	double elapsed() const {
		return std::chrono::duration_cast<Duration>(Clock::now() - start_).count();
	}
	
	double elapsed_ms() const { return elapsed() * 1000.0; }
	double elapsed_us() const { return elapsed() * 1e6; }
};

/* Benchmark function: returns average time per call in seconds */
template<typename Func>
double benchmark(Func &&f, int warmup_iters, int bench_iters) {
	/* Warmup */
	for (int i = 0; i < warmup_iters; ++i) {
		f();
	}
	
	/* Benchmark */
	Timer timer;
	for (int i = 0; i < bench_iters; ++i) {
		f();
	}
	return timer.elapsed() / bench_iters;
}

/* Calculate Mflop/s for FFT: 5 * N * log2(N) operations */
double mflops(int N, double time_seconds) {
	return (5.0 * N * std::log2(N)) / (time_seconds * 1e6);
}

/* ============================================================================
   Benchmarks
   ============================================================================ */

void bench_complex_dft() {
	std::cout << "\n=== Complex DFT (std::complex<double>) ===\n";
	std::cout << std::setw(10) << "N" 
			  << std::setw(15) << "minfft (us)"
			  << std::setw(15) << "notoriousfft (us)"
			  << std::setw(15) << "notoriousfft plan"
			  << std::setw(10) << "ratio"
			  << std::setw(12) << "Mflop/s"
			  << "\n";
	std::cout << std::string(77, '-') << "\n";
	
	int sizes[] = {64, 256, 1024, 4096, 16384, 65536};
	
	for (int N : sizes) {
		int warmup = 100;
		int iters = std::max(100, 10000 / N);
		
		/* Setup data */
		std::vector<minfft_cmpl> mx(N);
		std::vector<minfft_cmpl> my(N);
		std::vector<std::complex<double>> cx(N);
		std::vector<std::complex<double>> cy(N);
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (int i = 0; i < N; ++i) {
			double re = dist(rng);
			double im = dist(rng);
			reinterpret_cast<double*>(&mx[i])[0] = re;
			reinterpret_cast<double*>(&mx[i])[1] = im;
			cx[i] = std::complex<double>(re, im);
		}
		
		/* minfft reference */
		minfft_aux *ma = minfft_mkaux_dft_1d(N);
		auto t_minfft = benchmark([&]() {
			minfft_dft(mx.data(), my.data(), ma);
		}, warmup, iters);
		minfft_free_aux(ma);
		
		/* Notorious FFT vector interface (allocates aux each time) */
		auto t_notorious_fft_vec = benchmark([&]() {
			auto result = notorious_fft::dft(cx);
		}, warmup, iters);
		
		/* Notorious FFT reusable plan */
		notorious_fft::dft_plan plan(N);
		auto t_notorious_fft_plan = benchmark([&]() {
			auto result = plan.execute(cx);
		}, warmup, iters);
		
		std::cout << std::setw(10) << N
				  << std::fixed << std::setprecision(2)
				  << std::setw(15) << t_minfft * 1e6
				  << std::setw(15) << t_notorious_fft_vec * 1e6
				  << std::setw(15) << t_notorious_fft_plan * 1e6
				  << std::setw(10) << std::setprecision(2) << (t_minfft / t_notorious_fft_plan)
				  << std::setw(12) << std::setprecision(1) << mflops(N, t_notorious_fft_plan)
				  << "\n";
	}
}

void bench_real_dft() {
	std::cout << "\n=== Real DFT (double -> std::complex<double>) ===\n";
	std::cout << std::setw(10) << "N" 
			  << std::setw(15) << "minfft (us)"
			  << std::setw(15) << "notoriousfft (us)"
			  << std::setw(15) << "notoriousfft plan"
			  << std::setw(10) << "ratio"
			  << "\n";
	std::cout << std::string(65, '-') << "\n";
	
	int sizes[] = {64, 256, 1024, 4096, 16384, 65536};
	
	for (int N : sizes) {
		int warmup = 100;
		int iters = std::max(100, 10000 / N);
		
		std::vector<minfft_real> mx(N);
		std::vector<minfft_cmpl> my(N / 2 + 1);
		std::vector<double> cx(N);
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (int i = 0; i < N; ++i) {
			mx[i] = dist(rng);
			cx[i] = mx[i];
		}
		
		/* minfft reference */
		minfft_aux *ma = minfft_mkaux_realdft_1d(N);
		auto t_minfft = benchmark([&]() {
			minfft_realdft(mx.data(), my.data(), ma);
		}, warmup, iters);
		minfft_free_aux(ma);
		
		/* Notorious FFT vector interface */
		auto t_notorious_fft_vec = benchmark([&]() {
			auto result = notorious_fft::realdft(cx);
		}, warmup, iters);
		
		/* Notorious FFT reusable plan */
		notorious_fft::realdft_plan plan(N);
		auto t_notorious_fft_plan = benchmark([&]() {
			auto result = plan.execute(cx);
		}, warmup, iters);
		
		std::cout << std::setw(10) << N
				  << std::fixed << std::setprecision(2)
				  << std::setw(15) << t_minfft * 1e6
				  << std::setw(15) << t_notorious_fft_vec * 1e6
				  << std::setw(15) << t_notorious_fft_plan * 1e6
				  << std::setw(10) << std::setprecision(2) << (t_minfft / t_notorious_fft_plan)
				  << "\n";
	}
}

void bench_dct() {
	std::cout << "\n=== DCT-2 (double -> double) ===\n";
	std::cout << std::setw(10) << "N" 
			  << std::setw(15) << "minfft (us)"
			  << std::setw(15) << "notoriousfft (us)"
			  << std::setw(15) << "notoriousfft plan"
			  << std::setw(10) << "ratio"
			  << "\n";
	std::cout << std::string(65, '-') << "\n";
	
	int sizes[] = {64, 256, 1024, 4096, 16384, 65536};
	
	for (int N : sizes) {
		int warmup = 100;
		int iters = std::max(100, 10000 / N);
		
		std::vector<minfft_real> mx(N);
		std::vector<minfft_real> my(N);
		std::vector<double> cx(N);
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (int i = 0; i < N; ++i) {
			mx[i] = dist(rng);
			cx[i] = mx[i];
		}
		
		/* minfft reference */
		minfft_aux *ma = minfft_mkaux_t2t3_1d(N);
		auto t_minfft = benchmark([&]() {
			minfft_dct2(mx.data(), my.data(), ma);
		}, warmup, iters);
		minfft_free_aux(ma);
		
		/* Notorious FFT vector interface */
		auto t_notorious_fft_vec = benchmark([&]() {
			auto result = notorious_fft::dct2(cx);
		}, warmup, iters);
		
		/* Notorious FFT reusable plan */
		notorious_fft::dct2_plan plan(N);
		auto t_notorious_fft_plan = benchmark([&]() {
			auto result = plan.execute(cx);
		}, warmup, iters);
		
		std::cout << std::setw(10) << N
				  << std::fixed << std::setprecision(2)
				  << std::setw(15) << t_minfft * 1e6
				  << std::setw(15) << t_notorious_fft_vec * 1e6
				  << std::setw(15) << t_notorious_fft_plan * 1e6
				  << std::setw(10) << std::setprecision(2) << (t_minfft / t_notorious_fft_plan)
				  << "\n";
	}
}

void bench_batch_transforms() {
	std::cout << "\n=== Batch Transforms (1000 x size-1024 DFTs) ===\n";
	
	const int N = 1024;
	const int batch = 1000;
	
	std::vector<std::vector<std::complex<double>>> inputs(batch);
	for (auto &in : inputs) {
		in.resize(N);
		static std::mt19937 rng(42);
		static std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (auto &v : in) {
			v = std::complex<double>(dist(rng), dist(rng));
		}
	}
	
	/* Method 1: Create aux each time (worst case) */
	{
		Timer t;
		for (const auto &in : inputs) {
			auto out = notorious_fft::dft(in);
			volatile auto tmp = out[0]; /* prevent optimization */
			(void)tmp;
		}
		auto elapsed = t.elapsed_ms();
		std::cout << "Create aux each call: " << elapsed << " ms\n";
	}
	
	/* Method 2: Reusable plan (best case) */
	{
		notorious_fft::dft_plan plan(N);
		Timer t;
		for (const auto &in : inputs) {
			auto out = plan.execute(in);
			volatile auto tmp = out[0];
			(void)tmp;
		}
		auto elapsed = t.elapsed_ms();
		std::cout << "Reusable plan:        " << elapsed << " ms\n";
	}
	
	/* Method 3: In-place with reusable plan */
	{
		auto inputs_copy = inputs;
		notorious_fft::dft_plan plan(N);
		Timer t;
		for (auto &in : inputs_copy) {
			plan.execute_in_place(in);
		}
		auto elapsed = t.elapsed_ms();
		std::cout << "In-place with plan:   " << elapsed << " ms\n";
	}
}

void bench_2d_transforms() {
	std::cout << "\n=== 2D Transforms ===\n";
	std::cout << std::setw(15) << "Size" 
			  << std::setw(15) << "Time (ms)"
			  << std::setw(15) << "Mflop/s"
			  << "\n";
	std::cout << std::string(45, '-') << "\n";
	
	struct TestCase { int N1, N2; };
	TestCase cases[] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}};
	
	for (const auto &tc : cases) {
		int N = tc.N1 * tc.N2;
		std::vector<std::complex<double>> x(N);
		
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (auto &v : x) {
			v = std::complex<double>(dist(rng), dist(rng));
		}
		
		/* Warmup and benchmark */
		volatile int iters = 100;
		Timer timer;
		for (int i = 0; i < iters; ++i) {
			auto y = notorious_fft::dft_2d(x, tc.N1, tc.N2);
			volatile auto tmp = y[0];
			(void)tmp;
		}
		auto elapsed = timer.elapsed_ms() / iters;
		
		std::cout << std::setw(15) << (std::to_string(tc.N1) + "x" + std::to_string(tc.N2))
				  << std::fixed << std::setprecision(3)
				  << std::setw(15) << elapsed
				  << std::setw(15) << std::setprecision(1) << mflops(N, elapsed / 1000.0)
				  << "\n";
	}
}

/* ============================================================================
   Main
   ============================================================================ */
int main() {
	std::cout << "Notorious FFT C++17 Wrapper Benchmark\n";
	std::cout << "================================\n";
	std::cout << "Compiler: " << __VERSION__ << "\n";
	
	bench_complex_dft();
	bench_real_dft();
	bench_dct();
	bench_batch_transforms();
	bench_2d_transforms();
	
	std::cout << "\nDone.\n";
	return 0;
}
