/*
 * Comprehensive FFT Benchmark
 * 
 * Compares performance of:
 * - notoriousfft (NotoriousFFT)
 * - minfft (reference implementation)
 * - FFTW3 (highly optimized, system library)
 * - KissFFT (simple C library)
 * - muFFT (single-file C library)
 * 
 * Outputs JSON results for plotting with matplotlib.
 * 
 * Note: Not all libraries support all transform types.
 * This benchmark focuses on common functionality.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <cstdlib>

// External library includes
extern "C" {
#include "minfft.h"
}

#include "notorious_fft.h"

// FFTW3
#include <fftw3.h>

// KissFFT
#include "kiss_fft.h"

// muFFT - define our own cfloat type to match muFFT expectations
extern "C" {
#include "fft.h"
}

// muFFT uses complex float internally - define a compatible type
struct cfloat {
    float real;
    float imag;
};

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
    
    double elapsed_us() const { return elapsed() * 1e6; }
    double elapsed_ms() const { return elapsed() * 1e3; }
};

/* ============================================================================
   Benchmark configuration
   ============================================================================ */

struct BenchConfig {
    std::vector<int> sizes;
    int warmup_iters;
    int min_iters;
    int max_iters;
    double min_time_seconds;
};

static BenchConfig default_config() {
    return {
        // Sizes from small to large (power-of-2 for fair comparison)
        {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
        10,      // warmup iterations
        100,     // minimum iterations
        10000,   // maximum iterations
        0.1      // minimum time per benchmark (seconds)
    };
}

/* ============================================================================
   Result structure
   ============================================================================ */

struct BenchResult {
    std::string library;
    std::string transform;
    int N;
    double time_us;       // time per transform in microseconds
    double stddev_us;     // standard deviation
    double gflops;        // GFLOP/s (5 * N * log2(N) operations per FFT)
    int iterations;       // number of iterations run
};

/* ============================================================================
   Data generation
   ============================================================================ */

template<typename T>
std::vector<T> generate_signal(int N, uint32_t seed = 42) {
    std::vector<T> data(N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<T>(dist(rng));
    }
    return data;
}

/* ============================================================================
   Generic benchmark runner
   ============================================================================ */

template<typename Func>
BenchResult run_benchmark(const std::string& library, const std::string& transform, 
                          int N, Func&& func, const BenchConfig& config) {
    // Warmup
    for (int i = 0; i < config.warmup_iters; ++i) {
        func();
    }
    
    // Determine number of iterations
    int iters = config.min_iters;
    std::vector<double> times;
    times.reserve(config.max_iters);
    
    // Run benchmark, increasing iterations until we hit minimum time
    double total_time = 0.0;
    while (total_time < config.min_time_seconds && iters <= config.max_iters) {
        Timer timer;
        for (int i = 0; i < iters; ++i) {
            func();
        }
        double elapsed = timer.elapsed();
        times.push_back(elapsed / iters);
        total_time += elapsed;
        iters *= 2;
    }
    
    // Calculate statistics
    double mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean_time) * (t - mean_time);
    }
    variance /= times.size();
    double stddev = std::sqrt(variance);
    
    // Calculate GFLOP/s: 5 * N * log2(N) operations per FFT
    double flops = 5.0 * N * std::log2(N);
    double gflops = flops / (mean_time * 1e9);
    
    return {
        library,
        transform,
        N,
        mean_time * 1e6,  // convert to microseconds
        stddev * 1e6,
        gflops,
        static_cast<int>(times.size())
    };
}

/* ============================================================================
   Complex DFT Benchmarks
   ============================================================================ */

std::vector<BenchResult> bench_complex_dft(const BenchConfig& config) {
    std::vector<BenchResult> results;
    
    for (int N : config.sizes) {
        // Generate test data
        auto real_data = generate_signal<double>(2 * N);
        std::vector<std::complex<double>> cx_data(N);
        for (int i = 0; i < N; ++i) {
            cx_data[i] = std::complex<double>(real_data[2*i], real_data[2*i+1]);
        }
        
        // Copy for minfft (interleaved double[2])
        std::vector<minfft_cmpl> minfft_in(N), minfft_out(N);
        memcpy(minfft_in.data(), cx_data.data(), N * sizeof(minfft_cmpl));
        
        // Copy for NotoriousFFT
        std::vector<notorious_fft_cmpl> notorious_fft_in(N), notorious_fft_out(N);
        memcpy(notorious_fft_in.data(), cx_data.data(), N * sizeof(notorious_fft_cmpl));
        
        // Copy for FFTW3
        std::vector<std::complex<double>> fftw_in(cx_data);
        std::vector<std::complex<double>> fftw_out(N);
        
        // KissFFT (uses separate real/imag arrays)
        kiss_fft_cfg kiss_cfg = kiss_fft_alloc(N, 0, nullptr, nullptr);
        std::vector<kiss_fft_cpx> kiss_in(N), kiss_out(N);
        for (int i = 0; i < N; ++i) {
            kiss_in[i].r = static_cast<kiss_fft_scalar>(cx_data[i].real());
            kiss_in[i].i = static_cast<kiss_fft_scalar>(cx_data[i].imag());
        }
        
        // muFFT - allocate aligned buffers
        mufft_plan_1d* mufft_plan = mufft_create_plan_1d_c2c(N, MUFFT_FORWARD, MUFFT_FLAG_CPU_ANY);
        void* mufft_in = mufft_alloc(N * sizeof(cfloat));
        void* mufft_out = mufft_alloc(N * sizeof(cfloat));
        cfloat* mufft_in_f = static_cast<cfloat*>(mufft_in);
        cfloat* mufft_out_f = static_cast<cfloat*>(mufft_out);
        for (int i = 0; i < N; ++i) {
            mufft_in_f[i].real = static_cast<float>(cx_data[i].real());
            mufft_in_f[i].imag = static_cast<float>(cx_data[i].imag());
        }
        
        // --- minfft ---
        {
            minfft_aux* aux = minfft_mkaux_dft_1d(N);
            auto result = run_benchmark("minfft", "complex_dft", N, [&]() {
                minfft_dft(minfft_in.data(), minfft_out.data(), aux);
            }, config);
            results.push_back(result);
            minfft_free_aux(aux);
        }
        
        // --- notoriousfft ---
        {
            notorious_fft_aux* aux = notorious_fft_mkaux_dft_1d(N);
            auto result = run_benchmark("notoriousfft", "complex_dft", N, [&]() {
                notorious_fft_dft(notorious_fft_in.data(), notorious_fft_out.data(), aux);
            }, config);
            results.push_back(result);
            notorious_fft_free_aux(aux);
        }
        
        // --- FFTW3 ---
        {
            fftw_complex* in = reinterpret_cast<fftw_complex*>(fftw_in.data());
            fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_out.data());
            fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
            auto result = run_benchmark("FFTW3", "complex_dft", N, [&]() {
                fftw_execute(plan);
            }, config);
            results.push_back(result);
            fftw_destroy_plan(plan);
        }
        
        // --- KissFFT ---
        {
            auto result = run_benchmark("KissFFT", "complex_dft", N, [&]() {
                kiss_fft(kiss_cfg, kiss_in.data(), kiss_out.data());
            }, config);
            results.push_back(result);
        }
        
        // --- muFFT ---
        {
            auto result = run_benchmark("muFFT", "complex_dft", N, [&]() {
                mufft_execute_plan_1d(mufft_plan, mufft_out, mufft_in);
            }, config);
            results.push_back(result);
            mufft_free_plan_1d(mufft_plan);
        }
        
        free(kiss_cfg);
        mufft_free(mufft_in);
        mufft_free(mufft_out);
    }
    
    return results;
}

/* ============================================================================
   Real DFT Benchmarks
   ============================================================================ */

std::vector<BenchResult> bench_real_dft(const BenchConfig& config) {
    std::vector<BenchResult> results;
    
    for (int N : config.sizes) {
        auto real_data = generate_signal<double>(N);
        
        // minfft
        std::vector<minfft_real> minfft_in(real_data.begin(), real_data.end());
        std::vector<minfft_cmpl> minfft_out(N/2 + 1);
        
        // NotoriousFFT
        std::vector<notorious_fft_real> notorious_fft_in(real_data.begin(), real_data.end());
        std::vector<notorious_fft_cmpl> notorious_fft_out(N/2 + 1);
        
        // FFTW3
        std::vector<double> fftw_in(real_data);
        std::vector<std::complex<double>> fftw_out(N/2 + 1);
        
        // muFFT - r2c only takes flags, not direction
        mufft_plan_1d* mufft_plan = mufft_create_plan_1d_r2c(N, MUFFT_FLAG_CPU_ANY);
        void* mufft_in = mufft_alloc(N * sizeof(float));
        void* mufft_out = mufft_alloc((N/2 + 1) * sizeof(cfloat));
        float* mufft_in_f = static_cast<float*>(mufft_in);
        cfloat* mufft_out_f = static_cast<cfloat*>(mufft_out);
        for (int i = 0; i < N; ++i) {
            mufft_in_f[i] = static_cast<float>(real_data[i]);
        }
        
        // --- minfft ---
        {
            minfft_aux* aux = minfft_mkaux_realdft_1d(N);
            auto result = run_benchmark("minfft", "real_dft", N, [&]() {
                minfft_realdft(minfft_in.data(), minfft_out.data(), aux);
            }, config);
            results.push_back(result);
            minfft_free_aux(aux);
        }
        
        // --- notoriousfft ---
        {
            notorious_fft_aux* aux = notorious_fft_mkaux_realdft_1d(N);
            auto result = run_benchmark("notoriousfft", "real_dft", N, [&]() {
                notorious_fft_realdft(notorious_fft_in.data(), notorious_fft_out.data(), aux);
            }, config);
            results.push_back(result);
            notorious_fft_free_aux(aux);
        }
        
        // --- FFTW3 ---
        {
            fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_out.data());
            fftw_plan plan = fftw_plan_dft_r2c_1d(N, fftw_in.data(), out, FFTW_MEASURE);
            auto result = run_benchmark("FFTW3", "real_dft", N, [&]() {
                fftw_execute(plan);
            }, config);
            results.push_back(result);
            fftw_destroy_plan(plan);
        }
        
        // --- muFFT ---
        if (mufft_plan) {
            auto result = run_benchmark("muFFT", "real_dft", N, [&]() {
                mufft_execute_plan_1d(mufft_plan, mufft_out, mufft_in);
            }, config);
            results.push_back(result);
            mufft_free_plan_1d(mufft_plan);
        }
        
        mufft_free(mufft_in);
        mufft_free(mufft_out);
    }
    
    return results;
}

/* ============================================================================
   DCT-2 Benchmarks (only for libraries that support it)
   ============================================================================ */

std::vector<BenchResult> bench_dct2(const BenchConfig& config) {
    std::vector<BenchResult> results;
    
    for (int N : config.sizes) {
        auto real_data = generate_signal<double>(N);
        std::vector<double> out(N);
        
        // minfft
        std::vector<minfft_real> minfft_in(real_data.begin(), real_data.end());
        std::vector<minfft_real> minfft_out(N);
        
        // NotoriousFFT
        std::vector<notorious_fft_real> notorious_fft_in(real_data.begin(), real_data.end());
        std::vector<notorious_fft_real> notorious_fft_out(N);
        
        // FFTW3
        std::vector<double> fftw_in(real_data);
        std::vector<double> fftw_out(N);
        
        // --- minfft ---
        {
            minfft_aux* aux = minfft_mkaux_t2t3_1d(N);
            auto result = run_benchmark("minfft", "dct2", N, [&]() {
                minfft_dct2(minfft_in.data(), minfft_out.data(), aux);
            }, config);
            results.push_back(result);
            minfft_free_aux(aux);
        }
        
        // --- notoriousfft ---
        {
            notorious_fft_aux* aux = notorious_fft_mkaux_t2t3_1d(N);
            auto result = run_benchmark("notoriousfft", "dct2", N, [&]() {
                notorious_fft_dct2(notorious_fft_in.data(), notorious_fft_out.data(), aux);
            }, config);
            results.push_back(result);
            notorious_fft_free_aux(aux);
        }
        
        // --- FFTW3 ---
        {
            fftw_plan plan = fftw_plan_r2r_1d(N, fftw_in.data(), fftw_out.data(), 
                                               FFTW_REDFT10, FFTW_MEASURE);
            auto result = run_benchmark("FFTW3", "dct2", N, [&]() {
                fftw_execute(plan);
            }, config);
            results.push_back(result);
            fftw_destroy_plan(plan);
        }
    }
    
    return results;
}

/* ============================================================================
   JSON Output
   ============================================================================ */

void print_json(const std::vector<BenchResult>& results) {
    std::cout << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  {\n";
        std::cout << "    \"library\": \"" << r.library << "\",\n";
        std::cout << "    \"transform\": \"" << r.transform << "\",\n";
        std::cout << "    \"N\": " << r.N << ",\n";
        std::cout << "    \"time_us\": " << std::fixed << std::setprecision(4) << r.time_us << ",\n";
        std::cout << "    \"stddev_us\": " << r.stddev_us << ",\n";
        std::cout << "    \"gflops\": " << std::setprecision(3) << r.gflops << ",\n";
        std::cout << "    \"iterations\": " << r.iterations << "\n";
        std::cout << "  }";
        if (i < results.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "]\n";
}

/* ============================================================================
   Main
   ============================================================================ */

int main(int argc, char** argv) {
    // Parse command line options
    std::string output_format = "json";
    std::string selected_transform = "all";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--text" || arg == "-t") {
            output_format = "text";
        } else if (arg == "--json" || arg == "-j") {
            output_format = "json";
        } else if (arg == "--transform" && i + 1 < argc) {
            selected_transform = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --text, -t          Output in text format\n"
                      << "  --json, -j          Output in JSON format (default)\n"
                      << "  --transform <name>  Run only specific transform (complex_dft, real_dft, dct2)\n"
                      << "  --help, -h          Show this help\n";
            return 0;
        }
    }
    
    // Print header
    if (output_format == "text") {
        std::cout << "========================================\n";
        std::cout << "  Comprehensive FFT Benchmark\n";
        std::cout << "========================================\n\n";
        std::cout << "Libraries tested:\n";
        std::cout << "  - notoriousfft    (NotoriousFFT)\n";
        std::cout << "  - minfft    (reference)\n";
        std::cout << "  - FFTW3     (highly optimized)\n";
        std::cout << "  - KissFFT   (simple C)\n";
        std::cout << "  - muFFT     (single-file C)\n\n";
    }
    
    auto config = default_config();
    std::vector<BenchResult> all_results;
    
    // Run benchmarks
    if (selected_transform == "all" || selected_transform == "complex_dft") {
        if (output_format == "text") {
            std::cout << "Running Complex DFT benchmarks...\n";
        }
        auto results = bench_complex_dft(config);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    if (selected_transform == "all" || selected_transform == "real_dft") {
        if (output_format == "text") {
            std::cout << "Running Real DFT benchmarks...\n";
        }
        auto results = bench_real_dft(config);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    if (selected_transform == "all" || selected_transform == "dct2") {
        if (output_format == "text") {
            std::cout << "Running DCT-2 benchmarks...\n";
        }
        auto results = bench_dct2(config);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    // Output results
    if (output_format == "json") {
        print_json(all_results);
    } else {
        // Text output
        std::cout << "\n\nResults:\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << std::setw(12) << "Library" 
                  << std::setw(15) << "Transform"
                  << std::setw(10) << "N"
                  << std::setw(15) << "Time (us)"
                  << std::setw(15) << "GFLOP/s"
                  << "\n";
        std::cout << std::string(80, '=') << "\n";
        
        for (const auto& r : all_results) {
            std::cout << std::setw(12) << r.library
                      << std::setw(15) << r.transform
                      << std::setw(10) << r.N
                      << std::fixed << std::setprecision(2)
                      << std::setw(15) << r.time_us
                      << std::setw(15) << r.gflops
                      << "\n";
        }
    }
    
    return 0;
}
