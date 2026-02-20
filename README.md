# Notorious FFT

A header-only Fast Fourier Transform library for C/C++.

Notorious FFT provides a complete set of discrete transforms in a single header file with no dependencies beyond the C standard library.

## Features

- **Header-only**: single file `include/notorious_fft.h`, just `#include` and go
- **All transform types**: complex DFT, real DFT, DCT-2/3/4, DST-2/3/4
- **Any dimensionality**: 1D, 2D, 3D, and arbitrary N-dimensional transforms
- **SIMD support**: AVX2/AVX512/NEON vectorization for butterfly operations
- **OpenMP support**: automatic parallelization for large transforms
- **Fast approximations**: Bhaskara I sine/cosine approximations for faster twiddle computation
- **Single and double precision**: define `NOTORIOUS_FFT_SINGLE` for float, default is double

### Accuracy Modes

- **Default**: Machine-precision twiddle factors using standard library sin/cos
- **Fast math**: Define `NOTORIOUS_FFT_FAST_MATH` for Bhaskara I approximation (~0.1% max error). Only suitable for small N or applications tolerant of ~1% output error, as errors accumulate across FFT stages.

## Quick Start

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"

/* 1D complex DFT */
notorious_fft_cmpl x[1024], y[1024];
/* ... fill x ... */
notorious_fft_aux *a = notorious_fft_mkaux_dft_1d(1024);
notorious_fft_dft(x, y, a);
notorious_fft_free_aux(a);
```

### C++ Usage

```cpp
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"
#include "notorious_fft.hpp"

// RAII aux data - automatically freed
auto a = notorious_fft::mkaux_dft_1d(1024);
notorious_fft::dft(x, y, a);

// Or use the vector convenience API
std::vector<notorious_fft_cmpl> x(1024);
auto y = notorious_fft::dft(x);
```

## API Reference

### Transform Functions

| Function | Description |
|---|---|
| `notorious_fft_dft(x, y, a)` | Forward complex DFT |
| `notorious_fft_invdft(x, y, a)` | Inverse complex DFT (unnormalized) |
| `notorious_fft_realdft(x, z, a)` | Forward real DFT |
| `notorious_fft_invrealdft(z, y, a)` | Inverse real DFT (unnormalized) |
| `notorious_fft_dct2(x, y, a)` | Type-2 Discrete Cosine Transform |
| `notorious_fft_dst2(x, y, a)` | Type-2 Discrete Sine Transform |
| `notorious_fft_dct3(x, y, a)` | Type-3 Discrete Cosine Transform |
| `notorious_fft_dst3(x, y, a)` | Type-3 Discrete Sine Transform |
| `notorious_fft_dct4(x, y, a)` | Type-4 Discrete Cosine Transform |
| `notorious_fft_dst4(x, y, a)` | Type-4 Discrete Sine Transform |

### Auxiliary Data Creation

Auxiliary data contains precomputed twiddle factors and temporary buffers. It is reusable across multiple transforms of the same type and size.

| Function | Description |
|---|---|
| `notorious_fft_mkaux_dft_1d(N)` | Aux for 1D complex DFT/IDFT |
| `notorious_fft_mkaux_dft_2d(N1, N2)` | Aux for 2D complex DFT/IDFT |
| `notorious_fft_mkaux_dft_3d(N1, N2, N3)` | Aux for 3D complex DFT/IDFT |
| `notorious_fft_mkaux_dft(d, Ns)` | Aux for d-dimensional complex DFT/IDFT |
| `notorious_fft_mkaux_realdft_1d(N)` | Aux for 1D real DFT/IDFT |
| `notorious_fft_mkaux_realdft_2d(N1, N2)` | Aux for 2D real DFT/IDFT |
| `notorious_fft_mkaux_realdft_3d(N1, N2, N3)` | Aux for 3D real DFT/IDFT |
| `notorious_fft_mkaux_realdft(d, Ns)` | Aux for d-dimensional real DFT/IDFT |
| `notorious_fft_mkaux_t2t3_1d(N)` | Aux for 1D DCT-2/DCT-3/DST-2/DST-3 |
| `notorious_fft_mkaux_t4_1d(N)` | Aux for 1D DCT-4/DST-4 |
| `notorious_fft_free_aux(a)` | Free auxiliary data |

### Data Layout

Transform definitions and data layout are compatible with FFTW:
- Complex data is stored as interleaved real/imaginary pairs (`double[2]`)
- Real DFT output has `N/2+1` complex elements
- Multi-dimensional inverse real DFT does not preserve its input

## Building

This is a header-only library. To use it, define `NOTORIOUS_FFT_IMPLEMENTATION` in exactly one translation unit before including the header:

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"
```

### Compiler Flags

| Flag | Effect |
|---|---|
| `-DNOTORIOUS_FFT_SINGLE` | Use single precision (float) instead of double |
| `-mfpu=neon` or `-march=armv8-a` | Enable ARM NEON SIMD |
| `-mavx2` | Enable AVX2 SIMD (stubs) |
| `-mavx512f` | Enable AVX-512 SIMD (stubs) |
| `-fopenmp` | Enable OpenMP parallelization |

### Running Tests

```sh
gcc -O2 -I. -o tests/test_notorious_fft tests/test_notoriousfft.c minfft/minfft.c -lm
./tests/test_notoriousfft
```

### Running Benchmarks

```sh
gcc -O2 -I. -o benchmarks/bench benchmarks/bench.c minfft/minfft.c -lm
./benchmarks/bench
```

For comprehensive benchmarking of all API functions:

```sh
./benchmarks/bench --all
```

This runs benchmarks for all transform types:
- Complex DFT (forward/inverse)
- Real DFT (forward/inverse)
- DCT Type 2/3
- DST Type 2/3
- DCT Type 4 / DST Type 4

## Benchmark Results

Latest benchmark run on ARM Cortex-A78AE (Jetson Orin), GCC 11.4.0 -O3 -march=armv8.2-a+fp16, double precision.

### Quick Benchmark (vs minfft)

```bash
cd build
cmake .. -DNOTORIOUS_FFT_BUILD_BENCHMARKS=ON
make bench
./bench
```

**Results:**
```
--- Complex DFT (Forward) ---
N           minfft (us)   notoriousfft (us)  ratio   
------------------------------------------------------
16          0.10          0.11          0.944x
64          0.51          0.49          1.044x
256         2.72          2.59          1.049x
1024        14.12         13.33         1.059x
4096        71.72         66.06         1.086x
16384       350.14        332.94        1.052x
65536       1712.24       1638.09       1.045x

--- Complex DFT (Inverse) ---
16          0.08          0.09          0.870x
64          0.51          0.49          1.052x
256         2.91          2.60          1.120x
1024        15.14         13.39         1.131x
4096        76.86         66.21         1.161x
16384       376.24        329.73        1.141x
65536       1793.43       1598.81       1.122x

--- Real DFT (Forward) ---
16          0.04          0.05          0.846x
64          0.29          0.31          0.916x
256         1.56          1.59          0.983x
1024        7.86          7.71          1.019x
4096        38.38         37.45         1.025x
16384       184.63        181.71        1.016x
65536       878.21        845.70        1.038x

--- DCT Type 2 ---
16          0.08          0.09          0.876x
64          0.42          0.43          0.970x
256         2.07          2.06          1.005x
1024        9.94          9.69          1.026x
4096        46.62         44.70         1.043x
16384       226.05        215.25        1.050x
65536       1060.05       1011.72       1.048x
```

### Summary

- **NotoriousFFT** achieves **1.04–1.16x** speedup over minfft for N≥256 on complex transforms
- Real DFT and DCT-2 show consistent **1.02–1.05x** speedup at larger sizes
- Small N (16-64) has some overhead; the split-radix terminal cases are highly optimized
- **SIMD Accelerated** with NEON (ARM) and AVX2/AVX-512 (x86) intrinsics
- All memory allocated in a single 64-byte-aligned slab; cleanup is a single `free`

### Comprehensive Benchmark

For comparison against FFTW3, KissFFT, PocketFFT, and muFFT:

```bash
cd build
cmake .. -DNOTORIOUS_FFT_BUILD_COMPREHENSIVE_BENCHMARK=ON
make comprehensive_bench
./comprehensive_bench --text
```

### Visualization

Generate performance plots with matplotlib:

```bash
./comprehensive_bench > results.json
python3 ../benchmarks/plot_results.py results.json
```

This produces three plots:
- `fft_benchmark_summary.png` - Execution time and GFLOP/s for all transforms
- `fft_benchmark_speedup.png` - Speedup relative to NotoriousFFT
- `fft_benchmark_normalized.png` - Normalized execution time (μs / (N log N))

## Algorithm

Notorious FFT uses a split-radix (2/4) decimation-in-frequency algorithm with manually unrolled terminal cases for N=1,2,4,8. Real transforms are reduced to complex transforms of half length. DCT-2/3 are computed via real DFT with pre/post-processing twiddles. DCT-4/DST-4 use an O(N log N) reduction to an N/2-point complex DFT. All per-plan memory is managed through a bump allocator backed by a single 64-byte-aligned slab; destruction is a single `free`.

## License

MIT License.
