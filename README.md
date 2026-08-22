<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/images/logo.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/images/logo.png">
  <img src="/.github/images/logo.png" width="480" alt="Notorious FFT" >
</picture>	
</p>

A header-only Fast Fourier Transform library for C/C++.

Notorious FFT provides a complete set of discrete transforms in a single header file with no dependencies beyond the C standard library.

## Features

- **Header-only**: single file `include/notorious_fft.h`, just `#include` and go
- **FFTW-shaped planner**: `plan_dft_*` / `execute` / `destroy_io_plan`, plus `include/notorious_fft_fftw.h` aliases
- **All transform types**: complex DFT, real DFT (including 2D/3D r2c), DCT-2/3/4, DST-2/3/4
- **Any size**: split-radix for powers of two, mixed-radix 3/5/7, Rader for primes with smooth N−1, Bluestein otherwise
- **SIMD**: NEON, AVX2, AVX-512 selected at compile time (not stubs)
- **Single and double precision**: define `NOTORIOUS_FFT_SINGLE` for float, default is double
- **Version**: 1.0.0 (`NOTORIOUS_FFT_VERSION_*`)

### Accuracy Modes

- **Default**: Machine-precision twiddle factors using standard library sin/cos
- **Fast math**: Define `NOTORIOUS_FFT_FAST_MATH` for Bhaskara I approximation (~0.1% max error). Only suitable for small N or applications tolerant of ~1% output error, as errors accumulate across FFT stages.

## Quick Start

Planner API (recommended):

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"

notorious_fft_cmpl *x = notorious_fft_malloc(1024 * sizeof(notorious_fft_cmpl));
notorious_fft_cmpl *y = notorious_fft_malloc(1024 * sizeof(notorious_fft_cmpl));
/* ... fill x ... */
notorious_fft_io_plan *p = notorious_fft_plan_dft_1d(
    1024, x, y, NOTORIOUS_FFT_FORWARD, NOTORIOUS_FFT_ESTIMATE);
notorious_fft_execute(p);
notorious_fft_destroy_io_plan(p);
notorious_fft_free(x);
notorious_fft_free(y);
```

Drop-in FFTW3 subset (`examples/fftw_compat.c`):

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"
#include "notorious_fft_fftw.h"

fftw_complex *in = fftw_alloc_complex(1024);
fftw_complex *out = fftw_alloc_complex(1024);
fftw_plan p = fftw_plan_dft_1d(1024, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```

Do not mix `notorious_fft_fftw.h` with linking `libfftw3` in the same translation unit.

**Thread safety:** concurrent `execute` on the **same** plan is not supported (plans own scratch). Distinct plans may run in parallel.

Inverse DFTs are **unnormalized** (same as FFTW): a round-trip is `IDFT(DFT(x)) = N·x`.
Multi-dimensional c2r/invrealdft may overwrite its input.

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
| `-DNOTORIOUS_FFT_FAST_MATH` | Bhaskara I twiddles (~0.1% max error; accumulates with N) |
| `-march=armv8-a` / `-mavx2 -mfma` / `-mavx512f -mavx512dq` | Enable the matching SIMD path |
| `-fopenmp` | Enable OpenMP (independent batches / large 1D stages) |

Do **not** pass `-ffast-math` if you care about matching libm accuracy.

### Running Tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_notoriousfft
./build/test_notoriousfft_cpp
```

### Running Benchmarks

```sh
cmake -S . -B build -DNOTORIOUS_FFT_BUILD_BENCHMARKS=ON
cmake --build build --target bench -j
./build/bench
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

**Results (v1.0.0):**
```
--- Complex DFT (Forward) ---
N           minfft (us)   notoriousfft (us)  ratio
------------------------------------------------------
16          0.08          0.08          0.97x
64          0.52          0.50          1.04x
256         2.93          2.72          1.08x
1024        15.23         13.86         1.10x
4096        77.59         68.95         1.13x
16384       384.39        354.75        1.08x
65536       1896.76       1674.19       1.13x

--- Complex DFT (Inverse) ---
16          0.08          0.08          0.927x
64          0.52          0.49          1.050x
256         2.93          2.68          1.092x
1024        15.22         13.67         1.114x
4096        81.41         69.07         1.179x
16384       383.55        347.12        1.105x
65536       1836.34       1635.89       1.123x

--- Real DFT (Forward) ---
16          0.05          0.06          0.834x
64          0.31          0.33          0.961x
256         1.69          1.67          1.010x
1024        8.50          8.16          1.041x
4096        41.53         39.44         1.053x
16384       199.84        195.27        1.023x
65536       956.70        898.27        1.065x

--- DCT Type 2 ---
16          0.09          0.10          0.866x
64          0.46          0.46          1.002x
256         2.28          2.21          1.027x
1024        10.89         10.44         1.043x
4096        50.98         48.34         1.055x
16384       244.53        231.15        1.058x
65536       1181.50       1079.74       1.094x
```

### Summary

- **v1.0.0** beats minfft by **1.08–1.13×** on complex DFT for N≥256
- N=16 is essentially tied (0.97×) after FFmpeg-style hardcoded twiddles
- Mixed-radix **3/5/7**; **Rader** for primes with smooth N−1 (31, 127, 17, 257, …); Bluestein for awkward N
- Four-step at N≥2²⁰; recursive split-radix is faster at 64K
- SIMD: NEON (f64 + f32), AVX2 inverse, AVX-512 forward

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

Notorious FFT uses split-radix (2/4) decimation-in-frequency by default (no bit-reversal; unrolled terminals for N≤32). `NOTORIOUS_FFT_MEASURE` may select a DIT combine or iterative Cooley–Tukey instead. Real transforms reduce to half-length complex FFTs. Mixed-radix 3/5/7 peels those factors even when the cofactor is prime. Prime lengths whose N−1 is 2·3·5·7-smooth use Rader (an (N−1)-point FFT), including small Mersenne primes 31 and 127 and Fermat primes 17 and 257. Padding a Mersenne length 2^p−1 with one zero and taking a 2^p FFT is a different transform (bin frequencies 2πk/2^p, not 2πk/n) and is not used. Awkward sizes fall back to Bluestein. DCT-2/3 go through real DFT with pre/post twiddles; DCT-4/DST-4 reduce to an N/2-point complex DFT. All per-plan memory is a single 64-byte-aligned slab.

## License

MIT License

Copyright (c) 2025 adri4n <yo@adri4n.net>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
