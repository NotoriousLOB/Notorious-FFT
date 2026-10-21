<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/images/logo.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/images/logo.png">
  <img src="/.github/images/logo.png" width="480" alt="Notorious FFT" >
</picture>
</p>

# Notorious FFT

**Header-only FFT for C and C++. One mic, one file, an FFTW-shaped planner, and an O(N log N) cosine transform at every positive length.**

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"

notorious_fft_cmpl *x = notorious_fft_malloc(1024 * sizeof(notorious_fft_cmpl));
notorious_fft_cmpl *y = notorious_fft_malloc(1024 * sizeof(notorious_fft_cmpl));
notorious_fft_io_plan *p = notorious_fft_plan_dft_1d(
    1024, x, y, NOTORIOUS_FFT_FORWARD, NOTORIOUS_FFT_ESTIMATE);
notorious_fft_execute(p);
notorious_fft_destroy_io_plan(p);
```

It was all a dream: one translation unit defines the implementation macro, everyone else includes the header, you link `libm`. Plan, execute, destroy. That's the whole session.

---

## On the record

Notorious FFT is the vendorable FFTW dialect. You drop in a header, you keep the planner verbs your code already knows, and you get a real transform for sizes FFTW would also accept.

- **One file.** `include/notorious_fft.h` is amalgamated from `src/01_core.h`–`src/08_api.h`. C++17 lives in `notorious_fft.hpp`. The `fftw_*` aliases live in `notorious_fft_fftw.h` ([`examples/fftw_compat.c`](examples/fftw_compat.c)).
- **Any complex length.** Split-radix DIF on powers of two. Hard codelets for 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 43. Mixed-radix peels the largest of those (513 = 27×19, 129 = 43×3). Rader when N−1 is 2·3·5·7-smooth. Bluestein for the leftovers.
- **Real, cosine, sine.** Packed r2c/c2r including 2D/3D. DCT-II/III at every positive `int` length. DCT-IV and DST-II/III/IV on powers of two. Inverses are unnormalized, FFTW-style: `IDFT(DFT(x)) = N·x`.
- **SIMD you already asked for.** NEON, AVX2, AVX-512 follow `-march`. OpenMP follows `-fopenmp`.

Thirty-six chambers, one execute call. The planner picks the chamber.

## Studio rules

A plan owns its scratch. Two threads on the same plan will step on each other; two plans will not. **C.R.E.A.M.** — cache rules everything around me — which is why the engine lives in one 64-byte-aligned slab.

`ESTIMATE` is estimate. r2r `MEASURE` / `PATIENT` / `EXHAUSTIVE` are stored and then ignored as a search. Complex DFT has a limited MEASURE path. Nobody here writes a wisdom file.

The amalgam is fat on purpose: v1.1.2 is ~8.5k lines / 470 KB of unrolled codelets. You pay at compile time, once.

FFTW3 with a long `MEASURE` plan is still Dre in the booth. We print our ESTIMATE numbers next to ESTIMATE FFTW, which is the flag the compatibility header actually sends.

DST-II/III on the FFTW-shaped r2r API are power-of-two. DCT-IV/DST-IV are power-of-two. Partial buffer overlap is unsupported. Multi-dimensional c2r may overwrite its input; disjoint 1D/2D r2c input is preserved. Contracts: [docs/r2r.md](docs/r2r.md), [docs/real-dft.md](docs/real-dft.md).

---

## Life After Death — 1.1.2

The quadratic NPO2 DCT is dead. Even lengths pack through an N/2-point real DFT. Odd lengths run an N-point FFT and a precomputed phase. A 257-point DCT is still 257 points. REDFT10/01 scaling is still FFTW's unnormalized 2N.

Codelets and a greedy mixed-radix peel put 513 on 27×19 and packed 129 (DCT 258) on 43. SIMD still walks mixed-radix 3/5/7 across consecutive columns.

On Jetson Orin Nano, isolated REDFT10 at 257/258/511/513 stays inside ~1.6× FFTW 3.3.8 ESTIMATE. Complex C2C at 257, 511, and 513 is ahead of that same FFTW. Receipts below.

**1.1.1** — arbitrary-length 1D/2D r2c/c2r, padded in-place rows, disjoint C2R input kept. [docs/real-dft.md](docs/real-dft.md).

**1.1.0** — native 1D/2D/3D r2r, one kind per axis, planning-failure cleanup. [docs/core-validation.md](docs/core-validation.md).

---

## You Can Do It

C11 compiler, `libm`, and CMake ≥ 3.14 if you want tests.

```c
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"
```

Exactly one `.c` / `.cpp` gets the macro.

```bash
cc -O3 -march=native -o fft fft.c -lm
```

`-DNOTORIOUS_FFT_SINGLE` for float. `-DNOTORIOUS_FFT_FAST_MATH` for Bhaskara twiddles (~0.1% per factor, accumulates). Leave `-ffast-math` off if you want libm.

```bash
git clone https://github.com/NotoriousLOB/Notorious-FFT.git
cd Notorious-FFT
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/example_basic
```

Differential vs FFTW3: `-DNOTORIOUS_FFT_VALIDATE_FFTW=ON`.

---

## The Blueprint

```c
notorious_fft_io_plan *p = notorious_fft_plan_dft_1d(
    n, in, out, NOTORIOUS_FFT_FORWARD, NOTORIOUS_FFT_ESTIMATE);
if (!p) return 1;
notorious_fft_execute(p);
notorious_fft_destroy_io_plan(p);
```

FFTW aliases — keep `notorious_fft_fftw.h` and `<fftw3.h>` in separate translation units:

```c
#include "notorious_fft.h"
#include "notorious_fft_fftw.h"

fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```

C++:

```cpp
#include "notorious_fft.hpp"
auto a = notorious_fft::mkaux_dft_1d(1024);
notorious_fft::dft(x, y, a);
```

Three commandments:

1. One plan per concurrent worker.
2. You scale the inverse. Divide by N, or by Π 2N on r2r axes.
3. `notorious_fft_malloc` / `fftw_malloc` are 64-byte aligned. Use them on the hot buffers.

Awkward-length DCT-II is the same verb:

```c
notorious_fft_io_plan *p = notorious_fft_plan_r2r_1d(
    257, in, out, NOTORIOUS_FFT_REDFT10, NOTORIOUS_FFT_ESTIMATE);
notorious_fft_execute(p);
```

### Transforms

| Transform | Forward | Inverse | Lengths | Notes |
|-----------|:-------:|:-------:|---------|-------|
| Complex DFT | Yes | Yes, ×N | Any positive `int` | Split-radix / codelet / mixed / Rader / Bluestein |
| Real DFT (r2c/c2r) | Yes | Yes, ×N | Any positive `int` | Packed `floor(N/2)+1` bins |
| DCT-II / DCT-III | Yes | Yes, ×2N | Any positive `int` | FFT-backed for N≥3 |
| DCT-IV / DST-IV | Yes | Yes, ×2N | Powers of two | |
| DST-II / DST-III | Yes | Yes, ×2N | Powers of two in the FFTW r2r API | Legacy `mkaux_t2t3` still has a quadratic NPO2 DST kernel |

### Planner

| Group | Functions |
|---|---|
| Complex | `notorious_fft_plan_dft_1d/2d/3d`, `execute`, `execute_dft` |
| Real | `notorious_fft_plan_dft_r2c_1d/2d`, `plan_dft_c2r_1d/2d` |
| r2r | `notorious_fft_plan_r2r_1d/2d/3d`, `execute_r2r` |
| Destroy | `notorious_fft_destroy_io_plan` |
| Memory | `notorious_fft_malloc`, `notorious_fft_free` |
| Legacy aux | `mkaux_dft_*`, `mkaux_realdft_*`, `mkaux_t2t3_*`, `mkaux_t4_*`, `free_aux` |

| Flag | Effect |
|---|---|
| `-DNOTORIOUS_FFT_SINGLE` | `float` instead of `double` |
| `-DNOTORIOUS_FFT_FAST_MATH` | Bhaskara I twiddles |
| `-DNOTORIOUS_FFT_R2R_DCT_2N` | Optional 2N even-extension DCT |
| `-march=armv8.2-a` / `-mavx2 -mfma` / `-mavx512f -mavx512dq` | Matching SIMD |
| `-fopenmp` | Independent batches / large 1D stages |

---

## The Receipts

NVIDIA Jetson Orin Nano, Cortex-A78AE, Clang 22.1.3, `-O3 -DNDEBUG -march=armv8.2-a+dotprod+fp16`, double, `taskset -c 0`, 2026-10-21. FFTW 3.3.8 `FFTW_ESTIMATE`. Medians of 40–80 steady executes after warmup. Planning excluded.

**Complex C2C, FP64 (µs):**

| Size | Notorious FFT | FFTW3 ESTIMATE | Ratio |
|-----:|--------------:|---------------:|------:|
| 16 | 0.128 | 0.096 | 1.33× |
| 64 | 0.544 | 0.576 | 0.94× |
| 256 | 2.75 | 2.78 | 0.99× |
| 257 | 7.01 | 9.79 | 0.72× |
| 258 | 12.0 | 7.39 | 1.62× |
| 511 | 20.3 | 21.7 | 0.94× |
| 512 | 6.14 | 6.30 | 0.97× |
| 513 | 10.0 | 11.0 | 0.91× |
| 1024 | 13.7 | 14.6 | 0.93× |
| 4096 | 68.2 | 72.6 | 0.94× |
| 16384 | 336 | 377 | 0.89× |
| 65536 | 1660 | 2002 | 0.83× |

Mid-size powers of two sit on the same bar as ESTIMATE FFTW. 257, 511, and 513 go in front. 258 (2×3×43) is still the mo' problems size. CSV: [benchmarks/results/jetson_orin_nano_c2c.csv](benchmarks/results/jetson_orin_nano_c2c.csv).

**DCT-II (REDFT10), same host:**

| N | Notorious (µs) | FFTW3 (µs) | Ratio | Path |
|--:|---------------:|-----------:|------:|------|
| 256 | 2.30 | 2.24 | 1.03× | packed real DFT |
| 257 | 7.81 | 7.97 | 0.98× | Rader → 256-point FFT |
| 258 | 5.44 | 4.96 | 1.10× | packed real DFT of 129 = 43×3 |
| 511 | 22.0 | 14.3 | 1.54× | Rader-511, inner 510 = 17×30 |
| 512 | 4.83 | 4.64 | 1.04× | packed real DFT |
| 513 | 11.6 | 7.65 | 1.52× | mixed 27×19 |

v1.1.1 spent milliseconds at 257/511/513. v1.1.2 is O(N log N) and inside ~1.6× ESTIMATE FFTW. 257 is a good day. CSV: [benchmarks/results/jetson_orin_nano_r2r_dct.csv](benchmarks/results/jetson_orin_nano_r2r_dct.csv).

**Versus minfft** (powers of two, same host, mean of many repeats):

| N | minfft C2C (µs) | Notorious C2C (µs) | Ratio |
|--:|----------------:|-------------------:|------:|
| 256 | 3.54 | 2.71 | 1.31× |
| 1024 | 17.4 | 13.7 | 1.27× |
| 4096 | 85.8 | 68.9 | 1.24× |
| 16384 | 417 | 353 | 1.18× |
| 65536 | 2113 | 1649 | 1.28× |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench bench_r2r_dct -j
taskset -c 0 ./build/bench
taskset -c 0 ./build/bench_r2r_dct
```

---

## Build

| Option | Default | Description |
|--------|---------|-------------|
| `NOTORIOUS_FFT_BUILD_TESTS` | ON | `test_notoriousfft`, r2r, real-DFT, consumers |
| `NOTORIOUS_FFT_BUILD_BENCHMARKS` | ON | `bench`, `bench_r2r_dct`, mixed-size, OpenMP |
| `NOTORIOUS_FFT_BUILD_EXAMPLES` | ON | basic, FFTW compat, C++ |
| `NOTORIOUS_FFT_VALIDATE_FFTW` | OFF | FFTW3 differential r2r/r2c gates |
| `NOTORIOUS_FFT_BUILD_COMPREHENSIVE_BENCHMARK` | OFF | Also fetch KissFFT / PocketFFT / muFFT / FFTW |
| `NOTORIOUS_FFT_OPENMP` | ON | OpenMP when the compiler has it |
| `NOTORIOUS_FFT_SINGLE` | OFF | Library-wide `float` |
| `NOTORIOUS_FFT_FAST_MATH` | OFF | Bhaskara twiddles |

Release is `-O3`. CMake detects NEON on AArch64 and AVX2/AVX-512 on x86.

| Platform | NEON | SSE2 | AVX2 | AVX-512 |
|----------|:----:|:----:|:----:|:-------:|
| AArch64 | Yes | — | — | — |
| x86_64 | — | Yes | Yes | Yes |

```
include/     amalgam · C++ wrapper · fftw aliases
src/         01_core … 08_api · amalgamate.py
tests/       oracles, r2r, real DFT, consumers
benchmarks/  minfft, FFTW DCT, mixed sizes
examples/    basic.c · fftw_compat.c · cpp_advanced.cpp
docs/        r2r · real-DFT · v1.1.0 validation
```

## License

MIT — see [LICENSE](LICENSE). Protect ya neck, cite the copyright.
