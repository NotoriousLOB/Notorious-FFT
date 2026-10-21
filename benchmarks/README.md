# FFT Benchmarks

v1.1.2 numbers live in the [README receipts](../README.md#the-receipts)
and [results/jetson_orin_nano_r2r_dct.csv](results/jetson_orin_nano_r2r_dct.csv).
Run on Jetson Orin Nano / Cortex-A78AE, Clang 22, `-O3`, `taskset -c 0`.

## Available Benchmarks

### 0. Isolated DCT vs FFTW (`bench_r2r_dct.c`)

Execute-only REDFT10/REDFT01 medians at 256–513. Links FFTW3 when CMake finds it.

```bash
cmake --build build --target bench_r2r_dct
taskset -c 0 ./build/bench_r2r_dct
```

### 1. Basic Benchmark (`bench.c`)

Simple comparison of NotoriousFFT vs minfft for basic transform types.

```bash
cd build && make bench
./bench           # Run default benchmarks
./bench --all     # Run comprehensive benchmarks
```

### 2. Comprehensive Benchmark (`comprehensive_bench.cpp`)

Multi-library comparison including FFTW3, KissFFT, and muFFT.

```bash
cd build
cmake .. -DNOTORIOUS_FFT_BUILD_COMPREHENSIVE_BENCHMARK=ON
make comprehensive_bench

./comprehensive_bench              # JSON output
./comprehensive_bench --text       # Text output
./comprehensive_bench --transform complex_dft   # Specific transform
```

### 3. Mixed Sizes Benchmark (`bench_mixed_sizes.c`)

Non-power-of-two, prime, and multi-dimensional sizes (1D–4D). minfft only
covers power-of-two; Notorious FFT uses codelets, mixed-radix, Rader, or
Bluestein for the rest.

```bash
cd build && make bench_mixed_sizes

# Run all benchmarks
./bench_mixed_sizes

# Run specific dimensionalities
./bench_mixed_sizes --1d     # 1D transforms only
./bench_mixed_sizes --2d     # 2D transforms only (power-of-2 only)
./bench_mixed_sizes --3d     # 3D transforms only (experimental)
./bench_mixed_sizes --4d     # 4D transforms only (experimental)

# Run specific size types
./bench_mixed_sizes --power2  # Power-of-2 sizes only
./bench_mixed_sizes --npo2    # Non-power-of-2 sizes only
./bench_mixed_sizes --prime   # Prime sizes only (NotoriousFFT only)
./bench_mixed_sizes --mixed   # Mixed composite sizes only

# JSON output for plotting
./bench_mixed_sizes --json > results.json
python3 plot_results.py results.json
```

### 4. OpenMP Benchmark (`bench_openmp.c`)

Benchmark for multi-threaded performance.

```bash
cd build && make bench_openmp
./bench_openmp
```

## Tested Sizes

### Power-of-2 Sizes (all benchmarks)
```
16, 64, 256, 1024, 4096, 16384, 65536
```

### Non-Power-of-2 Sizes (`bench_mixed_sizes`)
Highly composite numbers with small prime factors:
```
12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576
20, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 30720
30, 72, 144, 288, 576, 1152, 2304, 4608, 9216, 18432
100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000
```

### Prime Sizes (`bench_mixed_sizes`)
```
17, 31, 61, 127, 257, 521, 1031, 2053, 4099, 8191, 16381, 32771, 65537
101, 211, 503, 1009, 2003, 5003, 10007, 20011, 50021
```

### Mixed/Composite Sizes (`bench_mixed_sizes`)
Numbers with medium prime factors (e.g., 2×3×5×7=210):
```
30, 60, 120, 210, 420, 840, 1680, 2310, 4620, 9240, 27720
90, 180, 360, 720, 1260, 2520, 5040, 7560, 15120, 30240
```

### 2D Sizes (`bench_mixed_sizes`)
```
# Square power-of-2
16×16, 32×32, 64×64, 128×128, 256×256, 512×512

# Non-square power-of-2
32×64, 64×128, 128×256, 256×512

# NPO2 sizes
24×24, 48×48, 96×96, 100×100, 120×120, 200×200

# Mixed aspect ratios
64×96, 128×192, 100×200, 120×240

# With prime dimension
64×127, 128×257, 100×101, 120×127

# Large rectangular
1024×64, 64×1024, 512×256, 256×512
```

### 3D Sizes (`bench_mixed_sizes`)
```
# Cubic power-of-2
8×8×8, 16×16×16, 32×32×32, 64×64×64

# Non-cubic power-of-2
8×16×32, 16×32×64, 32×64×128

# Mixed sizes
10×10×10, 12×12×12, 24×24×24, 30×30×30

# With prime dimension
16×16×17, 32×32×31, 64×64×61

# Large mixed
128×128×16, 256×64×64, 512×32×32
```

### 4D Sizes (`bench_mixed_sizes`)
```
# Hypercubic power-of-2
4×4×4×4, 8×8×8×8, 16×16×16×16, 32×32×32×32

# Non-hypercubic power-of-2
4×8×16×32, 8×16×32×64

# Mixed sizes
6×6×6×6, 10×10×10×10, 12×12×12×12

# With small prime dimension
8×8×8×7, 16×16×16×17, 32×32×32×31

# Large mixed
64×64×16×16, 128×32×32×32, 256×16×16×16
```

## Generating Plots

All benchmarks support JSON output for visualization:

```bash
# From bench_mixed_sizes with multi-dimensional support
./bench_mixed_sizes --json > results.json
python3 plot_results.py results.json

# From comprehensive_bench
./comprehensive_bench > results.json
python3 plot_results.py results.json

# Or pipe directly
./bench_mixed_sizes --json | python3 plot_results.py
```

### Generated Plots

The `plot_results.py` script generates:

1. **`fft_benchmark_summary.png`** - Execution time and performance for all transforms
2. **`fft_benchmark_speedup.png`** - Speedup relative to NotoriousFFT
3. **`fft_benchmark_normalized.png`** - Normalized execution time (time / N log N)

For multi-dimensional results (`bench_mixed_sizes`):
4. **`fft_benchmark_size_types.png`** - Performance comparison by size type (power2/npo2/prime/mixed)
5. **`fft_benchmark_dimensions.png`** - Multi-dimensional transform performance

## Output Format

### Standard Format (`comprehensive_bench.cpp`)
```json
[
  {
    "library": "notoriousfft",
    "transform": "complex_dft",
    "N": 1024,
    "time_us": 13.36,
    "stddev_us": 0.05,
    "gflops": 3.83,
    "iterations": 7
  }
]
```

### Multi-Dimensional Format (`bench_mixed_sizes.c`)
```json
[
  {
    "library": "notoriousfft",
    "transform": "dft",
    "size_type": "power2",
    "dim": 3,
    "n1": 32,
    "n2": 32,
    "n3": 32,
    "total_n": 32768,
    "time_us": 245.67,
    "mflops": 156.3
  }
]
```

## Supported Transforms

| Transform | Description | Supported Benchmarks |
|-----------|-------------|---------------------|
| Complex DFT | 1D Complex-to-Complex | All |
| Real DFT | 1D Real-to-Complex | All |
| Inverse Complex DFT | 1D Complex IFFT | `bench`, `bench_mixed_sizes` |
| Inverse Real DFT | 1D Complex-to-Real | `bench`, `bench_mixed_sizes` |
| DCT-2 | Type-2 Discrete Cosine Transform | All |
| DCT-3 | Type-3 Discrete Cosine Transform | `bench` |
| DCT-4 | Type-4 Discrete Cosine Transform | `bench` |
| DST-2 | Type-2 Discrete Sine Transform | `bench` |
| DST-3 | Type-3 Discrete Sine Transform | `bench` |
| DST-4 | Type-4 Discrete Sine Transform | `bench` |

## Implementation Notes

### Multi-Dimensional Support
- **minfft**: Power-of-2 only, any dimensionality
- **NotoriousFFT**: Any positive length (codelets, mixed-radix, Rader, Bluestein)

`bench_mixed_sizes` covers primes (127, 257, 1009), composites (100, 210),
and up to 4-D tensors.

### GFLOP/s Calculation
- **1D Complex DFT**: `5 * N * log2(N)` operations
- **1D Real DFT**: `2.5 * N * log2(N)` operations
- **Multi-Dimensional DFT**: `5 * N * (log2(n1) + log2(n2) + ...)` operations

### Statistical Rigor
- Warmup iterations before timing
- Multiple timing measurements with mean and standard deviation
- Minimum time threshold (typically 0.1s) to ensure accuracy
- Adaptive iteration count based on transform size

## Requirements

- CMake 3.14+
- C11/C++17 compiler
- Python 3 with matplotlib and numpy (for plotting)
- FFTW3 development libraries (for `comprehensive_bench`)

## Library-Specific Notes

- **FFTW3**: `comprehensive_bench` uses `FFTW_MEASURE`; `bench_r2r_dct` uses `FFTW_ESTIMATE`
- **muFFT**: Single-precision internally
- **KissFFT**: Portable, no explicit SIMD
- **minfft**: Split-radix, power-of-two only
- **NotoriousFFT**: Split-radix, codelets 3–43, mixed-radix, Rader, Bluestein; SIMD at compile time
