# FFT Benchmarks

This directory contains comprehensive benchmarks for the NotoriousFFT library.

## Available Benchmarks

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

### 3. Mixed Sizes Benchmark (`bench_mixed_sizes.c`) ⭐ NEW

Enhanced benchmark with non-power-of-2 sizes, prime sizes, and multi-dimensional transforms (1D-4D).

**Note:** This benchmark is designed to test NotoriousFFT's support for arbitrary sizes via Bluestein's algorithm.
- ✅ **1D Power-of-2**: Fully working (NotoriousFFT vs minfft comparison)
- ✅ **1D NPO2**: Supported if NotoriousFFT has Bluestein implementation
- ⚠️ **1D Prime**: Requires complete Bluestein implementation in NotoriousFFT
- ✅ **2D**: Working for power-of-2 sizes (minfft-compatible)
- ⚠️ **3D/4D**: Experimental - may be slow for large sizes

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
- **NotoriousFFT**: Arbitrary sizes via Bluestein's algorithm, any dimensionality

The `bench_mixed_sizes` benchmark demonstrates NotoriousFFT's ability to handle:
- Prime-sized dimensions (e.g., 127, 257, 1009)
- Non-power-of-2 dimensions (e.g., 100, 200, 500)
- Up to 4-dimensional transforms

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

- **FFTW3**: Uses `FFTW_MEASURE` planning mode for fair comparison
- **muFFT**: Uses single-precision floats internally
- **KissFFT**: Portable implementation without explicit SIMD
- **minfft**: Split-radix algorithm, power-of-2 only
- **NotoriousFFT**: Split-radix for power-of-2, Bluestein for arbitrary sizes, SIMD optimizations
