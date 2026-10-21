# Real DFTs (v1.1.1)

Packing, signs, and inverse scales are unchanged in v1.1.2. That release
speeds the child complex FFT (codelets, mixed-radix peel) used by odd-N
and composite r2c; it does not change this contract.

Native `notorious_fft_plan_dft_r2c_1d/2d` and
`notorious_fft_plan_dft_c2r_1d/2d`, and their `fftw_*` compatibility aliases,
accept positive lengths in prime, odd, composite, and power-of-two size
classes. Invalid dimensions, checked size/index overflow, and allocation
failure return NULL. The legacy `mkaux_realdft_*` functions share this backend.
No FFTW library is required by consumers.

## Definition and packing

For length `N`, the forward transform is

`X[k] = Σ_{j=0}^{N-1} x[j] exp(-2πi j k / N)`.

Real input satisfies `X[N-k] = conj(X[k])`, so R2C stores complex bins
`k=0,...,floor(N/2)` in increasing frequency order. The imaginary part of DC
is zero. For even lengths, the last stored bin is Nyquist and also real.
For odd lengths, the last stored bin is an ordinary complex bin: its
imaginary part must be retained.

C2R expands the Hermitian spectrum and evaluates the positive-sign inverse,
without division by `N`. Equivalently,

`y[j] = Re(X[0]) + 2 Σ_{k=1}^{floor((N-1)/2)} (Re(X[k]) cos(2πjk/N) - Im(X[k]) sin(2πjk/N))`

plus `(-1)^j Re(X[N/2])` when `N` is even. The inverse ignores imaginary
DC/Nyquist endpoint components. Consequently `C2R(R2C(x)) = N*x`.

In 2D, dimensions are row-major, with the last dimension contiguous. Input
shape `(n0,n1)` has `n0*n1` real samples; packed output has
`n0*(floor(n1/2)+1)` complex elements. Both frequency coordinates use the
same Fourier sign, and inverse composition scales by `n0*n1`.

Hermitian symmetry couples **all** coordinates:
`X[(-k0) mod n0, (-k1) mod n1] = conj(X[k0,k1])`.
The stored DC plane, and Nyquist plane when `n1` is even, must themselves
be Hermitian across `k0`. Only points self-conjugate in every coordinate
are necessarily real. Arbitrary C2R input must satisfy these boundary-plane
constraints; zeroing entire DC/Nyquist planes' imaginary components would
incorrectly discard valid coefficients.

## Arrays and execution

Disjoint real arrays are dense. Packed complex arrays are contiguous.
For in-place transforms, allocate `2*(floor(n1/2)+1)` **real slots per row**:
`n1+2` for even `n1`, or `n1+1` for odd `n1`. In 1D, use the same padded
allocation for the one row. The extra slots are physical storage; they do
not change the logical length or transform normalization. After in-place
C2R, logical rows retain that physical stride; padding values are unspecified.

Identical pointers and disjoint arrays are supported; partial overlap is not.
New-array execution accepts replacement buffers with the same dimensions
and packing, including a change between disjoint and identical pointers.
Ordinary scalar alignment suffices; no SIMD alignment-class match is needed.

Execution preserves disjoint input in both directions, including replacement
arrays. In particular, the implementation honors `PRESERVE_INPUT` for 2D C2R
as an extension: FFTW3 can reject that planning flag for multidimensional C2R.
In-place execution necessarily overwrites shared storage. Planning does not
touch bound arrays, and NULL-bound plans may be executed on replacement arrays.
NULL execution arrays are a no-op. Destroying a NULL plan is safe.

Workspaces are allocated during planning, with complete cleanup on failure.
Execution makes no heap allocations. Separate plans may execute concurrently;
sharing a plan requires caller synchronization. Real-DFT planning flags are
estimate-like and do not perform a timing search.

Power-of-two lengths retain their half-length complex FFT embedding.
Other lengths use the existing full-length mixed-radix, Rader, or Bluestein
complex FFT and a preallocated interleaved line. Multidimensional plans stage
the packed tensor before copying output or applying the inverse, preserving
input and preventing padded rows from overwriting unread samples.

Checked limits still apply: legacy multidimensional logical counts must fit
`int`, complex-engine interleaved indices must fit their existing integer
bounds, and all objects/aligned slabs must fit pointer-size arithmetic.
For Bluestein, its padded convolution must fit those bounds too. “Arbitrary”
describes supported size classes, not unlimited storage or index ranges.

## Validation

`test_real_dft` checks 31 1D lengths through 1024 and 16 asymmetric 2D shapes,
including units, reversed axes, primes, mixed-radix sizes, and Bluestein sizes.
An independent full complex DFT oracle evaluates both signs using long-double
trigonometric tables. C2R tests use independently generated Hermitian spectra,
not only forward/inverse round trips. The suite checks every packed bin,
expanded Hermitian bins, endpoint handling, inverse scales, disjoint input
preservation, padded in-place arrays, alternate arrays, and repeated execution.
Four small legacy 3D shapes additionally check the shared staging code against
a direct three-axis formula.

`test_real_dft_fftw` also compares out-of-place and padded in-place execution
with FFTW3. Inputs cover zero, constants, boundary impulses, alternating signs,
sinusoids, ramps, and seeded random values at scales `1e-8`, `1`, and `1e8`.
The finite-output gate is
`abs(error) <= 2e-9*input_scale + 2e-10*abs(reference)`.
Maximum absolute and scaled errors are reported by shape, with
`scaled = abs(error)/max(input_scale,abs(reference))`.

The allocator-interception test fails every allocation in representative
real-plan construction paths (mixed-radix, Rader, Bluestein, and 2D staging),
checks clean unwinding and invalid/overflow rejection, and verifies no
execution-time allocations. Strict C99/C11/C++17 consumers now execute an
odd-length R2C/C2R round trip too.

```sh
cmake -S . -B build -DNOTORIOUS_FFT_VALIDATE_FFTW=ON \
  -DNOTORIOUS_FFT_OPENMP=OFF -DNOTORIOUS_FFT_BUILD_BENCHMARKS=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

FFTW3 is required only when the differential gate is enabled. Independent
DFT tests remain available with the option off. The full compatibility gate
uses double precision and accurate math.

### v1.1.1 release checks

The release passed all 13 CTest targets in each of five configurations:
Clang 22 Debug NEON, Clang 22 Release NEON, GCC 11 Release scalar,
GCC 11 Release NEON with OpenMP, and Clang 22 ASan/UBSan with leak detection.
FFTW 3.3.8 differential checks were enabled in all five. The lifecycle gate
passed 179 individually injected allocation failures across 22 construction
paths, with no execution-time allocations or leaked objects.

GCC 11 and Clang 22 also passed 28 strict AVX2/AVX-512 compile checks and
12 emulated x86 SSE2 runtime checks, including independent real-DFT
validation with both compilers. AVX execution requires an x86 host; local AVX checks cover
compilation only. Installed pkg-config C99, C11, and C++17 consumers passed
with version `1.1.1`. A separate single-precision direct-DFT/round-trip smoke
check passed ten lengths through 129; this does not replace the double
precision differential gate.

Representative maxima from the optimized Clang NEON run, including FFTW
comparisons and inputs scaled through `1e8`, are:

| Shape | Maximum absolute error | Maximum scaled error |
| --- | ---: | ---: |
| 17 | 9.54e-7 | 2.22e-15 |
| 23 | 8.23e-6 | 4.61e-14 |
| 64 | 2.86e-6 | 4.38e-15 |
| 1024 | 6.87e-5 | 1.32e-13 |
| 32 × 48 | 1.53e-4 | 1.08e-13 |

Absolute and scaled maxima may occur on different inputs. The suite prints
both metrics for every tested 1D length and 2D shape.
