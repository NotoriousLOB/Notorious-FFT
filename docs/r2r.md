# Real-to-real transforms

`notorious_fft_plan_r2r_1d`, `_2d`, and `_3d` implement a tested subset of
FFTW's unnormalized real-to-real transforms. Include `notorious_fft_fftw.h`
to use the corresponding `fftw_plan_r2r_*`, `fftw_execute_r2r`,
`fftw_execute`, and `fftw_destroy_plan` aliases. No FFTW dependency is needed
by library consumers. The native header can be used alongside FFTW3 for
differential testing; the compatibility aliases and `<fftw3.h>` must be in
separate translation units.

## Definitions and lengths

For a length `N` input `x`, output index `k = 0,...,N-1`, and sums over
`j = 0,...,N-1` unless otherwise stated:

| Kind | Definition of `y[k]` | Inverse kind | Composition scale | Supported lengths |
| --- | --- | --- | --- | --- |
| `REDFT10` (DCT-II) | `2 Σ x[j] cos(π (j+1/2) k / N)` | `REDFT01` | `2N` | Every positive `int` length, subject to memory/size limits |
| `REDFT01` (DCT-III) | `x[0] + 2 Σ_{j=1}^{N-1} x[j] cos(π j (k+1/2) / N)` | `REDFT10` | `2N` | Every positive `int` length, subject to memory/size limits |
| `REDFT11` (DCT-IV) | `2 Σ x[j] cos(π (j+1/2) (k+1/2) / N)` | `REDFT11` | `2N` | Powers of two, `N <= INT_MAX/2` |
| `RODFT10` (DST-II) | `2 Σ x[j] sin(π (j+1/2) (k+1) / N)` | `RODFT01` | `2N` | Powers of two |
| `RODFT01` (DST-III) | `(-1)^k x[N-1] + 2 Σ_{j=0}^{N-2} x[j] sin(π (j+1) (k+1/2) / N)` | `RODFT10` | `2N` | Powers of two |
| `RODFT11` (DST-IV) | `2 Σ x[j] sin(π (j+1/2) (k+1/2) / N)` | `RODFT11` | `2N` | Powers of two, `N <= INT_MAX/2` |

In particular, length-one DCT-II returns `2*x[0]`, DCT-III returns `x[0]`,
and both type-IV transforms return `sqrt(2)*x[0]`. These transforms are not
orthonormal. Applying the inverse kinds along all axes gives a scale of
`Π_axis (2*n[axis])`; divide by this product to recover the original input.

The verified power-of-two DCT-II/III embeddings retain their FFT/SIMD
kernels. Length one and all other lengths use direct quadratic evaluation
with linear workspace. There is no measured crossover threshold or arbitrary
length fast-path claim in this milestone. Large non-power-of-two axes can be
slow. The legacy `mkaux_t2t3_*` constructors also accept these lengths;
their shared auxiliary additionally evaluates DST-II/III directly on fallback
lengths, but arbitrary-length DST is outside the FFTW-shaped support contract.

## Layout, arrays, and execution

Each axis has its own kind. Arrays are contiguous row-major tensors with the
last dimension contiguous: `(i,j)` is at `i*n1+j`, and `(i,j,k)` is at
`(i*n1+j)*n2+k`. Constructors do not require dimension reversal. Their
separable executor handles axes explicitly and does not change the legacy
auxiliary traversal or the layout of other transform families.

```c
fftw_plan p = fftw_plan_r2r_3d(n0, n1, n2, in, out,
    FFTW_REDFT10, FFTW_REDFT01, FFTW_RODFT11, FFTW_ESTIMATE);
if (!p) { /* stop: planning failed */ }
else {
    fftw_execute(p);
    fftw_execute_r2r(p, another_in, another_out);
    fftw_destroy_plan(p);
}
```

The two array pointers may be identical or disjoint. Partial overlap is
unsupported. Disjoint input is preserved. Replacement arrays have the same
shape and element type as the planned arrays; they can be in-place even when
the bound arrays were disjoint, or vice versa. Ordinary correctly aligned
`double` arrays suffice; replacement arrays need not share a SIMD alignment
class. `fftw_malloc` and `fftw_alloc_*` provide 64-byte alignment and reject
overflowing byte counts. Pair them with `fftw_free`.

All axis auxiliaries and line buffers are allocated while planning: one line
in 1D, two lines of the maximum axis length in 2D/3D. Contiguous rows transform
directly into output; in-place rows first copy into scratch. Execution,
including fallback execution, makes
no heap allocations. Repeated execution reuses plan scratch. Separate plans
can be constructed and executed concurrently. Access to a shared plan,
including execution or destruction, requires caller synchronization.

Planning does not read or overwrite bound arrays. NULL bound arrays are
accepted for use with `execute_r2r` replacements. Executing with a NULL plan
or NULL execution array is a no-op, as is NULL destruction. A successfully
constructed plan owns all nested allocations; construction failures clean up
before returning NULL.

## Planning failures and flags

Constructors preserve their pointer/NULL signatures. A NULL result can mean:

* A dimension is zero or negative.
* A kind is unsupported (`R2HC`, `HC2R`, `DHT`, `REDFT00`, `RODFT00`, or an
  invalid enum), or an axis violates the length restrictions above.
* The element/byte count exceeds representable object/pointer arithmetic.
* An allocation, including a nested FFT allocation, failed.

These conditions are distinguished in test diagnostics; there is no new
public error-code API. Products and aligned slab sizes are checked before
allocation. Legacy auxiliary total counts must fit `int` (complex auxiliary
counts and complex FFT engine lengths are limited to `INT_MAX/2` because
their interleaved indices use `int`). The FFTW-shaped r2r executor uses
`size_t` tensor offsets and does not impose a legacy `int` tensor-count limit.

All r2r planner flags are accepted and stored, with estimate-like behavior.
`MEASURE`, `PATIENT`, and `EXHAUSTIVE` do not run a search; `DESTROY_INPUT`
does not force input destruction, and `CONSERVE_MEMORY` does not change
workspace. The documented aliasing/input-preservation contract applies with
all flags. The existing complex DFT planner has separate limited timing
selection and is outside this r2r compatibility claim.

## Tested compatibility subset

| Function/family | Support in this milestone | Validation | Restrictions |
| --- | --- | --- | --- |
| `plan_r2r_1d/2d/3d` | Native constructors and FFTW aliases; independent kind per axis | Direct formulas, FFTW3, inverse scales, all PLAN.md reproductions | Six kinds and lengths above; contiguous rank 1–3 |
| `execute` / `execute_r2r` | Bound and replacement arrays | In-place, disjoint, input preservation, repeated execution, asymmetric axes | Partial overlap unsupported; shared plan needs synchronization |
| `destroy_plan` | Complete and partial construction cleanup | Every allocation failed in representative paths; ASan/UBSan/leak gate | NULL accepted |
| `malloc/free`, `alloc_real/complex` | Aligned allocations with checked byte counts | Consumer smoke builds, overflow and lifecycle checks | Available memory/object-size limits |
| Generic-rank, many, guru, strided or batched r2r | Not implemented | No compatibility claim | Deferred |
| Other FFTW APIs | Existing support retained | Existing C/C++ regression suite | No expanded FFTW differential claim in this milestone |

The numerical suite checks both directions independently, every ordered
pair/triple of six kinds on asymmetric small power-of-two tensors, prime
axes mixed with sine/type-IV axes, axis permutations, unit dimensions,
constants, zero, impulses at physical boundaries, small full basis sets,
alternating signs, ramps/sinusoids, and deterministic random data at scales
`1e-8`, `1`, and `1e8`. It also checks legacy in-place transforms, deliberately
unaligned replacement arrays, and 64 seeded rank/shape/kind property samples.
For every size it prints maximum absolute error,
relative error with denominator `max(input_scale, abs(reference))`, and
scaled RMS error. The per-element gate is
`abs(error) <= 2e-9*input_scale + 2e-10*abs(reference)`; finite output is
required. The absolute term covers cancellation near zero.

## Running the gate

```sh
cmake -S . -B build/core -DNOTORIOUS_FFT_VALIDATE_FFTW=ON \
  -DNOTORIOUS_FFT_OPENMP=OFF -DNOTORIOUS_FFT_BUILD_BENCHMARKS=OFF
cmake --build build/core -j4
ctest --test-dir build/core --output-on-failure
```

FFTW3 headers and the double-precision library are mandatory when
`NOTORIOUS_FFT_VALIDATE_FFTW=ON`; configuration fails if they are absent.
With the option off, the independent direct-formula suite still runs without
FFTW. Both gates require default double precision and accurate math. Strict
`-Wall -Wextra -Wpedantic -Werror` C99/C11/C++17 compatibility consumers and
a separate-plan concurrent execution test are included. A declaration-only
C++ consumer also links against a separate C implementation to check language
linkage and multi-translation-unit use.

For scalar ARM validation add `-DCMAKE_C_FLAGS=-DNOTORIOUS_FFT_HAS_NEON=0`
and the same C++ flag in a separate build directory. On x86, disable all
`NOTORIOUS_FFT_HAS_AVX512/AVX2/SSE2` macros for scalar kernel validation.
For sanitizers, use a separate Debug build with
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` in both
`CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS`, then run CTest with
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`.
LeakSanitizer must run in an environment permitting process/thread inspection.

`tests/test_r2r_memory.c` intercepts every malloc/calloc/aligned allocation in
the header. It checks NULL propagation, exact input preservation while
planning, zero live allocations after each failure, and zero allocations
during execution. A lowered four-step threshold reaches nested FFT ownership
branches with small fixtures. `tests/repro_r2r.c` can also be compiled against
the original header to reproduce the initial constructor rejections.
