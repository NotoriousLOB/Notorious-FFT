# v1.1.0 core correctness validation

This is the v1.1.0 r2r-correctness record. v1.1.1 added arbitrary-length
real DFTs ([real-dft.md](real-dft.md)). v1.1.2 replaced the quadratic
NPO2 DCT-II/III kernel with an FFT-backed path; the formulas, flags, and
aliasing contract in [r2r.md](r2r.md) still apply. Do not use the
“direct fallback remains quadratic” paragraph below as current behavior.

Baseline: `83142da40558f2242e7988c821207b60c8c7185d`, the pinned revision
named in PLAN.md. The repository started clean. No CHEAP repository or
dependency pins were changed. External CHEAP CI logs were not present; the local gate addresses
PLAN.md phases 0–2 and does not claim CHEAP CI parity.

Environment: ARM64 Cortex-A78AE, Ubuntu GCC/G++ 11.4, Clang/Clang++ 22.1.3,
CMake 3.22.1, FFTW3 3.3.8. SIMD validation uses NEON on this host; AVX and
other architectures are not validated here.

## Baseline diagnosis

Before implementation, compiling and running `tests/repro_r2r.c` against
the baseline rejected 20 of 31 requests:

| Family | Rejected requests | Successful requests |
| --- | --- | --- |
| 1D `mkaux_t2t3_1d` | 3, 5, 7, 13, 17, 31, 63, 65, 127 | 1, 2, 8, 16, 32, 64, 128, 1024 |
| 2D `mkaux_t2t3_2d` | 7×5, 17×64, 63×65, 32×48, 8×6 | 16×16, 128×256 |
| 3D `mkaux_t2t3_3d` | 4×8×3, 8×6×4, 8×10×6, 16×12×8, 64×48×32, 3×3×3 | 2×2×2 |

The last failing stage is `mkaux_realdft_1d`'s explicit power-of-two guard,
called by `mkaux_t2t3_1d`. These are unsupported lengths in that embedding,
not evidence of allocator exhaustion. `plan_r2r_1d` used the same auxiliary
path. Native `plan_r2r_2d/3d` and their FFTW aliases were absent.

Source review and direct-reference tests also found in-place hazards for
length-two type-II/III transforms, a skipped normalization factor when a
legacy tensor's inner subtransform had length one, a type-IV all-unit tensor
shortcut, signed products in legacy tensor sizing and twiddle sizing, and
unchecked aligned slab accounting. Those paths now have checked bounds and
failure unwinding. Nested FFT allocation failure propagates rather than
silently trying a different algorithm. Existing legacy axis traversal is
preserved: the real-to-real recursive code transforms the remaining axes in
contiguous blocks before gathering the first axis. The new row-major
constructors require no adapter or dimension reversal.

## Verification record

The reproducible test targets and commands are described in [r2r.md](r2r.md).
All final runs enabled `NOTORIOUS_FFT_VALIDATE_FFTW`; both the independent
direct-formula target and the FFTW differential target ran in each build.

| Configuration | Result |
| --- | --- |
| Clang 22 Debug, NEON | 11/11 CTest targets passed |
| Clang 22 Release (`-O3`), NEON | 11/11 CTest targets passed |
| GCC 11 Release, scalar kernels (`NOTORIOUS_FFT_HAS_NEON=0`) | 11/11 CTest targets passed |
| GCC 11 Release, NEON and OpenMP (`OMP_NUM_THREADS=2`) | 11/11 CTest targets passed |
| Clang 22, NEON, `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` | 11/11 CTest targets passed, with `detect_leaks=1` and sanitizer halt-on-error |

The sanitizer gate ran outside the workspace sandbox because LeakSanitizer's
process/thread inspection was blocked inside it. The final unrestricted run
reported no ASan, UBSan, or leak errors. Leak checking was not disabled.

Each numerical target covers 499 shape/kind cases, including all 216 ordered
triples of supported kinds, the full reproduction matrix, all-unit tensors,
64 seeded rank/shape/kind property samples, and inverse-scale
checks. Both DCT directions agree with independent direct formulas and FFTW.
The memory target failed each of 136 allocation positions across 17 planning
paths, with NULL propagation and no tracked live allocations after unwinding.
It also verified no allocations during representative r2r executions.
Existing C and C++ regressions, strict-warning C99/C11/C++17 consumers, and
separate-plan concurrent execution all passed in each configuration.
The C++ declaration-only consumer also linked against a separate C implementation
under strict warnings. Installed pkg-config consumers compiled and executed
under GCC/G++ in C99, C11, and C++17 modes with those flags, using only the
installed headers and pkg-config metadata. The consumer-only CMake configuration
and installation needed no downloaded test dependencies. Clang static analysis
of the fault-injection/planner harness completed without diagnostics. Fresh
amalgamation was byte-identical to the distributed
header, and `git diff --check` passed.
The existing basic example also compiled and ran with `NOTORIOUS_FFT_SINGLE`;
the full numerical compatibility gate remains double precision.

Selected DCT-II error rows from the optimized NEON differential run follow. Absolute
errors aggregate all patterns, including random inputs scaled by `1e8`;
relative errors use the near-zero floor documented in [r2r.md](r2r.md).
Full per-size/per-kind rows are emitted by the test executables and retained
in each build's `Testing/Temporary/LastTest.log`.

| Shape | Maximum absolute error | Maximum relative error | Scaled RMS error |
| --- | ---: | ---: | ---: |
| 3 | 2.98e-8 | 1.33e-15 | 4.78e-16 |
| 127 | 2.65e-5 | 5.63e-13 | 1.96e-14 |
| 1024 | 1.48e-5 | 1.25e-13 | 1.89e-15 |
| 128×256 | 9.92e-5 | 1.32e-11 | 7.35e-16 |
| 64×48×32 | 3.49e-3 | 5.32e-10 | 1.29e-14 |

Release polish removes the redundant tensor copy and gather/scatter operations
on contiguous rows, reduces 1D plans to one line buffer, and hoists fallback
direction/trigonometric selection and endpoint handling outside inner loops.
The direct fallback remains quadratic. CMake, public version macros, README,
and generated pkg-config metadata agree on v1.1.0. CI now invokes CTest with
the required FFTW differential gate and sanitizer leak checks; remote CI has
not been run locally.

The acceptance scope is arbitrary positive-length DCT-II/III with a direct
fallback, six-kind native rank-1/2/3 r2r planning, checked lifecycle handling,
and the documented subset. Fast arbitrary-length optimization, batched or
strided r2r, benchmarks, and CHEAP integration remain deferred.
