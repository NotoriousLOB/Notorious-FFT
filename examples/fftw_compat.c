/*
 * FFTW3 tutorial-style example using Notorious FFT as a drop-in.
 *
 * Build:
 *   cc -O3 -DNOTORIOUS_FFT_IMPLEMENTATION -Iinclude examples/fftw_compat.c -lm
 */

#ifndef NOTORIOUS_FFT_IMPLEMENTATION
#define NOTORIOUS_FFT_IMPLEMENTATION
#endif
#include "notorious_fft.h"
#include "notorious_fft_fftw.h"

#include <stdio.h>
#include <math.h>

int main(void) {
    const int N = 8;
    fftw_complex *in = fftw_alloc_complex((size_t)N);
    fftw_complex *out = fftw_alloc_complex((size_t)N);

    for (int n = 0; n < N; n++) {
        ((double *)in)[2 * n]     = cos(2.0 * 3.14159265358979323846 * n / N);
        ((double *)in)[2 * n + 1] = 0.0;
    }

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    printf("FFTW-compatible 1D DFT of cos(2*pi*n/N):\n");
    for (int k = 0; k < N; k++)
        printf("  out[%d] = (%.3f, %.3f)\n", k,
               ((double *)out)[2 * k], ((double *)out)[2 * k + 1]);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    fftw_cleanup();
    return 0;
}
