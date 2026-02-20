/*
	Basic usage example for NotoriousFFT
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>

#include "notorious_fft.h"

int main(void) {
	/* Example 1: 1D Complex DFT of a simple signal */
	printf("=== 1D Complex DFT ===\n");
	{
		int N = 8;
		notorious_fft_cmpl x[8], y[8];

		/* create a signal: x[n] = cos(2*pi*n/N) */
		for (int n = 0; n < N; n++) {
			((double*)&x[n])[0] = cos(2.0 * M_PI * n / N);
			((double*)&x[n])[1] = 0;
		}

		notorious_fft_aux *a = notorious_fft_mkaux_dft_1d(N);
		notorious_fft_dft(x, y, a);

		printf("Input: ");
		for (int n = 0; n < N; n++)
			printf("(%.2f,%.2f) ", ((double*)&x[n])[0], ((double*)&x[n])[1]);
		printf("\n");

		printf("DFT:   ");
		for (int n = 0; n < N; n++)
			printf("(%.2f,%.2f) ", ((double*)&y[n])[0], ((double*)&y[n])[1]);
		printf("\n\n");

		notorious_fft_free_aux(a);
	}

	/* Example 2: 1D Real DFT */
	printf("=== 1D Real DFT ===\n");
	{
		int N = 8;
		notorious_fft_real x[8];
		notorious_fft_cmpl z[5]; /* N/2+1 = 5 */

		/* create signal */
		for (int n = 0; n < N; n++)
			x[n] = sin(2.0 * M_PI * n / N);

		notorious_fft_aux *a = notorious_fft_mkaux_realdft_1d(N);
		notorious_fft_realdft(x, z, a);

		printf("Input: ");
		for (int n = 0; n < N; n++)
			printf("%.3f ", x[n]);
		printf("\n");

		printf("DFT:   ");
		for (int n = 0; n < N / 2 + 1; n++)
			printf("(%.3f,%.3f) ", ((double*)&z[n])[0], ((double*)&z[n])[1]);
		printf("\n\n");

		notorious_fft_free_aux(a);
	}

	/* Example 3: DCT-2 */
	printf("=== 1D DCT-2 ===\n");
	{
		int N = 8;
		notorious_fft_real x[8], y[8];

		for (int n = 0; n < N; n++)
			x[n] = n + 1.0;

		notorious_fft_aux *a = notorious_fft_mkaux_t2t3_1d(N);
		notorious_fft_dct2(x, y, a);

		printf("Input: ");
		for (int n = 0; n < N; n++)
			printf("%.2f ", x[n]);
		printf("\n");

		printf("DCT-2: ");
		for (int n = 0; n < N; n++)
			printf("%.2f ", y[n]);
		printf("\n\n");

		notorious_fft_free_aux(a);
	}

	printf("Done.\n");
	return 0;
}
