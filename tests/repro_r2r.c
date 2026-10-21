/* Minimal, allocation-free caller for the PLAN.md constructor matrix.
 * Run against the baseline or current header; NULL is never executed. */
#include "notorious_fft.h"

static int probe(int rank, const int *n) {
    notorious_fft_aux *a;
    int supported = 1;
    for (int i = 0; i < rank; ++i)
        if (n[i] <= 0 || (n[i] & (n[i] - 1))) supported = 0;
    if (rank == 1) a = notorious_fft_mkaux_t2t3_1d(n[0]);
    else if (rank == 2) a = notorious_fft_mkaux_t2t3_2d(n[0], n[1]);
    else a = notorious_fft_mkaux_t2t3_3d(n[0], n[1], n[2]);
    printf("mkaux_t2t3_%dd %d x %d x %d: %s\n", rank,
           n[0], rank > 1 ? n[1] : 1, rank > 2 ? n[2] : 1,
           a ? "OK" : supported ? "construction/allocation failure" :
           "unsupported by baseline realdft power-of-two gate");
    int failed = a == NULL;
    notorious_fft_free_aux(a);
    return failed;
}

int main(void) {
    const int lengths[] = {1,2,3,5,7,8,13,16,17,31,32,63,64,65,127,128,1024};
    const int shapes2[][2] = {{7,5},{16,16},{17,64},{63,65},{128,256},{32,48},{8,6}};
    const int shapes3[][3] = {{4,8,3},{8,6,4},{8,10,6},{16,12,8},{64,48,32},{2,2,2},{3,3,3}};
    int failures = 0;
    for (size_t i = 0; i < sizeof(lengths)/sizeof(*lengths); ++i)
        failures += probe(1, lengths+i);
    for (size_t i = 0; i < sizeof(shapes2)/sizeof(*shapes2); ++i)
        failures += probe(2, shapes2[i]);
    for (size_t i = 0; i < sizeof(shapes3)/sizeof(*shapes3); ++i)
        failures += probe(3, shapes3[i]);
    printf("%d rejected constructors\n", failures);
    return failures != 0;
}
