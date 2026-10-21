/* Also compiled as C++17. Uses only the public FFTW compatibility surface. */
#include "notorious_fft_fftw.h"

int main(void) {
    double input[6] = {1,1,1,1,1,1}, output[6], alternate[6];
    fftw_plan p = fftw_plan_r2r_2d(2,3,input,output,FFTW_REDFT10,FFTW_REDFT10,FFTW_ESTIMATE);
    fftw_plan q = fftw_plan_r2r_3d(2,3,1,alternate,alternate,
                                  FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,FFTW_PATIENT);
    if (!p || !q) { fftw_destroy_plan(p); fftw_destroy_plan(q); return 1; }
    fftw_execute(p);
    fftw_execute_r2r(p,input,alternate);
    fftw_execute(q);
    int failed = fabs(output[0]-24) > 1e-12;
#ifdef NOTORIOUS_TEST_EXPECTED_VERSION
    failed |= strcmp(NOTORIOUS_FFT_VERSION_STRING, NOTORIOUS_TEST_EXPECTED_VERSION) != 0;
#endif
    for (int i = 0; i < 6; ++i) failed |= fabs(alternate[i]-24) > 1e-12;
    fftw_destroy_plan(p); fftw_destroy_plan(q);
    void *buffer = fftw_alloc_real(6);
    if (!buffer || ((uintptr_t)buffer % 64)) failed = 1;
    fftw_free(buffer); fftw_cleanup();
    double real_input[7]={1,2,3,4,5,6,7}, real_output[7];
    fftw_complex spectrum[4];
    p=fftw_plan_dft_r2c_1d(7,real_input,spectrum,FFTW_ESTIMATE);
    q=fftw_plan_dft_c2r_1d(7,spectrum,real_output,FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
    if (!p || !q) failed=1;
    else {
        fftw_execute(p); fftw_execute(q);
        for (int i=0;i<7;++i) failed |= fabs(real_output[i]-7*real_input[i])>1e-10;
    }
    fftw_destroy_plan(p); fftw_destroy_plan(q);
    return failed;
}
