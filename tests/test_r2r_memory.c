/* Intercept every heap allocation in the amalgam, including nested FFT slabs.
 * Each successful construction is replayed with its kth allocation failing. */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <time.h>

/* x86 intrinsics include mm_malloc.h, which redeclares posix_memalign even
 * after stdlib.h. Load those system headers before allocator interception. */
#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static void *live[4096];
static size_t attempts, fail_at, live_count;
static int failed_line;

static int should_fail(int line) {
    ++attempts;
    if (attempts == fail_at) { failed_line = line; return 1; }
    return 0;
}
static void *record(void *p) {
    if (p) { REQUIRE(live_count < sizeof(live)/sizeof(*live)); live[live_count++] = p; }
    return p;
}
static void *test_malloc(size_t n, int line) {
    if (should_fail(line) || n > 64*1024*1024) return NULL;
    return record(malloc(n));
}
static void *test_calloc(size_t n, size_t elem, int line) {
    if (should_fail(line) || (elem && n > 64*1024*1024/elem)) return NULL;
    return record(calloc(n,elem));
}
static int test_aligned(void **p, size_t align, size_t n, int line) {
    if (should_fail(line) || n > 64*1024*1024) { *p = NULL; return 12; }
    int ret = posix_memalign(p,align,n);
    if (!ret) record(*p);
    return ret;
}
static void test_free(void *p) {
    if (!p) return;
    size_t i = 0;
    while (i < live_count && live[i] != p) ++i;
    REQUIRE(i < live_count);
    live[i] = live[--live_count];
    free(p);
}
#define malloc(n) test_malloc(n,__LINE__)
#define calloc(n,e) test_calloc(n,e,__LINE__)
#define posix_memalign(p,a,n) test_aligned(p,a,n,__LINE__)
#define free(p) test_free(p)
#include "notorious_fft_fftw.h"
#undef malloc
#undef calloc
#undef posix_memalign
#undef free

static double input[2048], output[2048];
static void *construct(int which) {
    switch (which) {
    case 0: return notorious_fft_plan_r2r_1d(17,input,output,NOTORIOUS_FFT_REDFT10,0);
    case 1: return notorious_fft_plan_r2r_1d(32,input,input,NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_PATIENT);
    case 2: return notorious_fft_plan_r2r_2d(7,8,input,output,NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_RODFT11,0);
    case 3: return notorious_fft_plan_r2r_3d(4,3,8,input,input,NOTORIOUS_FFT_RODFT01,NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT11,0);
    case 4: return notorious_fft_mkaux_t2t3_1d(1);
    case 5: return notorious_fft_mkaux_t2t3_1d(2);
    case 6: return notorious_fft_mkaux_t2t3_2d(7,8);
    case 7: return notorious_fft_mkaux_t2t3_3d(3,8,4);
    case 8: return notorious_fft_mkaux_t4_3d(4,8,2);
    case 9: return notorious_fft_mkaux_realdft_1d(32);
    case 10: return notorious_fft_mkaux_realdft_2d(8,16);
    case 11: return notorious_fft_mkaux_dft_1d(45); /* mixed */
    case 12: return notorious_fft_mkaux_dft_1d(17); /* Rader */
    case 13: return notorious_fft_mkaux_dft_1d(23); /* Bluestein */
    case 14: return notorious_fft_mkaux_dft_1d(128); /* four-step in this test */
    case 15: return notorious_fft_plan_r2r_1d(1,input,output,NOTORIOUS_FFT_REDFT10,0);
    case 16: return notorious_fft_plan_r2r_1d(2,input,input,NOTORIOUS_FFT_RODFT10,UINT_MAX);
    case 17: return notorious_fft_plan_dft_r2c_1d(23,input,(notorious_fft_cmpl*)output,0);
    case 18: return notorious_fft_plan_dft_c2r_1d(17,(notorious_fft_cmpl*)input,output,NOTORIOUS_FFT_PRESERVE_INPUT);
    case 19: return notorious_fft_plan_dft_r2c_2d(3,5,input,(notorious_fft_cmpl*)input,0);
    case 20: return notorious_fft_plan_dft_c2r_2d(7,6,(notorious_fft_cmpl*)input,output,NOTORIOUS_FFT_PRESERVE_INPUT);
    case 21: return notorious_fft_plan_dft_c2r_1d(6,(notorious_fft_cmpl*)input,input,0);
    default: abort();
    }
}
static void destroy(int which, void *p) {
    if (which < 4 || which >= 15) notorious_fft_destroy_io_plan((notorious_fft_io_plan*)p);
    else notorious_fft_free_aux((notorious_fft_aux*)p);
}
static void execute(int which, void *p) {
    if (which < 4 || which >= 15) {
        notorious_fft_execute((notorious_fft_io_plan*)p);
        notorious_fft_execute_r2r((notorious_fft_io_plan*)p,input,output);
        notorious_fft_execute_r2r((notorious_fft_io_plan*)p,output,output);
    } else if (which < 8) {
        notorious_fft_dct2(input,output,(notorious_fft_aux*)p);
        notorious_fft_dct3(output,output,(notorious_fft_aux*)p);
    } else if (which == 8) notorious_fft_dst4(input,output,(notorious_fft_aux*)p);
    else if (which == 9 || which == 10) {
        notorious_fft_realdft(input,(notorious_fft_cmpl*)output,(notorious_fft_aux*)p);
        notorious_fft_invrealdft((notorious_fft_cmpl*)output,input,(notorious_fft_aux*)p);
    }
}
static void injection(void) {
    for (int which = 0; which < 22; ++which) {
        attempts = 0; fail_at = 0;
        double before[2048];
        memcpy(before,input,sizeof(input));
        void *p = construct(which);
        REQUIRE(p);
        REQUIRE(!memcmp(before,input,sizeof(input)));
        size_t total = attempts;
        fail_at = attempts+1;
        execute(which,p); execute(which,p);
        REQUIRE(attempts == total); /* zero execution-time heap allocation */
        destroy(which,p);
        REQUIRE(live_count == 0);
        for (size_t k = 1; k <= total; ++k) {
            attempts = 0; fail_at = k; failed_line = 0;
            memcpy(before,input,sizeof(input));
            p = construct(which);
            REQUIRE(!p); /* every required allocation failure propagates */
            REQUIRE(!memcmp(before,input,sizeof(input)));
            REQUIRE(failed_line != 0);
            destroy(which,p); /* NULL destruction is safe */
            REQUIRE(live_count == 0);
            printf("fault path=%d allocation=%zu/%zu header-line=%d: allocation failure, clean unwind\n",
                   which,k,total,failed_line);
        }
    }
    fail_at = 0;
}

/* Rejected requests must not allocate and must never be executed. */
#define REJECT(expr,reason) do { attempts = 0; void *bad = (expr); REQUIRE(!bad); REQUIRE(attempts == 0); printf("rejected %s: %s\n",#expr,reason); } while (0)
static void invalid(void) {
    REJECT(fftw_plan_r2r_1d(0,input,output,FFTW_REDFT10,0),"invalid dimension");
    REJECT(fftw_plan_r2r_1d(-1,input,output,FFTW_REDFT01,0),"invalid dimension");
    REJECT(fftw_plan_dft_r2c_1d(0,input,(fftw_complex*)output,0),"invalid real dimension");
    REJECT(fftw_plan_dft_c2r_1d(-1,(fftw_complex*)input,output,0),"invalid real dimension");
    REJECT(fftw_plan_dft_r2c_2d(INT_MAX,2,input,(fftw_complex*)output,0),"real tensor count overflow");
    REJECT(fftw_plan_dft_c2r_2d(INT_MAX,INT_MAX,(fftw_complex*)input,output,0),"real tensor count overflow");
    REJECT(fftw_plan_r2r_2d(8,-3,input,output,FFTW_REDFT10,FFTW_REDFT10,0),"invalid dimension");
    REJECT(fftw_plan_r2r_3d(1,0,1,input,output,FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,0),"invalid dimension");
    for (int k = -2; k < 14; ++k) {
        if (k == 4 || k == 5 || k == 6 || k == 8 || k == 9 || k == 10) continue;
        REJECT(fftw_plan_r2r_1d(8,input,output,(fftw_r2r_kind)k,0),"unsupported kind");
        REJECT(fftw_plan_r2r_2d(8,4,input,output,FFTW_REDFT10,(fftw_r2r_kind)k,0),"unsupported axis kind");
        REJECT(fftw_plan_r2r_3d(8,4,2,input,output,FFTW_REDFT10,FFTW_REDFT01,(fftw_r2r_kind)k,0),"unsupported axis kind");
    }
    REJECT(fftw_plan_r2r_1d(7,input,output,FFTW_RODFT10,0),"unsupported DST length");
    REJECT(fftw_plan_r2r_2d(8,6,input,output,FFTW_REDFT10,FFTW_REDFT11,0),"unsupported DCT-IV length");
    REJECT(fftw_plan_r2r_3d(INT_MAX,INT_MAX,INT_MAX,input,output,FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT01,0),"element count overflow");
    REJECT(fftw_plan_r2r_2d(INT_MAX,INT_MAX,input,output,FFTW_REDFT10,FFTW_REDFT01,0),"byte count overflow");
    REJECT(fftw_alloc_real(SIZE_MAX),"allocation byte count overflow");
    REJECT(fftw_alloc_complex(SIZE_MAX),"allocation byte count overflow");
    int ns[3] = {INT_MAX,2,2};
    REJECT(notorious_fft_mkaux_t2t3(3,ns),"legacy int element count overflow");
    REJECT(notorious_fft_mkaux_realdft(3,ns),"legacy int element count overflow");
    REJECT(notorious_fft_mkaux_dft(0,ns),"invalid rank");
    REJECT(notorious_fft_mkaux_t2t3(1,NULL),"invalid dimensions pointer");
    REJECT(notorious_fft_create_plan(SIZE_MAX,0),"FFT index/size overflow");
    REJECT(notorious_fft_create_plan((size_t)INT_MAX,0),"FFT interleaved index overflow");
    REJECT(notorious_fft_create_plan_power2((size_t)1 << (sizeof(size_t)*CHAR_BIT-1)),"FFT index/size overflow");
    size_t total = 0;
    REQUIRE(!notorious_fft_slab_add(&total,SIZE_MAX,sizeof(double)));
    REQUIRE(!notorious_fft_slab_add(&total,(size_t)PTRDIFF_MAX,1));
    total = (size_t)PTRDIFF_MAX-32;
    REQUIRE(!notorious_fft_slab_add(&total,64,1));
    /* NULL-bound plans are useful with execute_r2r replacement arrays. */
    fftw_plan p = fftw_plan_r2r_3d(3,4,2,NULL,NULL,FFTW_REDFT10,FFTW_RODFT11,FFTW_REDFT01,FFTW_UNALIGNED);
    REQUIRE(p);
    fftw_execute(p); /* safe no-op for missing bound arrays */
    fftw_execute_r2r(p,input,output);
    fftw_destroy_plan(p);
    REQUIRE(!live_count);
    fftw_execute(NULL); fftw_destroy_plan(NULL); notorious_fft_free_aux(NULL);
    p=fftw_plan_dft_r2c_1d(7,NULL,NULL,0);
    REQUIRE(p);
    fftw_execute(p);
    fftw_execute_dft_r2c(p,input,(fftw_complex*)output);
    double before[2048]; memcpy(before,output,sizeof(output));
    fftw_execute_dft_c2r(p,(fftw_complex*)input,output); /* wrong plan kind */
    REQUIRE(!memcmp(before,output,sizeof(output)));
    fftw_destroy_plan(p);
    REQUIRE(!live_count);
}

int main(void) {
    for (size_t i = 0; i < sizeof(input)/sizeof(*input); ++i) input[i] = sin((double)i);
    invalid(); injection();
    puts("allocation failure and execution lifetime validation passed");
    return 0;
}
