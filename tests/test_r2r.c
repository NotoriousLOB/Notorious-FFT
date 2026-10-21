/* Independent direct-formula and optional FFTW differential validation.
 * No FFTW aliases here: native Notorious symbols and FFTW coexist. */
#include "notorious_fft.h"
#ifdef NOTORIOUS_TEST_FFTW
#include <fftw3.h>
#endif

static int failures;
static double worst_abs, worst_rel, worst_rms;
static const double pi = 3.141592653589793238462643383279502884;

static void *checked_alloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "test fixture allocation failure (%zu bytes)\n", bytes); exit(2); }
    return p;
}

static notorious_fft_io_plan *make_plan(int rank, const int *n, double *in, double *out,
                                        const notorious_fft_r2r_kind *k) {
    if (rank == 1) return notorious_fft_plan_r2r_1d(n[0], in, out, k[0], NOTORIOUS_FFT_ESTIMATE);
    if (rank == 2) return notorious_fft_plan_r2r_2d(n[0], n[1], in, out, k[0], k[1], NOTORIOUS_FFT_ESTIMATE);
    return notorious_fft_plan_r2r_3d(n[0], n[1], n[2], in, out, k[0], k[1], k[2], NOTORIOUS_FFT_ESTIMATE);
}

static void legacy_execute(notorious_fft_r2r_kind kind, double *in, double *out, const notorious_fft_aux *aux) {
    switch (kind) {
    case NOTORIOUS_FFT_REDFT10: notorious_fft_dct2(in,out,aux); break;
    case NOTORIOUS_FFT_REDFT01: notorious_fft_dct3(in,out,aux); break;
    case NOTORIOUS_FFT_REDFT11: notorious_fft_dct4(in,out,aux); break;
    case NOTORIOUS_FFT_RODFT10: notorious_fft_dst2(in,out,aux); break;
    case NOTORIOUS_FFT_RODFT01: notorious_fft_dst3(in,out,aux); break;
    case NOTORIOUS_FFT_RODFT11: notorious_fft_dst4(in,out,aux); break;
    default: abort();
    }
}

/* Matrix rows are output coefficients. This oracle never calls library trig,
 * FFT kernels, reorder operations, or the production separable executor. */
static double *matrix(int n, notorious_fft_r2r_kind kind) {
    double *m = (double*)checked_alloc((size_t)n*n*sizeof(*m));
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            long double a = acosl(-1.0L) / n, v = 0;
            switch (kind) {
            case NOTORIOUS_FFT_REDFT10: v = 2*cosl(a*(j+0.5L)*k); break;
            case NOTORIOUS_FFT_REDFT01: v = j ? 2*cosl(a*j*(k+0.5L)) : 1; break;
            case NOTORIOUS_FFT_REDFT11: v = 2*cosl(a*(j+0.5L)*(k+0.5L)); break;
            case NOTORIOUS_FFT_RODFT10: v = 2*sinl(a*(j+0.5L)*(k+1)); break;
            case NOTORIOUS_FFT_RODFT01: v = j == n-1 ? (k%2 ? -1 : 1) : 2*sinl(a*(j+1)*(k+0.5L)); break;
            case NOTORIOUS_FFT_RODFT11: v = 2*sinl(a*(j+0.5L)*(k+0.5L)); break;
            default: abort();
            }
            m[(size_t)k*n+j] = (double)v;
        }
    }
    return m;
}

static void reference(int rank, const int *n, size_t count, double **mat,
                       const double *in, double *out) {
    double *prev = (double*)checked_alloc(count*sizeof(*prev));
    memcpy(prev, in, count*sizeof(*prev));
    for (int axis = 0; axis < rank; ++axis) {
        size_t stride = 1;
        for (int q = axis+1; q < rank; ++q) stride *= (size_t)n[q];
        for (size_t i = 0; i < count; ++i) {
            size_t coord = (i/stride) % (size_t)n[axis];
            size_t start = i - coord*stride;
            double sum = 0;
            for (int j = 0; j < n[axis]; ++j)
                sum += mat[axis][coord*(size_t)n[axis]+j] * prev[start+(size_t)j*stride];
            out[i] = sum;
        }
        memcpy(prev, out, count*sizeof(*prev));
    }
    free(prev);
}

static void compare(const char *what, const double *actual, const double *expected,
                    size_t count, double input_scale) {
    double sq = 0, norm = 0;
    for (size_t i = 0; i < count; ++i) {
        double err = fabs(actual[i]-expected[i]);
        double rel = err / fmax(input_scale, fabs(expected[i]));
        double tol = input_scale*2e-9 + 2e-10*fabs(expected[i]);
        if (err > worst_abs) worst_abs = err;
        if (rel > worst_rel) worst_rel = rel;
        sq += err*err;
        norm += expected[i]*expected[i];
        if (!isfinite(actual[i]) || !isfinite(expected[i]) || err > tol) {
            if (failures < 12) fprintf(stderr, "%s [%zu]: %.17g expected %.17g, abs %.3g tol %.3g\n",
                                       what, i, actual[i], expected[i], err, tol);
            ++failures;
            break;
        }
    }
    double rms = sqrt(sq/fmax(input_scale*input_scale, norm));
    if (rms > worst_rms) worst_rms = rms;
}

static void fill(double *x, size_t n, int pattern, double scale) {
    uint32_t seed = 0x7c4a28u;
    for (size_t i = 0; i < n; ++i) {
        seed = seed*1664525u+1013904223u;
        switch (pattern) {
        case 0: x[i] = 0; break;
        case 1: x[i] = 1; break;
        case 2: x[i] = i == 0; break;
        case 3: x[i] = i == n-1; break;
        case 4: x[i] = i%2 ? -1 : 1; break;
        case 5: x[i] = (double)(seed >> 8)/8388608.0 - 1; break;
        default: x[i] = sin(2*pi*(double)i/(double)n) + (double)i/(double)n; break;
        }
        x[i] *= scale;
    }
}

static void run_case(int rank, const int *n, const notorious_fft_r2r_kind *k) {
    size_t count = 1;
    double *mat[3] = {NULL,NULL,NULL};
    for (int i = 0; i < rank; ++i) { count *= (size_t)n[i]; mat[i] = matrix(n[i],k[i]); }
    double *x = (double*)checked_alloc(count*sizeof(double));
    double *y = (double*)checked_alloc(count*sizeof(double));
    double *ref = (double*)checked_alloc(count*sizeof(double));
    double *alt_storage = (double*)checked_alloc((count+1)*sizeof(double));
    double *alt = alt_storage+1; /* valid double alignment, deliberately not SIMD aligned */
    notorious_fft_io_plan *p = make_plan(rank,n,x,y,k);
    notorious_fft_io_plan *ip = make_plan(rank,n,alt,alt,k);
    notorious_fft_aux *legacy = NULL;
    if (!p || !ip) {
        fprintf(stderr, "constructor failure: r2r_%dd (%d,%d,%d), kinds (%d,%d,%d); valid supported request => allocation/construction\n",
                rank,n[0],rank>1?n[1]:1,rank>2?n[2]:1,k[0],rank>1?k[1]:k[0],rank>2?k[2]:k[0]);
        ++failures;
        goto done;
    }
    if (rank > 1 && k[0] == k[1] && (rank == 2 || k[0] == k[2])) {
        int type4 = k[0] == NOTORIOUS_FFT_REDFT11 || k[0] == NOTORIOUS_FFT_RODFT11;
        if (type4) legacy = rank == 2 ? notorious_fft_mkaux_t4_2d(n[0],n[1]) :
                                       notorious_fft_mkaux_t4_3d(n[0],n[1],n[2]);
        else legacy = rank == 2 ? notorious_fft_mkaux_t2t3_2d(n[0],n[1]) :
                                  notorious_fft_mkaux_t2t3_3d(n[0],n[1],n[2]);
        if (!legacy) { fprintf(stderr,"legacy constructor failure\n"); ++failures; goto done; }
    }
#ifdef NOTORIOUS_TEST_FFTW
    fftw_r2r_kind fk[3];
    for (int i = 0; i < rank; ++i) fk[i] = (fftw_r2r_kind)k[i];
    fftw_plan fp = fftw_plan_r2r(rank,n,x,alt,fk,FFTW_ESTIMATE|FFTW_UNALIGNED);
    if (!fp) { fprintf(stderr,"FFTW oracle constructor failed\n"); exit(2); }
#endif
    worst_abs = worst_rel = worst_rms = 0;
    int patterns = 9 + rank + (rank == 1 && n[0] <= 17 ? n[0] : 0);
    for (int pattern = 0; pattern < patterns; ++pattern) {
        double scale = pattern == 7 ? 1e-8 : pattern == 8 ? 1e8 : 1;
        fill(x,count,pattern > 6 ? 5 : pattern,scale);
        if (pattern >= 9) {
            /* Basis vectors and impulses at each physical axis boundary. */
            size_t pos;
            if (pattern < 9+rank) {
                int axis = pattern-9;
                size_t stride = 1;
                for (int q = axis+1; q < rank; ++q) stride *= (size_t)n[q];
                pos = ((size_t)n[axis]-1)*stride;
            } else pos = (size_t)(pattern-9-rank);
            memset(x,0,count*sizeof(double));
            x[pos] = 1;
        }
        reference(rank,n,count,mat,x,ref);
        memcpy(alt,x,count*sizeof(double));
        notorious_fft_execute(p);
        compare("direct",y,ref,count,scale);
        /* Out-of-place execution must preserve every input element. */
        if (memcmp(x,alt,count*sizeof(double))) { fprintf(stderr,"input modified\n"); ++failures; }
        memcpy(alt,x,count*sizeof(double));
        notorious_fft_execute(ip);
        compare("in-place",alt,ref,count,scale);
        notorious_fft_execute_r2r(p,x,alt);
        compare("alternate arrays",alt,ref,count,scale);
        memcpy(alt,x,count*sizeof(double));
        notorious_fft_execute_r2r(p,alt,alt);
        compare("alternate in-place",alt,ref,count,scale);
        notorious_fft_execute(p);
        compare("repeat",y,ref,count,scale);
        if (rank == 1) {
            /* Exercise legacy entry points directly, including small kernels
             * and the alias-copy branch in the direct fallback. */
            memcpy(alt,x,count*sizeof(double));
            legacy_execute(k[0],alt,alt,p->axis_aux[0]);
            compare("legacy in-place",alt,ref,count,scale);
        }
        if (legacy) {
            legacy_execute(k[0],x,alt,legacy);
            compare("legacy multidimensional",alt,ref,count,scale);
            memcpy(alt,x,count*sizeof(double));
            legacy_execute(k[0],alt,alt,legacy);
            compare("legacy multidimensional in-place",alt,ref,count,scale);
        }
#ifdef NOTORIOUS_TEST_FFTW
        fftw_execute(fp);
        compare("FFTW",y,alt,count,scale);
#endif
    }
#ifdef NOTORIOUS_TEST_FFTW
    fftw_destroy_plan(fp);
#endif
    printf("r2r rank=%d shape=%dx%dx%d kinds=%d,%d,%d abs=%.3g rel=%.3g scaled-rms=%.3g\n",
           rank,n[0],rank>1?n[1]:1,rank>2?n[2]:1,k[0],rank>1?k[1]:k[0],rank>2?k[2]:k[0],
           worst_abs,worst_rel,worst_rms);
done:
    notorious_fft_free_aux(legacy);
    notorious_fft_destroy_io_plan(p); notorious_fft_destroy_io_plan(ip);
    free(x); free(y); free(ref); free(alt_storage);
    for (int i = 0; i < rank; ++i) free(mat[i]);
}

static void roundtrip(int rank, const int *n) {
    notorious_fft_r2r_kind k2[3] = {NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_REDFT10};
    notorious_fft_r2r_kind k3[3] = {NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT01};
    size_t count = 1;
    double scale = 1;
    for (int i = 0; i < rank; ++i) { count *= (size_t)n[i]; scale *= 2.0*n[i]; }
    double *x = (double*)checked_alloc(count*sizeof(double));
    double *y = (double*)checked_alloc(count*sizeof(double));
    double *ref = (double*)checked_alloc(count*sizeof(double));
    fill(x,count,5,1);
    for (size_t i = 0; i < count; ++i) ref[i] = scale*x[i];
    notorious_fft_io_plan *a = make_plan(rank,n,x,y,k2), *b = make_plan(rank,n,y,y,k3);
    if (!a || !b) { fprintf(stderr,"roundtrip constructor failure\n"); ++failures; }
    else { notorious_fft_execute(a); notorious_fft_execute(b); compare("inverse scale",y,ref,count,scale); }
    notorious_fft_destroy_io_plan(a); notorious_fft_destroy_io_plan(b);
    free(x); free(y); free(ref);
}

static void assert_fft_backed(int n) {
    notorious_fft_aux *a = notorious_fft_mkaux_t2t3_1d(n);
    if (!a) {
        fprintf(stderr, "mkaux_t2t3_1d(%d) failed\n", n);
        ++failures;
        return;
    }
    if (n >= 16 && (n & (n - 1)) && (a->direct_r2r || !a->plan ||
        a->r2r_algo == NOTORIOUS_R2R_ALGO_DIRECT)) {
        fprintf(stderr, "NPO2 DCT N=%d used quadratic path (direct=%d plan=%p algo=%d)\n",
                n, a->direct_r2r, (void*)a->plan, a->r2r_algo);
        ++failures;
    }
    notorious_fft_free_aux(a);
}

int main(void) {
    const int lengths[] = {1,2,3,4,5,6,7,8,10,12,13,16,17,31,32,48,63,64,65,127,128,
                           255,256,257,258,509,511,512,513,514,1021,1024,1025};
    const int path_ns[] = {16,17,31,63,64,65,127,128,255,256,257,258,509,511,512,513,514,
                           1021,1024,1025,4093,4096};
    for (size_t i = 0; i < sizeof(path_ns)/sizeof(*path_ns); ++i)
        assert_fft_backed(path_ns[i]);
    const int s2[][2] = {{7,5},{16,16},{17,64},{63,65},{128,256},{32,48},{8,6},{5,7},{64,17},{1,7},{7,1},{1,1},{1,4},{4,1}};
    const int s3[][3] = {{4,8,3},{8,6,4},{8,10,6},{16,12,8},{64,48,32},{2,2,2},{3,3,3},{3,4,8},{8,3,4},{1,3,2},{3,1,2},{3,2,1},{1,1,1},{1,1,4},{1,4,1},{4,1,1}};
    const notorious_fft_r2r_kind all[] = {NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT11,
                                         NOTORIOUS_FFT_RODFT10,NOTORIOUS_FFT_RODFT01,NOTORIOUS_FFT_RODFT11};
    for (size_t i = 0; i < sizeof(lengths)/sizeof(*lengths); ++i) {
        for (int k = 0; k < 6; ++k)
            if (k < 2 || !(lengths[i] & (lengths[i]-1))) run_case(1,lengths+i,all+k);
        roundtrip(1,lengths+i);
    }
    for (size_t i = 0; i < sizeof(s2)/sizeof(*s2); ++i) {
        int pow2 = !(s2[i][0] & (s2[i][0]-1)) && !(s2[i][1] & (s2[i][1]-1));
        for (int k = 0; k < (pow2 ? 6 : 2); ++k) { notorious_fft_r2r_kind ks[2] = {all[k],all[k]}; run_case(2,s2[i],ks); }
        roundtrip(2,s2[i]);
    }
    for (size_t i = 0; i < sizeof(s3)/sizeof(*s3); ++i) {
        int pow2 = 1;
        for (int q = 0; q < 3; ++q) pow2 &= !(s3[i][q] & (s3[i][q]-1));
        for (int k = 0; k < (pow2 ? 6 : 2); ++k) { notorious_fft_r2r_kind ks[3] = {all[k],all[k],all[k]}; run_case(3,s3[i],ks); }
        roundtrip(3,s3[i]);
    }
    /* Every ordered pair/triple of the six supported kinds, including mixed
     * sine/cosine axes. Also rotate a prime-length DCT axis through a tensor. */
    const int mixed2[2] = {4,8}, mixed3[3] = {2,4,8};
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        notorious_fft_r2r_kind ks[3] = {all[i],all[j],all[0]};
        run_case(2,mixed2,ks);
        for (int k = 0; k < 6; ++k) { ks[2] = all[k]; run_case(3,mixed3,ks); }
    }
    for (int axis = 0; axis < 3; ++axis) {
        int ns[3] = {4,4,4}; ns[axis] = 7;
        notorious_fft_r2r_kind ks[3] = {all[2],all[3],all[5]}; ks[axis] = all[axis%2];
        run_case(3,ns,ks);
    }
    /* Deterministic property samples vary rank, dimensions, kinds and unit axes. */
    uint32_t seed = 0x42a115u;
    for (int trial = 0; trial < 64; ++trial) {
        seed = seed*1664525u+1013904223u;
        int rank = 1+(int)((seed >> 8)%3), ns[3];
        notorious_fft_r2r_kind ks[3];
        for (int axis = 0; axis < rank; ++axis) {
            seed = seed*1664525u+1013904223u;
            int kind = (int)((seed >> 8)%6);
            ks[axis] = all[kind];
            ns[axis] = kind < 2 ? 1+(int)((seed >> 16)%17) : 1 << ((seed >> 16)%5);
        }
        run_case(rank,ns,ks);
    }
#ifdef NOTORIOUS_TEST_FFTW
    fftw_cleanup();
#endif
    printf("r2r numerical validation: %d failures\n",failures);
    return failures != 0;
}
