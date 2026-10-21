/* Independent complex DFT oracle for packed real transforms, with optional FFTW. */
#include "notorious_fft.h"
#ifdef NOTORIOUS_TEST_FFTW
#include <fftw3.h>
#endif

typedef struct { double re, im; } reference_complex;
static int failures;
static double max_abs, max_scaled;

static void *fixture(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fputs("real DFT fixture allocation failed\n",stderr); exit(2); }
    return p;
}

static void compare(double actual, double expected, double scale, const char *label, size_t i) {
    double error = fabs(actual-expected), scaled = error/fmax(scale,fabs(expected));
    if (error > max_abs) max_abs = error;
    if (scaled > max_scaled) max_scaled = scaled;
    if (!isfinite(actual) || !isfinite(expected) || error > 2e-9*scale+2e-10*fabs(expected)) {
        if (failures < 16) fprintf(stderr,"%s [%zu]: %.17g expected %.17g\n",label,i,actual,expected);
        ++failures;
    }
}

/* Positive-angle tables, built independently with long-double libm. */
static reference_complex *table(int n) {
    reference_complex *t = (reference_complex*)fixture((size_t)n*n*sizeof(*t));
    for (int k=0;k<n;++k) for (int j=0;j<n;++j) {
        long double angle = 2*acosl(-1.0L)*k*j/n;
        t[(size_t)k*n+j].re = (double)cosl(angle);
        t[(size_t)k*n+j].im = (double)sinl(angle);
    }
    return t;
}

static void direct(int n0, int n1, reference_complex **tables, const reference_complex *in,
                   reference_complex *out, int inverse) {
    size_t count = (size_t)n0*n1;
    reference_complex *prev = (reference_complex*)fixture(count*sizeof(*prev));
    memcpy(prev,in,count*sizeof(*prev));
    /* Leading axis first, unlike the production last-axis-first executor. */
    for (int axis=0;axis<2;++axis) {
        int len = axis ? n1 : n0;
        size_t stride = axis ? 1 : (size_t)n1;
        for (size_t i=0;i<count;++i) {
            size_t k = (i/stride)%(size_t)len, start = i-k*stride;
            double re=0, im=0;
            for (int j=0;j<len;++j) {
                reference_complex x=prev[start+(size_t)j*stride], w=tables[axis][k*(size_t)len+j];
                double s = inverse ? w.im : -w.im;
                re += x.re*w.re-x.im*s;
                im += x.re*s+x.im*w.re;
            }
            out[i].re=re; out[i].im=im;
        }
        memcpy(prev,out,count*sizeof(*prev));
    }
    free(prev);
}

static double random_real(uint32_t *seed) {
    *seed=*seed*1664525u+1013904223u;
    return (double)(*seed>>8)/8388608.0-1;
}

static void signal(double *x, size_t count, int pattern, double scale) {
    uint32_t seed=0x4a12c3u;
    for (size_t i=0;i<count;++i) {
        double value;
        switch (pattern) {
        case 0: value=0; break;
        case 1: value=1; break;
        case 2: value=i==0; break;
        case 3: value=i==count-1; break;
        case 4: value=i%2 ? -1 : 1; break;
        case 5: value=sin(2*acos(-1.0)*(double)i/(double)count); break;
        case 6: value=(double)i/(double)count; break;
        default: value=random_real(&seed); break;
        }
        x[i]=scale*value;
    }
}

static void expand(int n0, int n1, const double *packed, reference_complex *full) {
    size_t nc=(size_t)n1/2+1;
    for (int i=0;i<n0;++i) for (int j=0;j<n1;++j) {
        size_t row = j <= n1/2 ? (size_t)i : (size_t)((n0-i)%n0);
        size_t col = j <= n1/2 ? (size_t)j : (size_t)(n1-j);
        size_t k=row*nc+col;
        full[(size_t)i*n1+j].re=packed[2*k];
        full[(size_t)i*n1+j].im=(j <= n1/2 ? 1 : -1)*packed[2*k+1];
    }
    /* Self-conjugate points are real; the inverse ignores their imaginary parts. */
    for (int i=0;i<n0;++i) for (int j=0;j<n1;++j)
        if ((i==0 || (n0%2==0 && i==n0/2)) && (j==0 || (n1%2==0 && j==n1/2)))
            full[(size_t)i*n1+j].im=0;
}

static void spectrum(int n0, int n1, notorious_fft_cmpl *packed, double scale) {
    size_t nc=(size_t)n1/2+1;
    uint32_t seed=0x8b5431u;
    double *p=(double*)packed;
    for (size_t i=0;i<(size_t)n0*nc;++i) { p[2*i]=scale*random_real(&seed); p[2*i+1]=scale*random_real(&seed); }
    /* The DC/Nyquist planes must be Hermitian across the leading axis. */
    for (int j=0;j<(int)nc;++j) if (j==0 || (n1%2==0 && j==n1/2)) {
        for (int i=0;i<=n0/2;++i) {
            size_t a=(size_t)i*nc+j, b=(size_t)((n0-i)%n0)*nc+j;
            p[2*b]=p[2*a]; p[2*b+1]=a==b ? 0 : -p[2*a+1];
        }
    }
}

static notorious_fft_io_plan *forward(int rank,int n0,int n1,double *x,notorious_fft_cmpl *z) {
    return rank==1 ? notorious_fft_plan_dft_r2c_1d(n1,x,z,NOTORIOUS_FFT_ESTIMATE) :
                    notorious_fft_plan_dft_r2c_2d(n0,n1,x,z,NOTORIOUS_FFT_ESTIMATE);
}
static notorious_fft_io_plan *backward(int rank,int n0,int n1,notorious_fft_cmpl *z,double *x,unsigned flags) {
    return rank==1 ? notorious_fft_plan_dft_c2r_1d(n1,z,x,flags) : notorious_fft_plan_dft_c2r_2d(n0,n1,z,x,flags);
}

static void run_case(int rank,int n0,int n1) {
    size_t count=(size_t)n0*n1, nc=(size_t)n1/2+1, bins=(size_t)n0*nc, physical=2*bins;
    double *x=(double*)fixture(count*sizeof(double)), *y=(double*)fixture(count*sizeof(double));
    /* Exact allocation sizes let ASan detect packing overruns; offset buffers
     * exercise replacement arrays without a SIMD alignment guarantee. */
    double *storage=(double*)fixture((physical+1)*sizeof(double)), *alias=storage+1;
    notorious_fft_cmpl *z=(notorious_fft_cmpl*)fixture(bins*sizeof(*z));
    notorious_fft_cmpl *saved=(notorious_fft_cmpl*)fixture(bins*sizeof(*saved));
    reference_complex *input=(reference_complex*)fixture(count*sizeof(*input));
    reference_complex *ref=(reference_complex*)fixture(count*sizeof(*ref));
    reference_complex *tables[2]={table(n0),table(n1)};
    notorious_fft_io_plan *f=forward(rank,n0,n1,x,z);
    notorious_fft_io_plan *b=backward(rank,n0,n1,z,y,NOTORIOUS_FFT_ESTIMATE|NOTORIOUS_FFT_PRESERVE_INPUT);
    notorious_fft_io_plan *fi=forward(rank,n0,n1,alias,(notorious_fft_cmpl*)alias);
    notorious_fft_io_plan *bi=backward(rank,n0,n1,(notorious_fft_cmpl*)alias,alias,NOTORIOUS_FFT_ESTIMATE);
    if (!f || !b || !fi || !bi) {
        fprintf(stderr,"real planner rejected supported shape rank=%d %dx%d\n",rank,n0,n1); ++failures; goto done;
    }
#ifdef NOTORIOUS_TEST_FFTW
    fftw_complex *oracle_z=(fftw_complex*)fixture(bins*sizeof(*oracle_z));
    double *oracle_x=(double*)fixture(count*sizeof(double));
    double *oracle_alias=(double*)fixture(physical*sizeof(double));
    int ns[2]={n0,n1};
    fftw_plan ff=fftw_plan_dft_r2c(rank,rank==1?ns+1:ns,x,oracle_z,FFTW_ESTIMATE|FFTW_UNALIGNED);
    fftw_plan fb=fftw_plan_dft_c2r(rank,rank==1?ns+1:ns,oracle_z,oracle_x,FFTW_ESTIMATE|FFTW_UNALIGNED);
    fftw_plan ffi=fftw_plan_dft_r2c(rank,rank==1?ns+1:ns,oracle_alias,(fftw_complex*)oracle_alias,FFTW_ESTIMATE|FFTW_UNALIGNED);
    fftw_plan fbi=fftw_plan_dft_c2r(rank,rank==1?ns+1:ns,(fftw_complex*)oracle_alias,oracle_alias,FFTW_ESTIMATE|FFTW_UNALIGNED);
    if (!ff || !fb || !ffi || !fbi) { fputs("FFTW real planner failed\n",stderr); exit(2); }
#endif
    max_abs=max_scaled=0;
    for (int pattern=0;pattern<10;++pattern) {
        double scale=pattern==8 ? 1e-8 : pattern==9 ? 1e8 : 1;
        signal(x,count,pattern,scale);
        for (size_t i=0;i<count;++i) { input[i].re=x[i]; input[i].im=0; }
        direct(n0,n1,tables,input,ref,0);
        notorious_fft_execute(f);
        memcpy(saved,z,bins*sizeof(*z));
        for (size_t i=0;i<count;++i) compare(x[i],input[i].re,scale,"r2c input preserved",i);
        for (size_t row=0;row<(size_t)n0;++row) for (size_t k=0;k<nc;++k) {
            size_t i=row*nc+k, j=row*(size_t)n1+k;
            compare(((double*)z)[2*i],ref[j].re,scale,"r2c real",i);
            compare(((double*)z)[2*i+1],ref[j].im,scale,"r2c imaginary/sign",i);
        }
        expand(n0,n1,(double*)z,input);
        for (size_t i=0;i<count;++i) {
            compare(input[i].re,ref[i].re,scale,"full Hermitian real",i);
            compare(input[i].im,ref[i].im,scale,"full Hermitian imaginary",i);
        }
        for (size_t row=0;row<(size_t)n0;++row) memcpy(alias+row*2*nc,x+row*(size_t)n1,(size_t)n1*sizeof(double));
        notorious_fft_execute(fi);
        for (size_t i=0;i<physical;++i) compare(alias[i],((double*)saved)[i],scale,"padded in-place r2c",i);
        notorious_fft_execute(bi);
        for (size_t row=0;row<(size_t)n0;++row) for (int j=0;j<n1;++j)
            compare(alias[row*2*nc+j],count*x[row*(size_t)n1+j],scale*count,"inverse scale/in-place c2r",row*(size_t)n1+j);
#ifdef NOTORIOUS_TEST_FFTW
        fftw_execute(ff);
        for (size_t i=0;i<physical;++i) compare(((double*)z)[i],((double*)oracle_z)[i],scale,"FFTW r2c",i);
        for (size_t row=0;row<(size_t)n0;++row) memcpy(oracle_alias+row*2*nc,x+row*(size_t)n1,(size_t)n1*sizeof(double));
        fftw_execute(ffi);
        for (size_t i=0;i<physical;++i) compare(((double*)saved)[i],oracle_alias[i],scale,"FFTW padded r2c",i);
        fftw_execute(fbi);
        for (size_t row=0;row<(size_t)n0;++row) for (int j=0;j<n1;++j)
            compare(alias[row*2*nc+j],oracle_alias[row*2*nc+j],scale*count,"FFTW padded c2r",row*(size_t)n1+j);
#endif
        notorious_fft_execute_dft_r2c(f,x,(notorious_fft_cmpl*)alias);
        notorious_fft_execute_dft_r2c(f,x,(notorious_fft_cmpl*)alias);
        for (size_t i=0;i<physical;++i) compare(alias[i],((double*)saved)[i],scale,"alternate/repeated r2c",i);
        /* Unrelated packed coefficients validate inverse direction independently
         * of forward results; the highest odd bin remains complex. */
        spectrum(n0,n1,z,scale);
        if (rank==1) { ((double*)z)[1]=73*scale; if (n1%2==0) ((double*)z)[2*(nc-1)+1]=-41*scale; }
        memcpy(saved,z,bins*sizeof(*z));
        expand(n0,n1,(double*)z,input);
        direct(n0,n1,tables,input,ref,1);
        notorious_fft_execute(b);
        if (memcmp(saved,z,bins*sizeof(*z))) { fputs("c2r modified preserved input\n",stderr); ++failures; }
        for (size_t i=0;i<count;++i) compare(y[i],ref[i].re,scale,"c2r direct positive-sign DFT",i);
        memcpy(alias,saved,bins*sizeof(*z));
        notorious_fft_execute_dft_c2r(b,(notorious_fft_cmpl*)alias,y);
        if (memcmp(alias,saved,bins*sizeof(*z))) { fputs("c2r modified replacement input\n",stderr); ++failures; }
        for (size_t i=0;i<count;++i) compare(y[i],ref[i].re,scale,"alternate c2r",i);
        notorious_fft_execute_dft_c2r(b,(notorious_fft_cmpl*)alias,(double*)alias);
        for (size_t row=0;row<(size_t)n0;++row) for (int j=0;j<n1;++j)
            compare(alias[row*2*nc+j],ref[row*(size_t)n1+j].re,scale,"independent in-place c2r",row*(size_t)n1+j);
#ifdef NOTORIOUS_TEST_FFTW
        memcpy(oracle_z,saved,bins*sizeof(*z)); fftw_execute(fb);
        for (size_t i=0;i<count;++i) compare(y[i],oracle_x[i],scale,"FFTW c2r",i);
#endif
    }
    printf("real DFT rank=%d shape=%dx%d bins=%zu abs=%.3g scaled=%.3g\n",rank,n0,n1,bins,max_abs,max_scaled);
#ifdef NOTORIOUS_TEST_FFTW
    fftw_destroy_plan(ff); fftw_destroy_plan(fb); fftw_destroy_plan(ffi); fftw_destroy_plan(fbi);
    free(oracle_z); free(oracle_x); free(oracle_alias);
#endif
done:
    notorious_fft_destroy_io_plan(f); notorious_fft_destroy_io_plan(b);
    notorious_fft_destroy_io_plan(fi); notorious_fft_destroy_io_plan(bi);
    free(x); free(y); free(storage); free(z); free(saved); free(input); free(ref); free(tables[0]); free(tables[1]);
}

/* The shared legacy auxiliary also supports rank three. Guard the changed
 * packing/staging code with a small full 3D formula, independent of separability. */
static void legacy_3d(int n0,int n1,int n2) {
    size_t count=(size_t)n0*n1*n2, nc=(size_t)n2/2+1, bins=(size_t)n0*n1*nc;
    double *x=(double*)fixture(count*sizeof(double)), *out=(double*)fixture(count*sizeof(double));
    notorious_fft_cmpl *z=(notorious_fft_cmpl*)fixture(bins*sizeof(*z));
    double *alias=(double*)fixture(2*bins*sizeof(double));
    notorious_fft_aux *a=notorious_fft_mkaux_realdft_3d(n0,n1,n2);
    if (!a) { ++failures; goto done; }
    signal(x,count,7,1);
    notorious_fft_realdft(x,z,a);
    for (int k0=0;k0<n0;++k0) for (int k1=0;k1<n1;++k1) for (size_t k2=0;k2<nc;++k2) {
        long double re=0,im=0;
        for (int j0=0;j0<n0;++j0) for (int j1=0;j1<n1;++j1) for (int j2=0;j2<n2;++j2) {
            long double angle=2*acosl(-1.0L)*((long double)j0*k0/n0+(long double)j1*k1/n1+(long double)j2*k2/n2);
            double v=x[((size_t)j0*n1+j1)*n2+j2];
            re+=v*cosl(angle); im-=v*sinl(angle);
        }
        size_t k=((size_t)k0*n1+k1)*nc+k2;
        compare(((double*)z)[2*k],(double)re,1,"legacy 3D real",k);
        compare(((double*)z)[2*k+1],(double)im,1,"legacy 3D imaginary",k);
    }
    notorious_fft_invrealdft(z,out,a);
    for (size_t j=0;j<count;++j) compare(out[j],count*x[j],(double)count,"legacy 3D inverse scale",j);
    for (size_t row=0;row<(size_t)n0*n1;++row) memcpy(alias+row*2*nc,x+row*(size_t)n2,(size_t)n2*sizeof(double));
    notorious_fft_realdft(alias,(notorious_fft_cmpl*)alias,a);
    for (size_t j=0;j<2*bins;++j) compare(alias[j],((double*)z)[j],1,"legacy 3D padded r2c",j);
    notorious_fft_invrealdft((notorious_fft_cmpl*)alias,alias,a);
    for (size_t row=0;row<(size_t)n0*n1;++row) for (int j=0;j<n2;++j)
        compare(alias[row*2*nc+j],count*x[row*(size_t)n2+j],(double)count,"legacy 3D padded c2r",row*(size_t)n2+j);
done:
    notorious_fft_free_aux(a); free(x); free(out); free(z); free(alias);
}

int main(void) {
    const int ns[]={1,2,3,4,5,6,7,8,9,10,12,13,15,16,17,23,29,30,31,32,46,47,48,63,64,65,127,128,129,256,1024};
    const int shapes[][2]={{1,1},{1,7},{7,1},{3,5},{5,3},{4,6},{7,8},{8,7},{16,16},{17,6},{6,17},{4,23},{23,4},{17,13},{8,32},{32,48}};
    for (size_t i=0;i<sizeof(ns)/sizeof(*ns);++i) run_case(1,1,ns[i]);
    for (size_t i=0;i<sizeof(shapes)/sizeof(*shapes);++i) run_case(2,shapes[i][0],shapes[i][1]);
    legacy_3d(2,3,5); legacy_3d(3,2,4); legacy_3d(1,3,1); legacy_3d(3,1,2);
#ifdef NOTORIOUS_TEST_FFTW
    fftw_cleanup();
#endif
    printf("real DFT validation: %d failures\n",failures);
    return failures!=0;
}
