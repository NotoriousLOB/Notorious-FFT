/*
 * Notorious FFT - FFTW-shaped planner / execute API
 *
 * Native names use the notorious_fft_ prefix. include/notorious_fft_fftw.h
 * aliases the FFTW3 basic+advanced (many) symbols onto these.
 */

#ifndef NOTORIOUS_FFT_API_H
#define NOTORIOUS_FFT_API_H

#define NOTORIOUS_FFT_FORWARD          (-1)
#define NOTORIOUS_FFT_BACKWARD         (+1)

#define NOTORIOUS_FFT_MEASURE          (0U)
#define NOTORIOUS_FFT_DESTROY_INPUT    (1U << 0)
#define NOTORIOUS_FFT_UNALIGNED        (1U << 1)
#define NOTORIOUS_FFT_CONSERVE_MEMORY  (1U << 2)
#define NOTORIOUS_FFT_EXHAUSTIVE       (1U << 3)
#define NOTORIOUS_FFT_PRESERVE_INPUT   (1U << 4)
#define NOTORIOUS_FFT_PATIENT          (1U << 5)
#define NOTORIOUS_FFT_ESTIMATE         (1U << 6)

typedef enum {
    NOTORIOUS_FFT_R2HC    = 0,
    NOTORIOUS_FFT_HC2R    = 1,
    NOTORIOUS_FFT_DHT     = 2,
    NOTORIOUS_FFT_REDFT00 = 3,
    NOTORIOUS_FFT_REDFT01 = 4,  /* DCT-III */
    NOTORIOUS_FFT_REDFT10 = 5,  /* DCT-II  */
    NOTORIOUS_FFT_REDFT11 = 6,  /* DCT-IV  */
    NOTORIOUS_FFT_RODFT00 = 7,
    NOTORIOUS_FFT_RODFT01 = 8,  /* DST-III */
    NOTORIOUS_FFT_RODFT10 = 9,  /* DST-II  */
    NOTORIOUS_FFT_RODFT11 = 10  /* DST-IV  */
} notorious_fft_r2r_kind;

typedef enum {
    NOTORIOUS_FFT_IO_DFT = 0,
    NOTORIOUS_FFT_IO_R2C = 1,
    NOTORIOUS_FFT_IO_C2R = 2,
    NOTORIOUS_FFT_IO_R2R = 3
} notorious_fft_io_kind;

typedef struct notorious_fft_io_plan {
    int rank;
    int n[8];
    int sign;
    unsigned flags;
    notorious_fft_io_kind kind;
    notorious_fft_r2r_kind r2r_kind[8];
    void *in;
    void *out;
    notorious_fft_aux *aux;
    int howmany;
    int istride, ostride;
    int idist, odist;
    /* r2r only: independent 1D auxiliaries and reusable gather/scatter lines. */
    notorious_fft_aux *axis_aux[3];
    notorious_fft_real *axis_in, *axis_out;
    size_t r2r_count;
} notorious_fft_io_plan;

notorious_fft_io_plan *notorious_fft_plan_dft_1d(int n, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft_2d(int n0, int n1, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft_3d(int n0, int n1, int n2, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft(int rank, const int *n, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                              int sign, unsigned flags);

notorious_fft_io_plan *notorious_fft_plan_dft_r2c_1d(int n, notorious_fft_real *in, notorious_fft_cmpl *out,
                                                     unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft_c2r_1d(int n, notorious_fft_cmpl *in, notorious_fft_real *out,
                                                     unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft_r2c_2d(int n0, int n1, notorious_fft_real *in, notorious_fft_cmpl *out,
                                                     unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_dft_c2r_2d(int n0, int n1, notorious_fft_cmpl *in, notorious_fft_real *out,
                                                     unsigned flags);

/* Unnormalized separable real-to-real transforms, row-major (last axis
 * contiguous). REDFT10/01 accept all positive int lengths; REDFT11 and
 * RODFT10/01/11 require powers of two (type IV also N <= INT_MAX/2).
 * Other kinds are rejected. Inverse composition scale is the product of 2N
 * over axes. Flags are estimate-like; planning never touches user arrays.
 * NULL means invalid/unsupported dimensions or kinds, overflow, or OOM.
 * Arrays may be identical or disjoint, and may be replaced at execution;
 * partial overlap is unsupported. All scratch is allocated while planning.
 * Distinct plans may execute concurrently; shared plans need synchronization.
 * See docs/r2r.md for formulas and the tested FFTW compatibility subset. */
notorious_fft_io_plan *notorious_fft_plan_r2r_1d(int n, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind kind, unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_r2r_2d(int n0, int n1, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind k0, notorious_fft_r2r_kind k1, unsigned flags);
notorious_fft_io_plan *notorious_fft_plan_r2r_3d(int n0, int n1, int n2, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind k0, notorious_fft_r2r_kind k1,
                                                 notorious_fft_r2r_kind k2, unsigned flags);

notorious_fft_io_plan *notorious_fft_plan_many_dft(int rank, const int *n, int howmany,
                                                   notorious_fft_cmpl *in, const int *inembed,
                                                   int istride, int idist,
                                                   notorious_fft_cmpl *out, const int *onembed,
                                                   int ostride, int odist,
                                                   int sign, unsigned flags);

void notorious_fft_execute(const notorious_fft_io_plan *p);
void notorious_fft_execute_dft(const notorious_fft_io_plan *p, notorious_fft_cmpl *in, notorious_fft_cmpl *out);
void notorious_fft_execute_dft_r2c(const notorious_fft_io_plan *p, notorious_fft_real *in, notorious_fft_cmpl *out);
void notorious_fft_execute_dft_c2r(const notorious_fft_io_plan *p, notorious_fft_cmpl *in, notorious_fft_real *out);
void notorious_fft_execute_r2r(const notorious_fft_io_plan *p, notorious_fft_real *in, notorious_fft_real *out);
void notorious_fft_destroy_io_plan(notorious_fft_io_plan *p);
void notorious_fft_cleanup(void);

#ifdef NOTORIOUS_FFT_IMPLEMENTATION

static notorious_fft_io_plan *notorious_fft_io_plan_new(void) {
    notorious_fft_io_plan *p = (notorious_fft_io_plan *)calloc(1, sizeof(notorious_fft_io_plan));
    if (p) {
        p->howmany = 1;
        p->istride = 1;
        p->ostride = 1;
        p->idist = 1;
        p->odist = 1;
    }
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_dft(int rank, const int *n, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                              int sign, unsigned flags) {
    if (!n || rank < 1 || rank > 8) return NULL;
    for (int i = 0; i < rank; i++)
        if (n[i] <= 0) return NULL;

    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = rank;
    p->sign = sign;
    p->flags = flags;
    p->kind = NOTORIOUS_FFT_IO_DFT;
    p->in = in;
    p->out = out;
    for (int i = 0; i < rank; i++) p->n[i] = n[i];

    if (rank == 1)
        p->aux = notorious_fft_mkaux_dft_1d(n[0]);
    else if (rank == 2)
        p->aux = notorious_fft_mkaux_dft_2d(n[0], n[1]);
    else if (rank == 3)
        p->aux = notorious_fft_mkaux_dft_3d(n[0], n[1], n[2]);
    else {
        int ns[8];
        for (int i = 0; i < rank; i++) ns[i] = n[i];
        p->aux = notorious_fft_mkaux_dft(rank, ns);
    }
    if (!p->aux) {
        free(p);
        return NULL;
    }
    /* MEASURE (default FFTW flags=0): pick iterative vs split-radix on 1D pow2. */
    if (rank == 1 && p->aux->plan && !(flags & NOTORIOUS_FFT_ESTIMATE))
        notorious_fft_tune_plan(p->aux->plan);
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_dft_1d(int n, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags) {
    return notorious_fft_plan_dft(1, &n, in, out, sign, flags);
}

notorious_fft_io_plan *notorious_fft_plan_dft_2d(int n0, int n1, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags) {
    int ns[2] = {n0, n1};
    return notorious_fft_plan_dft(2, ns, in, out, sign, flags);
}

notorious_fft_io_plan *notorious_fft_plan_dft_3d(int n0, int n1, int n2, notorious_fft_cmpl *in, notorious_fft_cmpl *out,
                                                 int sign, unsigned flags) {
    int ns[3] = {n0, n1, n2};
    return notorious_fft_plan_dft(3, ns, in, out, sign, flags);
}

notorious_fft_io_plan *notorious_fft_plan_dft_r2c_1d(int n, notorious_fft_real *in, notorious_fft_cmpl *out,
                                                     unsigned flags) {
    if (n <= 0) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = 1;
    p->n[0] = n;
    p->sign = NOTORIOUS_FFT_FORWARD;
    p->flags = flags;
    p->kind = NOTORIOUS_FFT_IO_R2C;
    p->in = in;
    p->out = out;
    p->aux = notorious_fft_mkaux_realdft_1d(n);
    if (!p->aux) { free(p); return NULL; }
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_dft_c2r_1d(int n, notorious_fft_cmpl *in, notorious_fft_real *out,
                                                     unsigned flags) {
    if (n <= 0) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = 1;
    p->n[0] = n;
    p->sign = NOTORIOUS_FFT_BACKWARD;
    p->flags = flags;
    p->kind = NOTORIOUS_FFT_IO_C2R;
    p->in = in;
    p->out = out;
    p->aux = notorious_fft_mkaux_realdft_1d(n);
    if (!p->aux) { free(p); return NULL; }
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_dft_r2c_2d(int n0, int n1, notorious_fft_real *in, notorious_fft_cmpl *out,
                                                     unsigned flags) {
    if (n0 <= 0 || n1 <= 0) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = 2;
    p->n[0] = n0;
    p->n[1] = n1;
    p->kind = NOTORIOUS_FFT_IO_R2C;
    p->flags = flags;
    p->in = in;
    p->out = out;
    p->aux = notorious_fft_mkaux_realdft_2d(n0, n1);
    if (!p->aux) { free(p); return NULL; }
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_dft_c2r_2d(int n0, int n1, notorious_fft_cmpl *in, notorious_fft_real *out,
                                                     unsigned flags) {
    if (n0 <= 0 || n1 <= 0) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = 2;
    p->n[0] = n0;
    p->n[1] = n1;
    p->kind = NOTORIOUS_FFT_IO_C2R;
    p->flags = flags;
    p->in = in;
    p->out = out;
    p->aux = notorious_fft_mkaux_realdft_2d(n0, n1);
    if (!p->aux) { free(p); return NULL; }
    return p;
}

static notorious_fft_aux *notorious_fft_aux_for_r2r(int n, notorious_fft_r2r_kind k) {
    switch (k) {
    case NOTORIOUS_FFT_REDFT10:
    case NOTORIOUS_FFT_REDFT01:
    case NOTORIOUS_FFT_RODFT10:
    case NOTORIOUS_FFT_RODFT01:
        return notorious_fft_mkaux_t2t3_1d(n);
    case NOTORIOUS_FFT_REDFT11:
    case NOTORIOUS_FFT_RODFT11:
        return notorious_fft_mkaux_t4_1d(n);
    default:
        return NULL;
    }
}

static int notorious_fft_r2r_supported(int n, notorious_fft_r2r_kind kind) {
    if (n <= 0) return 0;
    switch (kind) {
    case NOTORIOUS_FFT_REDFT10:
    case NOTORIOUS_FFT_REDFT01: return 1;
    case NOTORIOUS_FFT_RODFT10:
    case NOTORIOUS_FFT_RODFT01: return !(n & (n - 1));
    case NOTORIOUS_FFT_REDFT11:
    case NOTORIOUS_FFT_RODFT11: return n <= INT_MAX / 2 && !(n & (n - 1));
    default: return 0;
    }
}

static notorious_fft_io_plan *notorious_fft_plan_r2r_axes(int rank, const int *n,
        notorious_fft_real *in, notorious_fft_real *out,
        const notorious_fft_r2r_kind *kinds, unsigned flags) {
    size_t count = 1, max_n = 0, bytes;
    /* Validate every axis and the entire array before allocating anything. */
    for (int i = 0; i < rank; ++i) {
        if (!notorious_fft_r2r_supported(n[i], kinds[i]) ||
            !notorious_fft_size_mul(count, (size_t)n[i], &count)) return NULL;
        if ((size_t)n[i] > max_n) max_n = (size_t)n[i];
    }
    if (!notorious_fft_size_mul(count, sizeof(notorious_fft_real), &bytes) ||
        !notorious_fft_size_mul(max_n, sizeof(notorious_fft_real), &bytes)) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) return NULL;
    p->rank = rank;
    p->r2r_count = count;
    p->kind = NOTORIOUS_FFT_IO_R2R;
    p->flags = flags;
    p->in = in;
    p->out = out;
    for (int i = 0; i < rank; ++i) {
        p->n[i] = n[i];
        p->r2r_kind[i] = kinds[i];
        p->axis_aux[i] = notorious_fft_aux_for_r2r(n[i], kinds[i]);
        if (!p->axis_aux[i]) { notorious_fft_destroy_io_plan(p); return NULL; }
    }
    p->axis_in = (notorious_fft_real*)notorious_fft_malloc(bytes);
    if (rank > 1) p->axis_out = (notorious_fft_real*)notorious_fft_malloc(bytes);
    if (!p->axis_in || (rank > 1 && !p->axis_out)) { notorious_fft_destroy_io_plan(p); return NULL; }
    return p;
}

notorious_fft_io_plan *notorious_fft_plan_r2r_1d(int n, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind kind, unsigned flags) {
    return notorious_fft_plan_r2r_axes(1, &n, in, out, &kind, flags);
}

notorious_fft_io_plan *notorious_fft_plan_r2r_2d(int n0, int n1, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind k0, notorious_fft_r2r_kind k1, unsigned flags) {
    const int ns[2] = {n0, n1};
    const notorious_fft_r2r_kind ks[2] = {k0, k1};
    return notorious_fft_plan_r2r_axes(2, ns, in, out, ks, flags);
}

notorious_fft_io_plan *notorious_fft_plan_r2r_3d(int n0, int n1, int n2, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind k0, notorious_fft_r2r_kind k1,
                                                 notorious_fft_r2r_kind k2, unsigned flags) {
    const int ns[3] = {n0, n1, n2};
    const notorious_fft_r2r_kind ks[3] = {k0, k1, k2};
    return notorious_fft_plan_r2r_axes(3, ns, in, out, ks, flags);
}

notorious_fft_io_plan *notorious_fft_plan_many_dft(int rank, const int *n, int howmany,
                                                   notorious_fft_cmpl *in, const int *inembed,
                                                   int istride, int idist,
                                                   notorious_fft_cmpl *out, const int *onembed,
                                                   int ostride, int odist,
                                                   int sign, unsigned flags) {
    (void)inembed;
    (void)onembed;
    if (howmany <= 0 || istride == 0 || ostride == 0) return NULL;
    notorious_fft_io_plan *p = notorious_fft_plan_dft(rank, n, in, out, sign, flags);
    if (!p) return NULL;
    p->howmany = howmany;
    p->istride = istride;
    p->ostride = ostride;
    p->idist = idist;
    p->odist = odist;
    return p;
}

static void notorious_fft_execute_one_dft(const notorious_fft_io_plan *p,
                                          notorious_fft_cmpl *in, notorious_fft_cmpl *out) {
    if (p->sign == NOTORIOUS_FFT_BACKWARD)
        notorious_fft_invdft(in, out, p->aux);
    else
        notorious_fft_dft(in, out, p->aux);
}

void notorious_fft_execute_dft(const notorious_fft_io_plan *p, notorious_fft_cmpl *in, notorious_fft_cmpl *out) {
    if (!p || !p->aux || !in || !out) return;
    if (p->kind != NOTORIOUS_FFT_IO_DFT) return;

    if (p->howmany == 1 && p->istride == 1 && p->ostride == 1) {
        notorious_fft_execute_one_dft(p, in, out);
        return;
    }

    /* Batched unit-geometry 1D: howmany independent transforms, dist apart. */
    if (p->rank == 1 && p->istride == 1 && p->ostride == 1) {
        int n = p->n[0];
        for (int k = 0; k < p->howmany; k++)
            notorious_fft_execute_one_dft(p, in + k * p->idist, out + k * p->odist);
        (void)n;
        return;
    }

    /* Strided many: gather → 1D → scatter using aux scratch. */
    if (p->rank == 1) {
        int n = p->n[0];
        notorious_fft_cmpl *tmp_in = (notorious_fft_cmpl *)p->aux->scratch_re;
        if (!tmp_in) return;
        /* Need a second scratch for out-of-place strided; reuse plan work via a stack-sized
         * cap only when n is small. For general n, transform in-place in a local buffer
         * allocated on the aux already (scratch_re holds n complexes). Copy in, FFT, scatter. */
        for (int k = 0; k < p->howmany; k++) {
            notorious_fft_cmpl *ik = in + k * p->idist;
            notorious_fft_cmpl *ok = out + k * p->odist;
            for (int i = 0; i < n; i++)
                memcpy(&tmp_in[i], &ik[i * p->istride], sizeof(notorious_fft_cmpl));
            notorious_fft_execute_one_dft(p, tmp_in, tmp_in);
            for (int i = 0; i < n; i++)
                memcpy(&ok[i * p->ostride], &tmp_in[i], sizeof(notorious_fft_cmpl));
        }
        return;
    }

    notorious_fft_execute_one_dft(p, in, out);
}

void notorious_fft_execute_dft_r2c(const notorious_fft_io_plan *p, notorious_fft_real *in, notorious_fft_cmpl *out) {
    if (!p || !p->aux || !in || !out) return;
    notorious_fft_realdft(in, out, p->aux);
}

void notorious_fft_execute_dft_c2r(const notorious_fft_io_plan *p, notorious_fft_cmpl *in, notorious_fft_real *out) {
    if (!p || !p->aux || !in || !out) return;
    if ((p->flags & NOTORIOUS_FFT_PRESERVE_INPUT) && in == (notorious_fft_cmpl *)p->in) {
        /* Preserve by copying into aux scratch when possible. */
        int n = p->n[0];
        size_t bins = (size_t)(n / 2 + 1);
        notorious_fft_cmpl *tmp = (notorious_fft_cmpl *)p->aux->scratch_re;
        if (tmp && p->rank == 1) {
            memcpy(tmp, in, bins * sizeof(notorious_fft_cmpl));
            notorious_fft_invrealdft(tmp, out, p->aux);
            return;
        }
    }
    notorious_fft_invrealdft(in, out, p->aux);
}

static void notorious_fft_execute_axis(notorious_fft_r2r_kind kind, notorious_fft_aux *aux,
                                       notorious_fft_real *in, notorious_fft_real *out) {
    switch (kind) {
    case NOTORIOUS_FFT_REDFT10: notorious_fft_dct2(in, out, aux); break;
    case NOTORIOUS_FFT_REDFT01: notorious_fft_dct3(in, out, aux); break;
    case NOTORIOUS_FFT_REDFT11: notorious_fft_dct4(in, out, aux); break;
    case NOTORIOUS_FFT_RODFT10: notorious_fft_dst2(in, out, aux); break;
    case NOTORIOUS_FFT_RODFT01: notorious_fft_dst3(in, out, aux); break;
    case NOTORIOUS_FFT_RODFT11: notorious_fft_dst4(in, out, aux); break;
    default: break;
    }
}

void notorious_fft_execute_r2r(const notorious_fft_io_plan *p, notorious_fft_real *in, notorious_fft_real *out) {
    if (!p || p->kind != NOTORIOUS_FFT_IO_R2R || !in || !out) return;
    int last = p->rank - 1;
    size_t stride = (size_t)p->n[last];
    /* Transform contiguous rows directly into output. Only exact aliasing
     * needs an input copy; kernels always receive disjoint input and output. */
    for (size_t base = 0; base < p->r2r_count; base += stride) {
        notorious_fft_real *src = in + base;
        if (in == out) {
            memcpy(p->axis_in, src, stride * sizeof(*src));
            src = p->axis_in;
        }
        notorious_fft_execute_axis(p->r2r_kind[last], p->axis_aux[last], src, out + base);
    }
    /* Gather complete strided lines before scattering into the same tensor. */
    for (int axis = last - 1; axis >= 0; --axis) {
        size_t len = (size_t)p->n[axis];
        size_t block = len * stride;
        for (size_t base = 0; base < p->r2r_count; base += block) {
            for (size_t offset = 0; offset < stride; ++offset) {
                for (size_t j = 0; j < len; ++j)
                    p->axis_in[j] = out[base + offset + j * stride];
                notorious_fft_execute_axis(p->r2r_kind[axis], p->axis_aux[axis], p->axis_in, p->axis_out);
                for (size_t j = 0; j < len; ++j)
                    out[base + offset + j * stride] = p->axis_out[j];
            }
        }
        stride = block;
    }
}

void notorious_fft_execute(const notorious_fft_io_plan *p) {
    if (!p) return;
    switch (p->kind) {
    case NOTORIOUS_FFT_IO_DFT:
        notorious_fft_execute_dft(p, (notorious_fft_cmpl *)p->in, (notorious_fft_cmpl *)p->out);
        break;
    case NOTORIOUS_FFT_IO_R2C:
        notorious_fft_execute_dft_r2c(p, (notorious_fft_real *)p->in, (notorious_fft_cmpl *)p->out);
        break;
    case NOTORIOUS_FFT_IO_C2R:
        notorious_fft_execute_dft_c2r(p, (notorious_fft_cmpl *)p->in, (notorious_fft_real *)p->out);
        break;
    case NOTORIOUS_FFT_IO_R2R:
        notorious_fft_execute_r2r(p, (notorious_fft_real *)p->in, (notorious_fft_real *)p->out);
        break;
    }
}

void notorious_fft_destroy_io_plan(notorious_fft_io_plan *p) {
    if (!p) return;
    notorious_fft_free_aux(p->aux);
    for (int i = 0; i < 3; ++i) notorious_fft_free_aux(p->axis_aux[i]);
    notorious_fft_free(p->axis_in);
    notorious_fft_free(p->axis_out);
    free(p);
}

void notorious_fft_cleanup(void) {
    /* No global state. */
}

#endif /* NOTORIOUS_FFT_IMPLEMENTATION */

#endif /* NOTORIOUS_FFT_API_H */
