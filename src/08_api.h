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

notorious_fft_io_plan *notorious_fft_plan_r2r_1d(int n, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind kind, unsigned flags);

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

notorious_fft_io_plan *notorious_fft_plan_r2r_1d(int n, notorious_fft_real *in, notorious_fft_real *out,
                                                 notorious_fft_r2r_kind kind, unsigned flags) {
    if (n <= 0) return NULL;
    notorious_fft_aux *aux = notorious_fft_aux_for_r2r(n, kind);
    if (!aux) return NULL;
    notorious_fft_io_plan *p = notorious_fft_io_plan_new();
    if (!p) { notorious_fft_free_aux(aux); return NULL; }
    p->rank = 1;
    p->n[0] = n;
    p->kind = NOTORIOUS_FFT_IO_R2R;
    p->r2r_kind[0] = kind;
    p->flags = flags;
    p->in = in;
    p->out = out;
    p->aux = aux;
    return p;
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

void notorious_fft_execute_r2r(const notorious_fft_io_plan *p, notorious_fft_real *in, notorious_fft_real *out) {
    if (!p || !p->aux || !in || !out) return;
    switch (p->r2r_kind[0]) {
    case NOTORIOUS_FFT_REDFT10: notorious_fft_dct2(in, out, p->aux); break;
    case NOTORIOUS_FFT_REDFT01: notorious_fft_dct3(in, out, p->aux); break;
    case NOTORIOUS_FFT_REDFT11: notorious_fft_dct4(in, out, p->aux); break;
    case NOTORIOUS_FFT_RODFT10: notorious_fft_dst2(in, out, p->aux); break;
    case NOTORIOUS_FFT_RODFT01: notorious_fft_dst3(in, out, p->aux); break;
    case NOTORIOUS_FFT_RODFT11: notorious_fft_dst4(in, out, p->aux); break;
    default: break;
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
    free(p);
}

void notorious_fft_cleanup(void) {
    /* No global state. */
}

#endif /* NOTORIOUS_FFT_IMPLEMENTATION */

#endif /* NOTORIOUS_FFT_API_H */
