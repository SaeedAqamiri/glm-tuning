from pathlib import Path
root=Path('/home/saeed/glm-tuning-20261011/llama/ggml/src/ggml-cpu/amx')
p=root/'amx.cpp';s=p.read_text()
s=s.replace('''        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);''','''        if (op->op == GGML_OP_MUL_MAT_ID) {
            ggml_backend_amx_mul_mat_id(params, op);
            return true;
        }
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);''')
s=s.replace('''        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }''','''        const bool moe = op->op == GGML_OP_MUL_MAT_ID;
        if (op->op != GGML_OP_MUL_MAT && !moe) {
            return false;
        }
        if (moe && !getenv("GGML_AMX_MOE")) {
            return false;
        }''')
s=s.replace('''        if (qtype_has_amx_kernels(src0->type) && (src0->ne[2] != 1 || src0->ne[3] != 1)) {
            return false;
        }''','''        if (qtype_has_amx_kernels(src0->type) && ((!moe && src0->ne[2] != 1) || src0->ne[3] != 1)) {
            return false;
        }
        if (moe && (!qtype_has_amx_kernels(src0->type) || src1->ne[3] != 1 ||
                    (src1->ne[1] != 1 && src1->ne[1] != op->src[2]->ne[0]))) {
            return false;
        }''')
p.write_text(s)
p=root/'mmq.h';s=p.read_text();s+='\nvoid ggml_backend_amx_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);\n';p.write_text(s)
p=root/'mmq.cpp';s=p.read_text();s=s.replace('return N * row_size_B;', 'return N * row_size_B * tensor->ne[2] * tensor->ne[3];')
s=s.replace('''        convert_B_packed_format<type, blck_size>((void *)((char *)tensor->data + offset), (const type *)data, N, K);''','''        const size_t packed_bytes = N * get_row_size<type, blck_size>(K);
        for (int64_t e = 0; e < tensor->ne[2] * tensor->ne[3]; ++e) {
            convert_B_packed_format<type, blck_size>((char *) tensor->data + e * packed_bytes,
                (const type *) ((const char *) data + e * tensor->nb[2]), N, K);
        }''')
s=s.replace('''    const int M = dst->ne[1];
    const int K = src0->ne[0];''','''    if (dst->op == GGML_OP_MUL_MAT_ID) {
        const int64_t rows = dst->src[2]->ne[0] * dst->src[2]->ne[1];
        const int64_t experts = src0->ne[2];
        const size_t meta = GGML_PAD((3 * experts + 2 + rows) * sizeof(int32_t), 64);
        size_t row_bytes = 0;
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_bytes = src0->ne[0] / blck_size * sizeof(vec_dot_type);
        });
        return meta + rows * row_bytes;
    }

    const int M = dst->ne[1];
    const int K = src0->ne[0];''',1)
idx=s.rindex('#endif // if defined(__AMX_INT8__)')
code=r'''
// Use the existing packed kernels for each expert's selected token rows.
void ggml_backend_amx_mul_mat_id(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int K = w->ne[0], N = w->ne[1], E = w->ne[2];
    const int S = ids->ne[0], T = ids->ne[1], R = S * T;
    const int TYPE = w->type;
    constexpr int BM = 32, BN = 32;
    const int NB = N / BN;
    int32_t * counts = (int32_t *) params->wdata;
    int32_t * starts = counts + E;
    int32_t * tasks = starts + E + 1;
    int32_t * rows = tasks + E + 1;
    const size_t meta = GGML_PAD((3 * E + 2 + R) * sizeof(int32_t), 64);
    char * qa = (char *) params->wdata + meta;
    if (params->ith == 0) {
        std::fill(counts, counts + E, 0);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < S; ++j) {
                int e = *(int32_t *) ((char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
                GGML_ASSERT(e >= 0 && e < E);
                ++counts[e];
            }
        }
        starts[0] = tasks[0] = 0;
        for (int e = 0; e < E; ++e) {
            starts[e + 1] = starts[e] + counts[e];
            tasks[e + 1] = tasks[e] + div_up(counts[e], BM) * NB;
            counts[e] = 0;
        }
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < S; ++j) {
                int e = *(int32_t *) ((char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
                rows[starts[e] + counts[e]++] = t * S + j;
            }
        }
    }
    ggml_barrier(params->threadpool);
    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int KB = K / blck_size;
        const size_t row_bytes = KB * sizeof(vec_dot_type);
        const size_t expert_bytes = N * get_row_size<type, blck_size>(K);
        const int tile_bytes = get_tile_size<type>();
        GGML_ASSERT(params->wsize >= meta + R * row_bytes);
        for (int r = params->ith; r < R; r += params->nth) {
            int t = rows[r] / S, j = rows[r] % S;
            const float * xr = (const float *) ((const char *) x->data + t * x->nb[2] + (j % x->ne[1]) * x->nb[1]);
            from_float<vec_dot_type>(xr, qa + r * row_bytes, K);
        }
        ggml_barrier(params->threadpool);
        ggml_tile_config_init();
        alignas(64) float result[BM * BN];
        const int begin = (int64_t) tasks[E] * params->ith / params->nth;
        const int end = (int64_t) tasks[E] * (params->ith + 1) / params->nth;
        for (int job = begin; job < end; ++job) {
            int e = std::upper_bound(tasks, tasks + E + 1, job) - tasks - 1;
            int local = job - tasks[e];
            int mb = local / NB, nb = local % NB;
            int nr = std::min(BM, counts[e] - mb * BM);
            const char * ap = qa + (starts[e] + mb * BM) * row_bytes;
            const char * bp = (const char *) w->data + e * expert_bytes + PACKED_INDEX(nb * 2, 0, KB, tile_bytes);
            if (nr == 1) {
                tinygemm_kernel_vnni<vec_dot_type, type, float, 1, BN, blck_size>::apply(KB, ap, bp, result, BN);
            } else {
                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(nr, BN, KB, ap, bp, result, BN);
            }
            for (int m = 0; m < nr; ++m) {
                int r = rows[starts[e] + mb * BM + m];
                char * yp = (char *) dst->data + (r / S) * dst->nb[2] + (r % S) * dst->nb[1] + nb * BN * sizeof(float);
                memcpy(yp, result + m * BN, BN * sizeof(float));
            }
        }
        _tile_release();
    });
}

'''
s=s[:idx]+code+s[idx:];p.write_text(s)
