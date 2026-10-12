// Apply integer group scales before the dot product, using two byte planes.
template <typename TB>
void tinygemm_scaled_k(int M, int KB, const block_q8_K * A, const char * B, float * C, int ldc) {
    const int TS = get_tile_size<TB>();
    alignas(64) static thread_local uint8_t planes[2][2][4096];
    alignas(64) int8_t raw[512];
    alignas(64) int32_t low0[256], high0[256], low1[256], high1[256];
    const __m512i mask = _mm512_set1_epi16(255);
    const int rows = std::min(M, 16);
    ggml_tile_config_init(rows, 64);
    for (int i = 0; i < KB; ++i) {
        for (int n = 0; n < 2; ++n) {
            const char * bp = B + PACKED_INDEX(n, i, KB, TS);
            const uint8_t * scales = (const uint8_t *) bp + get_quants_size<TB>();
            for (int g = 0; g < 8; ++g) {
                unpack_B<TB>(raw, bp, g);
                __m512i scale = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(scales + g * 16)));
                scale = _mm512_or_si512(scale, _mm512_slli_epi32(scale, 16));
                for (int k = 0; k < 8; ++k) {
                    const __m512i q = _mm512_load_si512(raw + k * 64);
                    const __m512i even = _mm512_mullo_epi16(_mm512_and_si512(q, mask), scale);
                    const __m512i odd = _mm512_mullo_epi16(_mm512_srli_epi16(q, 8), scale);
                    const __m512i lo = _mm512_or_si512(_mm512_and_si512(even, mask), _mm512_slli_epi16(odd, 8));
                    const __m512i hi = _mm512_or_si512(_mm512_srli_epi16(even, 8), _mm512_andnot_si512(mask, odd));
                    _mm512_store_si512(planes[n][0] + g * 512 + k * 64, lo);
                    _mm512_store_si512(planes[n][1] + g * 512 + k * 64, hi);
                }
            }
        }
        for (int m = 0; m < M; m += 16) {
            const int nr = std::min(16, M - m);
            // The final partial tile uses the same rows; padded results are ignored.
            _tile_zero(TMM4); _tile_zero(TMM5); _tile_zero(TMM6); _tile_zero(TMM7);
            for (int k = 0; k < 4; ++k) {
                if (nr == rows) {
                    _tile_loadd(TMM2, A[m * KB + i].qs + k * 64, KB * sizeof(block_q8_K));
                } else {
                    alignas(64) int8_t tail[16][64] = {};
                    for (int r = 0; r < nr; ++r) memcpy(tail[r], A[(m+r) * KB + i].qs + k*64, 64);
                    _tile_loadd(TMM2, tail, 64);
                }
                _tile_loadd(TMM0, planes[0][0] + k * 1024, 64);
                _tile_loadd(TMM1, planes[0][1] + k * 1024, 64);
                _tile_dpbsud(TMM4, TMM2, TMM0); _tile_dpbsud(TMM5, TMM2, TMM1);
                _tile_loadd(TMM0, planes[1][0] + k * 1024, 64);
                _tile_loadd(TMM1, planes[1][1] + k * 1024, 64);
                _tile_dpbsud(TMM6, TMM2, TMM0); _tile_dpbsud(TMM7, TMM2, TMM1);
            }
            _tile_stored(TMM4, low0, 64); _tile_stored(TMM5, high0, 64);
            _tile_stored(TMM6, low1, 64); _tile_stored(TMM7, high1, 64);
            for (int r = 0; r < nr; ++r) {
                _mm512_store_si512(low0+r*16, _mm512_add_epi32(_mm512_load_si512(low0+r*16), _mm512_slli_epi32(_mm512_load_si512(high0+r*16), 8)));
                _mm512_store_si512(low1+r*16, _mm512_add_epi32(_mm512_load_si512(low1+r*16), _mm512_slli_epi32(_mm512_load_si512(high1+r*16), 8)));
            }
            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<block_q8_K, TB, is_acc>::apply(C+m*ldc, ldc, low0, &A[m*KB+i], KB, B+PACKED_INDEX(0,i,KB,TS), nr);
                acc_C<block_q8_K, TB, is_acc>::apply(C+m*ldc+16, ldc, low1, &A[m*KB+i], KB, B+PACKED_INDEX(1,i,KB,TS), nr);
            });
        }
    }
}
