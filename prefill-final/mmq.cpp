#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "amx.h"
#include "mmq.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-quants.h"
#include <algorithm>
#include <type_traits>

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/mempolicy.h>
#include <cerrno>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

namespace {

// Forced unrolling
template <int n>
struct Unroll {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<int, n - 1>{}, args...);
    }
};

template <>
struct Unroll<1> {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        f(std::integral_constant<int, 0>{}, args...);
    }
};

#if defined(GGML_AMX_Q4_EXPANDED)
constexpr int q4_weight_bytes = QK_K;
#else
constexpr int q4_weight_bytes = QK_K/2;
#endif

#if defined(GGML_AMX_Q8_UNSIGNED)
constexpr bool q8_unsigned = true;
#else
constexpr bool q8_unsigned = false;
#endif

// Exact 128*sum(qs), shared across every output column of an input block.
static inline int32_t q8_input_compensation(const block_q8_0 * a) {
 const __m256i pairs=_mm256_maddubs_epi16(_mm256_set1_epi8(1),_mm256_loadu_si256((const __m256i *)a->qs));
 const __m256i sums=_mm256_madd_epi16(pairs,_mm256_set1_epi16(1));
 __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1));
 sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);
 return _mm_cvtsi128_si32(sum)*128;
}

// type traits
template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> { using type = int8_t; };
template <> struct PackedTypes<block_q4_1> { using type = uint8_t; };
template <> struct PackedTypes<block_q8_0> { using type = int8_t; };
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

template <typename T>
struct do_compensate : std::integral_constant<bool,
    std::is_same<T, block_q8_0>::value && !q8_unsigned> {};

template <typename T>
struct do_unpack : std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value> {};

template <typename T>
struct is_type_qkk : std::integral_constant<bool,
    std::is_same<T, block_q4_K>::value ||
    std::is_same<T, block_q5_K>::value ||
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_iq4_xs>::value> {};

#define GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        switch (TYPE) {                                                                \
            case GGML_TYPE_F16: {                                                      \
                using type = ggml_fp16_t;                                              \
                constexpr int blck_size = 16;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_BF16: {                                                     \
                using type = ggml_bf16_t;                                              \
                constexpr int blck_size = 32;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#define GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        switch (QT) {                                                                  \
            case GGML_TYPE_Q4_0: {                                                     \
                using type = block_q4_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK4_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_1: {                                                     \
                using type = block_q4_1;                                               \
                using vec_dot_type = block_q8_1;                                       \
                constexpr int blck_size = QK4_1;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q8_0: {                                                     \
                using type = block_q8_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK8_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_K: {                                                     \
                using type = block_q4_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q5_K: {                                                     \
                using type = block_q5_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q6_K: {                                                     \
                using type = block_q6_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ4_XS: {                                                   \
                using type = block_iq4_xs;                                             \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", int(TYPE));   \
        }                                                                              \
    }()

#define GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        if (BOOL_V) {                                                                  \
            constexpr bool BOOL_NAME = true;                                           \
            return __VA_ARGS__();                                                      \
        } else {                                                                       \
            constexpr bool BOOL_NAME = false;                                          \
            return __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
struct tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//    Block A: {16, 32}, dtype = int8_t
//    Block B: {16, 32}, dtype = uint8_t/int8_t
//    Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//    packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the single batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//    advanced-matrix-extensions-intrinsics-functions.html
//

inline void ggml_tile_config_init(int rows = 16, int width = 32) {
    static thread_local int previous_config = 0;

    if (previous_config == rows + width * 32) {
        return;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row = 0;
    tc.rows[0] = width / 4;   tc.colsb[0] = 64;
    tc.rows[1] = width / 4;   tc.colsb[1] = 64;
    tc.rows[2] = rows;  tc.colsb[2] = width;
    tc.rows[3] = rows;  tc.colsb[3] = width;
    tc.rows[4] = rows;  tc.colsb[4] = 64;
    tc.rows[5] = rows;  tc.colsb[5] = 64;
    tc.rows[6] = rows;  tc.colsb[6] = 64;
    tc.rows[7] = rows;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    previous_config = rows + width * 32;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
template <typename TB>
int get_tile_size() {
    int tile_size = TILE_N * sizeof(TB);
    if (do_compensate<TB>::value) {
        tile_size += TILE_N * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        tile_size += TILE_N * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        tile_size += TILE_N * 2;
    }
    if (std::is_same<TB, block_q4_K>::value) tile_size += TILE_N*(q4_weight_bytes - QK_K/2);
    return tile_size;
}

template <typename TB, int BLOCK_K>
int get_row_size(int K) {
    int KB = K / BLOCK_K;
    int row_size = KB * sizeof(TB);
    if (do_compensate<TB>::value) {
        row_size += KB * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        row_size += KB * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        row_size += KB * 2;
    }
    if (std::is_same<TB, block_q4_K>::value) row_size += KB*(q4_weight_bytes - QK_K/2);
    return row_size;
}

// transpose utils
#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    // unpacking and 32-bit elements
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    // shuffling the 32-bit elements
    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    // shuffling 128-bit elements
    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

inline void transpose_16x4_32bit(__m512i * r, __m512i * d) {

    static const __m512i index1 = _mm512_set_epi32(
        0x0f, 0x0b, 0x07, 0x03,
        0x0e, 0x0a, 0x06, 0x02,
        0x0d, 0x09, 0x05, 0x01,
        0x0c, 0x08, 0x04, 0x00);

    d[0] = _mm512_permutexvar_epi32(index1, r[0]);
    d[1] = _mm512_permutexvar_epi32(index1, r[1]);
    d[2] = _mm512_permutexvar_epi32(index1, r[2]);
    d[3] = _mm512_permutexvar_epi32(index1, r[3]);

    r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
    r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
    r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
    r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

    d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
    d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
    d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
    d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);
}

inline void transpose_16x16_32bit(__m512i * v) {
    __m512i v1[16];
    v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
    v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
    v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
    v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
    v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
    v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
    v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
    v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
    v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

    v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
    v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
    v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
    v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
    v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
    v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
    v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
    v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
    v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
    v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
    v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
    v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
    v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
    v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
    v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
    v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

    v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
    v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
    v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
    v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
    v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
    v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
    v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
    v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
    v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
    v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
    v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
    v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
    v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
    v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
    v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
    v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

    v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
    v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
    v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
    v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
    v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
    v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
    v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
    v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
    v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
    v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
    v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
    v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
    v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
    v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
    v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
    v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

void quantize_row_q8_K_vnni(const float * RESTRICT x, void * RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int KB = k / QK_K;
    constexpr int kVecs = QK_K / 16;

    block_q8_K * y = reinterpret_cast<block_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    const __m512 signBit = _mm512_set1_ps(-0.f);

    for (int i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(signBit, v[j]));
        }
        const float amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        const float iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_TO_FP16(1 / iscale);
        const float id = ( amax != 0.0f ) ? iscale : 0.f;
        const __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        for (int j = 0; j < kVecs / 4; ++j) {
            __m128i q8_0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 0]));
            __m128i q8_1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 1]));
            __m128i q8_2 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 2]));
            __m128i q8_3 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 3]));

            __m256i q8_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_0), (q8_1), 1);
            __m256i q8_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_2), (q8_3), 1);

            vq[j] = _mm512_inserti32x8(_mm512_castsi256_si512(q8_01), q8_23, 1);
            _mm512_storeu_si512((__m512i *)(y[i].qs + j * 64), vq[j]);
        }

        // Compute the bsums with vnni
        transpose_16x4_32bit(vq, vq_packed);

        const __m512i one = _mm512_set1_epi8(1);
        __m512i sum = _mm512_setzero_si512();
        for (int k = 0; k < 4; ++k) {
            sum = _mm512_dpbusd_epi32(sum, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(sum));
    }
}

// quantize A from float to `vec_dot_type`
template <typename T>
inline void from_float(const float * x, char * vy, int64_t k);

template <>
inline void from_float<block_q8_0>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_0(x, (block_q8_0 *)vy, k);
}

template <>
inline void from_float<block_q8_1>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_1(x, (block_q8_1 *)vy, k);
}

template <>
inline void from_float<block_q8_K>(const float * x, char * vy, int64_t k) {
#if 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (block_q8_K *)vy, k);
#else
    quantize_row_q8_K_vnni(x, vy, k);
#endif
}

// load A from memory to array when nrows can not fill in whole tile
void unpack_A(int8_t * RESTRICT tile, const block_q8_0 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

void unpack_A(int8_t * RESTRICT tile, const block_q8_1 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <typename TB>
void unpack_A(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <>
void unpack_A<block_q6_K>(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    const __m128i zero = _mm_setzero_si128();
    for (int m = 0; m < nr; ++m) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(A[m * lda].qs + k * 16));
        const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// used for block_q4_K
inline __m512i bytes_from_nibbles_64(const uint8_t * rsi) {
    const __m256i tmp = _mm256_loadu_si256((const __m256i *)rsi);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    const __m256i q4l = _mm256_and_si256(tmp, lowMask);
    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), lowMask);
    return _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
inline __m512i bytes_from_nibbles_64(const uint8_t * qs, const uint8_t * qh, int k) {
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    const __m256i q5bits = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i hbits = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q5l_0 = _mm256_and_si256(q5bits, lowMask);
    const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), lowMask);
    const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);

    return _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
inline void bytes_from_nibbles_128(__m512i& r0, __m512i& r1, const uint8_t * qs, const uint8_t * qh) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x3);

    const __m256i q6bits1 = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i q6bits2 = _mm256_loadu_si256((const __m256i *)(qs + 32));
    const __m256i q6bitsH = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    const __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    const __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    const __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

inline __m512i packNibbles(__m512i r0, __m512i r1) {
    return _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

template <typename TB>
inline void pack_qs(void * RESTRICT packed_B, const TB * RESTRICT B, int KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    for (int n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
    }
}

template <>
inline void pack_qs<block_q8_0>(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64 + 32), v2[n]);
    }
}

template <>
inline void pack_qs<block_q4_K>(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        if (q4_weight_bytes == QK_K) {
            for (int n = 0; n < TILE_N; ++n) { _mm512_storeu_si512((__m512i *)pb, v[n]); pb += 64; }
        } else {
            for (int n = 0; n < TILE_N; n += 2) {
                _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1])); pb += 64;
            }
        }
    }
}

template <>
inline void pack_qs<block_q5_K>(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    __m512i v[16];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /* group */2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        for (int n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        const __m512i hmask = _mm512_set1_epi8(0x10);
        for (int g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_q6_K>(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    __m512i v[32];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 128; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            bytes_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        for (int n = 0; n < 32; n += 2) {
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        const __m512i hmask = _mm512_set1_epi8(0x30);
        for (int g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_iq4_xs>(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    __m512i v[16];
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            __m256i r0 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        for (int n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
void pack_B(void * RESTRICT packed_B, const block_q4_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

void pack_B(void * RESTRICT packed_B, const block_q4_1 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

inline void s8s8_compensation(void * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    const int offset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((char *)(packed_B) + offset), vcomp);
}

void pack_B(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    if constexpr(q8_unsigned) {
        const __m512i sign=_mm512_set1_epi8((char)0x80);
        for(int k=0;k<8;++k) {
            char * ptr=(char *)packed_B+k*64;
            _mm512_storeu_si512((__m512i *)ptr,_mm512_xor_si512(_mm512_loadu_si512((const __m512i *)ptr),sign));
        }
    } else s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
inline void unpack_mins_and_scales(const uint8_t * scales, uint32_t * utmp) {
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + q4_weight_bytes * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   qh     {8, TILE_N,  4}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {16, TILE_N, 8}  uint8
//   qh     {16, TILE_N, 4}  uint8
//   scales {16, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        const int8_t * ps = B[n * KB].scales;
        for (int k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    for (int n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        for (int k = 0; k < 8; k += 2) {
            const int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

template<typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t * RESTRICT tile, const void * RESTRICT packed_B) {
    GGML_UNUSED(tile);
    GGML_UNUSED(packed_B);
}

template <>
void unpack_B<block_q8_0>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
    const __m512i offset = _mm512_set1_epi8((char) 0x80);
    for (int i = 0; i < 8; ++i) {
        const __m512i bytes = _mm512_loadu_si512((const char *) packed_B + i * 64);
        _mm512_storeu_si512(tile + i * 64, _mm512_xor_si512(bytes, offset));
    }
    // Tile loads do not expose their source memory to GCC.
    __asm__ volatile("" ::: "memory");
}

template <>
void unpack_B<block_q4_0>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
  const __m512i off = _mm512_set1_epi8(8);
  const __m512i lowMask = _mm512_set1_epi8(0xF);
  for (int n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
    const __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(bytes, lowMask), off);
    const __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

template <>
void unpack_B<block_q4_1>(uint8_t * RESTRICT tile, const void * RESTRICT packed_B) {
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
template <typename TB>
void unpack_B(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    if (std::is_same<TB, block_q4_K>::value && q4_weight_bytes == QK_K) {
        memcpy(tile, (const char *) packed_B + k*TILE_N*32, TILE_N*32);
        return;
    }
    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * packed_B_group = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(packed_B_group + n * 32);
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q5_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 256 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 1bit, stride 64 bytes
    const int packed_h1_group_size = QK_K / 8 * TILE_N / 8;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(bytes, lowMask);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = _mm512_add_epi8(r0, h0);
        r1 = _mm512_add_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q6_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 128 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 16;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 2bits, stride 64 bytes
    const int packed_h2_group_size = QK_K / 4 * TILE_N / 16;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i off = _mm512_set1_epi8(32);
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i bytes = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(bytes, lowMask);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    bytes = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(bytes, lowMask);
    r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));
}

template <>
void unpack_B<block_iq4_xs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    static const __m512i values128 = _mm512_set_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        const __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(bytes, lowMask));
        const __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <typename TA, typename TB, bool is_acc>
struct acc_C {};

template <bool is_acc>
struct acc_C<block_q8_0, block_q4_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_1, block_q4_1, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_1 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));
        const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset + TILE_N * sizeof(ggml_half))));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].s));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_0, block_q8_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr, bool signed_q8 = false) {
        const int offset = TILE_N * TILE_K;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            __m512i vi=_mm512_loadu_si512(tile+m*TILE_N);
            if (q8_unsigned && !signed_q8) vi=_mm512_sub_epi32(vi,_mm512_set1_epi32(q8_input_compensation(&A[m*lda])));
            const __m512 vtile=_mm512_cvtepi32_ps(vi);

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q4_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + q4_weight_bytes * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q5_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q6_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 16 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_iq4_xs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const int8_t * scales = reinterpret_cast<const int8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 8 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <typename TB> constexpr int get_quants_size();
template <> constexpr int get_quants_size<block_q4_K>() { return q4_weight_bytes * TILE_N; }
template <> constexpr int get_quants_size<block_q5_K>() { return (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
template <> constexpr int get_quants_size<block_q6_K>() { return (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
template <> constexpr int get_quants_size<block_iq4_xs>() { return (QK_K / 2) * TILE_N; }

// used for QKK format
template <typename TB, bool is_acc,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
inline void scale_C(const int32_t * RESTRICT tile, int32_t * RESTRICT sumi, const void * packed_B, int k, int nr) {
    const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + get_quants_size<TB>());
    const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(scales + k * TILE_N)));

    for (int m = 0; m < nr; ++m) {
        __m512i vsumi;
        if (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } else {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = _mm512_add_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}


template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx {
    static void apply(int K, const TA * RESTRICT A, const TB * RESTRICT B, TC * RESTRICT C, int ldc) {
        GGML_UNUSED(K);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        GGML_UNUSED(C);
        GGML_UNUSED(ldc);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx<float, ggml_fp16_t, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int K, const float * RESTRICT A, const ggml_fp16_t * RESTRICT B, float * RESTRICT C, int ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N;
        assert(BLOCK_K == 16);

        __m512 va;
        __m512 vb[COLS];
        __m512 vc[ROWS * COLS];

        auto loadc = [&](auto idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ROWS * COLS>{}(loadc);

        auto compute = [&](auto idx, auto k) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;

            if constexpr (col == 0) {
                va = _mm512_loadu_ps(A + row * K + k);
            }
            if constexpr (row == 0) {
                vb[col] =  _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(B + col * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        };

        for (int k = 0; k < K; k += 16) {
            Unroll<ROWS * COLS>{}(compute, k);
        }

        auto storec = [&](auto idx) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;
            C[row * ldc + col] = _mm512_reduce_add_ps(vc[idx]);
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_AVX(MB_SIZE, NB_SIZE)                                \
    tinygemm_kernel_avx<float, type, float, MB_SIZE, NB_SIZE, blck_size>::apply(    \
        K, (const float *)((const char *)src1->data + src1_offset) + mb_start * K,                  \
        (const type *)((const char *)src0->data + src0_offset) + nb_start * K,                      \
        (float *)((char *)dst->data + dst_offset) + mb_start * ldc + nb_start, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (((int64_t)(n) * (KB) + (k)) * (int64_t)(tile_size))

template<int BN>
void tinygemm_q8_unsigned_m1(int KB,const void * aa,const void * bb,float * C) {
 constexpr int COLS=BN/16;const int TS=get_tile_size<block_q8_0>();
 const auto * A=(const block_q8_0 *)aa;const char * B=(const char *)bb;
 __m512 c[COLS];Unroll<COLS>{}([&](auto col){c[col]=_mm512_setzero_ps();});
 for(int i=0;i<KB;++i){
  __m512i e[COLS],o[COLS];Unroll<COLS>{}([&](auto col){e[col]=o[col]=_mm512_setzero_si512();});
  const int32_t * ap=(const int32_t *)A[i].qs;
  for(int k=0;k<8;k+=2){
   const __m512i a0=_mm512_set1_epi32(ap[k]),a1=_mm512_set1_epi32(ap[k+1]);
   Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);
    e[col]=_mm512_dpbusd_epi32(e[col],_mm512_loadu_si512((const __m512i *)(bp+k*64)),a0);
    o[col]=_mm512_dpbusd_epi32(o[col],_mm512_loadu_si512((const __m512i *)(bp+(k+1)*64)),a1);
   });
  }
  const __m512i comp=_mm512_set1_epi32(q8_input_compensation(&A[i]));
  const __m512 ad=_mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[i].d));
  Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const __m512 wd=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+512)));
   c[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_add_epi32(e[col],o[col]),comp)),_mm512_mul_ps(wd,ad),c[col]);
  });
 }
 Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,c[col]);});
}


template<typename TB, int BLOCK_K>
void convert_B_packed_format(void * RESTRICT packed_B, const TB * RESTRICT B, int N, int K) {
    const int NB = N / TILE_N;
    const int KB = K / BLOCK_K;
    const int TILE_SIZE = get_tile_size<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](int begin, int end) {
        for (int n = begin; n < end; ++n) {
            for (int k = 0; k < KB; ++k) {
                int n0 = n * TILE_N;
                pack_B((char *)packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
            }
        }
    });
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni {};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q4_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_0);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512 vc[COLS];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        const __m512i off = _mm512_set1_epi8(8);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and compute compensation
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_1, block_q4_1, float, 1, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_1);

        const block_q8_1 * RESTRICT A = static_cast<const block_q8_1 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1, vs1;

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].s));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(bytes, lowMask);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset + TILE_N * sizeof(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
        if constexpr(q8_unsigned) { tinygemm_q8_unsigned_m1<BLOCK_N>(KB,_A,_B,C); return; }

        const bool integer_chains = getenv("GGML_Q8_INT_CHAINS");
        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        if (getenv("GGML_Q8_INTERLEAVE")) {
            const char * pf_env=getenv("GGML_Q8_PREFETCH");
            const int pf_dist=pf_env?std::max(0,atoi(pf_env)):0;
            for(int i=0;i<KB;++i){
                if(pf_dist > 0 && pf_dist < KB-i) Unroll<COLS>{}([&](auto col){
                    const char * pf=B+PACKED_INDEX(col,i+pf_dist,KB,TILE_SIZE);
                    _mm_prefetch(pf+0,_MM_HINT_T0);
                    _mm_prefetch(pf+256,_MM_HINT_T0);
                    _mm_prefetch(pf+512,_MM_HINT_T0);
                });

                __m512i sums[COLS];
                Unroll<COLS>{}([&](auto col){sums[col]=_mm512_setzero_si512();});
                const int32_t * ap = (const int32_t *)A[i].qs;
                const __m512 ad = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[i].d));
                for(int k=0;k<8;++k){
                    const __m512i av=_mm512_add_epi8(_mm512_set1_epi32(ap[k]),off);
                    Unroll<COLS>{}([&](auto col){
                        const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                        const __m512i bv=_mm512_loadu_si512((const __m512i *)(bp+k*64));
                        sums[col]=_mm512_dpbusd_epi32(sums[col],av,bv);
                    });
                }
                Unroll<COLS>{}([&](auto col){
                    const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                    const int offd=TILE_N*TILE_K;
                    const __m512 wd=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+offd)));
                    const __m512i comp=_mm512_loadu_si512((const __m512i *)(bp+offd+TILE_N*sizeof(ggml_half)));
                    const __m512i sum=_mm512_sub_epi32(sums[col],comp);
                    vc[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(sum),_mm512_mul_ps(wd,ad),vc[col]);
                });
            }
            Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,vc[col]);});
            return;
        }

        const bool four_chains = integer_chains && getenv("GGML_Q8_FOUR_CHAINS");
        auto compute = [&](auto col, auto i) {
            // load a and add offset 128
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = _mm512_add_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
            }
            const int offset = TILE_N * TILE_K;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const int offset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
            const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            if (four_chains) {
                __m512i s1 = _mm512_setzero_si512(), s2 = s1, s3 = s1;
                for (int k = 0; k < 2; ++k) {
                    vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
                    s1 = _mm512_dpbusd_epi32(s1, va[k+2], vb[k+2]);
                    s2 = _mm512_dpbusd_epi32(s2, va[k+4], vb[k+4]);
                    s3 = _mm512_dpbusd_epi32(s3, va[k+6], vb[k+6]);
                }
                vsum = _mm512_add_epi32(_mm512_add_epi32(vsum,s1),_mm512_add_epi32(s2,s3));
            } else if (integer_chains) {
                __m512i other = _mm512_setzero_si512();
                for (int k = 0; k < 4; ++k) {
                    vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
                    other = _mm512_dpbusd_epi32(other, va[k + 4], vb[k + 4]);
                }
                vsum = _mm512_add_epi32(vsum, other);
            } else {
                for (int k = 0; k < 8; ++k) vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q4_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_K) + TILE_N * 4 + TILE_N*(q4_weight_bytes - QK_K/2);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = q4_weight_bytes * TILE_N;
        const int offset_mins   = q4_weight_bytes * TILE_N +  8 * TILE_N;
        const int offset_d0     = q4_weight_bytes * TILE_N + 16 * TILE_N;
        const int offset_dmin   = q4_weight_bytes * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        if (getenv("GGML_K_INTERLEAVE")) {
            const char * pf_env=getenv("GGML_K_PREFETCH");
            const int pf_dist=pf_env?std::max(0,atoi(pf_env)):0;
            for (int i=0;i<KB;++i) {
                if(pf_dist > 0 && pf_dist < KB-i) Unroll<COLS>{}([&](auto col){
                    const char * pf=B+PACKED_INDEX(col,i+pf_dist,KB,TILE_SIZE);
                    _mm_prefetch(pf+0,_MM_HINT_T0);
                    _mm_prefetch(pf+512,_MM_HINT_T0);
                    _mm_prefetch(pf+1024,_MM_HINT_T0);
                    _mm_prefetch(pf+1536,_MM_HINT_T0);
                    _mm_prefetch(pf+offset_scales,_MM_HINT_T0);
                });

                __m512i acc[COLS];
                Unroll<COLS>{}([&](auto col){acc[col]=_mm512_setzero_si512();});
                const __m512 ad = _mm512_set1_ps(A[i].d);
                const __m256i sums = _mm256_loadu_si256((const __m256i *)A[i].bsums);
                const __m512i absum = _mm512_castsi128_si512(_mm_hadd_epi16(_mm256_extracti128_si256(sums,0),_mm256_extracti128_si256(sums,1)));
                for (int kg=0;kg<8;++kg) {
                    __m512i even[COLS], odd[COLS];
                    Unroll<COLS>{}([&](auto col){even[col]=odd[col]=_mm512_setzero_si512();});
                    const __m512i ax = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[i].qs+kg*32)));
                    for (int k=0;k<8;k+=2) {
                        const __m512i a0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k),ax);
                        const __m512i a1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k+1),ax);
                        Unroll<COLS>{}([&](auto col){
                            const char * bp = B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                            const char * qp = bp+kg * (q4_weight_bytes == QK_K ? 512 : 256) + (k/2) * (q4_weight_bytes == QK_K ? 128 : 64);
                            const __m512i bytes = _mm512_loadu_si512((const __m512i *)qp);
                            const __m512i vb0 = q4_weight_bytes == QK_K ? bytes : _mm512_and_si512(bytes,lowMask);
                            const __m512i vb1 = q4_weight_bytes == QK_K ? _mm512_loadu_si512((const __m512i *)(qp+64)) : _mm512_and_si512(_mm512_srli_epi16(bytes,4),lowMask);
                            even[col]=_mm512_dpbusd_epi32(even[col],vb0,a0);
                            odd[col]=_mm512_dpbusd_epi32(odd[col],vb1,a1);
                        });
                    }
                    Unroll<COLS>{}([&](auto col){
                        const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                        const __m512i scale=_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bp+offset_scales+kg*TILE_N)));
                        acc[col]=_mm512_add_epi32(acc[col],_mm512_mullo_epi32(_mm512_add_epi32(even[col],odd[col]),scale));
                    });
                }
                Unroll<COLS>{}([&](auto col){
                    const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                    const __m512 d=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+offset_d0)));
                    vc[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[col]),_mm512_mul_ps(d,ad),vc[col]);
                    __m512i am=_mm512_setzero_si512();
                    for(int k=0;k<4;++k){
                        const __m512i aa=_mm512_permutexvar_epi32(_mm512_set1_epi32(k),absum);
                        const __m512i bb=_mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(bp+offset_mins+k*32)));
                        am=_mm512_dpwssds_epi32(am,aa,bb);
                    }
                    const __m512 dm=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+offset_dmin)));
                    vc[col]=_mm512_fnmadd_ps(_mm512_cvtepi32_ps(am),_mm512_mul_ps(dm,ad),vc[col]);
                });
            }
            Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,vc[col]);});
            return;
        }

        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i vsum_odd = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    const __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    const __m512i vb0 = q4_weight_bytes == QK_K ? bytes : _mm512_and_si512(bytes, lowMask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    const __m512i vb1 = q4_weight_bytes == QK_K ? _mm512_loadu_si512((const __m512i *)(b_qs + 64)) : _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                    vsum_odd = _mm512_dpbusd_epi32(vsum_odd, vb1, va1);
                    b_qs += q4_weight_bytes == QK_K ? 128 : 64;
                }
                vsum = _mm512_add_epi32(vsum, vsum_odd);
                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q5_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q5_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        const int offset_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        const int offset_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        if (getenv("GGML_K_INTERLEAVE")) {
            const char * pf_env=getenv("GGML_K_PREFETCH");
            const int pf_dist=pf_env?std::max(0,atoi(pf_env)):0;
            for (int i=0;i<KB;++i) {
                if(pf_dist > 0 && pf_dist < KB-i) Unroll<COLS>{}([&](auto col){
                    const char * pf=B+PACKED_INDEX(col,i+pf_dist,KB,TILE_SIZE);
                    _mm_prefetch(pf+0,_MM_HINT_T0);
                    _mm_prefetch(pf+512,_MM_HINT_T0);
                    _mm_prefetch(pf+1024,_MM_HINT_T0);
                    _mm_prefetch(pf+1536,_MM_HINT_T0);
                    _mm_prefetch(pf+offset_scales,_MM_HINT_T0);
                });

                __m512i acc[COLS];
                Unroll<COLS>{}([&](auto col){acc[col]=_mm512_setzero_si512();});
                const __m512 ad = _mm512_set1_ps(A[i].d);
                const __m256i sums = _mm256_loadu_si256((const __m256i *)A[i].bsums);
                const __m512i absum = _mm512_castsi128_si512(_mm_hadd_epi16(_mm256_extracti128_si256(sums,0),_mm256_extracti128_si256(sums,1)));
                for (int kg=0;kg<8;++kg) {
                    __m512i even[COLS], odd[COLS];
                    Unroll<COLS>{}([&](auto col){even[col]=odd[col]=_mm512_setzero_si512();});
                    const __m512i ax = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[i].qs+kg*32)));
                    for (int k=0;k<8;k+=2) {
                        const __m512i a0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k),ax);
                        const __m512i a1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k+1),ax);
                        Unroll<COLS>{}([&](auto col){
                            const char * bp = B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                            const char * qp = bp+kg*256 + (k/2)*64;
                            const __m512i bytes = _mm512_loadu_si512((const __m512i *)qp);
                            const __m512i hi = _mm512_loadu_si512((const __m512i *)(bp+offset_qh+kg*64));
                            const __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hi,_mm512_set1_epi8(1<<k)),k),4);
                            const __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hi,_mm512_set1_epi8(1<<(k+1))),k+1),4);
                            const __m512i vb0 = _mm512_add_epi8(_mm512_and_si512(bytes,lowMask),vh0);
                            const __m512i vb1 = _mm512_add_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes,4),lowMask),vh1);
                            even[col]=_mm512_dpbusd_epi32(even[col],vb0,a0);
                            odd[col]=_mm512_dpbusd_epi32(odd[col],vb1,a1);
                        });
                    }
                    Unroll<COLS>{}([&](auto col){
                        const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                        const __m512i scale=_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bp+offset_scales+kg*TILE_N)));
                        acc[col]=_mm512_add_epi32(acc[col],_mm512_mullo_epi32(_mm512_add_epi32(even[col],odd[col]),scale));
                    });
                }
                Unroll<COLS>{}([&](auto col){
                    const char * bp=B+PACKED_INDEX(col,i,KB,TILE_SIZE);
                    const __m512 d=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+offset_d0)));
                    vc[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[col]),_mm512_mul_ps(d,ad),vc[col]);
                    __m512i am=_mm512_setzero_si512();
                    for(int k=0;k<4;++k){
                        const __m512i aa=_mm512_permutexvar_epi32(_mm512_set1_epi32(k),absum);
                        const __m512i bb=_mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(bp+offset_mins+k*32)));
                        am=_mm512_dpwssds_epi32(am,aa,bb);
                    }
                    const __m512 dm=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+offset_dmin)));
                    vc[col]=_mm512_fnmadd_ps(_mm512_cvtepi32_ps(am),_mm512_mul_ps(dm,ad),vc[col]);
                });
            }
            Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,vc[col]);});
            return;
        }

        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            const char * b_qh  = b_ptr + offset_qh;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i vsum_odd = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((const __m512i *)(b_qh + k_group * 64));
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = _mm512_add_epi8(vb0, vh0);
                    vb1 = _mm512_add_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum_odd = _mm512_dpbusd_epi32(vsum_odd, vb1, va1);

                    b_qs += 64;
                }
                vsum = _mm512_add_epi32(vsum, vsum_odd);
                // vacc += scale * (q8 @ q5)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template<int BN>
void tinygemm_q8_rows2(int KB,const void * aa,const void * bb,float * C,int ldc){
 constexpr int COLS=BN/16;const int TS=get_tile_size<block_q8_0>();
 const auto * A=(const block_q8_0 *)aa;const char * B=(const char *)bb;
 __m512 c0[COLS],c1[COLS];const __m512i off=_mm512_set1_epi8((char)0x80);
 Unroll<COLS>{}([&](auto col){c0[col]=c1[col]=_mm512_setzero_ps();});
 for(int i=0;i<KB;++i){
  __m512i s0[COLS],s1[COLS];Unroll<COLS>{}([&](auto col){s0[col]=s1[col]=_mm512_setzero_si512();});
  const int32_t * a0=(const int32_t *)A[i].qs,*a1=(const int32_t *)A[KB+i].qs;
  for(int k=0;k<8;++k){
   const __m512i v0=q8_unsigned?_mm512_set1_epi32(a0[k]):_mm512_add_epi8(_mm512_set1_epi32(a0[k]),off),v1=q8_unsigned?_mm512_set1_epi32(a1[k]):_mm512_add_epi8(_mm512_set1_epi32(a1[k]),off);
   Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const __m512i v=_mm512_loadu_si512((const __m512i *)(bp+k*64));if constexpr(q8_unsigned){s0[col]=_mm512_dpbusd_epi32(s0[col],v,v0);s1[col]=_mm512_dpbusd_epi32(s1[col],v,v1);}else{s0[col]=_mm512_dpbusd_epi32(s0[col],v0,v);s1[col]=_mm512_dpbusd_epi32(s1[col],v1,v);}});
  }
  const __m512 d0=_mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[i].d)),d1=_mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[KB+i].d));
  const __m512i acomp0=_mm512_set1_epi32(q8_unsigned?q8_input_compensation(&A[i]):0),acomp1=_mm512_set1_epi32(q8_unsigned?q8_input_compensation(&A[KB+i]):0);
  Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const __m512 wd=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+512)));__m512i comp=_mm512_setzero_si512();if constexpr(!q8_unsigned)comp=_mm512_loadu_si512((const __m512i *)(bp+512+16*sizeof(ggml_half)));
   c0[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s0[col],q8_unsigned?acomp0:comp)),_mm512_mul_ps(wd,d0),c0[col]);
   c1[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s1[col],q8_unsigned?acomp1:comp)),_mm512_mul_ps(wd,d1),c1[col]);});
 }
 Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,c0[col]);_mm512_storeu_ps(C+ldc+col*16,c1[col]);});
}

template<typename TB,int BN>
void tinygemm_k_rows2(int KB,const void * aa,const void * bb,float * C,int ldc){
 static_assert(std::is_same<TB,block_q4_K>::value || std::is_same<TB,block_q5_K>::value);
 constexpr bool q4=std::is_same<TB,block_q4_K>::value;constexpr int COLS=BN/16;
 const int TS=get_tile_size<TB>(),qs_bytes=q4?q4_weight_bytes:QK_K/2;
 const int oqh=128*16,osc=qs_bytes*16+(q4?0:32*16),omn=osc+8*16,od=osc+16*16,odm=od+16*sizeof(ggml_half);
 const auto * A=(const block_q8_K *)aa;const char * B=(const char *)bb;const __m512i mask=_mm512_set1_epi8(15);
 __m512 c0[COLS],c1[COLS];Unroll<COLS>{}([&](auto col){c0[col]=c1[col]=_mm512_setzero_ps();});
 for(int i=0;i<KB;++i){
  __m512i ac0[COLS],ac1[COLS];Unroll<COLS>{}([&](auto col){ac0[col]=ac1[col]=_mm512_setzero_si512();});
  for(int kg=0;kg<8;++kg){
   __m512i e0[COLS],o0[COLS],e1[COLS],o1[COLS];Unroll<COLS>{}([&](auto col){e0[col]=o0[col]=e1[col]=o1[col]=_mm512_setzero_si512();});
   const __m512i ax0=_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[i].qs+kg*32))),ax1=_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[KB+i].qs+kg*32)));
   for(int k=0;k<8;k+=2){
    const __m512i a00=_mm512_permutexvar_epi32(_mm512_set1_epi32(k),ax0),a01=_mm512_permutexvar_epi32(_mm512_set1_epi32(k+1),ax0),a10=_mm512_permutexvar_epi32(_mm512_set1_epi32(k),ax1),a11=_mm512_permutexvar_epi32(_mm512_set1_epi32(k+1),ax1);
    Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const char * qp=bp+kg*(q4&&qs_bytes==256?512:256)+(k/2)*(q4&&qs_bytes==256?128:64);const __m512i bytes=_mm512_loadu_si512((const __m512i *)qp);__m512i b0,b1;
     if constexpr(q4){b0=qs_bytes==256?bytes:_mm512_and_si512(bytes,mask);b1=qs_bytes==256?_mm512_loadu_si512((const __m512i *)(qp+64)):_mm512_and_si512(_mm512_srli_epi16(bytes,4),mask);}
     else{const __m512i hi=_mm512_loadu_si512((const __m512i *)(bp+oqh+kg*64));const __m512i h0=_mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hi,_mm512_set1_epi8(1<<k)),k),4),h1=_mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hi,_mm512_set1_epi8(1<<(k+1))),k+1),4);b0=_mm512_add_epi8(_mm512_and_si512(bytes,mask),h0);b1=_mm512_add_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes,4),mask),h1);}
     e0[col]=_mm512_dpbusd_epi32(e0[col],b0,a00);o0[col]=_mm512_dpbusd_epi32(o0[col],b1,a01);e1[col]=_mm512_dpbusd_epi32(e1[col],b0,a10);o1[col]=_mm512_dpbusd_epi32(o1[col],b1,a11);
    });
   }
   Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const __m512i sc=_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bp+osc+kg*16)));ac0[col]=_mm512_add_epi32(ac0[col],_mm512_mullo_epi32(_mm512_add_epi32(e0[col],o0[col]),sc));ac1[col]=_mm512_add_epi32(ac1[col],_mm512_mullo_epi32(_mm512_add_epi32(e1[col],o1[col]),sc));});
  }
  const __m512 ad0=_mm512_set1_ps(A[i].d),ad1=_mm512_set1_ps(A[KB+i].d);
  const __m256i bs0=_mm256_loadu_si256((const __m256i *)A[i].bsums),bs1=_mm256_loadu_si256((const __m256i *)A[KB+i].bsums);
  const __m512i ab0=_mm512_castsi128_si512(_mm_hadd_epi16(_mm256_extracti128_si256(bs0,0),_mm256_extracti128_si256(bs0,1))),ab1=_mm512_castsi128_si512(_mm_hadd_epi16(_mm256_extracti128_si256(bs1,0),_mm256_extracti128_si256(bs1,1)));
  Unroll<COLS>{}([&](auto col){const char * bp=B+PACKED_INDEX(col,i,KB,TS);const __m512 wd=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+od)));
   c0[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(ac0[col]),_mm512_mul_ps(wd,ad0),c0[col]);c1[col]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(ac1[col]),_mm512_mul_ps(wd,ad1),c1[col]);
   __m512i am0=_mm512_setzero_si512(),am1=am0;for(int k=0;k<4;++k){const __m512i mb=_mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(bp+omn+k*32)));am0=_mm512_dpwssds_epi32(am0,_mm512_permutexvar_epi32(_mm512_set1_epi32(k),ab0),mb);am1=_mm512_dpwssds_epi32(am1,_mm512_permutexvar_epi32(_mm512_set1_epi32(k),ab1),mb);}
   const __m512 wm=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(bp+odm)));c0[col]=_mm512_fnmadd_ps(_mm512_cvtepi32_ps(am0),_mm512_mul_ps(wm,ad0),c0[col]);c1[col]=_mm512_fnmadd_ps(_mm512_cvtepi32_ps(am1),_mm512_mul_ps(wm,ad1),c1[col]);
  });
 }
 Unroll<COLS>{}([&](auto col){_mm512_storeu_ps(C+col*16,c0[col]);_mm512_storeu_ps(C+ldc+col*16,c1[col]);});
}

template<typename TB,int BN>
bool tinygemm_rows2(int KB,const void * A,const void * B,float * C,int ldc){
 if(!getenv("GGML_VNNI_ROWS2"))return false;
 if constexpr(std::is_same<TB,block_q4_K>::value && q4_weight_bytes==QK_K) {
  if(getenv("GGML_EXPANDED_ROWS2_AMX")) return false;
 }
 if constexpr(std::is_same<TB,block_q8_0>::value){tinygemm_q8_rows2<BN>(KB,A,B,C,ldc);return true;}
 else if constexpr(std::is_same<TB,block_q4_K>::value || std::is_same<TB,block_q5_K>::value){tinygemm_k_rows2<TB,BN>(KB,A,B,C,ldc);return true;}
 return false;
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q6_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q6_K);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m512i m32s = _mm512_set1_epi32(32);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            const char * b_qh = b_ptr + offset_qh;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 16; ++k_group) {
                int r = k_group >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i bytes = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                bytes = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(bytes, lowMask);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_group);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](int col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq4_xs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_iq4_xs) + TILE_N * 2;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = (QK_K / 2) * TILE_N ;
        const int offset_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m256i m128s = _mm256_set1_epi16(128);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        const __m512i values128 = _mm512_set_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m512i values256 = _mm512_add_epi8(values128, off);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                int r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                    __m512i bytes = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(bytes, lowMask));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_VNNI(NB_SIZE)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, NB_SIZE, blck_size>::apply(             \
        KB, wdata_batch,                                                                       \
        (const char *)src0->data + src0_offset + PACKED_INDEX(nb * kTilesN, 0, KB, TILE_SIZE), \
        (float *) ((char *) dst->data + dst_offset) + nb_start, ldc)

template <typename TA, typename TB, bool is_acc>
inline void acc_amx(float * C, int ldc, const int32_t * tile, const TA * A, int lda, const void * B, int nr, bool signed_q8) {
    if constexpr (std::is_same<TB, block_q8_0>::value) {
        acc_C<TA, TB, is_acc>::apply(C, ldc, tile, A, lda, B, nr, signed_q8);
    } else {
        acc_C<TA, TB, is_acc>::apply(C, ldc, tile, A, lda, B, nr);
    }
}

#define AMX_WEIGHT_DOT(D, A, B) do { \
 if constexpr(q8_unsigned && std::is_same<TB,block_q8_0>::value) { if (signed_q8) _tile_dpbssd(D,A,B); else _tile_dpbsud(D,A,B); } \
 else _tile_dpbssd(D,A,B); \
} while (0)

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<!is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, TC * RESTRICT C, int ldc) {
    using packed_B_t = packed_B_type<TB>;
    const int TILE_SIZE = get_tile_size<TB>();
    const bool signed_q8 = q8_unsigned && std::is_same<TB, block_q8_0>::value && M > 2 && getenv("GGML_Q8_PREFILL_SIGNED");
    const bool need_unpack = do_unpack<TB>::value || signed_q8;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    ggml_tile_config_init(getenv("GGML_AMX_SMALL_ROWS") ? std::min(M, TILE_M) : (getenv("GGML_AMX_SINGLE_ROW") && M == 1 ? 1 : 16));
    const int lda = KB * sizeof(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) static thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) static thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    auto Tile4 = [&](int32_t * base) { return base; };
    auto Tile5 = [&](int32_t * base) { return base + TILE_M * TILE_N; };
    auto Tile6 = [&](int32_t * base) { return base + 2 * TILE_M * TILE_N; };
    auto Tile7 = [&](int32_t * base) { return base + 3 * TILE_M * TILE_N; };

    if (M == 2 * TILE_M) {
        // i = 0
        const char * B_blk0 = B + PACKED_INDEX(0, 0, KB, TILE_SIZE);
        const char * B_blk1 = B + PACKED_INDEX(1, 0, KB, TILE_SIZE);
        if (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        AMX_WEIGHT_DOT(TMM4,TMM2,TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        AMX_WEIGHT_DOT(TMM5,TMM3,TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * sizeof(int32_t));

        if (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        AMX_WEIGHT_DOT(TMM6,TMM2,TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM7);
        AMX_WEIGHT_DOT(TMM7,TMM3,TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * sizeof(int32_t));

        for (int i = 1; i < KB; ++i) {
            // index of previous iter
            const int ii = i - 1;
            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                if (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_amx<TA, TB, is_acc>(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M, signed_q8);

                AMX_WEIGHT_DOT(TMM4,TMM2,TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_amx<TA, TB, is_acc>(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M, signed_q8);

                AMX_WEIGHT_DOT(TMM5,TMM3,TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));

                if (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_amx<TA, TB, is_acc>(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M, signed_q8);

                AMX_WEIGHT_DOT(TMM6,TMM2,TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM7);
                acc_amx<TA, TB, is_acc>(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M, signed_q8);

                AMX_WEIGHT_DOT(TMM7,TMM3,TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            int ii = KB - 1;
            acc_amx<TA, TB, true>(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M, signed_q8);
            acc_amx<TA, TB, true>(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M, signed_q8);
            acc_amx<TA, TB, true>(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M, signed_q8);
            acc_amx<TA, TB, true>(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M, signed_q8);
        }
    } else {
        for (int i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            if (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            if (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            if (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            if (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } else {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            AMX_WEIGHT_DOT(TMM4,TMM2,TMM0);
            AMX_WEIGHT_DOT(TMM6,TMM2,TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_amx<TA, TB, is_acc>(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0, signed_q8);
                acc_amx<TA, TB, is_acc>(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0, signed_q8);
            });

            if (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                AMX_WEIGHT_DOT(TMM5,TMM3,TMM0);
                AMX_WEIGHT_DOT(TMM7,TMM3,TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_amx<TA, TB, is_acc>(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1, signed_q8);
                    acc_amx<TA, TB, is_acc>(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1, signed_q8);
                });
            }
        }
    }
    return;
}

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
        __asm__ volatile("" ::: "memory");
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
                    // GCC tile loads do not declare their memory inputs.
                    __asm__ volatile("" ::: "memory");
                    _tile_loadd(TMM2, tail, 64);
                    __asm__ volatile("" ::: "memory");
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

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    const bool signed_q8 = false;
    static_assert(std::is_same<TA, block_q8_K>::value);
    if constexpr (std::is_same<TB, block_q4_K>::value || std::is_same<TB, block_q5_K>::value) {
        if (M > 2 && getenv("GGML_PREFILL_SCALED_K")) {
            tinygemm_scaled_k<TB>(M, KB, (const block_q8_K *) _A, (const char *) _B, C, ldc);
            return;
        }
    }

    const int TILE_SIZE = get_tile_size<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    ggml_tile_config_init(getenv("GGML_AMX_SMALL_ROWS") ? std::min(M, TILE_M) : (getenv("GGML_AMX_SINGLE_ROW") && M == 1 ? 1 : 16));
    //const int lda = KB * sizeof(TA);

    alignas(64) static thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) static thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) static thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi7[TILE_M * TILE_N];

    const int k_group_size = std::is_same<TB, block_q6_K>::value ? 16 : 32;
    for (int i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        for (int k = 0; k < QK_K / k_group_size; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                if constexpr (std::is_same<TB, block_q4_K>::value && q4_weight_bytes == QK_K) {
                    _tile_loadd(TMM0, B + PACKED_INDEX(0, i, KB, TILE_SIZE) + k*TILE_N*32, TILE_N * VNNI_BLK);
                } else {
                    unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                }

                if constexpr (std::is_same<TB, block_q4_K>::value && q4_weight_bytes == QK_K) {
                    _tile_loadd(TMM1, B + PACKED_INDEX(1, i, KB, TILE_SIZE) + k*TILE_N*32, TILE_N * VNNI_BLK);
                } else {
                    unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                }

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                AMX_WEIGHT_DOT(TMM4,TMM2,TMM0);
                AMX_WEIGHT_DOT(TMM6,TMM2,TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m0);

                if (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    AMX_WEIGHT_DOT(TMM5,TMM3,TMM0);
                    AMX_WEIGHT_DOT(TMM7,TMM3,TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
            if (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
            }
        });
    }
    return;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor) {
    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    auto get_tensor_size = [&] {
        size_t row_size_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_size_B = get_row_size<type, blck_size>(K);
        });
        return N * row_size_B * tensor->ne[2] * tensor->ne[3];
    };

    if (qtype_has_amx_kernels(TYPE)) {
        return get_tensor_size();
    } else {
        // for f16, bf16 we don't do packing
        return ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        GGML_ASSERT(ggml_is_contiguous(tensor));
#if defined(__gnu_linux__)
        if ((getenv("GGML_AMX_NUMA_ROWS") && tensor->ne[2] > 1 && tensor->ne[3] == 1 && N % 64 == 0) ||
            (getenv("GGML_AMX_NUMA_DENSE") && N % 128 == 0)) {
            const size_t half_bytes = (N / 2) * get_row_size<type, blck_size>(K);
            const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
            for (int64_t e = 0; e < tensor->ne[2] * tensor->ne[3]; ++e) {
                for (int node = 0; node < 2; ++node) {
                    const uintptr_t begin = (uintptr_t) tensor->data + (2 * e + node) * half_bytes;
                    const uintptr_t lo = (begin + page - 1) & ~(page - 1);
                    const uintptr_t hi = (begin + half_bytes) & ~(page - 1);
                    unsigned long mask = 1UL << node;
                    if (hi > lo && syscall(SYS_mbind, (void *) lo, hi - lo, MPOL_BIND, &mask,
                                           sizeof(mask) * 8, 0) != 0) {
                        fprintf(stderr, "AMX row mbind failed: %s\n", strerror(errno));
                    }
                }
            }
        }
#endif
        convert_B_packed_format<type, blck_size>(tensor->data, (const type *) data,
            N * tensor->ne[2] * tensor->ne[3], K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
inline int64_t ggml_batch_offset(const ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    const int64_t i2 = batch_idx % ne2;
    const int64_t i3 = batch_idx / ne2;
    return i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];

    const enum ggml_type TYPE = src0->type;

    const bool is_floating_type = TYPE == GGML_TYPE_F16;
    if (is_floating_type) {
        return 0;
    }

    if (dst->op == GGML_OP_MUL_MAT_ID) {
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
    const int K = src0->ne[0];
    const int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
        desired_wsize = n_batch * M * row_size_A;
    });

    return desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
void ggml_backend_amx_mul_mat(const ggml_compute_params * params, struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    const enum ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    const bool is_floating_type = TYPE == GGML_TYPE_F16;

    const int M = dst->ne[1];
    const int N = dst->ne[0];
    const int K = src0->ne[0];
    const int ldc = dst->nb[1] / dst->nb[0];

    const int64_t ne2 = dst->ne[2];
    const int64_t n_batch = ne2 * dst->ne[3];

    if (is_floating_type) {
        constexpr int BLOCK_M = 4;
        constexpr int BLOCK_N = 6;
        const int MB = div_up(M, BLOCK_M);
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / (MB * NB);
                    int remaining = i % (MB * NB);
                    int mb = remaining / NB;
                    int nb = remaining % NB;

                    int64_t src0_batch = (batch_idx % ne2) / (ne2 / src0->ne[2]) +
                        (batch_idx / ne2) / (dst->ne[3] / src0->ne[3]) * src0->ne[2];
                    int64_t src0_offset = ggml_batch_offset(src0, src0_batch, src0->ne[2]);
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);

                    int mb_start = mb * BLOCK_M;
                    int mb_size = std::min(BLOCK_M, M - mb_start);
                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start);

                    switch (mb_size << 4 | nb_size) {
                        case 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); break;
                        case 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); break;
                        case 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); break;
                        case 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); break;
                        case 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); break;
                        case 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); break;
                        case 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); break;
                        case 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); break;
                        case 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); break;
                        case 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); break;
                        case 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); break;
                        case 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); break;
                        default: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        return;
    }

    // pointer to work space, used convert A from float to quantized type
    void * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
            const size_t desired_wsize = n_batch * M * row_size_A;
            if (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_size || TILE_K * 8 == blck_size);

            parallel_for_ggml(params, n_batch * M, [&](int begin, int end) {
                for (int idx = begin; idx < end; ++idx) {
                    int batch_idx = idx / M;
                    int m         = idx % M;
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    const float * A_data = (const float *)((const char *)src1->data + src1_offset);
                    char * wdata_batch = (char *)wdata + batch_idx * M * row_size_A;
                    from_float<vec_dot_type>((const float *) ((const char *) A_data + m * src1->nb[1]),
                        wdata_batch + m * row_size_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    if(M==2 && TYPE==GGML_TYPE_Q8_0 && N%32==0 && getenv("GGML_VNNI_ROWS2")) {
        const int BN=64,NB=div_up(N,BN),KB=K/QK8_0,rb=KB*sizeof(block_q8_0),ts=get_tile_size<block_q8_0>();
        const bool numa=getenv("GGML_AMX_NUMA_DENSE") && N%128==0 && params->nth>1;
        const int node=numa?params->ith%2:0,wn=numa?NB/2:NB;ggml_compute_params lp=*params;
        if(numa){lp.ith=params->ith/2;lp.nth=(params->nth+1-node)/2;}
        parallel_for_ggml(&lp,n_batch*wn,[&](int begin,int end){for(int job=begin;job<end;++job){
            const int batch=job/wn,nb=job%wn+node*wn;
            const int64_t wb=(batch%ne2)/(ne2/src0->ne[2])+(batch/ne2)/(dst->ne[3]/src0->ne[3])*src0->ne[2];
            const char * ap=(const char *)wdata+batch*2*rb;
            const char * bp=(const char *)src0->data+wb*N*get_row_size<block_q8_0,QK8_0>(K)+PACKED_INDEX(nb*4,0,KB,ts);
            float * cp=(float *)((char *)dst->data+ggml_batch_offset(dst,batch,ne2))+nb*64;
            const int ldc=dst->nb[1]/sizeof(float);
            if(N-nb*64>=64)tinygemm_q8_rows2<64>(KB,ap,bp,cp,ldc);else tinygemm_q8_rows2<32>(KB,ap,bp,cp,ldc);
        }});
        return;
    }

    const char * small_dense_env = getenv("GGML_AMX_DENSE_SMALL");
    const int small_dense_max = small_dense_env ? atoi(small_dense_env) : 1;
    if (M <= small_dense_max) {
        // MB = 1 and handle 8 tiles in each block
        constexpr int kTilesN = 4;
        constexpr int BLOCK_N = TILE_N * kTilesN;
        const int NB = div_up(N, BLOCK_N);
        const bool numa_rows = getenv("GGML_AMX_NUMA_DENSE") && N % (2 * BLOCK_N) == 0 && params->nth > 1;
        const int node = numa_rows ? params->ith % 2 : 0;
        const int work_nb = numa_rows ? NB / 2 : NB;
        ggml_compute_params local_params = *params;
        if (numa_rows) {
            local_params.ith = params->ith / 2;
            local_params.nth = (params->nth + 1 - node) / 2;
        }
        parallel_for_ggml(&local_params, n_batch * M * work_nb, [&](int begin, int end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                const int KB = K / blck_size;
                const int TILE_SIZE = get_tile_size<type>();
                const int row_size_A = KB * sizeof(vec_dot_type);
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / (M * work_nb);
                    int m = (i / work_nb) % M;
                    int nb = i % work_nb + node * work_nb;

                    int64_t src0_batch = (batch_idx % ne2) / (ne2 / src0->ne[2]) +
                        (batch_idx / ne2) / (dst->ne[3] / src0->ne[3]) * src0->ne[2];
                    int64_t src0_offset = src0_batch * N * get_row_size<type, blck_size>(K);
                    int64_t dst_offset  = ggml_batch_offset(dst, batch_idx, ne2) + m * dst->nb[1];
                    const char * wdata_batch = (const char *)wdata + (batch_idx * M + m) * row_size_A;

                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start); // 32, 64, 96

                    switch (nb_size) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        case 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); break;
                        case 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); break;
                        case 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); break;
                        case 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); break;
                        default: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        return;
    }

    // handle 4 tiles at a tile
    constexpr int BLOCK_M = TILE_M * 2;
    constexpr int BLOCK_N = TILE_N * 2;
    const int MB = div_up(M, BLOCK_M);
    const int NB = div_up(N, BLOCK_N);

    const bool numa_rows = getenv("GGML_AMX_NUMA_DENSE") && N % 128 == 0 && params->nth > 1;
    const int node = numa_rows ? params->ith % 2 : 0;
    const int work_nb = numa_rows ? NB / 2 : NB;
    ggml_compute_params local_params = *params;
    if (numa_rows) {
        local_params.ith = params->ith / 2;
        local_params.nth = (params->nth + 1 - node) / 2;
    }
    parallel_for_ggml(&local_params, n_batch * MB * work_nb, [&](int begin, int end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const int KB = K / blck_size;
            const int TILE_SIZE = get_tile_size<type>();
            const int row_size_A = KB * sizeof(vec_dot_type);

            for (int i = begin; i < end; ++i) {
                int batch_idx = i / (MB * work_nb);
                int remaining = i % (MB * work_nb);
                int mb = remaining / work_nb;
                int nb = remaining % work_nb + node * work_nb;

                int64_t src0_batch = (batch_idx % ne2) / (ne2 / src0->ne[2]) +
                        (batch_idx / ne2) / (dst->ne[3] / src0->ne[3]) * src0->ne[2];
                    int64_t src0_offset = src0_batch * N * get_row_size<type, blck_size>(K);
                int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                const char * wdata_batch = (const char *)wdata + batch_idx * M * row_size_A;

                int mb_start = mb * BLOCK_M;
                int mb_size = std::min(BLOCK_M, M - mb_start);
                int nb_start = nb * BLOCK_N;
                int nb_size = BLOCK_N;

                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
                    mb_size, nb_size, KB,
                    wdata_batch + mb_start * row_size_A,
                    (const char *)src0->data + src0_offset + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
                    (float *) ((char *) dst->data + dst_offset) + mb_start * N + nb_start, ldc);
            }
        });
    });
}


// Use the existing packed kernels for each expert's selected token rows.
void ggml_backend_amx_mul_mat_id(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int K = w->ne[0], N = w->ne[1], E = w->ne[2];
    const int S = ids->ne[0], T = ids->ne[1], R = S * T;
    const bool shared_decode = getenv("GGML_AMX_SHARED_DECODE") && T == 1 && x->ne[1] == 1;
    const int TYPE = w->type;
    constexpr int BM = 32, BN = 32;
    const int NB = N / BN;
    const bool local_rows = getenv("GGML_AMX_NUMA_ROWS") && N % 64 == 0 && params->nth > 1;
    const int work_nb = local_rows ? NB / 2 : NB;
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
            tasks[e + 1] = tasks[e] + div_up(counts[e], BM) * work_nb;
            counts[e] = 0;
        }
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < S; ++j) {
                int e = *(int32_t *) ((char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
                rows[starts[e] + counts[e]++] = t * S + j;
            }
        }
        if (shared_decode) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                const size_t rb = (K / blck_size) * sizeof(vec_dot_type);
                GGML_ASSERT(params->wsize >= meta + R * rb);
                from_float<vec_dot_type>((const float *) x->data, qa, K);
                for (int r = 1; r < R; ++r) memcpy(qa + r * rb, qa, rb);
            });
        }
    }
    ggml_barrier(params->threadpool);
    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int KB = K / blck_size;
        const size_t row_bytes = KB * sizeof(vec_dot_type);
        const size_t expert_bytes = N * get_row_size<type, blck_size>(K);
        const int tile_bytes = get_tile_size<type>();
        GGML_ASSERT(params->wsize >= meta + R * row_bytes);
        if (!shared_decode) {
        for (int r = params->ith; r < R; r += params->nth) {
            int t = rows[r] / S, j = rows[r] % S;
            const float * xr = (const float *) ((const char *) x->data + t * x->nb[2] + (j % x->ne[1]) * x->nb[1]);
            from_float<vec_dot_type>(xr, qa + r * row_bytes, K);
        }
        ggml_barrier(params->threadpool);
        }
        ggml_tile_config_init();
        alignas(64) float result[BM * BN];
        const int node = local_rows ? params->ith % 2 : 0;
        const int rank = local_rows ? params->ith / 2 : params->ith;
        const int workers = local_rows ? (params->nth + 1 - node) / 2 : params->nth;
        const int begin = (int64_t) tasks[E] * rank / workers;
        const int end = (int64_t) tasks[E] * (rank + 1) / workers;
        for (int job = begin; job < end; ++job) {
            int e = std::upper_bound(tasks, tasks + E + 1, job) - tasks - 1;
            int local = job - tasks[e];
            int mb = local / work_nb, nb = local % work_nb + node * work_nb;
            int nr = std::min(BM, counts[e] - mb * BM);
            const char * ap = qa + (starts[e] + mb * BM) * row_bytes;
            const char * bp = (const char *) w->data + e * expert_bytes + PACKED_INDEX(nb * 2, 0, KB, tile_bytes);
            const char * small_env = getenv("GGML_AMX_MOE_SMALL");
            const int small_max = small_env ? atoi(small_env) : 1;
            if (nr == 2 && tinygemm_rows2<type, BN>(KB, ap, bp, result, BN)) {
                if(params->ith==0 && getenv("GGML_VNNI_ROWS2_TRACE"))fprintf(stderr,"ROWS2_MOE type=%d\n",TYPE);
            } else if (nr <= small_max) {
                for (int m = 0; m < nr; ++m) {
                    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, BN, blck_size>::apply(
                        KB, ap + m * row_bytes, bp, result + m * BN, BN);
                }
            } else {
                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(nr, BN, KB, ap, bp, result, BN);
            }
            for (int m = 0; m < nr; ++m) {
                int r = rows[starts[e] + mb * BM + m];
                char * yp = (char *) dst->data + (r / S) * dst->nb[2] + (r % S) * dst->nb[1] + nb * BN * sizeof(float);
                memcpy(yp, result + m * BN, BN * sizeof(float));
            }
        }
    });
}

void ggml_backend_amx_mul_mat_id_pair(const ggml_compute_params * params, ggml_tensor * dst, ggml_tensor * dst2) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int K = w->ne[0], N = w->ne[1], E = w->ne[2];
    const int S = ids->ne[0], T = ids->ne[1], R = S * T;
    const bool shared_decode = getenv("GGML_AMX_SHARED_DECODE") && T == 1 && x->ne[1] == 1;
    const int TYPE = w->type;
    constexpr int BM = 32, BN = 32;
    const int NB = N / BN;
    const bool local_rows = getenv("GGML_AMX_NUMA_ROWS") && N % 64 == 0 && params->nth > 1;
    const int work_nb = local_rows ? NB / 2 : NB;
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
            tasks[e + 1] = tasks[e] + div_up(counts[e], BM) * work_nb;
            counts[e] = 0;
        }
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < S; ++j) {
                int e = *(int32_t *) ((char *) ids->data + t * ids->nb[1] + j * ids->nb[0]);
                rows[starts[e] + counts[e]++] = t * S + j;
            }
        }
        if (shared_decode) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                const size_t rb = (K / blck_size) * sizeof(vec_dot_type);
                GGML_ASSERT(params->wsize >= meta + R * rb);
                from_float<vec_dot_type>((const float *) x->data, qa, K);
                for (int r = 1; r < R; ++r) memcpy(qa + r * rb, qa, rb);
            });
        }
    }
    ggml_barrier(params->threadpool);
    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int KB = K / blck_size;
        const size_t row_bytes = KB * sizeof(vec_dot_type);
        const size_t expert_bytes = N * get_row_size<type, blck_size>(K);
        const int tile_bytes = get_tile_size<type>();
        GGML_ASSERT(params->wsize >= meta + R * row_bytes);
        if (!shared_decode) {
        for (int r = params->ith; r < R; r += params->nth) {
            int t = rows[r] / S, j = rows[r] % S;
            const float * xr = (const float *) ((const char *) x->data + t * x->nb[2] + (j % x->ne[1]) * x->nb[1]);
            from_float<vec_dot_type>(xr, qa + r * row_bytes, K);
        }
        ggml_barrier(params->threadpool);
        }
        ggml_tile_config_init();
        alignas(64) float result[BM * BN];
        const int node = local_rows ? params->ith % 2 : 0;
        const int rank = local_rows ? params->ith / 2 : params->ith;
        const int workers = local_rows ? (params->nth + 1 - node) / 2 : params->nth;
        const int begin = (int64_t) (2 * tasks[E]) * rank / workers;
        const int end = (int64_t) (2 * tasks[E]) * (rank + 1) / workers;
        for (int pair_job = begin; pair_job < end; ++pair_job) {
            const int which = pair_job / tasks[E];
            const int job = pair_job % tasks[E];
            ggml_tensor * target = which ? dst2 : dst;
            const ggml_tensor * weight = target->src[0];
            int e = std::upper_bound(tasks, tasks + E + 1, job) - tasks - 1;
            int local = job - tasks[e];
            int mb = local / work_nb, nb = local % work_nb + node * work_nb;
            int nr = std::min(BM, counts[e] - mb * BM);
            const char * ap = qa + (starts[e] + mb * BM) * row_bytes;
            const char * bp = (const char *) weight->data + e * expert_bytes + PACKED_INDEX(nb * 2, 0, KB, tile_bytes);
            const char * small_env = getenv("GGML_AMX_MOE_SMALL");
            const int small_max = small_env ? atoi(small_env) : 1;
            if (nr == 2 && tinygemm_rows2<type, BN>(KB, ap, bp, result, BN)) {
                if(params->ith==0 && getenv("GGML_VNNI_ROWS2_TRACE"))fprintf(stderr,"ROWS2_MOE type=%d\n",TYPE);
            } else if (nr <= small_max) {
                for (int m = 0; m < nr; ++m) {
                    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, BN, blck_size>::apply(
                        KB, ap + m * row_bytes, bp, result + m * BN, BN);
                }
            } else {
                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(nr, BN, KB, ap, bp, result, BN);
            }
            for (int m = 0; m < nr; ++m) {
                int r = rows[starts[e] + mb * BM + m];
                char * yp = (char *) target->data + (r / S) * target->nb[2] + (r % S) * target->nb[1] + nb * BN * sizeof(float);
                memcpy(yp, result + m * BN, BN * sizeof(float));
            }
        }
    });
}

#endif // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
// Q/K/V share one activation quantization and write directly into concatenated rows.
void ggml_backend_amx_qkv(const ggml_compute_params * params, const ggml_tensor * const weights[3], const ggml_tensor * x, ggml_tensor * out) {
    const int K = x->ne[0], M = x->ne[1], N = weights[0]->ne[1];
    const int KB = K/QK8_0, row_size = KB*sizeof(block_q8_0), ldc = out->nb[1]/sizeof(float);
    GGML_ASSERT(params->wsize >= size_t(M)*row_size);
    char * a = (char *) params->wdata;
    parallel_for_ggml(params, M, [&](int begin, int end) {
        for (int m = begin; m < end; ++m) from_float<block_q8_0>((const float *) ((const char *) x->data + m*x->nb[1]), a + m*row_size, K);
    });
    ggml_barrier(params->threadpool);
    const bool numa = getenv("GGML_AMX_NUMA_DENSE") && params->nth > 1;
    const int node = numa ? params->ith % 2 : 0;
    ggml_compute_params local = *params;
    if (numa) { local.ith = params->ith/2; local.nth = (params->nth + 1 - node)/2; }
    const int tile_size = get_tile_size<block_q8_0>();
    if (M == 2 && getenv("GGML_VNNI_ROWS2")) {
        const int NB=N/64,work_nb=numa?NB/2:NB;
        parallel_for_ggml(&local,3*work_nb,[&](int begin,int end){for(int i=begin;i<end;++i){int w=i/work_nb,nb=i%work_nb+node*work_nb;
            tinygemm_q8_rows2<64>(KB,a,(const char *)weights[w]->data+PACKED_INDEX(nb*4,0,KB,tile_size),(float *)out->data+w*N+nb*64,ldc);
        }});
    } else if (M == 1 && !getenv("GGML_AMX_SINGLE_ROW")) {
        const int NB = N/64, work_nb = numa ? NB/2 : NB;
        parallel_for_ggml(&local, 3*work_nb, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const int w = i/work_nb, nb = i % work_nb + node*work_nb;
                tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, QK8_0>::apply(KB, a,
                    (const char *) weights[w]->data + PACKED_INDEX(nb*4, 0, KB, tile_size),
                    (float *) out->data + w*N + nb*64, ldc);
            }
        });
    } else {
        const int MB = div_up(M, 32), NB = N/32, work_nb = numa ? NB/2 : NB;
        parallel_for_ggml(&local, 3*MB*work_nb, [&](int begin, int end) {
            ggml_tile_config_init();
            for (int i = begin; i < end; ++i) {
                const int w = i/(MB*work_nb), mb = (i/work_nb) % MB, nb = i % work_nb + node*work_nb;
                const int m = mb*32;
                tinygemm_kernel_amx<block_q8_0, block_q8_0, float, QK8_0>(std::min(32, M-m), 32, KB,
                    a + m*row_size, (const char *) weights[w]->data + PACKED_INDEX(nb*2, 0, KB, tile_size),
                    (float *) out->data + m*ldc + w*N + nb*32, ldc);
            }
        });
    }
}

#endif
