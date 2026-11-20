#pragma once

#include <cstdint>

// SIMD backend selection by processor architecture. Only x86-64 currently
// provides vectorized kernels (AVX2); every other architecture uses the
// scalar baseline below, which is bit-identical by construction.
// RASTER_SIMD_FORCE_SCALAR overrides to the baseline.
#if defined(RASTER_SIMD_FORCE_SCALAR)
#define RASTER_SIMD_BACKEND_SCALAR 1
#elif defined(__x86_64__)
#define RASTER_SIMD_BACKEND_X86_64 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
// NEON kernels are future work; the scalar baseline below covers ARM.
#define RASTER_SIMD_BACKEND_ARM 1
#include <arm_neon.h>
#else
#define RASTER_SIMD_BACKEND_SCALAR 1
#endif

#if defined(RASTER_SIMD_BACKEND_X86_64)
#include <cstring>
#else
#include <algorithm>
#include <cstddef>
#endif

namespace raster::simd {
    // Runtime AVX2 detection with process-wide caching. All wide helpers
    // below require AVX2; callers must check this first and use the scalar
    // path otherwise so one binary runs on any x86-64 CPU.
    [[nodiscard]]
    inline bool cpu_has_avx2() noexcept {
#if defined(RASTER_SIMD_BACKEND_X86_64)
        static const bool cached = [] {
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") != 0;
        }();
        return cached;
#else
        return false;
#endif
    }

#if defined(RASTER_SIMD_BACKEND_X86_64)
    // Eight-lane barycentric weights matching the scalar derivation exactly:
    // l0 = e1 * inv_area, l1 = e2 * inv_area, l2 = 1 - (l0 + l1), with the
    // same operation order, so results are bit-identical to scalar code.
    // Lanes hold two horizontally adjacent 2x2 quads: lanes 0-3 are the left
    // quad's corners (TL, TR, BL, BR) and lanes 4-7 the right quad's.
    __attribute__((target("avx2"))) inline void barycentric_weights8(
        const float* edge1, const float* edge2, float inv_area, float* l0, float* l1, float* l2
    ) noexcept {
        const __m256 e1 = _mm256_loadu_ps(edge1);
        const __m256 e2 = _mm256_loadu_ps(edge2);
        const __m256 inv = _mm256_set1_ps(inv_area);
        const __m256 w0 = _mm256_mul_ps(e1, inv);
        const __m256 w1 = _mm256_mul_ps(e2, inv);
        const __m256 w2 = _mm256_sub_ps(_mm256_set1_ps(1.0F), _mm256_add_ps(w0, w1));
        _mm256_storeu_ps(l0, w0);
        _mm256_storeu_ps(l1, w1);
        _mm256_storeu_ps(l2, w2);
    }

    namespace detail {
        // Shared per-channel quantization used by quantize_pack8 below.
        __attribute__((target("avx2"))) inline __m256i quantize_channel8(
            __m256 value, __m256 zero, __m256 one, __m256 scale, __m256 half
        ) noexcept {
            const __m256 clamped = _mm256_min_ps(_mm256_max_ps(value, zero), one);
            return _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(clamped, scale), half));
        }
    } // namespace detail

    // Eight-lane sRGB packing matching to_argb exactly: clamp to [0, 1],
    // scale by 255, round half up, pack 0xFFRRGGBB.
    __attribute__((target("avx2"))) inline void quantize_pack8(
        const float* red, const float* green, const float* blue, std::uint32_t* out
    ) noexcept {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0F);
        const __m256 scale = _mm256_set1_ps(255.0F);
        const __m256 half = _mm256_set1_ps(0.5F);
        __m256i packed = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_slli_epi32(
                    detail::quantize_channel8(_mm256_loadu_ps(red), zero, one, scale, half),
                    16
                ),
                _mm256_slli_epi32(
                    detail::quantize_channel8(_mm256_loadu_ps(green), zero, one, scale, half),
                    8
                )
            ),
            _mm256_or_si256(
                detail::quantize_channel8(_mm256_loadu_ps(blue), zero, one, scale, half),
                _mm256_set1_epi32(static_cast<int>(0xFF000000U))
            )
        );

        static_assert(sizeof(packed) == 8U * sizeof(std::uint32_t));
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Four-lane variant for a single 2x2 quad.
    inline void quantize_pack4(
        const float* red, const float* green, const float* blue, std::uint32_t* out
    ) noexcept {
        const __m128 zero = _mm_setzero_ps();
        const __m128 one = _mm_set1_ps(1.0F);
        const __m128 scale = _mm_set1_ps(255.0F);
        const __m128 half = _mm_set1_ps(0.5F);
        const auto quantize = [&](__m128 value) noexcept {
            const __m128 clamped = _mm_min_ps(_mm_max_ps(value, zero), one);
            return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(clamped, scale), half));
        };

        __m128i packed = _mm_or_si128(
            _mm_or_si128(
                _mm_slli_epi32(quantize(_mm_loadu_ps(red)), 16),
                _mm_slli_epi32(quantize(_mm_loadu_ps(green)), 8)
            ),
            _mm_or_si128(
                quantize(_mm_loadu_ps(blue)),
                _mm_set1_epi32(static_cast<int>(0xFF000000U))
            )
        );

        static_assert(sizeof(packed) == 4U * sizeof(std::uint32_t));
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Four-lane bilinear blend with the exact scalar operation order:
    // out = (v00 * (1 - fx) + v10 * fx) * (1 - fy)
    //     + (v01 * (1 - fx) + v11 * fx) * fy, one channel at a time.
    inline void bilinear_blend4(
        const float* v00,
        const float* v10,
        const float* v01,
        const float* v11,
        const float* fx,
        const float* fy,
        float* out
    ) noexcept {
        const __m128 one = _mm_set1_ps(1.0F);
        const __m128 fx_vec = _mm_loadu_ps(fx);
        const __m128 fy_vec = _mm_loadu_ps(fy);
        const __m128 wx0 = _mm_sub_ps(one, fx_vec);
        const __m128 wy0 = _mm_sub_ps(one, fy_vec);
        const __m128 top =
            _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(v00), wx0), _mm_mul_ps(_mm_loadu_ps(v10), fx_vec));
        const __m128 bottom =
            _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(v01), wx0), _mm_mul_ps(_mm_loadu_ps(v11), fx_vec));
        _mm_storeu_ps(out, _mm_add_ps(_mm_mul_ps(top, wy0), _mm_mul_ps(bottom, fy_vec)));
    }

    // Four-lane trilinear level mix: out = low + (high - low) * fraction.
    // The fraction is a per-quad scalar broadcast.
    inline void trilinear_mix4(
        const float* low, const float* high, float fraction, float* out
    ) noexcept {
        const __m128 frac = _mm_set1_ps(fraction);
        _mm_storeu_ps(
            out,
            _mm_add_ps(
                _mm_loadu_ps(low),
                _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(high), _mm_loadu_ps(low)), frac)
            )
        );
    }

    // Four-lane pack of byte-range channel blends, matching pack_channels
    // exactly: round half up per channel, keep the low 8 bits, pack 0xFFRGB.
    inline void pack_channels4(
        const float* c0, const float* c1, const float* c2, std::uint32_t* out
    ) noexcept {
        const __m128 half = _mm_set1_ps(0.5F);
        const __m128i mask = _mm_set1_epi32(0xFF);
        const auto quantize = [&](__m128 value) noexcept {
            return _mm_and_si128(_mm_cvttps_epi32(_mm_add_ps(value, half)), mask);
        };

        __m128i packed = _mm_or_si128(
            _mm_or_si128(quantize(_mm_loadu_ps(c0)), _mm_slli_epi32(quantize(_mm_loadu_ps(c1)), 8)),
            _mm_or_si128(
                _mm_slli_epi32(quantize(_mm_loadu_ps(c2)), 16),
                _mm_set1_epi32(static_cast<int>(0xFF000000U))
            )
        );

        static_assert(sizeof(packed) == 4U * sizeof(std::uint32_t));
        std::memcpy(out, &packed, sizeof(packed));
    }
#else
    // Non-x86_64 fallback with identical semantics.
    inline void barycentric_weights8(
        const float* edge1, const float* edge2, float inv_area, float* l0, float* l1, float* l2
    ) noexcept {
        for (std::size_t lane = 0; lane < 8U; ++lane) {
            const auto w0 = edge1[lane] * inv_area;
            const auto w1 = edge2[lane] * inv_area;
            l0[lane] = w0;
            l1[lane] = w1;
            l2[lane] = 1.0F - (w0 + w1);
        }
    }

    inline void quantize_pack8(
        const float* red, const float* green, const float* blue, std::uint32_t* out
    ) noexcept {
        for (std::size_t lane = 0; lane < 8U; ++lane) {
            const auto clamp = [](float channel) noexcept {
                return std::clamp(channel, 0.0F, 1.0F);
            };
            const auto pack = [](float channel) noexcept {
                // NOLINTNEXTLINE(bugprone-incorrect-roundings): non-negative bounded domain is exact.
                return static_cast<std::uint32_t>(channel + 0.5F);
            };
            out[lane] = 0xFF000000U
                | (pack(clamp(red[lane]) * 255.0F) << 16U)
                | (pack(clamp(green[lane]) * 255.0F) << 8U)
                | pack(clamp(blue[lane]) * 255.0F);
        }
    }

    inline void quantize_pack4(
        const float* red, const float* green, const float* blue, std::uint32_t* out
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            const auto clamp = [](float channel) noexcept {
                return std::clamp(channel, 0.0F, 1.0F);
            };
            const auto pack = [](float channel) noexcept {
                // NOLINTNEXTLINE(bugprone-incorrect-roundings): non-negative bounded domain is exact.
                return static_cast<std::uint32_t>(channel + 0.5F);
            };
            out[lane] = 0xFF000000U
                | (pack(clamp(red[lane]) * 255.0F) << 16U)
                | (pack(clamp(green[lane]) * 255.0F) << 8U)
                | pack(clamp(blue[lane]) * 255.0F);
        }
    }

    inline void bilinear_blend4(
        const float* v00,
        const float* v10,
        const float* v01,
        const float* v11,
        const float* fx,
        const float* fy,
        float* out
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            const auto wx0 = 1.0F - fx[lane];
            const auto wy0 = 1.0F - fy[lane];
            out[lane] = (v00[lane] * wx0 + v10[lane] * fx[lane])
                * wy0
                + (v01[lane] * wx0 + v11[lane] * fx[lane])
                * fy[lane];
        }
    }

    inline void trilinear_mix4(
        const float* low, const float* high, float fraction, float* out
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            out[lane] = low[lane] + (high[lane] - low[lane]) * fraction;
        }
    }

    inline void pack_channels4(
        const float* c0, const float* c1, const float* c2, std::uint32_t* out
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            const auto pack = [](float channel) noexcept {
                // NOLINTNEXTLINE(bugprone-incorrect-roundings): non-negative bounded domain is exact.
                return static_cast<std::uint32_t>(channel + 0.5F) & 0xFFU;
            };
            out[lane] =
                0xFF000000U | pack(c0[lane]) | (pack(c1[lane]) << 8U) | (pack(c2[lane]) << 16U);
        }
    }
#endif
} // namespace raster::simd
