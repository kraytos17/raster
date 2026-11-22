#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

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
#include <cmath>
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
        const std::array<float, 8>& edge1,
        const std::array<float, 8>& edge2,
        float inv_area,
        std::array<float, 8>& l0,
        std::array<float, 8>& l1,
        std::array<float, 8>& l2
    ) noexcept {
        const __m256 e1 = _mm256_loadu_ps(edge1.data());
        const __m256 e2 = _mm256_loadu_ps(edge2.data());
        const __m256 inv = _mm256_set1_ps(inv_area);
        const __m256 w0 = _mm256_mul_ps(e1, inv);
        const __m256 w1 = _mm256_mul_ps(e2, inv);
        const __m256 w2 = _mm256_sub_ps(_mm256_set1_ps(1.0F), _mm256_add_ps(w0, w1));
        _mm256_storeu_ps(l0.data(), w0);
        _mm256_storeu_ps(l1.data(), w1);
        _mm256_storeu_ps(l2.data(), w2);
    }

    namespace detail {
        // Lane-count-preserving vector types shared by the wide helpers.
        // GNU vector_size works on GCC and Clang with no guards; the scalar
        // fallback below never names them.
        using f32x8 = float __attribute__((vector_size(32)));
        using i32x8 = std::int32_t __attribute__((vector_size(32)));
        using f32x4 = float __attribute__((vector_size(16)));
        using i32x4 = std::int32_t __attribute__((vector_size(16)));
    } // namespace detail

    // Eight-lane sRGB packing matching to_argb exactly: clamp to [0, 1],
    // scale by 255, round half up, pack 0xFFRRGGBB. The truncating conversion
    // is __builtin_convertvector (C-cast semantics, same cvttps2dq instruction
    // on both compilers); shifts and ors are lane-wise integer ops.
    __attribute__((target("avx2"))) inline void quantize_pack8(
        const std::array<float, 8>& red,
        const std::array<float, 8>& green,
        const std::array<float, 8>& blue,
        std::array<std::uint32_t, 8>& out
    ) noexcept {
        const detail::f32x8 zero = _mm256_setzero_ps();
        const detail::f32x8 one = _mm256_set1_ps(1.0F);
        const detail::f32x8 scale = _mm256_set1_ps(255.0F);
        const detail::f32x8 half = _mm256_set1_ps(0.5F);
        // Inlined per channel (rather than a lambda returning the vector) so
        // no vector crosses a function boundary without AVX enabled.
        const detail::f32x8 clamped_red =
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(red.data()), zero), one);
        const detail::f32x8 clamped_green =
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(green.data()), zero), one);
        const detail::f32x8 clamped_blue =
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(blue.data()), zero), one);
        const detail::i32x8 quantized_red = __builtin_convertvector(
            _mm256_add_ps(_mm256_mul_ps(clamped_red, scale), half),
            detail::i32x8
        );
        const detail::i32x8 quantized_green = __builtin_convertvector(
            _mm256_add_ps(_mm256_mul_ps(clamped_green, scale), half),
            detail::i32x8
        );
        const detail::i32x8 quantized_blue = __builtin_convertvector(
            _mm256_add_ps(_mm256_mul_ps(clamped_blue, scale), half),
            detail::i32x8
        );

        detail::i32x8 packed = (quantized_red << 16) | (quantized_green << 8) | quantized_blue;
        packed |= static_cast<std::int32_t>(0xFF000000U);
        static_assert(sizeof(packed) == 8U * sizeof(std::uint32_t));
        std::memcpy(out.data(), &packed, sizeof(packed));
    }

    // Four-lane variant for a single 2x2 quad.
    inline void quantize_pack4(
        const std::array<float, 4>& red,
        const std::array<float, 4>& green,
        const std::array<float, 4>& blue,
        std::array<std::uint32_t, 4>& out
    ) noexcept {
        const detail::f32x4 zero = _mm_setzero_ps();
        const detail::f32x4 one = _mm_set1_ps(1.0F);
        const detail::f32x4 scale = _mm_set1_ps(255.0F);
        const detail::f32x4 half = _mm_set1_ps(0.5F);
        const auto quantize = [&](detail::f32x4 value) noexcept {
            const detail::f32x4 clamped = _mm_min_ps(_mm_max_ps(value, zero), one);
            return __builtin_convertvector(
                _mm_add_ps(_mm_mul_ps(clamped, scale), half),
                detail::i32x4
            );
        };

        detail::i32x4 packed = (quantize(_mm_loadu_ps(red.data())) << 16)
            | (quantize(_mm_loadu_ps(green.data())) << 8)
            | quantize(_mm_loadu_ps(blue.data()));

        packed |= static_cast<std::int32_t>(0xFF000000U);
        static_assert(sizeof(packed) == 4U * sizeof(std::uint32_t));
        std::memcpy(out.data(), &packed, sizeof(packed));
    }

    // Four-lane bilinear blend with the exact scalar operation order:
    // out = (v00 * (1 - fx) + v10 * fx) * (1 - fy)
    //     + (v01 * (1 - fx) + v11 * fx) * fy, one channel at a time.
    inline void bilinear_blend4(
        const std::array<float, 4>& v00,
        const std::array<float, 4>& v10,
        const std::array<float, 4>& v01,
        const std::array<float, 4>& v11,
        const std::array<float, 4>& fx,
        const std::array<float, 4>& fy,
        std::array<float, 4>& out
    ) noexcept {
        const __m128 one = _mm_set1_ps(1.0F);
        const __m128 fx_vec = _mm_loadu_ps(fx.data());
        const __m128 fy_vec = _mm_loadu_ps(fy.data());
        const __m128 wx0 = _mm_sub_ps(one, fx_vec);
        const __m128 wy0 = _mm_sub_ps(one, fy_vec);
        const __m128 top = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(v00.data()), wx0),
            _mm_mul_ps(_mm_loadu_ps(v10.data()), fx_vec)
        );
        const __m128 bottom = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(v01.data()), wx0),
            _mm_mul_ps(_mm_loadu_ps(v11.data()), fx_vec)
        );
        _mm_storeu_ps(out.data(), _mm_add_ps(_mm_mul_ps(top, wy0), _mm_mul_ps(bottom, fy_vec)));
    }

    // Four-lane trilinear level mix: out = low + (high - low) * fraction.
    // The fraction is a per-quad scalar broadcast.
    inline void trilinear_mix4(
        const std::array<float, 4>& low,
        const std::array<float, 4>& high,
        float fraction,
        std::array<float, 4>& out
    ) noexcept {
        const __m128 frac = _mm_set1_ps(fraction);
        _mm_storeu_ps(
            out.data(),
            _mm_add_ps(
                _mm_loadu_ps(low.data()),
                _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(high.data()), _mm_loadu_ps(low.data())), frac)
            )
        );
    }

    // Non-temporal 32-bit fill for write-once framebuffer planes: sequential
    // movnti stores bypass the cache (no prior fetch) and combine into full
    // lines in the store buffer, halving clear bandwidth. Any 4-byte alignment
    // works; the fence drains before raster threads read the plane.
    inline void stream_fill_u32(std::span<std::uint32_t> out, std::uint32_t value) noexcept {
        const int pattern = static_cast<int>(value);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* destination = reinterpret_cast<int*>(out.data());
        for (std::size_t index = 0; index < out.size(); ++index) {
            _mm_stream_si32(destination + index, pattern);
        }
        _mm_sfence();
    }

    // Four-lane pack of byte-range channel blends, matching pack_channels
    // exactly: round half up per channel, keep the low 8 bits, pack 0xFFRGB.
    inline void pack_channels4(
        const std::array<float, 4>& c0,
        const std::array<float, 4>& c1,
        const std::array<float, 4>& c2,
        std::array<std::uint32_t, 4>& out
    ) noexcept {
        const __m128 half = _mm_set1_ps(0.5F);
        const __m128i mask = _mm_set1_epi32(0xFF);
        const auto quantize = [&](__m128 value) noexcept {
            return _mm_and_si128(_mm_cvttps_epi32(_mm_add_ps(value, half)), mask);
        };

        __m128i packed = _mm_or_si128(
            _mm_or_si128(
                quantize(_mm_loadu_ps(c0.data())),
                _mm_slli_epi32(quantize(_mm_loadu_ps(c1.data())), 8)
            ),
            _mm_or_si128(
                _mm_slli_epi32(quantize(_mm_loadu_ps(c2.data())), 16),
                _mm_set1_epi32(static_cast<int>(0xFF000000U))
            )
        );

        static_assert(sizeof(packed) == 4U * sizeof(std::uint32_t));
        std::memcpy(out.data(), &packed, sizeof(packed));
    }

    // Four-lane ambient + Lambertian diffuse, matching FloorShader::shade_lit
    // exactly: normalize via sqrt + per-component divide (zero stays zero),
    // max against +0, gain, then per-channel multiply — same operation order.
    inline void apply_light4(
        const std::array<float, 4>& red,
        const std::array<float, 4>& green,
        const std::array<float, 4>& blue,
        const std::array<float, 4>& nx,
        const std::array<float, 4>& ny,
        const std::array<float, 4>& nz,
        float sun_x,
        float sun_y,
        float sun_z,
        float ambient,
        const std::array<float, 4>& sun_scale,
        std::array<float, 4>& out_red,
        std::array<float, 4>& out_green,
        std::array<float, 4>& out_blue
    ) noexcept {
        const __m128 zero = _mm_setzero_ps();
        const __m128 vx = _mm_loadu_ps(nx.data());
        const __m128 vy = _mm_loadu_ps(ny.data());
        const __m128 vz = _mm_loadu_ps(nz.data());
        const __m128 dot_nn =
            _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)), _mm_mul_ps(vz, vz));

        const __m128 len = _mm_sqrt_ps(dot_nn);
        const __m128 nonzero = _mm_cmpneq_ps(len, zero);
        // Masked divide: zero lanes yield exactly +0, matching the scalar
        // zero-vector branch; nonzero lanes keep the quotient untouched.
        const __m128 hx = _mm_and_ps(_mm_div_ps(vx, len), nonzero);
        const __m128 hy = _mm_and_ps(_mm_div_ps(vy, len), nonzero);
        const __m128 hz = _mm_and_ps(_mm_div_ps(vz, len), nonzero);
        const __m128 diffuse = _mm_max_ps(
            _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(hx, _mm_set1_ps(sun_x)), _mm_mul_ps(hy, _mm_set1_ps(sun_y))),
                _mm_mul_ps(hz, _mm_set1_ps(sun_z))
            ),
            zero
        );

        const __m128 gain =
            _mm_add_ps(_mm_set1_ps(ambient), _mm_mul_ps(diffuse, _mm_loadu_ps(sun_scale.data())));
        _mm_storeu_ps(out_red.data(), _mm_mul_ps(_mm_loadu_ps(red.data()), gain));
        _mm_storeu_ps(out_green.data(), _mm_mul_ps(_mm_loadu_ps(green.data()), gain));
        _mm_storeu_ps(out_blue.data(), _mm_mul_ps(_mm_loadu_ps(blue.data()), gain));
    }
#else
    // Non-x86_64 fallback with identical semantics.
    inline void barycentric_weights8(
        const std::array<float, 8>& edge1,
        const std::array<float, 8>& edge2,
        float inv_area,
        std::array<float, 8>& l0,
        std::array<float, 8>& l1,
        std::array<float, 8>& l2
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
        const std::array<float, 8>& red,
        const std::array<float, 8>& green,
        const std::array<float, 8>& blue,
        std::array<std::uint32_t, 8>& out
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
        const std::array<float, 4>& red,
        const std::array<float, 4>& green,
        const std::array<float, 4>& blue,
        std::array<std::uint32_t, 4>& out
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
        const std::array<float, 4>& v00,
        const std::array<float, 4>& v10,
        const std::array<float, 4>& v01,
        const std::array<float, 4>& v11,
        const std::array<float, 4>& fx,
        const std::array<float, 4>& fy,
        std::array<float, 4>& out
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
        const std::array<float, 4>& low,
        const std::array<float, 4>& high,
        float fraction,
        std::array<float, 4>& out
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            out[lane] = low[lane] + (high[lane] - low[lane]) * fraction;
        }
    }

    inline void pack_channels4(
        const std::array<float, 4>& c0,
        const std::array<float, 4>& c1,
        const std::array<float, 4>& c2,
        std::array<std::uint32_t, 4>& out
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

    inline void stream_fill_u32(std::span<std::uint32_t> out, std::uint32_t value) noexcept {
        for (auto& element : out) {
            element = value;
        }
    }

    inline void apply_light4(
        const std::array<float, 4>& red,
        const std::array<float, 4>& green,
        const std::array<float, 4>& blue,
        const std::array<float, 4>& nx,
        const std::array<float, 4>& ny,
        const std::array<float, 4>& nz,
        float sun_x,
        float sun_y,
        float sun_z,
        float ambient,
        const std::array<float, 4>& sun_scale,
        std::array<float, 4>& out_red,
        std::array<float, 4>& out_green,
        std::array<float, 4>& out_blue
    ) noexcept {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            const auto dot_nn = nx[lane] * nx[lane] + ny[lane] * ny[lane] + nz[lane] * nz[lane];
            const auto len = std::sqrt(dot_nn);
            const auto hx = len == 0.0F ? 0.0F : nx[lane] / len;
            const auto hy = len == 0.0F ? 0.0F : ny[lane] / len;
            const auto hz = len == 0.0F ? 0.0F : nz[lane] / len;
            const auto diffuse = std::max(hx * sun_x + hy * sun_y + hz * sun_z, 0.0F);
            const auto gain = ambient + diffuse * sun_scale[lane];

            out_red[lane] = red[lane] * gain;
            out_green[lane] = green[lane] * gain;
            out_blue[lane] = blue[lane] * gain;
        }
    }
#endif
} // namespace raster::simd
