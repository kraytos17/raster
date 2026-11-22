#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "raster/io/pbr_material_loader.hpp"
#include "raster/render/scene_renderer.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    template <typename T> const T& checked(const std::optional<T>& value) {
        if (!value.has_value()) {
            throw std::runtime_error("expected loaded material texture");
        }
        return value.value();
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("material sidecar loads factors and optional texture paths", "[asset][material]") {
        const auto material = raster::load_pbr_material(
            std::filesystem::path{RASTER_ASSET_DIR} / "materials/pbr_demo.rmat"
        );
        REQUIRE(material.has_value());
        REQUIRE(material->factors.base_color.x == 1.0F);
        REQUIRE(material->factors.base_color.y == 1.0F);
        REQUIRE(material->factors.base_color.z == 1.0F);
        REQUIRE(material->factors.metallic == 0.0F);
        REQUIRE(material->factors.perceptual_roughness == 0.65F);
        REQUIRE(material->base_color_texture.has_value());
        REQUIRE(material->metallic_roughness_texture.has_value());
        REQUIRE(
            checked(material->base_color_texture).color_space() == raster::TextureColorSpace::Srgb
        );
        REQUIRE(
            checked(material->metallic_roughness_texture).color_space()
            == raster::TextureColorSpace::EncodedBytes
        );
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("material sidecar defaults factors and leaves textures absent", "[asset][material]") {
        const auto material = raster::load_pbr_material_buffer(
            "# factors use Material defaults\n",
            std::filesystem::path{"assets"}
        );
        REQUIRE(material.has_value());
        REQUIRE(material->factors.base_color.x == 1.0F);
        REQUIRE(material->factors.base_color.y == 1.0F);
        REQUIRE(material->factors.base_color.z == 1.0F);
        REQUIRE(material->factors.metallic == 0.0F);
        REQUIRE(material->factors.perceptual_roughness == 1.0F);
        REQUIRE_FALSE(material->base_color_texture.has_value());
        REQUIRE_FALSE(material->metallic_roughness_texture.has_value());

        const auto factors_only =
            raster::load_pbr_material_buffer("metallic 0.4\n", std::filesystem::path{"assets"});
        REQUIRE(factors_only.has_value());
        REQUIRE(factors_only->factors.metallic == 0.4F);
        REQUIRE_FALSE(factors_only->base_color_texture.has_value());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("material sidecar buffer resolves sibling texture paths", "[asset][material]") {
        constexpr std::string_view source = "base_color_texture ../textures/pbr_base.tga\n"
                                            "metallic_roughness_texture ../textures/pbr_mr.tga\n";
        const auto material = raster::load_pbr_material_buffer(
            source,
            std::filesystem::path{RASTER_ASSET_DIR} / "materials"
        );
        REQUIRE(material.has_value());
        REQUIRE(material->base_color_texture.has_value());
        REQUIRE(material->metallic_roughness_texture.has_value());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "material sidecar rejects malformed and absolute texture paths", "[asset][material]"
    ) {
        const auto malformed = raster::load_pbr_material_buffer(
            "metallic 0.3 extra\n",
            std::filesystem::path{"assets"}
        );
        REQUIRE_FALSE(malformed.has_value());
        REQUIRE(malformed.error() == raster::AssetError::MalformedMaterial);

        const auto traversal = raster::load_pbr_material_buffer(
            "base_color_texture /secret.tga\n",
            std::filesystem::path{"assets/materials"}
        );
        REQUIRE_FALSE(traversal.has_value());
        REQUIRE(traversal.error() == raster::AssetError::InvalidMaterialPath);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "loaded material textures use sRGB only for base color", "[asset][material][texture]"
    ) {
        const auto material_path =
            std::filesystem::path{RASTER_ASSET_DIR} / "materials/pbr_demo.rmat";
        const auto material = raster::load_pbr_material(material_path);
        REQUIRE(material.has_value());
        REQUIRE(material->base_color_texture.has_value());
        REQUIRE(material->metallic_roughness_texture.has_value());
        REQUIRE(
            checked(material->base_color_texture).color_space() == raster::TextureColorSpace::Srgb
        );
        REQUIRE(
            checked(material->metallic_roughness_texture).color_space()
            == raster::TextureColorSpace::EncodedBytes
        );
        REQUIRE(checked(material->base_color_texture).width() == 2U);
        REQUIRE(checked(material->metallic_roughness_texture).width() == 2U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "loaded sidecar textures and factors produce same serial and tiled PBR frame",
        "[asset][material][scene]"
    ) {
        const auto loaded = raster::load_pbr_material(
            std::filesystem::path{RASTER_ASSET_DIR} / "materials/pbr_demo.rmat"
        );
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->base_color_texture.has_value());

        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        constexpr std::size_t width = 80U;
        constexpr std::size_t height = 64U;
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.5F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        raster::Framebuffer serial{width, height};
        raster::Framebuffer tiled{width, height};
        REQUIRE(
            raster::render::render_scene(
                serial,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                checked(loaded->base_color_texture),
                camera,
                raster::render::SceneShading::Pbr,
                loaded->factors,
                loaded->metallic_roughness_texture.has_value()
                    ? &checked(loaded->metallic_roughness_texture)
                    : nullptr
            )
        );
        raster::render::TileRenderer renderer{3U};
        REQUIRE(renderer.render_scene(
            tiled,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            checked(loaded->base_color_texture),
            camera,
            raster::render::SceneShading::Pbr,
            loaded->factors,
            loaded->metallic_roughness_texture.has_value()
                ? &checked(loaded->metallic_roughness_texture)
                : nullptr
        ));
        REQUIRE(tiled.color_storage == serial.color_storage);
        REQUIRE(tiled.depth_storage == serial.depth_storage);
        std::size_t non_clear_pixels = 0U;
        for (const auto pixel : serial.color_storage) {
            non_clear_pixels += pixel != 0xFF101018U ? 1U : 0U;
        }
        REQUIRE(non_clear_pixels > 100U);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
