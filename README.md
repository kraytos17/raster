# raster

A C++23 CPU software rasterizer. Uses SDL3's callback lifecycle for windowing and presentation; everything else—transforms, homogeneous clipping, depth buffering, and shading—runs on the CPU.

## Features

- **Watertight 28.4 fixed-point rasterization:** 64-bit edge equations, top-left fill rule, 2×2 quad traversal, and separate coverage/interpolation barycentrics.
- **Reverse-Z 3D pipeline:** Right-handed transforms, reverse-Z depth (`1.0` near, `0.0` far, `>` depth test), cached `1/w`, and perspective-correct varying interpolation.
- **Homogeneous clipping:** Stack-allocated, fixed-capacity 6-plane Sutherland–Hodgman clipping and fan triangulation prior to perspective divide.
- **Zero-overhead shaders:** Static dispatch via C++23 concepts (no RTTI or virtual calls).
- **Asset loading:** Bounds-checked OBJ parser (`std::expected`, `std::generator`, `std::from_chars`, `std::flat_map`) and TGA type-2 (24/32-bit) loader using byte spans, `std::bit_cast`, and `std::byteswap`.
- **Memory layout:** 64-byte-aligned flat color/depth buffers exposed via `std::mdspan` views.

## Requirements

- **Build:** CMake 3.28+, Ninja
- **Compiler:** GCC 14.2+ or Clang 18.1+ with C++23 stdlib support (`mdspan`, `expected`, `generator`, `flat_map`, `print`). Tested on GCC 16.2 and Clang 23.1.
- **Dependencies:** SDL3 (CMake config package). Catch2 3.16.1 is fetched automatically via `FetchContent` if not installed locally.
- **Optional:** `just`, `clang-format`, `clang-tidy`, `cppcheck`, `ccache`/`sccache`, `mold`/`lld`.

## Building & Running

Configure, build, and run tests using CMake workflow presets:

```sh
cmake --workflow --preset debug-sanitized
./build/debug-sanitized/raster
```

### Presets

| Preset | Compiler | Configuration |
| --- | --- | --- |
| `debug-sanitized` | GCC | Debug + ASan & UBSan |
| `release-native` | GCC | Release + `-march=native` + LTO |
| `profile-telemetry` | Clang | RelWithDebInfo + `-march=native` + LTO + frame pointers |
| `tidy` | Clang | Debug + `clang-tidy` during build |
| `analyze` | GCC | Debug + `-fanalyzer` |

### `just` Recipes

A `Justfile` wraps `CMakePresets.json` for common tasks:

```sh
just                                      # list all recipes
just doctor                               # verify toolchain and SDL3
just preset=release-native build          # build a specific preset
just preset=debug-sanitized test          # run Catch2 test suite
just preset=debug-sanitized resize-gate   # headless SDL resize/minimize test under ASan
just preset=debug-sanitized clip-stress-gate # high-poly camera-through + allocation trap
just free-look                             # release-native; 1/3-res mouse look + WASD, F2 toggles guard band
just verify                               # format check, cppcheck, clang-tidy, sanitized tests
just run                                  # build and run
```

Override defaults locally by setting `RASTER_PRESET` / `RASTER_JOBS` or copying `.env.example` to `.env`. An opt-in pre-commit format hook is available via `just hooks-install`.

## Roadmap

- [x] **M1 — Framebuffer & SDL3 shell:** Callback lifecycle, HiDPI resize/minimize handling, aligned `mdspan` buffers.
- [x] **M2 — Fixed-point rasterizer:** 28.4 coordinates, top-left rule, 2×2 quads, seam/coverage tests.
- [x] **M3 — 3D & perspective interpolation:** Reverse-Z depth, shader concepts, procedural checker scene, and a Catch2 near-to-far render-path test. Interactive visual inspection remains outstanding.
- [ ] **M4 — Clipping & asset loaders** *(in progress)*: Six-plane clipping remains the default correctness path. The high-poly sanitizer test compares it with an experimental 2× X/Y guard band; for the bundled camera-through traversal the guard band generates fewer output triangles. `just preset=debug-sanitized clip-stress-gate` verifies 120 frames and zero C++ allocations during rendering. Run `just free-look` on a display for interactive visual inspection; mouse moves the view, WASD moves horizontally, Q/E moves vertically, F2 switches clipping policies, and Escape exits.
- [ ] **M5 — Textures & mipmapping** *(started; incomplete)*: Morton and 4×4 tiled `mdspan` mappings, TGA-to-texture conversion, mip pyramid generation, nearest sampling, and a tested quad derivative/LOD helper are in place. Remaining work: compare layout performance, integrate derivatives and mip LOD selection into the shader, and test partially occluded silhouette quads for mip seams.
- [ ] **M6 — Multithreaded tile engine:** 16×16 tile binning, `std::jthread` work-stealing, per-thread transient arenas, zero-allocation frame verification, and TSan validation.

### Future Ideas

- 256-bit SIMD quad evaluation (2 quads / 8 pixels per vector)
- Hierarchical-Z per-tile rejection
- Shadow mapping, Cook–Torrance GGX PBR, or tiled deferred shading
- Terminal presenter (UTF-8 half-blocks + 24-bit TrueColor)
