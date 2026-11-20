# raster

A C++23 CPU software rasterizer. Uses SDL3's callback lifecycle for windowing and presentation; everything else—transforms, homogeneous clipping, depth buffering, and shading—runs on the CPU.

## Features

- **Watertight 28.4 fixed-point rasterization:** 64-bit edge equations, top-left fill rule, 2×2 quad traversal, and separate coverage/interpolation barycentrics.
- **Reverse-Z 3D pipeline:** Right-handed transforms, reverse-Z depth (`1.0` near, `0.0` far, `>` depth test), cached `1/w`, and perspective-correct varying interpolation.
- **Homogeneous clipping:** Stack-allocated, fixed-capacity 6-plane Sutherland–Hodgman clipping and fan triangulation prior to perspective divide.
- **Zero-overhead shaders:** Static dispatch via C++23 concepts (no RTTI or virtual calls).
- **Asset loading:** Bounds-checked OBJ parser (`std::expected`, `std::generator`, `std::from_chars`, `std::flat_map`) and TGA type-2 (24/32-bit) loader using byte spans, `std::bit_cast`, and `std::byteswap`.
- **Texturing:** Morton and 4×4 tiled texture layouts, mip pyramids, bilinear/trilinear filtering, and 2×2 quad derivative-based mip selection.
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
just preset=release-native texture-benchmark # quad-fetch microbenchmark, sequential/shuffled
just preset=release-native texture-benchmark-scene # sphere render: 3,968 triangles, 4K texture, all layouts
just preset=release-native texture-benchmark-scene morton # isolate one scene layout for perf stat
just preset=release-native texture-benchmark-threads # serial vs 1/2/4/8 tile workers
just preset=debug-sanitized test          # run Catch2 test suite
just preset=debug-sanitized resize-gate   # headless SDL resize/minimize test under ASan
just preset=debug-sanitized clip-stress-gate # high-poly camera-through + allocation trap
just tsan                                  # ThreadSanitizer workflow for tile rendering
just free-look                             # release-native; 60 FPS-paced adaptive-res mouse look + distance-scaled WASD
just verify                               # format check, cppcheck, clang-tidy, sanitized tests
just run                                  # build and run
```

Override defaults locally by setting `RASTER_PRESET` / `RASTER_JOBS` or copying `.env.example` to `.env`. An opt-in pre-commit format hook is available via `just hooks-install`.

## Roadmap

- [x] **M1 — Framebuffer & SDL3 shell:** Callback lifecycle, HiDPI resize/minimize handling, aligned `mdspan` buffers.
- [x] **M2 — Fixed-point rasterizer:** 28.4 coordinates, top-left rule, 2×2 quads, seam/coverage tests.
- [x] **M3 — 3D & perspective interpolation:** Reverse-Z depth, shader concepts, procedural checker scene, and a Catch2 near-to-far render-path test. The checker-floor render test passes; continue to use it as a reference when changing texture filtering.
- [x] **M4 — Clipping & asset loaders:** Six-plane clipping remains the default correctness path; the experimental 2× X/Y guard band is compared against it in tests. The bundled high-poly camera-through stress gate passes 120 frames with zero C++ render allocations under ASan/UBSan. The bundled UV sphere has exact single-point poles, one nondegenerate cap triangle per sector, midpoint pole UVs, and a duplicated longitude seam; tests validate topology/winding, pole coverage, and center-ray coverage as the camera recedes. Release and sanitizer suites pass, including optimized clipping-plane intersection regressions. Manually inspected both poles and confirmed the clipping-mode comparison does not change the intended visible result. Controls: mouse look, WASD movement, Q/E vertical, F2 switches clipping policies, F3 toggles solid-color geometry diagnostics, Escape exits.
- [x] **M5 — Textures & mipmapping:** Morton and 4×4 tiled `mdspan` mappings, TGA loading, mip pyramids, bilinear/trilinear sampling, and 2×2 quad derivative-driven shader LOD are implemented. Tests check analytic derivatives/LOD and mip-filtered output on partially covered, partially depth-occluded helper quads. The replacement checker’s orientation marker, pole marker, wrap seam, and minification behavior were visually checked in free-look; the intentional marker/seam artifacts are diagnostic texture features. `texture-benchmark` is a memory-access microbenchmark; `texture-benchmark-scene` renders the bundled sphere with a generated 4K checker and each layout, with texture construction outside the timed frame loop. Latest CachyOS/GCC 16.2 native Release/LTO medians at 640×480 (3 frames/layout): `layout_right` 41.0 ms/frame, tiled 42.0 ms, Morton 54.3 ms. These are one-host, scene-specific results, not a general ranking.

  Initial measurement on CachyOS/GCC 16.2, native Release/LTO (64 MiB texture; median ns/quad; misses from isolated `perf stat -e cache-references,cache-misses` runs):

  | Layout | Sequential | Shuffled |
  | --- | ---: | ---: |
  | `layout_right` | 1.57 ns/quad; 9.79M misses | 16.24 ns/quad; 85.40M misses |
  | `tiled4x4` | 3.84 ns/quad; 12.50M misses | 25.34 ns/quad; 88.02M misses |
  | Morton | 35.74 ns/quad; 28.17M misses | 109.12 ns/quad; 73.88M misses |

  These numbers are specific to this CPU, compiler, and benchmark access pattern—not a general layout ranking. Row-major was fastest in this run; Morton had fewer shuffled cache misses but much higher time, indicating mapping overhead matters too. The benchmark excludes rasterization and shader work.
- [x] **M6 — Multithreaded tile engine:** 16×16 stable-order tile bins, a persistent `std::jthread` work-stealing executor, and tile-bounded rasterization that preserves global 2×2 helper quads. `TileRenderer` is integrated into the app; serial/threaded framebuffer equivalence is tested, including color and depth, and the allocation probe counts across worker threads. `just preset=debug-sanitized clip-stress-gate` passes 120 high-poly frames with zero render allocations; `just tsan` runs the full 62-test suite under ThreadSanitizer. `texture-benchmark-threads` (3,968-triangle sphere, 4K tiled checker, 640×480, 7 measured frames) scales stably across six host runs: 1 worker stays ~0.9× serial, 2/4/8 workers improve, 8 workers reach ~3.6–4.3× serial (latest three runs: serial 33.1/33.1/32.8 ms, 8-worker 7.6/8.0/7.6 ms). Per-thread transient arenas were intentionally omitted: per-tile work is stack-only and frame buffers are reused, so the zero-allocation gate already passes.

### Future Ideas

- 256-bit SIMD quad evaluation (2 quads / 8 pixels per vector)
- Hierarchical-Z per-tile rejection
- Shadow mapping, Cook–Torrance GGX PBR, or tiled deferred shading
- Terminal presenter (UTF-8 half-blocks + 24-bit TrueColor)

Free-look targets 60 FPS, renders at an adaptive internal resolution (initially 1/3 scale), and reports frame-time p95/p99 plus the active scale. It reduces resolution after repeated budget misses and restores detail only after sustained headroom. Forward/back movement speed scales with camera distance, within a bounded range, so zooming out remains responsive. This is best-effort pacing, not a hard real-time guarantee.
