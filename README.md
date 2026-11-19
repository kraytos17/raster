# raster

`raster` is a dependency-light C++23 software rasterizer project. It is built incrementally,
starting with a contiguous cache-line-aligned framebuffer and an SDL3 callback application.

## Requirements

- GCC 14.2+ or Clang 18.1+ with support for the C++23 features used by the current milestone
- CMake 3.28 or newer
- Ninja (for the supplied presets)
- SDL3
- Internet access on the first configure if Catch2 v3 is not installed system-wide
- Optional developer tools: ccache/sccache, mold/lld, clang-format, cppcheck

## Build and run

```sh
cmake --workflow --preset debug-sanitized
./build/debug-sanitized/raster
```

The required build profiles are `debug-sanitized`, `release-native`, and `profile-telemetry`.
Also available: `tidy` and `analyze`. List them with `cmake --list-presets`.

Run the same configure, build, and test loop with `just workflow`. Choose another profile with
`just preset=release-native workflow` or `RASTER_PRESET=tidy just workflow`.

The `Justfile` provides shortcuts for build, test, run, lint, install, and cleanup. `just verify`
runs the local quality gate. Copy `.env.example` to `.env` to set local defaults.
Common commands: `just build`, `just test`, `just run`, `just lint`, `just doctor`, and
`just presets`. Override the build profile with `just preset=release-native build`.

## C++ tooling

The repository includes `.clang-format`, `.clang-tidy`, `.clangd`, and `.editorconfig` settings.
With the optional tools installed, use `just format`, `just check-format`, `just cppcheck`, or
`just lint`. Configure with `--preset tidy` to run clang-tidy as part
of compilation. Run `scripts/hooks-install.sh` to opt in to the staged C++ formatting hook.

The SDL3 callback application displays the current software-rendered scene in a resizable,
high-DPI window. The 64-byte-aligned framebuffer supports depth storage and tile-local clearing.

The current renderer implements the first three milestones: a watertight 28.4 fixed-point
2x2-quad triangle rasterizer; constexpr vector/matrix arithmetic with portable compile-time
reverse-Z projection checks; and a reverse-Z perspective-correct checkerboard floor. Runtime
norms, camera rotations, field-of-view conversion, and checker indexing use `std::` math.
The constexpr reverse-Z projection constructor accepts the already-computed focal scale so its
near/far depth invariant can be checked during constant evaluation without requiring constexpr
`std::tan` on every supported C++23 compiler. Shader interpolation is constrained by
`ShaderProgram` and implemented by the `ShaderBase` tuple mixin.

Run the automated framebuffer resize/minimize/restore gate with
`just preset=debug-sanitized resize-gate`; it uses SDL3's dummy video driver and exercises the
real callback lifecycle under ASan/UBSan.
