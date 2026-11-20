# raster — CMake + Ninja task runner
#
# Keep CMakePresets.json as the source of truth for compiler, flags, and build
# directories. This Justfile only provides short, consistent commands around it.
#
# Quick start:
#   just                         list recipes
#   just build                   configure and build current preset
#   just test                    build and run current preset's tests
#   just workflow                configure, build, and test in one CMake workflow
#   just verify                  format, static checks, and sanitized tests
#   just doctor                  check required and optional tools
#
# Select a preset with `just preset=release-native build`, or set RASTER_PRESET
# in the environment / ignored .env file. Append native tool arguments after
# recipe names, e.g. `just test -R framebuffer` or `just build raster`.

set shell := ['bash', '-euo', 'pipefail', '-c']
set dotenv-load

preset := env('RASTER_PRESET', 'debug-sanitized')
jobs := env('RASTER_JOBS', num_cpus())
build_dir := shell('python3 scripts/presets.py "$1" --build-dir', preset)
just_self := quote(just_executable()) + ' --justfile ' + quote(justfile())

alias b := build
alias t := test
alias r := run
alias ci := verify

[default]
help:
    @{{ just_self }} --list --unsorted

# List public configure presets and their associated test availability.
presets:
    python3 scripts/presets.py {{ quote(preset) }}

# Print selected preset, resolved build directory, and primary tool versions.
info:
    @echo "preset      = {{ preset }}"
    @echo "build_dir   = {{ build_dir }}"
    @echo "parallelism = {{ jobs }}"
    @echo "just        = $(just --version)"
    @echo "cmake       = $(cmake --version | sed -n '1p')"
    @echo "ninja       = $(ninja --version)"
    @echo "compiler    = $(c++ --version | sed -n '1p')"

# Report required and available optional development tools.
doctor:
    python3 scripts/check_tools.py

# Configure the current CMake preset.
configure: (_need-preset 'configure' preset)
    cmake --preset {{ quote(preset) }}

# Configure, then build all or selected targets with Ninja.
build *targets: (_need-preset 'configure' preset) (_configure preset)
    cmake --build --preset {{ quote(preset) }} --parallel {{ jobs }}{{ if targets == '' { '' } else { ' --target ' + targets } }}

# Build and run this preset's tests; forwards additional arguments to CTest.
[positional-arguments]
test *args: (_need-preset 'test' preset) (_build preset)
    ctest --preset {{ quote(preset) }} --parallel {{ jobs }} --no-tests=error "$@"

# Run CMake's preset workflow (configure + build + test).
workflow: (_need-preset 'configure' preset) (_need-preset 'test' preset)
    cmake --workflow --preset {{ quote(preset) }}

# Build and launch the SDL3 app; extra arguments are passed to the application.
[positional-arguments]
run *args: (_build preset 'raster')
    ./{{ build_dir }}/raster "$@"

# Headless resize/minimize/restore verification gate driving the real SDL3
# callbacks with the dummy video driver; fails the app if the framebuffer does
# not track the final window size. Pair with debug-sanitized for the ASan gate.
resize-gate: (_build preset 'raster')
    RASTER_AUTO_RESIZE_TEST=1 SDL_VIDEODRIVER=dummy ./{{ build_dir }}/raster

# Load the bundled high-poly OBJ and sweep a deterministic camera through it
# while clipping/rasterizing each frame; fail on nonzero allocations in render.
# Run with debug-sanitized to include UBSan/ASan coverage.
clip-stress-gate: (_build preset 'raster')
    RASTER_CLIP_STRESS_TEST=1 SDL_VIDEODRIVER=dummy ./{{ build_dir }}/raster

# Interactive loaded-sphere mode, built optimized for responsive controls.
# Mouse-look, WASD strafe, Q/E vertical, F2 toggles guard band, Escape exits.
free-look:
    cmake --preset release-native
    cmake --build --preset release-native --parallel {{ jobs }} --target raster
    RASTER_FREE_LOOK_TEST=1 ./build/release-native/raster

# Compare quad-local texture fetches, or render the sphere scene with each texture layout.
texture-benchmark: (_build preset 'raster_texture_benchmark')
    ./{{ build_dir }}/raster_texture_benchmark

[positional-arguments]
texture-benchmark-scene *args: (_build preset 'raster_texture_benchmark')
    ./{{ build_dir }}/raster_texture_benchmark scene {{ args }}

# Compare serial and persistent tiled rendering at several worker counts.
texture-benchmark-threads: (_build preset 'raster_texture_benchmark')
    ./{{ build_dir }}/raster_texture_benchmark scene-threads

# Reformat all project C++ files.
format: (_build preset 'format')

# Verify formatting without modifying files.
check-format: (_build preset 'check-format')

# Run cppcheck using the selected preset's compile_commands.json.
cppcheck: (_build preset 'cppcheck')

# Run clang-tidy and the tests using the dedicated Clang preset.
tidy:
    cmake --workflow --preset tidy

# Run ThreadSanitizer on the multithreaded tile renderer.
tsan:
    cmake --workflow --preset thread-sanitized

# Full local quality gate: format, cppcheck, clang-tidy, and sanitized tests.
verify: check-format cppcheck tidy
    cmake --workflow --preset debug-sanitized
    @echo "VERIFY PASSED: format, cppcheck, clang-tidy, ASan/UBSan tests"

# Install the current build under the CMake default prefix or a supplied path.
install prefix='': (_build preset)
    cmake --install {{ quote(build_dir) }}{{ if prefix == '' { '' } else { ' --prefix ' + quote(prefix) } }}

# Install the repository-local staged C++ format hook (opt-in).
hooks-install:
    scripts/hooks-install.sh

# Run the staged-file hook checks manually without changing Git config.
hooks-run:
    ./.githooks/pre-commit

# Remove only the selected preset's generated build tree.
[confirm('Remove generated build directory ' + build_dir + '?')]
clean:
    python3 -c 'import pathlib, shutil, sys; p=pathlib.Path(sys.argv[1]); shutil.rmtree(p) if p.exists() else None' {{ quote(build_dir) }}

# Remove all generated CMake/Ninja build trees.
[confirm('Remove every generated build tree under build/?')]
clean-all:
    python3 -c 'import pathlib, shutil; p=pathlib.Path("build"); shutil.rmtree(p) if p.exists() else None'

# Private helpers.
[private]
_configure p: (_need-preset 'configure' p)
    cmake --preset {{ quote(p) }}

_build p *targets: (_configure p)
    cmake --build --preset {{ quote(p) }} --parallel {{ jobs }}{{ if targets == '' { '' } else { ' --target ' + targets } }}

[positional-arguments]
[script]
_need-preset kind name:
    python3 scripts/check_preset.py "$1" "$2"
