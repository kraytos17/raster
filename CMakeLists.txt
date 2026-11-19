cmake_minimum_required(VERSION 3.28)

project(raster VERSION 0.1.0
    DESCRIPTION "A C++23 data-oriented software rasterizer"
    LANGUAGES CXX
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.2")
    message(FATAL_ERROR "raster requires GCC >= 14.2 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.1")
    message(FATAL_ERROR "raster requires Clang >= 18.1 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

option(RASTER_BUILD_TESTS "Build raster tests" ON)
option(RASTER_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(RASTER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(RASTER_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(RASTER_ENABLE_LTO "Enable link-time optimization" OFF)
option(RASTER_ENABLE_CLANG_TIDY "Run clang-tidy during compilation" OFF)
option(RASTER_ENABLE_IWYU "Run include-what-you-use during compilation" OFF)
option(RASTER_ENABLE_GCC_ANALYZER "Run GCC static analyzer" OFF)
option(RASTER_USE_PIPE "Use -pipe during compilation" ON)
option(RASTER_USE_FAST_LINKER "Prefer mold, then lld, over the default" ON)
option(RASTER_KEEP_FRAME_POINTER "Keep frame pointers for profiling" OFF)
option(RASTER_ENABLE_NATIVE "Tune optimized code for the build host CPU" OFF)
set(RASTER_CACHE_TOOL "AUTO" CACHE STRING "Compiler cache: AUTO, CCACHE, SCCACHE or NONE")
set_property(CACHE RASTER_CACHE_TOOL PROPERTY STRINGS AUTO CCACHE SCCACHE NONE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RasterCompilerCache.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RasterLinker.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RasterSanitizers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RasterOptimization.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RasterLint.cmake)

find_package(SDL3 CONFIG REQUIRED)

add_library(raster_engine INTERFACE)
add_library(raster::engine ALIAS raster_engine)
target_compile_features(raster_engine INTERFACE cxx_std_23)
target_sources(raster_engine INTERFACE
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/raster/memory/framebuffer.hpp
        include/raster/platform/sdl3_presenter.hpp
)
target_include_directories(raster_engine INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(raster_engine INTERFACE SDL3::SDL3)
target_compile_options(raster_engine INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-fno-rtti>
)

add_executable(raster
    src/main.cpp
    src/platform/sdl3_presenter.cpp
)
target_link_libraries(raster PRIVATE raster::engine)
raster_apply_optimization(raster)
raster_apply_sanitizers(raster)
raster_apply_static_analysis(raster)
raster_apply_clang_tidy(raster)

if(RASTER_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
        set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG 08092139210881f01a34e60c5f2fd6afa0ea4024
            GIT_SHALLOW FALSE
            GIT_SUBMODULES ""
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_executable(raster_tests tests/test_framebuffer.cpp)
    target_link_libraries(raster_tests PRIVATE raster::engine Catch2::Catch2WithMain)
    raster_apply_optimization(raster_tests)
    raster_apply_sanitizers(raster_tests)
    raster_apply_static_analysis(raster_tests)
    raster_apply_clang_tidy(raster_tests)
    include(Catch)
    catch_discover_tests(raster_tests)
endif()

install(TARGETS raster RUNTIME DESTINATION bin)
install(TARGETS raster_engine EXPORT rasterTargets FILE_SET HEADERS DESTINATION include)
install(EXPORT rasterTargets NAMESPACE raster:: DESTINATION lib/cmake/raster)
