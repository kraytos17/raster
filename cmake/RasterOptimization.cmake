# Small, opt-in build tuning shared by project targets.
include_guard(GLOBAL)

if(RASTER_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _raster_ipo_supported OUTPUT _raster_ipo_error LANGUAGES CXX)
    if(NOT _raster_ipo_supported)
        message(FATAL_ERROR "raster LTO requested but unavailable: ${_raster_ipo_error}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "raster: link-time optimization enabled")
endif()

if(RASTER_KEEP_FRAME_POINTER)
    message(STATUS "raster: frame pointers preserved for profiling")
endif()

if(RASTER_ENABLE_NATIVE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "RASTER_ENABLE_NATIVE is unsupported by ${CMAKE_CXX_COMPILER_ID}")
elseif(RASTER_ENABLE_NATIVE)
    message(STATUS "raster: native CPU tuning enabled")
endif()

function(raster_apply_optimization target)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-O0>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-g3>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3>
        $<$<BOOL:${RASTER_USE_PIPE}>:-pipe>
        $<$<BOOL:${RASTER_KEEP_FRAME_POINTER}>:-fno-omit-frame-pointer>
        $<$<BOOL:${RASTER_ENABLE_NATIVE}>:-march=native;-fno-math-errno;-fno-trapping-math;-DNDEBUG>
    )
    if(RASTER_ENABLE_LTO)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-g>
        )
    endif()
endfunction()
