# Keep sanitizer combinations explicit and reject incompatible modes early.
include_guard(GLOBAL)

if(RASTER_ENABLE_ASAN AND RASTER_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be enabled together")
endif()

set(_raster_sanitizers "")
if(RASTER_ENABLE_ASAN)
    list(APPEND _raster_sanitizers address)
endif()
if(RASTER_ENABLE_UBSAN)
    list(APPEND _raster_sanitizers undefined)
endif()
if(RASTER_ENABLE_TSAN)
    list(APPEND _raster_sanitizers thread)
endif()

if(_raster_sanitizers AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "raster sanitizers are supported only with GCC and Clang")
endif()

if(_raster_sanitizers)
    list(JOIN _raster_sanitizers "," _raster_sanitizer_arg)
    message(STATUS "raster: sanitizers enabled (${_raster_sanitizer_arg})")
endif()

function(raster_apply_sanitizers target)
    if(_raster_sanitizers)
        target_compile_options(${target} PRIVATE
            -fsanitize=${_raster_sanitizer_arg}
            -fno-omit-frame-pointer
            -g3
        )
        target_link_options(${target} PRIVATE -fsanitize=${_raster_sanitizer_arg})
    endif()
endfunction()
