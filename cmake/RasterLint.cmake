# Add optional lint targets only when the corresponding tool is installed.
include_guard(GLOBAL)

file(GLOB_RECURSE RASTER_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/raster/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
)

find_program(RASTER_CLANG_FORMAT_PROGRAM clang-format)
if(RASTER_CLANG_FORMAT_PROGRAM AND RASTER_FORMAT_SOURCES)
    add_custom_target(format
        COMMAND "${RASTER_CLANG_FORMAT_PROGRAM}" -i -style=file ${RASTER_FORMAT_SOURCES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Formatting raster sources"
    )
    add_custom_target(check-format
        COMMAND "${RASTER_CLANG_FORMAT_PROGRAM}" --dry-run --Werror -style=file ${RASTER_FORMAT_SOURCES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Checking raster source formatting"
    )
endif()

find_program(RASTER_CPPCHECK_PROGRAM cppcheck)
if(RASTER_CPPCHECK_PROGRAM)
    add_custom_target(cppcheck
        COMMAND "${RASTER_CPPCHECK_PROGRAM}" --enable=warning,performance,portability
            --std=c++23 --language=c++ --error-exitcode=1
            --project=${PROJECT_BINARY_DIR}/compile_commands.json
            --suppressions-list=${PROJECT_SOURCE_DIR}/suppressions/cppcheck.supp
            -i ${PROJECT_BINARY_DIR}/_deps
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Running cppcheck"
        VERBATIM
    )
endif()

set(_raster_lint_targets "")
if(TARGET check-format)
    list(APPEND _raster_lint_targets check-format)
endif()
if(TARGET cppcheck)
    list(APPEND _raster_lint_targets cppcheck)
endif()
if(_raster_lint_targets)
    add_custom_target(lint DEPENDS ${_raster_lint_targets})
endif()

if(RASTER_ENABLE_GCC_ANALYZER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "raster: GCC static analyzer enabled")
    else()
        message(WARNING "RASTER_ENABLE_GCC_ANALYZER only applies to GCC")
    endif()
endif()

function(raster_apply_static_analysis target)
    if(RASTER_ENABLE_GCC_ANALYZER AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -fanalyzer)
    endif()
endfunction()

if(RASTER_ENABLE_CLANG_TIDY)
    find_program(RASTER_CLANG_TIDY_PROGRAM clang-tidy)
    if(RASTER_CLANG_TIDY_PROGRAM)
        message(STATUS "raster: clang-tidy enabled (${RASTER_CLANG_TIDY_PROGRAM})")
        function(raster_apply_clang_tidy target)
            set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY
                "${RASTER_CLANG_TIDY_PROGRAM};--header-filter=^${PROJECT_SOURCE_DIR}/(include|src|tests)/raster/.*"
            )
        endfunction()
    else()
        message(WARNING "RASTER_ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
    endif()
endif()

if(NOT COMMAND raster_apply_clang_tidy)
    function(raster_apply_clang_tidy target)
    endfunction()
endif()

if(RASTER_ENABLE_IWYU)
    find_program(RASTER_IWYU_PROGRAM include-what-you-use)
    if(RASTER_IWYU_PROGRAM)
        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${RASTER_IWYU_PROGRAM}")
        message(STATUS "raster: include-what-you-use enabled (${RASTER_IWYU_PROGRAM})")
    else()
        message(WARNING "RASTER_ENABLE_IWYU is ON but include-what-you-use was not found")
    endif()
endif()
