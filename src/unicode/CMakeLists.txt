if(IS_DIRECTORY "${LIBUNICODE_UCD_PATH}")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/ucd.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/ucd.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/ucd_enums.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/ucd_ostream.h"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/mktables.py" "${LIBUNICODE_UCD_PATH}"
        DEPENDS mktables.py
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating UCD API and tables from ${LIBUNICODE_UCD_PATH}"
        VERBATIM
    )
endif()

add_library(unicode STATIC
    capi.h capi.cpp
    convert.h
    utf8.h
    width.h width.cpp
    emoji_segmenter.h emoji_segmenter.cpp
    grapheme_segmenter.h
    run_segmenter.h
    script_segmenter.h script_segmenter.cpp
    ucd.h ucd.cpp
    ucd_enums.h
    ucd_ostream.h
    word_segmenter.h
)
add_library(unicode::core ALIAS unicode)

target_include_directories(unicode PUBLIC ${PROJECT_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(unicode PUBLIC fmt::fmt-header-only)

# TODO: if ARCH == x86-64
target_compile_options(unicode PUBLIC -mavx512bw)

# --------------------------------------------------------------------------------------------------------
# unicode_test
if(LIBUNICODE_TESTING)
    enable_testing()
    add_executable(unicode_test
        capi_test.cpp
        convert_test.cpp
        emoji_segmenter_test.cpp
        grapheme_segmenter_test.cpp
        run_segmenter_test.cpp
        script_segmenter_test.cpp
        test_main.cpp
        unicode_test.cpp
        utf8_test.cpp
        width_test.cpp
        word_segmenter_test.cpp
    )
    target_link_libraries(unicode_test fmt::fmt-header-only Catch2::Catch2 unicode)
    add_test(unicode_test unicode_test)
endif()
