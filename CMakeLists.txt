cmake_minimum_required(VERSION 3.16)

project(rpi5_thermal_segmentation
    VERSION 1.0.0
    DESCRIPTION "Raspberry Pi 5 thermal image acquisition and multithreshold segmentation platform"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Build configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_XLSXWRITER "Enable XLSX export through libxlsxwriter when available" ON)
option(ENABLE_WARNINGS "Enable common compiler warnings" ON)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

if(USE_XLSXWRITER)
    find_path(XLSXWRITER_INCLUDE_DIR
        NAMES xlsxwriter.h
        PATH_SUFFIXES include
    )

    find_library(XLSXWRITER_LIBRARY
        NAMES xlsxwriter
        PATH_SUFFIXES lib
    )

    if(XLSXWRITER_INCLUDE_DIR AND XLSXWRITER_LIBRARY)
        message(STATUS "libxlsxwriter found: ${XLSXWRITER_LIBRARY}")
        add_compile_definitions(HAVE_XLSXWRITER=1)
    else()
        message(WARNING "libxlsxwriter was not found. XLSX export will be disabled and CSV fallback will be used. To install it on Raspberry Pi OS: sudo apt install libxlsxwriter-dev")
        set(USE_XLSXWRITER OFF)
    endif()
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(SEGM_SOURCES
    repo/src/core/main.cpp
    repo/src/core/batch.cpp
    repo/src/core/de.cpp
    repo/src/core/dessa.cpp
    repo/src/core/scheduled_de.cpp
    repo/src/core/gwo.cpp
    repo/src/core/pso.cpp
    repo/src/core/csa.cpp
    repo/src/core/firefly.cpp
    repo/src/core/kapur_entropy.cpp
    repo/src/core/image_gray.cpp
    repo/src/core/metrics.cpp
    repo/src/core/phasecong2.cpp
    repo/src/core/histogram_plot.cpp
    repo/src/core/plot.cpp
    repo/src/core/xlsx.cpp
    repo/src/core/thermal_capture.cpp
)

add_executable(segm ${SEGM_SOURCES})

target_include_directories(segm
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/repo/src/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/repo/src/core
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(segm
    PRIVATE
        ${OpenCV_LIBS}
)

if(USE_XLSXWRITER AND XLSXWRITER_INCLUDE_DIR AND XLSXWRITER_LIBRARY)
    target_include_directories(segm PRIVATE ${XLSXWRITER_INCLUDE_DIR})
    target_link_libraries(segm PRIVATE ${XLSXWRITER_LIBRARY})
endif()

if(ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(segm PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(segm PRIVATE /W4)
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation helper
# -----------------------------------------------------------------------------
install(TARGETS segm RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Build summary
# -----------------------------------------------------------------------------
message(STATUS "Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "XLSX export enabled: ${USE_XLSXWRITER}")
