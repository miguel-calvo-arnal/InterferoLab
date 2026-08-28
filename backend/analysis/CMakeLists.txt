# Minimum CMake version required to process this file.
# 3.16 is reasonable for modern Linux/Windows systems without requiring very new versions.
cmake_minimum_required(VERSION 3.16)

# Project name and languages used (C++ only).
project(interferometry LANGUAGES CXX)

# C++ standard to use (C++17) and enforce it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force -fPIC to avoid failures when building Python modules (mandatory on Linux).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to a Release build when the user does not say otherwise (O-5).
# Single-config generators (Makefiles/Ninja) would otherwise build with NO
# optimisation flags at all; multi-config generators (MSVC) ignore this.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (default Release; use -DCMAKE_BUILD_TYPE=... to override)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Release Debug RelWithDebInfo MinSizeRel)
endif()

# ------------------------------------------------------------------
# Compilation options: warnings and diagnostics
# ------------------------------------------------------------------
# INTERFEROLAB_NATIVE_ARCH: opt-in for local builds only.
# OFF (default): -march=x86-64-v2 — safe baseline for DISTRIBUTED binaries.
#   x86-64-v3 emits AVX2/FMA instructions that cause an instant SIGILL on CPUs
#   without AVX2 (pre-2013 Intel/AMD, Atoms/Celerons, some VMs).  The committed
#   .so/.pyd are distributed, so they must stay at the v2 baseline (SSE4/AVX).
# ON: -march=native — full optimisation (v3+) for a binary that will only run
#   on the machine that compiled it.  Configure with:
#     cmake -DINTERFEROLAB_NATIVE_ARCH=ON ...
# Note: enabling FMA (v3/native) can change floating-point results at the bit
# level (fused multiply-add rounds once instead of twice).
option(INTERFEROLAB_NATIVE_ARCH
       "Optimise for the build machine's CPU (-march=native). OFF = portable x86-64-v2 baseline."
       OFF)

if (MSVC)
    add_compile_options(/W4)
else()
    # On GCC/Clang: strict warnings to catch issues early.
    add_compile_options(-Wall -Wextra -Wpedantic)

    if (INTERFEROLAB_NATIVE_ARCH)
        # Local-only build: maximum optimisation for this machine.
        add_compile_options(-O3 -march=native)
    else()
        # Distributable build: portable baseline (see comment above).
        add_compile_options(-O3 -march=x86-64-v2 -mtune=generic)
    endif()
endif()

# ------------------------------------------------------------------
# OpenMP: CPU parallelisation
# ------------------------------------------------------------------
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    if (MSVC)
        # MSVC's default /openmp only implements OpenMP 2.0, which has no
        # '#pragma omp simd' -> error C7660 in filters.cpp.
        # /openmp:experimental keeps the same vcomp runtime (auto-linked by the
        # compiler, so no extra library is needed here) and adds the SIMD
        # directives.  We do NOT link OpenMP::OpenMP_CXX in this case because it
        # would also inject plain /openmp and MSVC would warn D9025 (flag
        # override) in every translation unit.
        add_compile_options(/openmp:experimental)
    else()
        set(OPENMP_LIB OpenMP::OpenMP_CXX)
    endif()
endif()

# ------------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------------
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs imgproc)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ------------------------------------------------------------------
# FFTW3 (float): library for FFT (single precision, 'fftwf_…')
# ------------------------------------------------------------------
find_library(FFTW3F_LIB NAMES fftw3f)
if(NOT FFTW3F_LIB)
    message(FATAL_ERROR "Could not find fftw3f. Install FFTW (float) dev package.")
endif()
message(STATUS "FFTW3F: ${FFTW3F_LIB}")

# fftwf_make_planner_thread_safe() lives in the threads library
# (libfftw3f_threads on Linux; combined into fftw3f in some Windows builds).
# It closes the race between plan creation and the thread_local plan-cache
# destruction that happens when OpenMP worker threads die (see hilbert.cpp).
find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads)
if(FFTW3F_THREADS_LIB)
    message(STATUS "FFTW3F threads: ${FFTW3F_THREADS_LIB}")
else()
    # Some distributions ship the threads API inside libfftw3f itself.
    # Verify the symbol resolves from the main library before giving up.
    include(CheckLibraryExists)
    check_library_exists("${FFTW3F_LIB}" fftwf_make_planner_thread_safe ""
                         FFTW3F_HAS_PLANNER_THREAD_SAFE)
    if(NOT FFTW3F_HAS_PLANNER_THREAD_SAFE)
        message(FATAL_ERROR
            "Could not find fftwf_make_planner_thread_safe "
            "(libfftw3f_threads). Install the FFTW threads library.")
    endif()
    set(FFTW3F_THREADS_LIB "")
endif()

# ------------------------------------------------------------------
# libtiff (for scanline-based reading)
# ------------------------------------------------------------------
find_package(TIFF REQUIRED)

# ------------------------------------------------------------------
# Header include paths
# ------------------------------------------------------------------
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------
# Common sources (WITHOUT main.cpp)
# These .cpp files are used both by the executable and by the Python module.
# ------------------------------------------------------------------
set(COMMON_SOURCES
    src/filters.cpp
    src/hilbert.cpp
    src/reconstruction.cpp
    src/utils.cpp
    src/analysis_api.cpp
)

# ------------------------------------------------------------------
# Original executable (with main.cpp)
# ------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/main.cpp
    ${COMMON_SOURCES}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

# Linking of libraries
# Note: the FFTW threads library must precede the main FFTW library.
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${FFTW3F_THREADS_LIB}
        ${FFTW3F_LIB}
        ${OpenCV_LIBS}
        TIFF::TIFF
        ${OPENMP_LIB}
)

# On Linux/UNIX, linking with libm may be required for mathematical functions.
if (UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

# ------------------------------------------------------------------
# Python module: analysis_backend (pybind11)
# ------------------------------------------------------------------

# Find the Python used to build the module.  Do NOT pin an exact version:
# the module must be buildable against whatever interpreter the project venv
# provides.  To select a specific interpreter (e.g. the project venv, or the
# real Python instead of a vcpkg one on Windows) configure with:
#   cmake -DPython3_EXECUTABLE=/path/to/.venv/bin/python ...
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION}")

# Instruct pybind11 to use FindPython (Python3)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# This creates the Python module analysis_backend (analysis_backend.cp313-...pyd/.so)
pybind11_add_module(analysis_backend
    analysis_backend.cpp
    ${COMMON_SOURCES}
)

target_include_directories(analysis_backend
    PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(analysis_backend
    PRIVATE
        ${FFTW3F_THREADS_LIB}
        ${FFTW3F_LIB}
        ${OpenCV_LIBS}
        TIFF::TIFF
        ${OPENMP_LIB}
)
