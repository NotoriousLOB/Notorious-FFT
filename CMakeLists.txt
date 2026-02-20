cmake_minimum_required(VERSION 3.14)
project(notoriousfft C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release if no build type specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options ----
option(NOTORIOUS_FFT_SINGLE    "Use single precision (float)"              OFF)
option(NOTORIOUS_FFT_OPENMP    "Enable OpenMP parallelization"             ON)
option(NOTORIOUS_FFT_FAST_MATH "Use fast (inaccurate) sin/cos approximation" OFF)
option(NOTORIOUS_FFT_BUILD_TESTS       "Build tests"       ON)
option(NOTORIOUS_FFT_BUILD_BENCHMARKS  "Build benchmarks"  ON)
option(NOTORIOUS_FFT_BUILD_EXAMPLES    "Build examples"    ON)
option(NOTORIOUS_FFT_BUILD_COMPREHENSIVE_BENCHMARK "Build comprehensive benchmark (FFTW3, KissFFT, PocketFFT, muFFT)" OFF)

# ---- Global optimization flags ----
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:RelWithDebInfo>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-ffast-math>
)

# ---- SIMD auto-detection ----
include(CheckCCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    check_c_compiler_flag("-march=armv8.2-a+dotprod+fp16" HAVE_ARMV82_DOTPROD)
    if(HAVE_ARMV82_DOTPROD)
        set(NOTORIOUS_FFT_ARCH_FLAGS -march=armv8.2-a+dotprod+fp16)
    else()
        check_c_compiler_flag("-march=armv8-a+simd" HAVE_ARMV8_SIMD)
        if(HAVE_ARMV8_SIMD)
            set(NOTORIOUS_FFT_ARCH_FLAGS -march=armv8-a+simd)
        endif()
    endif()
    message(STATUS "AArch64 detected — SIMD flags: ${NOTORIOUS_FFT_ARCH_FLAGS}")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    check_c_compiler_flag("-mfpu=neon-vfpv4" HAVE_NEON_VFPV4)
    if(HAVE_NEON_VFPV4)
        set(NOTORIOUS_FFT_ARCH_FLAGS -mfpu=neon-vfpv4 -mfloat-abi=hard)
    else()
        check_c_compiler_flag("-mfpu=neon" HAVE_NEON)
        if(HAVE_NEON)
            set(NOTORIOUS_FFT_ARCH_FLAGS -mfpu=neon -mfloat-abi=hard)
        endif()
    endif()
    message(STATUS "ARM 32-bit detected — SIMD flags: ${NOTORIOUS_FFT_ARCH_FLAGS}")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    check_c_compiler_flag("-mavx512f" HAVE_AVX512F)
    if(HAVE_AVX512F)
        set(NOTORIOUS_FFT_ARCH_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
        message(STATUS "x86-64 detected — AVX-512 available")
    else()
        check_c_compiler_flag("-mavx2" HAVE_AVX2)
        if(HAVE_AVX2)
            set(NOTORIOUS_FFT_ARCH_FLAGS -mavx2 -mfma)
            message(STATUS "x86-64 detected — AVX2 available")
        else()
            check_c_compiler_flag("-msse4.2" HAVE_SSE42)
            if(HAVE_SSE42)
                set(NOTORIOUS_FFT_ARCH_FLAGS -msse4.2)
                message(STATUS "x86-64 detected — SSE4.2 available")
            endif()
        endif()
    endif()
endif()

# ---- Amalgamation: generate include/notorious_fft.h from module files ----
set(NOTORIOUS_FFT_MODULES_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NOTORIOUS_FFT_AMALGAM_OUT  ${CMAKE_CURRENT_SOURCE_DIR}/include/notorious_fft.h)
set(NOTORIOUS_FFT_AMALGAM_SCRIPT ${NOTORIOUS_FFT_MODULES_DIR}/amalgamate.py)

file(GLOB NOTORIOUS_FFT_MODULE_SOURCES ${NOTORIOUS_FFT_MODULES_DIR}/[0-9][0-9]_*.h)
list(SORT NOTORIOUS_FFT_MODULE_SOURCES)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT  ${NOTORIOUS_FFT_AMALGAM_OUT}
    COMMAND ${Python3_EXECUTABLE} ${NOTORIOUS_FFT_AMALGAM_SCRIPT}
            ${NOTORIOUS_FFT_MODULES_DIR}
            ${NOTORIOUS_FFT_AMALGAM_OUT}
    DEPENDS ${NOTORIOUS_FFT_AMALGAM_SCRIPT} ${NOTORIOUS_FFT_MODULE_SOURCES}
    COMMENT "Amalgamating notoriousfft modules → include/notorious_fft.h"
    VERBATIM
)

add_custom_target(notoriousfft_amalgam ALL DEPENDS ${NOTORIOUS_FFT_AMALGAM_OUT})

# ---- Fetch minfft (reference implementation, used for tests/benchmarks) ----
include(FetchContent)
FetchContent_Declare(
    minfft
    GIT_REPOSITORY https://github.com/aimukhin/minfft.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(minfft)

# minfft doesn't have a CMakeLists, build it as a library ourselves
add_library(minfft STATIC ${minfft_SOURCE_DIR}/minfft.c)
target_include_directories(minfft PUBLIC ${minfft_SOURCE_DIR})
target_compile_options(minfft PRIVATE
    $<$<CONFIG:Release>:-O3 -ffast-math>
    $<$<CONFIG:RelWithDebInfo>:-O3 -ffast-math>
    ${NOTORIOUS_FFT_ARCH_FLAGS}
)

# ---- Header-only interface library ----
add_library(notoriousfft INTERFACE)
add_dependencies(notoriousfft notoriousfft_amalgam)
target_include_directories(notoriousfft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(notoriousfft INTERFACE ${NOTORIOUS_FFT_ARCH_FLAGS})
# Required for posix_memalign
target_compile_definitions(notoriousfft INTERFACE _POSIX_C_SOURCE=200112L)

if(NOTORIOUS_FFT_SINGLE)
    target_compile_definitions(notoriousfft INTERFACE NOTORIOUS_FFT_SINGLE)
endif()

if(NOTORIOUS_FFT_FAST_MATH)
    target_compile_definitions(notoriousfft INTERFACE NOTORIOUS_FFT_FAST_MATH)
endif()

if(NOTORIOUS_FFT_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(notoriousfft INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    target_compile_definitions(notoriousfft INTERFACE NOTORIOUS_FFT_OPENMP)
endif()

# ---- Helper: build a notoriousfft C consumer (defines NOTORIOUS_FFT_IMPLEMENTATION) ----
function(notoriousfft_add_executable target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE notoriousfft m)
    target_compile_definitions(${target} PRIVATE NOTORIOUS_FFT_IMPLEMENTATION)
    if(NOTORIOUS_FFT_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    endif()
endfunction()

# ---- Helper: build a notoriousfft C++ consumer ----
function(notoriousfft_add_executable_cpp target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE notoriousfft m)
    target_compile_definitions(${target} PRIVATE NOTORIOUS_FFT_IMPLEMENTATION)
    if(NOTORIOUS_FFT_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# ---- Tests ----
if(NOTORIOUS_FFT_BUILD_TESTS)
    enable_testing()

    # C tests
    notoriousfft_add_executable(test_notoriousfft tests/test_notoriousfft.c)
    target_link_libraries(test_notoriousfft PRIVATE minfft)
    target_include_directories(test_notoriousfft PRIVATE ${minfft_SOURCE_DIR})
    add_test(NAME test_notoriousfft COMMAND test_notoriousfft)

    # C++ tests
    notoriousfft_add_executable_cpp(test_notoriousfft_cpp tests/test_notoriousfft.cpp)
    target_link_libraries(test_notoriousfft_cpp PRIVATE minfft)
    target_include_directories(test_notoriousfft_cpp PRIVATE ${minfft_SOURCE_DIR})
    add_test(NAME test_notoriousfft_cpp COMMAND test_notoriousfft_cpp)
endif()

# ---- Benchmarks ----
if(NOTORIOUS_FFT_BUILD_BENCHMARKS)
    notoriousfft_add_executable(bench benchmarks/bench.c)
    target_link_libraries(bench PRIVATE minfft)
    target_include_directories(bench PRIVATE ${minfft_SOURCE_DIR})

    notoriousfft_add_executable(bench_openmp benchmarks/bench_openmp.c)
    if(NOTORIOUS_FFT_OPENMP)
        target_link_libraries(bench_openmp PRIVATE OpenMP::OpenMP_C)
    endif()

    # C++ benchmark
    notoriousfft_add_executable_cpp(bench_cpp benchmarks/bench_cpp.cpp)
    target_link_libraries(bench_cpp PRIVATE minfft)
    target_include_directories(bench_cpp PRIVATE ${minfft_SOURCE_DIR})
    
    # Mixed sizes benchmark (NPO2, prime, multi-dimensional)
    notoriousfft_add_executable(bench_mixed_sizes benchmarks/bench_mixed_sizes.c)
    target_link_libraries(bench_mixed_sizes PRIVATE minfft)
    target_include_directories(bench_mixed_sizes PRIVATE ${minfft_SOURCE_DIR})
endif()

# ---- Examples ----
if(NOTORIOUS_FFT_BUILD_EXAMPLES)
    notoriousfft_add_executable(example_basic examples/basic.c)
    notoriousfft_add_executable_cpp(example_cpp_advanced examples/cpp_advanced.cpp)
endif()

# ---- Comprehensive Benchmark (not built by default) ----
if(NOTORIOUS_FFT_BUILD_COMPREHENSIVE_BENCHMARK)
    message(STATUS "Building comprehensive FFT benchmark")
    
    # Find FFTW3 (system library)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    
    # Fetch KissFFT
    FetchContent_Declare(
        kissfft
        GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
        GIT_TAG        131.1.0
    )
    set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
    set(KISSFFT_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kissfft)
    
    # Fetch PocketFFT (C version)
    FetchContent_Declare(
        pocketfft
        GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(pocketfft)
    
    # Fetch muFFT
    FetchContent_Declare(
        mufft
        GIT_REPOSITORY https://github.com/Themaister/muFFT.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(mufft)
    # muFFT doesn't have CMakeLists, build it ourselves
    add_library(mufft STATIC 
        ${mufft_SOURCE_DIR}/fft.c
        ${mufft_SOURCE_DIR}/kernel.c
        ${mufft_SOURCE_DIR}/cpu.c
    )
    target_include_directories(mufft PUBLIC ${mufft_SOURCE_DIR})
    target_compile_options(mufft PRIVATE -O3)
    
    # Comprehensive benchmark executable
    add_executable(comprehensive_bench benchmarks/comprehensive_bench.cpp)
    target_link_libraries(comprehensive_bench PRIVATE 
        notoriousfft
        minfft
        ${FFTW3_LIBRARIES}
        kissfft
        mufft
        m
    )
    target_include_directories(comprehensive_bench PRIVATE 
        ${FFTW3_INCLUDE_DIRS}
        ${pocketfft_SOURCE_DIR}
        ${mufft_SOURCE_DIR}
        ${minfft_SOURCE_DIR}
    )
    target_compile_definitions(comprehensive_bench PRIVATE NOTORIOUS_FFT_IMPLEMENTATION)
    target_compile_options(comprehensive_bench PRIVATE 
        ${FFTW3_CFLAGS_OTHER}
        ${NOTORIOUS_FFT_ARCH_FLAGS}
        -O3
    )
    
    # Copy Python plotting script to build directory
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/plot_results.py
        ${CMAKE_CURRENT_BINARY_DIR}/plot_results.py
        COPYONLY
    )
    
    message(STATUS "Comprehensive benchmark: comprehensive_bench")
    message(STATUS "Python plotting script: plot_results.py")
endif()
