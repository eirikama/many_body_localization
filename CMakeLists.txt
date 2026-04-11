cmake_minimum_required(VERSION 3.16)
project(heisenberg_diag LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(
    -O2
    -march=native
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)
add_link_options(
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# ── Dependencies ──────────────────────────────────────────────────────────────

# Eigen (header-only)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# LAPACK / OpenBLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# LAPACKE (C interface) — find header and shared lib
find_path(LAPACKE_INCLUDE_DIR lapacke.h
    PATHS /usr/include /usr/local/include
)
find_library(LAPACKE_LIBRARY
    NAMES lapacke
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

if(NOT LAPACKE_INCLUDE_DIR OR NOT LAPACKE_LIBRARY)
    message(FATAL_ERROR "Could not find LAPACKE. Install liblapacke-dev.")
endif()

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(solve_random_heisenberg src/heisenberg_diag.cpp)

target_include_directories(solve_random_heisenberg PRIVATE
    ${LAPACKE_INCLUDE_DIR}
)

target_link_libraries(solve_random_heisenberg PRIVATE
    Eigen3::Eigen
    ${LAPACKE_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

# ── Status summary ────────────────────────────────────────────────────────────
message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen3       : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "LAPACKE lib  : ${LAPACKE_LIBRARY}")
message(STATUS "LAPACK libs  : ${LAPACK_LIBRARIES}")
message(STATUS "BLAS libs    : ${BLAS_LIBRARIES}")