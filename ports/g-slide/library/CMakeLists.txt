cmake_minimum_required(VERSION 3.20)
# Header-only public targets. Compiler/runtime integration belongs to the caller;
# consuming the host contracts/control target never requires a SYCL SDK.
add_library(gslide_control INTERFACE)
add_library(gslide::control ALIAS gslide_control)
target_compile_features(gslide_control INTERFACE cxx_std_17)
target_include_directories(gslide_control INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/../sycl")
add_library(gslide_sycl INTERFACE)
add_library(gslide::sycl ALIAS gslide_sycl)
target_link_libraries(gslide_sycl INTERFACE gslide::control)
