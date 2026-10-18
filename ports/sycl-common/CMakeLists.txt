cmake_minimum_required(VERSION 3.20)
# Consume through add_subdirectory; compiler integration remains caller-owned.
add_library(hashing_sycl_common INTERFACE)
add_library(hashing_sycl::common ALIAS hashing_sycl_common)
target_include_directories(hashing_sycl_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(hashing_sycl_common INTERFACE cxx_std_17)
