cmake_minimum_required(VERSION 3.24)
project(gslide_cuda_validation LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_LIST_DIR}/../../tools/generate_gslide_cuda_original.py
  ${CMAKE_CURRENT_BINARY_DIR}/original.cu COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/../../tools/generate_gslide_cuda_original.py
  ${CMAKE_CURRENT_LIST_DIR}/../../third_party/g-slide/src/kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../third_party/g-slide/src/lshKnl.cu)
set(PORT "${CMAKE_CURRENT_LIST_DIR}/../../ports/g-slide")
add_library(gslide_native STATIC
  ${PORT}/src/CscActNodes.cu ${PORT}/src/GPUMultiLinkedHashTable.cu
  ${PORT}/src/LSH.cu ${PORT}/src/Layer.cu ${PORT}/src/ReluLayer.cu
  ${PORT}/src/SoftmaxLayer.cu ${PORT}/src/Network.cu
  ${PORT}/src/kernel.cu ${PORT}/src/lshKnl.cu)
target_include_directories(gslide_native PUBLIC ${PORT}/include)
target_link_libraries(gslide_native PUBLIC CUDA::cublas)
set_target_properties(gslide_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(gslide_native PUBLIC cxx_std_14 cuda_std_14)
target_compile_options(gslide_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_executable(gslide_cuda_test test_native.cu ${CMAKE_CURRENT_BINARY_DIR}/original.cu)
target_link_libraries(gslide_cuda_test PRIVATE gslide_native)
set_target_properties(gslide_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gslide_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_test(NAME native COMMAND gslide_cuda_test)
set_tests_properties(native PROPERTIES TIMEOUT 120)
