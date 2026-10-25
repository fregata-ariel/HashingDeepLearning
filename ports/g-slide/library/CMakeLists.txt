cmake_minimum_required(VERSION 3.20)
project(GSlide VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Experimental header-only package. Compiler/runtime integration belongs to
# the consumer; configuring/installing this project never needs a SYCL SDK.
get_filename_component(GSLIDE_PORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(GSLIDE_PORTS_ROOT "${GSLIDE_PORT_ROOT}/.." ABSOLUTE)
add_library(gslide_control INTERFACE)
add_library(gslide::control ALIAS gslide_control)
set_target_properties(gslide_control PROPERTIES EXPORT_NAME control)
target_compile_features(gslide_control INTERFACE cxx_std_17)
target_include_directories(gslide_control INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${GSLIDE_PORTS_ROOT}>
  $<BUILD_INTERFACE:${GSLIDE_PORT_ROOT}/include>
  $<BUILD_INTERFACE:${GSLIDE_PORT_ROOT}/sycl>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/g-slide/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/g-slide/sycl>)
add_library(gslide_sycl INTERFACE)
add_library(gslide::sycl ALIAS gslide_sycl)
set_target_properties(gslide_sycl PROPERTIES EXPORT_NAME sycl)
target_link_libraries(gslide_sycl INTERFACE gslide::control)

install(TARGETS gslide_control gslide_sycl EXPORT GSlideTargets)
install(EXPORT GSlideTargets NAMESPACE gslide:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GSlide)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Preserve the maintained implementation's relative include structure; these
# are the actual tested headers, with no second maintained copy or rewrite.
install(DIRECTORY "${GSLIDE_PORT_ROOT}/sycl/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/g-slide/sycl FILES_MATCHING PATTERN "*.hpp")
install(FILES "${GSLIDE_PORT_ROOT}/include/TrainingControl.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/g-slide/include)
install(FILES "${GSLIDE_PORT_ROOT}/LICENSE" DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/GSlide)
install(FILES "${GSLIDE_PORT_ROOT}/../../papers/g-slide-source-pin.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/GSlide)
configure_package_config_file(cmake/GSlideConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GSlideConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GSlide)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GSlideConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GSlideConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/GSlideConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GSlide)
