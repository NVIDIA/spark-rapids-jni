# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake
)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
include(rapids-cpm)
rapids_cpm_init()

# Use GPU_ARCHS if it is defined
if(DEFINED GPU_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}")
endif()

rapids_cuda_init_architectures(SPARK_RAPIDS_JNI)

project(
  SPARK_RAPIDS_JNI
  VERSION 21.12.00
  LANGUAGES C CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" ON)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)

message(VERBOSE "SPARK_RAPIDS_JNI: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")

set(SPARK_RAPIDS_JNI_CXX_FLAGS "")
set(SPARK_RAPIDS_JNI_CUDA_FLAGS "")
set(SPARK_RAPIDS_JNI_CXX_DEFINITIONS "")
set(SPARK_RAPIDS_JNI_CUDA_DEFINITIONS "")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "SPARK_RAPIDS_JNI: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

set(CUDF_DIR
  "${PROJECT_BINARY_DIR}/../../thirdparty/cudf"
  CACHE STRING "path to cudf repository"
)

# Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
rapids_find_package(CUDAToolkit REQUIRED)
include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

# cudf
set(CUDF_BUILD_DIR "${CUDF_DIR}/cpp/build")
set(CUDF_DEPS_DIR "${CUDF_BUILD_DIR}/_deps")
set(CUDF_INCLUDE_DIRS
  "${CUDF_DIR}/cpp/include"
  "${CUDF_DIR}/java/src/main/native/include"
  "${CUDF_DIR}/java/src/main/native/src"
  "${CUDF_DEPS_DIR}/arrow-src/cpp/src"
  "${CUDF_DEPS_DIR}/nvcomp-src/include"
)

# Thrust
include(${CUDF_DIR}/cpp/cmake/thirdparty/get_thrust.cmake)

# RMM
include(${CUDF_DIR}/cpp/cmake/thirdparty/get_rmm.cmake)

# Arrow
find_library(ARROW_LIB "libarrow.a" REQUIRED
  HINTS "${CUDF_DEPS_DIR}/arrow-build/release"
        "${CUDF_DEPS_DIR}/arrow-build/debug"
)

# cudf
find_library(CUDF_LIB "cudf" REQUIRED HINTS ${CUDF_BUILD_DIR})

# cudfjni
find_library(CUDFJNI_LIB "libcudfjni.a" REQUIRED
  HINTS "${PROJECT_BINARY_DIR}/../libcudfjni"
)

# ZLIB
rapids_find_package(ZLIB REQUIRED)

# ##################################################################################################
# * nvcomp ----------------------------------------------------------------------------------------

find_path(NVCOMP_INCLUDE "nvcomp" HINTS "${CUDF_BUILD_DIR}/_deps/nvcomp-src/include")

message(STATUS "NVCOMP: NVCOMP_INCLUDE set to ${NVCOMP_INCLUDE}")

set(NVCOMP_LIBNAME "libnvcomp.a")
find_library(NVCOMP_LIB "libnvcomp.a" REQUIRED HINTS "${CUDF_BUILD_DIR}/lib")

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  spark_rapids_jni SHARED
  src/RowConversionJni.cpp
  src/row_conversion.cu
)

set_target_properties(
  spark_rapids_jni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             CUDA_RUNTIME_LIBRARY Static
)

target_compile_options(
  spark_rapids_jni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  spark_rapids_jni
  PRIVATE cudf::Thrust
          "${NVCOMP_INCLUDE}"
          "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
          "${CUDF_INCLUDE_DIRS}"
          "${JNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(
  spark_rapids_jni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_DEFINITIONS}>"
                          "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_DEFINITIONS}>>"
)

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(spark_rapids_jni PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

target_link_libraries(
  spark_rapids_jni PRIVATE
  -Wl,--whole-archive
    ${CUDFJNI_LIB}
    ${CUDF_LIB}
  -Wl,--no-whole-archive
  rmm::rmm
  ${ARROW_LIB}
  ${NVCOMP_LIB}
  ZLIB::ZLIB
  CUDA::cuda_driver
)

# For backwards-compatibility with the cudf Java bindings and RAPIDS accelerated UDFs,
# all of the code is built into libcudf.so that is statically linked to the CUDA runtime library.
# libcudfjni.so is a stub library dynamically-linked to libcudf.so.
set_target_properties(spark_rapids_jni PROPERTIES OUTPUT_NAME "cudf")
add_library(cudfjnistub SHARED src/emptyfile.cpp)
set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
target_link_libraries(cudfjnistub -Wl,--no-as-needed spark_rapids_jni -Wl,--as-needed)

# ##################################################################################################
# * cuFile JNI -------------------------------------------------------------------------------------

if(USE_GDS)
  include(${CUDF_DIR}/cpp/cmake/Modules/FindcuFile.cmake)
  find_library(CUFILEJNI_LIB "libcufilejni.a" REQUIRED HINTS "${PROJECT_BINARY_DIR}/../libcudfjni")
  add_library(cufilejni SHARED src/emptyfile.cpp)
  set_target_properties(
    cufilejni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    cufilejni PRIVATE
    -Wl,--whole-archive
      ${CUFILEJNI_LIB}
    -Wl,--no-whole-archive
    spark_rapids_jni
    ${cuFile_LIBRARIES}
  )
endif()
