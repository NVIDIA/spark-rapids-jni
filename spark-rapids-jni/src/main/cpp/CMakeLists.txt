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
  "${PROJECT_BINARY_DIR}/../../../thirdparty/cudf"
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
set(CUDF_DEPS_DIR "${CUDF_DIR}/cpp/build/_deps")
set(CUDF_INCLUDE_DIRS
  "${CUDF_DIR}/cpp/include"
  "${CUDF_DIR}/java/src/main/native/include"
  "${CUDF_DIR}/java/src/main/native/src"
  "${CUDF_DEPS_DIR}/arrow-src/cpp/src"
  "${CUDF_DEPS_DIR}/nvcomp-src/include"
  "${CUDF_DEPS_DIR}/thrust-src"
)

# RMM
find_path(
  RMM_INCLUDE "rmm"
  HINTS "${CUDF_DEPS_DIR}/rmm-src/include"
        "$ENV{RMM_ROOT}/include"
        "$ENV{RMM_HOME}/include"
        "$ENV{CONDA_PREFIX}/include"
)

find_path(
  SPDLOG_INCLUDE "spdlog"
  HINTS "${CUDF_DEPS_DIR}/spdlog-src/include"
        "$ENV{RMM_ROOT}/_deps/spdlog-src/include"
        "$ENV{RMM_ROOT}/include"
        "$ENV{CONDA_PREFIX}/include"
)

# cudfjni
find_library(CUDFJNI_LIB "cudfjni" REQUIRED
  HINTS "${PROJECT_BINARY_DIR}/../../../spark-rapids-cudf/target/cmake-build"
)

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
)

target_compile_options(
  spark_rapids_jni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  spark_rapids_jni
  PRIVATE "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
          "${CUDF_INCLUDE_DIRS}"
          "${RMM_INCLUDE}"
          "${SPDLOG_INCLUDE}"
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
  spark_rapids_jni
  PRIVATE ${CUDFJNI_LIB}
)
