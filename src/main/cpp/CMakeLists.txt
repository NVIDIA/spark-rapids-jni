# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

include(../../../thirdparty/cudf/rapids_config.cmake)
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
  VERSION 24.04.00
  LANGUAGES C CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" ON)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(BUILD_FAULTINJ "Configure CMake to build fault injection" ON)

message(
  VERBOSE "SPARK_RAPIDS_JNI: Build with per-thread default stream:
            ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}"
)
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build (nvbench) benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build fault injection: ${BUILD_FAULTINJ}")

set(SPARK_RAPIDS_JNI_CXX_FLAGS "")
set(SPARK_RAPIDS_JNI_CUDA_FLAGS "")
set(SPARK_RAPIDS_JNI_CXX_DEFINITIONS "")
set(SPARK_RAPIDS_JNI_CUDA_DEFINITIONS "")
set(SPARK_RAPIDS_JNI_BUILD_TESTS ${BUILD_TESTS})
set(SPARK_RAPIDS_JNI_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
set(SPARK_RAPIDS_JNI_BUILD_FAULTINJ ${BUILD_FAULTINJ})

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "SPARK_RAPIDS_JNI: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

set(CUDF_INSTALL_DIR
  "${PROJECT_BINARY_DIR}/../libcudf-install"
  CACHE STRING "path to libcudf install root"
)

set(CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR})

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

# find CCCL
include(${CUDF_DIR}/cpp/cmake/thirdparty/get_cccl.cmake)

# JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

# cudf
if(BUILD_TESTS)
  rapids_find_package(cudf REQUIRED COMPONENTS testing)
else()
  rapids_find_package(cudf REQUIRED)
endif()

# cudfjni
find_library(CUDFJNI_LIB "libcudfjni.a" REQUIRED NO_DEFAULT_PATH
  HINTS "${PROJECT_BINARY_DIR}/../libcudfjni"
)

# parquet
find_library(PARQUET_LIB "libparquet.a" REQUIRED NO_DEFAULT_PATH
  HINTS "${PROJECT_BINARY_DIR}/../libcudf-install/lib64"
  HINTS "${PROJECT_BINARY_DIR}/../libcudf-install/lib"
)

# Internal parquet headers
set (GENERATED_PARQUET_INCLUDE
    "${CUDF_DIR}/cpp/build/_deps/arrow-src/cpp/src/"
    CACHE STRING "generated parquet thrift headers"
)

# thrift
find_library(THRIFT_LIB "libthrift.a" REQUIRED NO_DEFAULT_PATH
    HINTS "${CUDF_DIR}/cpp/build/_deps/arrow-build/thrift_ep-install/lib/"
)

set(CUDFJNI_INCLUDE_DIRS
  "${CUDF_DIR}/java/src/main/native/include"
  "${CUDF_DIR}/java/src/main/native/src"
  "${GENERATED_PARQUET_INCLUDE}"
  "${CUDF_DIR}/cpp/build/_deps/arrow-build/thrift_ep-install/include/"
)

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  spark_rapids_jni SHARED
  src/BloomFilterJni.cpp
  src/CastStringJni.cpp
  src/DateTimeRebaseJni.cpp
  src/DecimalUtilsJni.cpp
  src/GpuTimeZoneDBJni.cpp
  src/HashJni.cpp
  src/HistogramJni.cpp
  src/MapUtilsJni.cpp
  src/NativeParquetJni.cpp
  src/ParseURIJni.cpp
  src/RowConversionJni.cpp
  src/SparkResourceAdaptorJni.cpp
  src/ZOrderJni.cpp
  src/bloom_filter.cu
  src/cast_decimal_to_string.cu
  src/format_float.cu
  src/cast_float_to_string.cu
  src/cast_string.cu
  src/cast_string_to_float.cu
  src/datetime_rebase.cu
  src/decimal_utils.cu
  src/get_json_object.cu
  src/histogram.cu
  src/map_utils.cu
  src/murmur_hash.cu
  src/parse_uri.cu
  src/timezones.cu
  src/utilities.cu
  src/xxhash64.cu
  src/zorder.cu
)

set_target_properties(
  spark_rapids_jni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
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
  PRIVATE "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
          "${JNI_INCLUDE_DIRS}"
          "${CUDFJNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(
  spark_rapids_jni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_DEFINITIONS}>"
                          "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_DEFINITIONS}>>"
)

# Per-thread default stream
if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(
    spark_rapids_jni PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM  CUDF_USE_PER_THREAD_DEFAULT_STREAM
  )
endif()

target_link_libraries(
  spark_rapids_jni PRIVATE
  -Wl,--whole-archive
    ${CUDFJNI_LIB}
    cudf::cudf
  -Wl,--no-whole-archive
    ${PARQUET_LIB}
    ${THRIFT_LIB}
)
set_target_properties(spark_rapids_jni PROPERTIES LINK_LANGUAGE "CXX")
# For backwards-compatibility with the cudf Java bindings and RAPIDS accelerated UDFs,
# all of the code is built into libcudf.so that is statically linked to the CUDA runtime library.
# libcudfjni.so is a stub library dynamically-linked to libcudf.so.
set_target_properties(spark_rapids_jni PROPERTIES OUTPUT_NAME "cudf")


add_library(cudfjnistub SHARED src/emptyfile.cpp)
set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
target_link_libraries(cudfjnistub -Wl,--no-as-needed $<TARGET_FILE:spark_rapids_jni> -Wl,--as-needed)
add_dependencies(cudfjnistub spark_rapids_jni)

# ##################################################################################################
# * cuFile JNI -------------------------------------------------------------------------------------

if(USE_GDS)
  include(${CUDF_DIR}/cpp/cmake/Modules/FindcuFile.cmake)
  find_library(CUFILEJNI_LIB "libcufilejni.a" REQUIRED NO_DEFAULT_PATH
    HINTS "${PROJECT_BINARY_DIR}/../libcudfjni"
  )
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
    CUDA::cudart_static
  )
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(SPARK_RAPIDS_JNI_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(SPARK_RAPIDS_JNI_BUILD_BENCHMARKS)
  # Find or install NVBench
  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  rapids_cpm_nvbench()
  add_subdirectory(benchmarks)
endif()

if(SPARK_RAPIDS_JNI_BUILD_FAULTINJ)
  add_subdirectory(faultinj)
endif()
