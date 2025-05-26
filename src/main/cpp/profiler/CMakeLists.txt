# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

# Ensure flatbuffers is built as static library
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_SHAREDLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)

include(../cmake/get_flatbuffers.cmake)

# ##################################################################################################
# * flatbuffer generation---------------------------------------------------------------------------

set(SPARK_RAPIDS_JNI_FBS_DIR "${SPARK_RAPIDS_JNI_SOURCE_DIR}/../fbs")
add_custom_command(
  OUTPUT ${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h
  DEPENDS "${SPARK_RAPIDS_JNI_FBS_DIR}/profiler.fbs"
  WORKING_DIRECTORY "${SPARK_RAPIDS_JNI_FBS_DIR}"
  VERBATIM
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
  COMMAND
    $<TARGET_FILE:flatbuffers::flatc> --cpp -o "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}" profiler.fbs
  COMMENT "Generating profiler flatbuffer code"
)

# ##################################################################################################
# * profiler JNI -----------------------------------------------------------------------------------

# Note: The spark_rapids_profile_converter is built with static linking to avoid runtime
# dependency issues. When building on older systems, consider using devtoolset to ensure
# compatibility. For example:
#   scl enable devtoolset-11 -- cmake ...
# This will use a newer GCC version while maintaining compatibility with older systems.

add_library(profilerjni SHARED
  ProfilerJni.cpp
  profiler_debug.cpp
  profiler_serializer.cpp
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/spark_rapids_jni_version.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h"
)

set_target_properties(
  profilerjni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_VISIBILITY_PRESET "hidden"
             VISIBILITY_INLINES_HIDDEN TRUE
)

target_include_directories(
  profilerjni
  PRIVATE "${JNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
          "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
          "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
)

find_library(CUPTI_LIBRARY_PATH cupti_static PATHS
  "/usr/local/cuda/lib64"
  "/usr/local/cuda/extras/CUPTI/lib64"
  "${CUDAToolkit_LIBRARY_DIR}"
  "${CUDAToolkit_LIBRARY_DIR}/extras/CUPTI/lib64"
)

if(NOT CUPTI_LIBRARY_PATH)
  message(FATAL_ERROR "Could not find static CUPTI library (cupti_static)")
endif()

target_link_libraries(profilerjni
  PRIVATE ${CUPTI_LIBRARY_PATH} nvtx3::nvtx3-cpp flatbuffers::flatbuffers
)

file(READ "${SPARK_RAPIDS_JNI_FBS_DIR}/profiler.fbs" SPARK_RAPIDS_JNI_PROFILER_SCHEMA)
configure_file(
  profiler_schema.cpp.in
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/profiler_schema.cpp"
  @ONLY
)

add_executable(spark_rapids_profile_converter
  spark_rapids_profile_converter.cpp
  initialize_nvtxw.cpp
  nvtxw3.cpp
  nvtxw3.h
  NvtxwEvents.cpp
  NvtxwEvents.h
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/profiler_schema.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/spark_rapids_jni_version.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h"
)

# Set properties for static linking
set_target_properties(
  spark_rapids_profile_converter
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  # Enable static linking of C++ runtime
  LINK_FLAGS "-static-libgcc -static-libstdc++"
  # Prefer static libraries
  CMAKE_FIND_LIBRARY_SUFFIXES ".a"
)

target_include_directories(
  spark_rapids_profile_converter
  PRIVATE
  "${CUDAToolkit_INCLUDE_DIRS}"
  "${SPARK_RAPIDS_JNI_SOURCE_DIR}"
  "${SPARK_RAPIDS_JNI_SOURCE_DIR}/profiler"
  "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
)

# Link statically with all dependencies except system libraries
target_link_libraries(spark_rapids_profile_converter
  ${CUPTI_LIBRARY_PATH}
  flatbuffers::flatbuffers
  # These are typically dynamically linked system libraries
  dl
  pthread
  rt
)

# If using devtoolset, we might want to add specific flags
if(DEFINED ENV{DEVTOOLSET_ROOT})
  target_compile_options(spark_rapids_profile_converter PRIVATE
    "-I$ENV{DEVTOOLSET_ROOT}/usr/include/c++/11"
    "-I$ENV{DEVTOOLSET_ROOT}/usr/include/c++/11/x86_64-redhat-linux"
  )
  target_link_options(spark_rapids_profile_converter PRIVATE
    "-L$ENV{DEVTOOLSET_ROOT}/usr/lib/gcc/x86_64-redhat-linux/11"
    "-Wl,-rpath,$ENV{DEVTOOLSET_ROOT}/usr/lib/gcc/x86_64-redhat-linux/11"
  )
endif()
