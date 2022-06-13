#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})
    target_compile_options(${CMAKE_BENCH_NAME}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_FLAGS}>")

    target_include_directories(${CMAKE_BENCH_NAME}
            PRIVATE "$<BUILD_INTERFACE:${CUDF_DIR}/cpp>"
                    "$<BUILD_INTERFACE:${CUDF_DIR}/cpp/include>"
                    "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}>"
                    "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}/src>")
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/benchmarks>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
                   CXX_STANDARD 17
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS ON
                   CUDA_STANDARD 17
                   CUDA_STANDARD_REQUIRED ON
        )
    target_link_libraries(${CMAKE_BENCH_NAME} nvbench::main cudf_datagen spark_rapids_jni Threads::Threads cudf_benchmark_common)
    install(
        TARGETS ${CMAKE_BENCH_NAME}
        COMPONENT testing
        DESTINATION bin/benchmarks/spark-rapids-jni
        EXCLUDE_FROM_ALL
    )
endfunction(ConfigureBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - hausdorff benchmark ---------------------------------------------------------------------------

ConfigureBench(ROW_CONVERSION_BENCH
    row_conversion.cpp)