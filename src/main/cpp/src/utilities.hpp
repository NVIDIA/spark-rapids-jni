/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

namespace spark_rapids_jni {

constexpr char const* JNI_ERROR_AT_ROW_CLASS = "com/nvidia/spark/rapids/jni/ExceptionAtRow";

// catch a cudf::jni::error_at_row exception and throw a Java exception with the row index
// This macro is used in JNI functions to throw an ExceptionAtRow if error occurs
// ExceptionAtRow contains the row number that caused the exception
#define CATCH_EXCEPTION_AT_ROW(env, ret_val)                                    \
  catch (const spark_rapids_jni::error_at_row& e)                               \
  {                                                                             \
    if (env->ExceptionOccurred()) { return ret_val; }                           \
    jclass ex_class = env->FindClass(spark_rapids_jni::JNI_ERROR_AT_ROW_CLASS); \
    if (ex_class != NULL) {                                                     \
      jmethodID ctor_id = env->GetMethodID(ex_class, "<init>", "(I)V");         \
      if (ctor_id != NULL) {                                                    \
        jint e_row         = static_cast<jint>(e.get_row_with_error());         \
        jobject cuda_error = env->NewObject(ex_class, ctor_id, e_row);          \
        if (cuda_error != NULL) { env->Throw((jthrowable)cuda_error); }         \
      }                                                                         \
    }                                                                           \
    return ret_val;                                                             \
  }

/**
 * An runtime error that includes a row index in a column where the error occurred.
 */
struct error_at_row : public std::runtime_error {
  /**
   * @brief Constructs a error_at_row with the row index.
   *
   */
  error_at_row(cudf::size_type row_index) : std::runtime_error(""), _row_index(row_index) {}

  /**
   * @brief Get the row index with error
   *
   * @return cudf::size_type row index
   */
  [[nodiscard]] cudf::size_type get_row_with_error() const { return _row_index; }

 private:
  cudf::size_type _row_index;
};

/**
 * @brief Bitwise-or an array of equally-sized bitmask buffers into a single output buffer
 *
 * @param input The array of input bitmask buffers.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned bloom filter's memory.
 *
 */
std::unique_ptr<rmm::device_buffer> bitmask_bitwise_or(
  std::vector<cudf::device_span<cudf::bitmask_type const>> const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Throws an error with the row index if has any row is invalid for a unary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 * @param input The input column view.
 * @param result The result column view.
 */
void throw_row_error_if_has(cudf::column_view const& input,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Throws an error with the row index if has any row is invalid for a binary operation.
 * If the inputs are not null and the result is null, it means the row is invalid.
 * @param input1 The first input column view.
 * @param input2 The second input column view.
 * @param result The result column view.
 */
void throw_row_error_if_has(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Throws an error with the row index if has any row is invalid for a ternary operation.
 * If the inputs are not null and the result is null, it means the row is invalid.
 *
 * @param input1 The first input column view.
 * @param input2 The second input column view.
 * @param input3 The third input column view.
 * @param result The result column view.
 */
void throw_row_error_if_has(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& input3,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream = cudf::get_default_stream());

}  // namespace spark_rapids_jni
