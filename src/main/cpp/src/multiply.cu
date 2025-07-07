/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "multiply.hpp"
#include "utilities.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

namespace spark_rapids_jni {

namespace {

// Functor to perform multiplication on two columns with overflow checks if ANSI mode is enabled.
struct MultiplyFn {
  cudf::column_device_view left_cdv;
  cudf::column_device_view right_cdv;
  bool is_ansi_mode;
  int* results;
  bool* validity;

  __device__ void operator()(int row_idx)
  {
    if (left_cdv.is_null(row_idx) || right_cdv.is_null(row_idx)) {
      validity[row_idx] = false;
      return;
    }

    if (is_ansi_mode) {
      long a = left_cdv.element<int>(row_idx);
      long b = right_cdv.element<int>(row_idx);
      long r = a * b;
      if (static_cast<int>(r) != r) {
        // Overflow detected
        validity[row_idx] = false;
      } else {
        validity[row_idx] = true;
        results[row_idx]  = static_cast<int>(r);
      }
    } else {
      validity[row_idx] = true;
      results[row_idx]  = left_cdv.element<int>(row_idx) * right_cdv.element<int>(row_idx);
    }
  }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> multiply(cudf::column_view const& left_input,
                                       cudf::column_view const& right_input,
                                       bool is_ansi_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  // Check input types
  if (left_input.type() != cudf::data_type(cudf::type_id::INT32) ||
      right_input.type() != cudf::data_type(cudf::type_id::INT32)) {
    throw cudf::logic_error("Both inputs must be of type INT32");
  }

  // Check input sizes
  if (left_input.size() != right_input.size()) {
    throw cudf::logic_error("Input columns must have the same size");
  }

  auto result = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32),
                                          left_input.size(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto validity =
    rmm::device_uvector<bool>(left_input.size(), stream, cudf::get_current_device_resource_ref());
  auto dcv_left  = cudf::column_device_view::create(left_input, stream);
  auto dcv_right = cudf::column_device_view::create(right_input, stream);

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    left_input.size(),
    MultiplyFn{
      *dcv_left, *dcv_right, is_ansi_mode, result->mutable_view().data<int>(), validity.data()});

  // make null mask and null count
  auto [null_mask, null_count] = cudf::detail::valid_if(
    validity.data(), validity.data() + validity.size(), cuda::std::identity{}, stream, mr);
  result->set_null_mask(std::move(null_mask), null_count, stream);

  if (is_ansi_mode) {
    // throw an error if any row has an overflow if ANSI mode is enabled
    spark_rapids_jni::throw_row_error_if_has(left_input, right_input, result->view(), stream);
  }

  return result;
}

}  // namespace spark_rapids_jni