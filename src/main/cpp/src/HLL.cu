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

#include "HLL.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

struct estimate_fn {
  int8_t const* const sketch_begin_ptr;
  int const sketch_size;
  int const precision;
  int64_t* out;

  __device__ void operator()(cudf::size_type const idx) const
  {
    int const* reg_ptr  = reinterpret_cast<int const*>(sketch_begin_ptr + idx * sketch_size);
    auto const num_regs = 1ull << precision;
    double sum          = 0;
    int zeroes          = 0;

    for (auto reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
      int reg = reg_ptr[reg_idx];
      sum += double{1} / static_cast<double>(1ull << reg);
      zeroes += reg == 0;
    }

    auto const finalize = cuco::hyperloglog_ns::detail::finalizer(precision);
    out[idx]            = finalize(sum, zeroes);
  }
};

}  // end anonymous namespace

std::unique_ptr<cudf::column> estimate_from_hll_sketches(cudf::column_view const& input,
                                                         int precision,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  // // get HLL sketch INT8 column from list<INT8> column
  auto const hll_bytes_input = input.child(cudf::lists_column_view::child_column_index);
  int8_t const* const d_hll_bytes_input =
    cudf::column_device_view::create(hll_bytes_input, stream)->data<int8_t>();

  // Here use mask_state::ALL_VALID because Spark APPROX_COUNT_DISTINCT returns 0 for NULL values
  auto result = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, input.size(), cudf::mask_state::ALL_VALID, stream);

  // TODO pass in sketch size
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    input.size(),
    estimate_fn{d_hll_bytes_input, 32 * 1024, precision, result->mutable_view().data<int64_t>()});
  return result;
}

}  // namespace spark_rapids_jni
