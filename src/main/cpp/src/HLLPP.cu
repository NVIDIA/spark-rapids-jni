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
#include "HLLPP.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

// The number of bits required by register value. Register value stores num of zeros.
// XXHash64 value is 64 bits, it's safe to use 6 bits to store a register value.
constexpr int REGISTER_VALUE_BITS = 6;

// MASK binary 6 bits: 111111
constexpr uint64_t MASK = (1L << REGISTER_VALUE_BITS) - 1L;

// One long stores 10 register values
constexpr int REGISTERS_PER_LONG = 64 / REGISTER_VALUE_BITS;

__device__ inline int get_register_value(int64_t const long_10_registers, int reg_idx)
{
  int64_t shift_mask = MASK << (REGISTER_VALUE_BITS * reg_idx);
  int64_t v          = (long_10_registers & shift_mask) >> (REGISTER_VALUE_BITS * reg_idx);
  return static_cast<int>(v);
}

struct estimate_fn {
  cudf::device_span<int64_t const*> sketch_longs;
  int const precision;
  int64_t* const out;

  __device__ void operator()(cudf::size_type const idx) const
  {
    auto const num_regs = 1ull << precision;
    double sum          = 0;
    int zeroes          = 0;

    for (auto reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
      // each long contains 10 register values
      int long_col_idx    = reg_idx / REGISTERS_PER_LONG;
      int reg_idx_in_long = reg_idx % REGISTERS_PER_LONG;
      int reg             = get_register_value(sketch_longs[long_col_idx][idx], reg_idx_in_long);
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
  CUDF_EXPECTS(precision >= 4 && precision <= 18, "HLL++ requires precision in range: [4, 18]");
  auto const input_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](int i) { return input.child(i).begin<int64_t>(); });
  auto input_cols = std::vector<int64_t const*>(input_iter, input_iter + input.num_children());
  auto d_inputs   = cudf::detail::make_device_uvector_async(input_cols, stream, mr);
  auto result     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, input.size(), cudf::mask_state::ALL_VALID, stream);
  // evaluate from struct<long, ..., long>
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(0),
                     input.size(),
                     estimate_fn{d_inputs, precision, result->mutable_view().data<int64_t>()});
  return result;
}

}  // namespace spark_rapids_jni
