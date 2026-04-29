/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "map_utils.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <cstdint>
#include <memory>

namespace spark_rapids_jni {

namespace {

// Per-row state — ordering MUST be NULL=0 < VALID=1 < NULL_KEY=2 so that:
//   cub::DeviceReduce::Max(row_state) == 2  ⇒  some row would throw
//   cub::DeviceReduce::Min(row_state) == 1  ⇒  every row valid ⇒ fast-path signal
constexpr std::uint8_t STATE_NULL  = 0;  // outer-null OR row contains a null struct entry
constexpr std::uint8_t STATE_VALID = 1;  // row valid; no null key OR throw policy off
constexpr std::uint8_t STATE_NULL_KEY =
  2;  // row valid + null key in valid struct + throw policy on

constexpr char kNullKeyError[] = "Cannot use null as map key.";

// Phase 1: one thread per row. No atomics, no shared memory, no __syncthreads.
__global__ void compute_row_state_kernel(cudf::bitmask_type const* list_null_mask,
                                         cudf::size_type const* offsets,
                                         cudf::bitmask_type const* struct_null_mask,
                                         cudf::bitmask_type const* key_null_mask,
                                         bool throw_on_null_key,
                                         cudf::size_type num_rows,
                                         std::uint8_t* row_state_out,
                                         cudf::size_type* row_size_out)
{
  auto const tid = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  auto const i = tid;

  bool const outer_valid = (list_null_mask == nullptr) || cudf::bit_is_set(list_null_mask, i);
  if (!outer_valid) {
    row_state_out[i] = STATE_NULL;
    row_size_out[i]  = 0;
    return;
  }

  auto const start = offsets[i];
  auto const end   = offsets[i + 1];

  bool any_null_struct       = false;
  bool any_null_key_in_valid = false;

  for (auto j = start; j < end; ++j) {
    bool const struct_valid =
      (struct_null_mask == nullptr) || cudf::bit_is_set(struct_null_mask, j);
    if (!struct_valid) {
      any_null_struct = true;
      break;  // STATE_NULL wins; no need to inspect remaining entries
    }
    if (throw_on_null_key) {
      bool const key_valid = (key_null_mask == nullptr) || cudf::bit_is_set(key_null_mask, j);
      if (!key_valid) any_null_key_in_valid = true;
    }
  }

  auto const row_extent = end - start;
  if (any_null_struct) {
    row_state_out[i] = STATE_NULL;
    row_size_out[i]  = 0;
  } else if (any_null_key_in_valid) {
    row_state_out[i] = STATE_NULL_KEY;
    row_size_out[i]  = row_extent;
  } else {
    row_state_out[i] = STATE_VALID;
    row_size_out[i]  = row_extent;
  }
}

// Phase 2: one thread per row writes its own contiguous segment of source indices.
__global__ void build_gather_map_kernel(std::uint8_t const* row_state,
                                        cudf::size_type const* in_offsets,
                                        cudf::size_type const* out_offsets,
                                        cudf::size_type num_rows,
                                        cudf::size_type* gather_map_out)
{
  auto const tid = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  auto const i = tid;
  if (row_state[i] != STATE_VALID) return;

  auto const src_start = in_offsets[i];
  auto const dst_start = out_offsets[i];
  auto const n         = in_offsets[i + 1] - src_start;
  for (cudf::size_type k = 0; k < n; ++k) {
    gather_map_out[dst_start + k] = src_start + k;
  }
}

}  // namespace

std::unique_ptr<cudf::column> map_from_entries(cudf::column_view const& input,
                                               bool throw_on_null_key,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "map_from_entries: input must be a LIST column");
  CUDF_EXPECTS(input.offset() == 0,
               "map_from_entries: sliced input not supported; materialize first");

  auto const lists_cv = cudf::lists_column_view(input);
  auto const structs  = lists_cv.child();
  CUDF_EXPECTS(structs.type().id() == cudf::type_id::STRUCT,
               "map_from_entries: list child must be a STRUCT column");
  CUDF_EXPECTS(structs.num_children() == 2,
               "map_from_entries: struct must have exactly 2 children (KEY, VALUE)");
  CUDF_EXPECTS(structs.offset() == 0, "map_from_entries: list struct child must not be sliced");
  // The Phase 1 kernel reads `keys.null_mask()` directly using positions taken from the list
  // offsets. cuDF list offsets are stored as physical positions when the child is at offset 0,
  // but become logical (offset-aware) when the child carries its own non-zero offset — in that
  // case a raw `bit_is_set(keys.null_mask(), j)` would be misaligned by `keys.offset()` bits.
  // Reject up front so the kernel never has to reason about it.
  auto const keys = structs.child(0);
  CUDF_EXPECTS(keys.offset() == 0, "map_from_entries: struct key child must not be sliced");

  // Empty input ⇒ fast-path signal (caller incRefCount the empty input as the result).
  if (input.size() == 0) { return nullptr; }

  auto const num_rows = input.size();
  auto const temp_mr  = cudf::get_current_device_resource_ref();

  rmm::device_uvector<std::uint8_t> row_state(num_rows, stream, temp_mr);
  rmm::device_uvector<cudf::size_type> row_size(num_rows, stream, temp_mr);

  // ── Phase 1: per-row state collection ──────────────────────────────────────
  {
    constexpr int block_size = 256;
    auto const grid_size     = cudf::util::div_rounding_up_safe(num_rows, block_size);
    compute_row_state_kernel<<<grid_size, block_size, 0, stream.value()>>>(input.null_mask(),
                                                                           lists_cv.offsets_begin(),
                                                                           structs.null_mask(),
                                                                           keys.null_mask(),
                                                                           throw_on_null_key,
                                                                           num_rows,
                                                                           row_state.data(),
                                                                           row_size.data());
    CUDF_CHECK_CUDA(stream.value());
  }

  // ── Phase 1.5: device reductions + scan + bundled host pull ────────────────
  // Output offsets: scanned row_size with leading 0.  Allocated from `mr` because the
  // buffer is released into the returned column at the end of Phase 2; only true scratch
  // (row_state, row_size, gather_map, cub temp) uses temp_mr.
  rmm::device_uvector<cudf::size_type> out_offsets(num_rows + 1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(out_offsets.data(), 0, sizeof(cudf::size_type), stream.value()));
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         row_size.begin(),
                         row_size.end(),
                         out_offsets.begin() + 1);

  // Max + Min reductions on row_state — the state ordering encodes both "must throw?" and
  // "all valid?" in a single byte.
  rmm::device_scalar<std::uint8_t> max_state_d(stream, temp_mr);
  rmm::device_scalar<std::uint8_t> min_state_d(stream, temp_mr);
  {
    std::size_t bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceReduce::Max(
      nullptr, bytes, row_state.data(), max_state_d.data(), num_rows, stream.value()));
    rmm::device_buffer tmp(bytes, stream, temp_mr);
    CUDF_CUDA_TRY(cub::DeviceReduce::Max(
      tmp.data(), bytes, row_state.data(), max_state_d.data(), num_rows, stream.value()));
  }
  {
    std::size_t bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceReduce::Min(
      nullptr, bytes, row_state.data(), min_state_d.data(), num_rows, stream.value()));
    rmm::device_buffer tmp(bytes, stream, temp_mr);
    CUDF_CUDA_TRY(cub::DeviceReduce::Min(
      tmp.data(), bytes, row_state.data(), min_state_d.data(), num_rows, stream.value()));
  }

  // Single bundled D→H pull — three async copies, one stream sync.
  std::uint8_t max_state{};
  std::uint8_t min_state{};
  cudf::size_type total_entries{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &max_state, max_state_d.data(), sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &min_state, min_state_d.data(), sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(&total_entries,
                                out_offsets.data() + num_rows,
                                sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  if (max_state == STATE_NULL_KEY) { throw cudf::logic_error(kNullKeyError); }
  if (min_state == STATE_VALID) { return nullptr; }  // every row valid ⇒ input == output

  // ── Phase 2: clean output construction (no dirty intermediate result) ─────
  // 2a. Null mask from row_state via an explicit `state == STATE_VALID` predicate.  Avoids
  //     reinterpreting uint8_t bytes (which can hold 0/1/2) as `bool` lvalues — the
  //     STATE_NULL_KEY=2 case is filtered by the throw above, but the invariant should not
  //     live only as a comment: a future refactor that reorders the throw must not silently
  //     introduce UB ([conv.bool] forbids reading non-canonical bool representations).
  auto [null_mask_buf, null_count_sz] = cudf::detail::valid_if(
    row_state.begin(),
    row_state.end(),
    [] __device__(std::uint8_t state) { return state == STATE_VALID; },
    stream,
    mr);
  auto const null_count = static_cast<cudf::size_type>(null_count_sz);

  // 2b. Gather map: each STATE_VALID row writes its source indices to its scanned slot.
  rmm::device_uvector<cudf::size_type> gather_map(total_entries, stream, temp_mr);
  {
    constexpr int block_size = 256;
    auto const grid_size     = cudf::util::div_rounding_up_safe(num_rows, block_size);
    build_gather_map_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      row_state.data(), lists_cv.offsets_begin(), out_offsets.data(), num_rows, gather_map.data());
    CUDF_CHECK_CUDA(stream.value());
  }

  // 2c. Single gather over the struct child — handles arbitrary nested key/value types.
  auto const gather_map_view = cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                                                 total_entries,
                                                 static_cast<void const*>(gather_map.data()),
                                                 nullptr,
                                                 0);
  auto gathered_table        = cudf::gather(cudf::table_view{{structs}},
                                     gather_map_view,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  auto gathered_struct       = std::move(gathered_table->release()[0]);

  // 2d. Output offsets column directly from out_offsets.
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    out_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

  // 2e. Assemble the LIST<STRUCT> result with the new null mask.  The public
  // make_lists_column overload only wraps the supplied buffers — no device work — so the
  // explicit null_count keeps it off the default stream.
  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_col),
                                 std::move(gathered_struct),
                                 null_count,
                                 std::move(null_mask_buf));
}

}  // namespace spark_rapids_jni
