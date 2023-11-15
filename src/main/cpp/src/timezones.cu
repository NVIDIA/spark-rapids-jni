/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "timezones.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

using column = cudf::column;
using column_device_view = cudf::column_device_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using size_type = cudf::size_type;
using struct_view = cudf::struct_view;

namespace {

/**
 * @brief adjust timestamp value by offset
 * 
 * @tparam typestamp_type type of the input and output timestamp
 * @param timestamp input timestamp
 * @param transitions the transitions 
 */
template <typename timestamp_type>
__device__ timestamp_type convert_timestamp_timezone(timestamp_type const& timestamp,
    lists_column_device_view const& transitions,
    size_type tz_index,
    bool to_utc) {

  using duration_type = typename timestamp_type::duration;
  using cuda::std::chrono::duration_cast;

  auto epoch_seconds = static_cast<int64_t>(duration_cast<cudf::duration_s>(timestamp.time_since_epoch()).count());

  auto const tz_transitions = cudf::list_device_view{transitions, tz_index};

  auto size = tz_transitions.size();

  cudf::device_span<int64_t const> transition_times(
    &(transitions.child().child(to_utc ? 1 : 0).data<int64_t>()[tz_transitions.element_offset(0)]),
    static_cast<size_t>(size));
    

  auto idx = thrust::upper_bound(thrust::seq, 
    transition_times.begin(),
    transition_times.end(),
    epoch_seconds) - transition_times.begin();

  auto const list_offset = tz_transitions.element_offset(size_type(idx-1));
  auto const utc_offset = duration_cast<duration_type>(
    cudf::duration_s{static_cast<int64_t>(
      transitions.child().child(2).element<int32_t>(list_offset)
    )});
  return to_utc ? timestamp - utc_offset : timestamp + utc_offset;
} 

}

namespace spark_rapids_jni {

std::unique_ptr<column> convert_timestamp_to_utc(cudf::column_view const& input,
  cudf::table_view const& transitions,
  size_type tz_index,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) {

  auto type = input.type().id();
  auto num_rows = input.size();

  // get the fixed transitions
  auto const ft_cdv_ptr = column_device_view::create(transitions.column(0), stream);
  lists_column_device_view fixed_transitions = cudf::detail::lists_column_device_view{*ft_cdv_ptr};

  auto results = cudf::make_timestamp_column(input.type(), input.size(),
                                            cudf::detail::copy_bitmask(input, stream, mr),
                                            input.null_count(), stream, mr);

  switch (type) {
    case cudf::type_id::TIMESTAMP_DAYS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_D>(),
        [input_data = input.begin<cudf::timestamp_D>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_D>(timestamp, fixed_transitions, tz_index, true);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_SECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_s>(),
        [input_data = input.begin<cudf::timestamp_s>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_s>(timestamp, fixed_transitions, tz_index, true);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_ms>(),
        [input_data = input.begin<cudf::timestamp_ms>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_ms>(timestamp, fixed_transitions, tz_index, true);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_us>(),
        [input_data = input.begin<cudf::timestamp_us>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_us>(timestamp, fixed_transitions, tz_index, true);
          }
      );
      break;
    default: 
      CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }


  return results;
}

std::unique_ptr<column> convert_utc_timestamp_to_timezone(cudf::column_view const& input,
  cudf::table_view const& transitions,
  size_type tz_index,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) {

  auto const type = input.type().id();
  auto num_rows = input.size();

  // get the fixed transitions
  auto const ft_cdv_ptr = column_device_view::create(transitions.column(0), stream);
  lists_column_device_view fixed_transitions = cudf::detail::lists_column_device_view{*ft_cdv_ptr};

  auto results = cudf::make_timestamp_column(input.type(), input.size(),
                                            cudf::detail::copy_bitmask(input, stream, mr),
                                            input.null_count(), stream, mr);

  switch (type) {
    case cudf::type_id::TIMESTAMP_DAYS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_D>(),
        [input_data = input.begin<cudf::timestamp_D>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_D>(timestamp, fixed_transitions, tz_index, false);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_SECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_s>(),
        [input_data = input.begin<cudf::timestamp_s>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_s>(timestamp, fixed_transitions, tz_index, false);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_ms>(),
        [input_data = input.begin<cudf::timestamp_ms>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_ms>(timestamp, fixed_transitions, tz_index, false);
          }
      );
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: 
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        results->mutable_view().begin<cudf::timestamp_us>(),
        [input_data = input.begin<cudf::timestamp_us>(), fixed_transitions, tz_index] __device__ (auto const i) {
            auto const timestamp = input_data[i];
            return convert_timestamp_timezone<cudf::timestamp_us>(timestamp, fixed_transitions, tz_index, false);
          }
      );
      break;
    default: 
      CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }
  return results;
}

}
