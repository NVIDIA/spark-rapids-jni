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
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <thrust/binary_search.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <optional>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using scalar_i64               = cudf::numeric_scalar<int64_t>;
using size_type                = cudf::size_type;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

namespace {

// This device functor uses a binary search to find the instant of the transition
// to find the right offset to do the transition.
// To transition to UTC: do a binary search on the tzInstant child column and subtract
// the offset.
// To transition from UTC: do a binary search on the utcInstant child column and add
// the offset.
template <typename timestamp_type>
struct convert_timestamp_tz_functor {
  using duration_type = typename timestamp_type::duration;

  // The list column of transitions to figure out the correct offset
  // to adjust the timestamp. The type of the values in this column is
  // LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const transitions;
  // the index of the specified zone id in the transitions table
  size_type const tz_index;
  // whether we are converting to UTC or converting to the timezone
  bool const to_utc;

  /**
   * @brief Convert the timestamp value to either UTC or a specified timezone
   * @param timestamp input timestamp
   *
   */
  __device__ timestamp_type operator()(timestamp_type const& timestamp) const
  {
    auto const utc_instants = transitions.child().child(0);
    auto const tz_instants  = transitions.child().child(1);
    auto const utc_offsets  = transitions.child().child(2);

    auto const epoch_seconds = static_cast<int64_t>(
      cuda::std::chrono::duration_cast<cudf::duration_s>(timestamp.time_since_epoch()).count());
    auto const tz_transitions = cudf::list_device_view{transitions, tz_index};
    auto const list_size      = tz_transitions.size();

    auto const transition_times = cudf::device_span<int64_t const>(
      (to_utc ? tz_instants : utc_instants).data<int64_t>() + tz_transitions.element_offset(0),
      static_cast<size_t>(list_size));

    auto const it = thrust::upper_bound(
      thrust::seq, transition_times.begin(), transition_times.end(), epoch_seconds);
    auto const idx         = static_cast<size_type>(thrust::distance(transition_times.begin(), it));
    auto const list_offset = tz_transitions.element_offset(idx - 1);
    auto const utc_offset  = cuda::std::chrono::duration_cast<duration_type>(
      cudf::duration_s{static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset))});
    return to_utc ? timestamp - utc_offset : timestamp + utc_offset;
  }
};

template <typename timestamp_type>
auto convert_timestamp_tz(column_view const& input,
                          table_view const& transitions,
                          size_type tz_index,
                          bool to_utc,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::detail::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  thrust::transform(
    rmm::exec_policy(stream),
    input.begin<timestamp_type>(),
    input.end<timestamp_type>(),
    results->mutable_view().begin<timestamp_type>(),
    convert_timestamp_tz_functor<timestamp_type>{fixed_transitions, tz_index, to_utc});

  return results;
}

template <typename timestamp_type>
struct time_add_functor {
  using duration_type = typename timestamp_type::duration;

  lists_column_device_view const transitions;

  size_type const tz_index;

  int64_t const duration_scalar;

  __device__ inline timestamp_type gao(timestamp_type const& timestamp,
                                       int64_t const& duration) const
  {
    if (duration == 0L) { return timestamp; }

    auto const utc_instants = transitions.child().child(0);
    auto const tz_instants  = transitions.child().child(1);
    auto const utc_offsets  = transitions.child().child(2);

    auto const epoch_seconds_utc = static_cast<int64_t>(
      cuda::std::chrono::duration_cast<cudf::duration_s>(timestamp.time_since_epoch()).count());
    // input_timestamp

    auto const tz_transitions = cudf::list_device_view{transitions, tz_index};
    auto const list_size      = tz_transitions.size();

    auto const transition_times_utc = cudf::device_span<int64_t const>(
      utc_instants.data<int64_t>() + tz_transitions.element_offset(0),
      static_cast<size_t>(list_size));

    auto const transition_times_tz = cudf::device_span<int64_t const>(
      tz_instants.data<int64_t>() + tz_transitions.element_offset(0),
      static_cast<size_t>(list_size));

    // step 1: Get offset when converting local timestamp to utc
    auto const utc_it = thrust::upper_bound(
      thrust::seq, transition_times_utc.begin(), transition_times_utc.end(), epoch_seconds_utc);
    auto utc_idx = static_cast<size_type>(thrust::distance(transition_times_utc.begin(), utc_it));
    auto const utc_list_offset = tz_transitions.element_offset(utc_idx - 1);
    auto const to_local_offset =
      static_cast<int64_t>(utc_offsets.element<int32_t>(utc_list_offset));
    // offset

    auto const to_local_offset_duration = cuda::std::chrono::duration_cast<duration_type>(
      cudf::duration_s{static_cast<int64_t>(to_local_offset)});

    auto const duration_typed = cuda::std::chrono::duration_cast<duration_type>(
      cudf::duration_us{static_cast<int64_t>(duration)});

    auto const local_timestamp_res = timestamp + to_local_offset_duration + duration_typed;

    auto const result_epoch_seconds = static_cast<int64_t>(
      cuda::std::chrono::duration_cast<cudf::duration_s>(local_timestamp_res.time_since_epoch())
        .count());

    auto const local_it = thrust::upper_bound(
      thrust::seq, transition_times_tz.begin(), transition_times_tz.end(), result_epoch_seconds);
    auto local_idx =
      static_cast<size_type>(thrust::distance(transition_times_tz.begin(), local_it));
    auto const temp_list_offset = tz_transitions.element_offset(local_idx);
    auto const temp_offset = static_cast<int64_t>(utc_offsets.element<int32_t>(temp_list_offset));
    if (local_idx != 0 && transition_times_utc[local_idx] != INT64_MAX &&
        transition_times_utc[local_idx] + temp_offset <= result_epoch_seconds) {
      local_idx += 1;
    }
    auto const local_list_offset = tz_transitions.element_offset(local_idx - 1);
    auto to_utc_offset = static_cast<int64_t>(utc_offsets.element<int32_t>(local_list_offset));
    auto const upper_bound_epoch = transition_times_tz[local_idx - 1];
    auto const upper_bound_utc   = transition_times_utc[local_idx - 1];

    auto const early_offset = static_cast<int64_t>(upper_bound_epoch - upper_bound_utc);

    bool const is_gap     = (upper_bound_utc + to_utc_offset == upper_bound_epoch);
    bool const is_overlap = !is_gap && upper_bound_utc != INT64_MIN;
    if (is_overlap) {  // overlap
      auto const overlap_before = static_cast<int64_t>(upper_bound_utc + to_utc_offset);
      auto const overlap_after  = static_cast<int64_t>(upper_bound_epoch);
      if (result_epoch_seconds >= overlap_before && result_epoch_seconds <= overlap_after) {
        if (to_local_offset == early_offset) { to_utc_offset = early_offset; }
      }
    }

    auto const to_utc_offset_duration = cuda::std::chrono::duration_cast<duration_type>(
      cudf::duration_s{static_cast<int64_t>(to_utc_offset)});

    return local_timestamp_res - to_utc_offset_duration;
  }

  __device__ timestamp_type operator()(timestamp_type const& timestamp) const
  {
    return gao(timestamp, duration_scalar);
  }

  __device__ timestamp_type operator()(timestamp_type const& timestamp,
                                       int64_t const& interval) const
  {
    return gao(timestamp, interval);
  }
};

template <typename timestamp_type>
auto time_add_with_tz(column_view const& input,
                      scalar_i64 const& duration,
                      table_view const& transitions,
                      size_type tz_index,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  if (!duration.is_valid()) {
    // return a column of nulls
    auto results = cudf::make_timestamp_column(
      input.type(), input.size(), cudf::mask_state::ALL_NULL, stream, mr);
    return results;
  }

  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::detail::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  thrust::transform(
    rmm::exec_policy(stream),
    input.begin<timestamp_type>(),
    input.end<timestamp_type>(),
    results->mutable_view().begin<timestamp_type>(),
    time_add_functor<timestamp_type>{fixed_transitions, tz_index, duration.value()});

  return results;
}

template <typename timestamp_type>
auto time_add_with_tz(column_view const& input,
                      column_view const& duration,
                      table_view const& transitions,
                      size_type tz_index,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(cudf::table_view{{input, duration}}, stream, mr);

  auto results = cudf::make_timestamp_column(
    input.type(), input.size(), rmm::device_buffer(null_mask, stream), null_count, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<timestamp_type>(),
                    input.end<timestamp_type>(),
                    duration.begin<int64_t>(),
                    results->mutable_view().begin<timestamp_type>(),
                    time_add_functor<timestamp_type>{fixed_transitions, tz_index, 0L});

  return results;
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<column> convert_timestamp(column_view const& input,
                                          table_view const& transitions,
                                          size_type tz_index,
                                          bool to_utc,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  auto const type = input.type().id();

  switch (type) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return convert_timestamp_tz<cudf::timestamp_s>(
        input, transitions, tz_index, to_utc, stream, mr);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return convert_timestamp_tz<cudf::timestamp_ms>(
        input, transitions, tz_index, to_utc, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return convert_timestamp_tz<cudf::timestamp_us>(
        input, transitions, tz_index, to_utc, stream, mr);
    default: CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }
}

std::unique_ptr<column> convert_timestamp_to_utc(column_view const& input,
                                                 table_view const& transitions,
                                                 size_type tz_index,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  return convert_timestamp(input, transitions, tz_index, true, stream, mr);
}

std::unique_ptr<column> convert_utc_timestamp_to_timezone(column_view const& input,
                                                          table_view const& transitions,
                                                          size_type tz_index,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::mr::device_memory_resource* mr)
{
  return convert_timestamp(input, transitions, tz_index, false, stream, mr);
}

std::unique_ptr<column> time_add(column_view const& input,
                                 scalar_i64 const& duration,
                                 table_view const& transitions,
                                 size_type tz_index,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto const type = input.type().id();

  switch (type) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return time_add_with_tz<cudf::timestamp_s>(
        input, duration, transitions, tz_index, stream, mr);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return time_add_with_tz<cudf::timestamp_ms>(
        input, duration, transitions, tz_index, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return time_add_with_tz<cudf::timestamp_us>(
        input, duration, transitions, tz_index, stream, mr);
    default: CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }
}

std::unique_ptr<column> time_add(column_view const& input,
                                 column_view const& duration,
                                 table_view const& transitions,
                                 size_type tz_index,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto const type = input.type().id();

  switch (type) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return time_add_with_tz<cudf::timestamp_s>(
        input, duration, transitions, tz_index, stream, mr);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return time_add_with_tz<cudf::timestamp_ms>(
        input, duration, transitions, tz_index, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return time_add_with_tz<cudf::timestamp_us>(
        input, duration, transitions, tz_index, stream, mr);
    default: CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }
}

}  // namespace spark_rapids_jni
