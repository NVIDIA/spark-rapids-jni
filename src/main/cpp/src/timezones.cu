/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "cast_string_to_timestamp_common.hpp"
#include "datetime_utils.cuh"
#include "timezones.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/binary_search.h>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using size_type                = cudf::size_type;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

namespace {

/**
 * Functor to convert timestamps between UTC and a specific timezone.
 */
template <typename timestamp_type>
struct convert_timestamp_tz_functor {
  using duration_type = typename timestamp_type::duration;

  // Fixed offset transitions in the timezone table
  // Column type is LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const fixed_transitions;

  // DST rules in the timezone table
  // column type is LIST<INT>, if it's DST, 12 integers defines two rules
  lists_column_device_view const dst_rules;

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
    return spark_rapids_jni::convert_timestamp(
      timestamp, fixed_transitions, dst_rules, tz_index, to_utc);
  }
};

template <typename timestamp_type>
auto convert_timestamp_tz(column_view const& input,
                          table_view const& transitions,
                          size_type tz_index,
                          bool to_utc,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get the DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(transitions.column(1), stream);
  auto const dst_rules   = cudf::detail::lists_column_device_view{*dst_cdv_ptr};

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
    convert_timestamp_tz_functor<timestamp_type>{fixed_transitions, dst_rules, tz_index, to_utc});

  return results;
}

/**
 * Functor to convert timestamps between UTC and a specific timezone.
 * This is used for casting string(with timezone) to timestamp.
 * This functor can handle multiple timezones for each row.
 */
struct convert_with_timezones_fn {
  // inputs
  int64_t const* input_seconds;
  int32_t const* input_microseconds;
  uint8_t const* invalid;
  uint8_t const* tz_type;
  int32_t const* tz_offset;
  // Fixed offset transitions in the timezone table
  // Column type is LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const fixed_transitions;
  // DST rules in the timezone table
  // column type is LIST<INT>, if it's DST, 12 integers defines two rules
  lists_column_device_view const dst_rules;
  int32_t const* tz_indices;

  // outputs
  cudf::timestamp_us* output;
  uint8_t* output_mask;

  /**
   * @brief Convert the timestamp from UTC to a specified timezone
   * @param row_idx row index of the input column
   *
   */
  __device__ void operator()(cudf::size_type row_idx) const
  {
    // 1. check if the input is invalid
    if (invalid[row_idx]) {
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
      output_mask[row_idx] = 0;
      return;
    }

    // 2. convert seconds part first
    int64_t epoch_seconds = input_seconds[row_idx];
    int64_t converted_seconds;
    if (static_cast<spark_rapids_jni::TZ_TYPE>(tz_type[row_idx]) ==
        spark_rapids_jni::TZ_TYPE::FIXED_TZ) {
      // Fixed offset, offset is in seconds, add the offset
      // E.g: A valid offset +01:02:03, it's not in the transition table
      // We need to handle it here.
      converted_seconds = epoch_seconds - tz_offset[row_idx];
    } else {
      // not fixed offset, use the fixed_transitions and dst_rules to convert
      cudf::timestamp_s converted_ts = spark_rapids_jni::convert_timestamp<cudf::timestamp_s>(
        cudf::timestamp_s{cudf::duration_s{epoch_seconds}},
        fixed_transitions,
        dst_rules,
        tz_indices[row_idx],
        true);
      converted_seconds = converted_ts.time_since_epoch().count();
    }

    // 3. Adding the microseconds part, this may cause overflow
    int64_t result;
    if (spark_rapids_jni::overflow_checker::get_timestamp_overflow(
          converted_seconds, input_microseconds[row_idx], result)) {
      // overflowed
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
      output_mask[row_idx] = 0;
    } else {
      // not overflowed
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{result}};
      output_mask[row_idx] = 1;
    }
  }
};

std::unique_ptr<column> convert_to_utc_with_multiple_timezones(
  column_view const& input_seconds,
  column_view const& input_microseconds,
  column_view const& invalid,
  column_view const& tz_type,
  column_view const& tz_offset,
  table_view const& transitions,
  column_view const tz_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input_seconds.type().id() == cudf::type_id::INT64,
               "seconds column must be of type INT64");
  CUDF_EXPECTS(input_microseconds.type().id() == cudf::type_id::INT32,
               "microseconds column must be of type INT32");

  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(transitions.column(1), stream);
  auto const dst_rules   = cudf::detail::lists_column_device_view{*dst_cdv_ptr};

  auto result = cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
                                            input_seconds.size(),
                                            rmm::device_buffer{},
                                            0,
                                            stream,
                                            mr);
  auto null_mask = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::UINT8},
                                                 input_seconds.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     input_seconds.size(),
                     convert_with_timezones_fn{input_seconds.begin<int64_t>(),
                                               input_microseconds.begin<int32_t>(),
                                               invalid.begin<uint8_t>(),
                                               tz_type.begin<uint8_t>(),
                                               tz_offset.begin<int32_t>(),
                                               fixed_transitions,
                                               dst_rules,
                                               tz_indices.begin<int32_t>(),
                                               result->mutable_view().begin<cudf::timestamp_us>(),
                                               null_mask->mutable_view().begin<uint8_t>()});

  auto [output_bitmask, null_count] = cudf::detail::valid_if(null_mask->view().begin<uint8_t>(),
                                                             null_mask->view().end<uint8_t>(),
                                                             cuda::std::identity{},
                                                             stream,
                                                             mr);
  if (null_count) { result->set_null_mask(std::move(output_bitmask), null_count); }

  return result;
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<column> convert_timestamp(column_view const& input,
                                          table_view const& transitions,
                                          size_type tz_index,
                                          bool to_utc,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
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
                                                 rmm::device_async_resource_ref mr)
{
  return convert_timestamp(input, transitions, tz_index, true, stream, mr);
}

std::unique_ptr<column> convert_utc_timestamp_to_timezone(column_view const& input,
                                                          table_view const& transitions,
                                                          size_type tz_index,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  return convert_timestamp(input, transitions, tz_index, false, stream, mr);
}

std::unique_ptr<column> convert_timestamp_to_utc(column_view const& input_seconds,
                                                 column_view const& input_microseconds,
                                                 column_view const& invalid,
                                                 column_view const& tz_type,
                                                 column_view const& tz_offset,
                                                 table_view const& transitions,
                                                 column_view const tz_indices,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  return convert_to_utc_with_multiple_timezones(input_seconds,
                                                input_microseconds,
                                                invalid,
                                                tz_type,
                                                tz_offset,
                                                transitions,
                                                tz_indices,
                                                stream,
                                                mr);
}

}  // namespace spark_rapids_jni
