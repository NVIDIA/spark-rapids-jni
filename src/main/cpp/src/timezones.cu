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
    return spark_rapids_jni::convert_timestamp(timestamp, transitions, tz_index, to_utc);
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

struct convert_with_timezones_fn {
  // inputs
  int64_t const* input_seconds;
  int32_t const* input_microseconds;
  uint8_t const* invalid;
  uint8_t const* tz_type;
  int32_t const* tz_offset;
  // The list column of transitions to figure out the correct offset
  // to adjust the timestamp. The type of the values in this column is
  // LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const transitions;
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

    int64_t epoch_seconds      = input_seconds[row_idx];
    int64_t epoch_microseconds = static_cast<int64_t>(input_microseconds[row_idx]);

    // 2. fixed offset conversion
    if (static_cast<spark_rapids_jni::TZ_TYPE>(tz_type[row_idx]) ==
        spark_rapids_jni::TZ_TYPE::FIXED_TZ) {
      // Fixed offset, offset is in seconds, add the offset
      int64_t converted_seconds = epoch_seconds - tz_offset[row_idx];
      int64_t result;
      // after the shift, the result maybe overflow
      if (spark_rapids_jni::overflow_checker::get_timestamp_overflow(
            converted_seconds, epoch_microseconds, result)) {
        output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
        output_mask[row_idx] = 0;
      } else {
        output[row_idx]      = cudf::timestamp_us{cudf::duration_us{result}};
        output_mask[row_idx] = 1;
      }
      return;
    }

    // 3. not fixed offset, use the transition table
    auto const tz_index     = tz_indices[row_idx];
    auto const utc_instants = transitions.child().child(0);
    auto const tz_instants  = transitions.child().child(1);
    auto const utc_offsets  = transitions.child().child(2);

    auto const tz_transitions = cudf::list_device_view{transitions, tz_index};
    auto const list_size      = tz_transitions.size();

    auto const transition_times = cudf::device_span<int64_t const>(
      tz_instants.data<int64_t>() + tz_transitions.element_offset(0),
      static_cast<size_t>(list_size));

    auto const it = thrust::upper_bound(
      thrust::seq, transition_times.begin(), transition_times.end(), epoch_seconds);
    auto const idx = static_cast<size_type>(cuda::std::distance(transition_times.begin(), it));
    auto const list_offset = tz_transitions.element_offset(idx - 1);
    auto const utc_offset  = utc_offsets.element<int32_t>(list_offset);

    int64_t converted_seconds2 = epoch_seconds - utc_offset;
    int64_t result2;
    // after the shift, the result maybe overflow
    if (spark_rapids_jni::overflow_checker::get_timestamp_overflow(
          converted_seconds2, epoch_microseconds, result2)) {
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
      output_mask[row_idx] = 0;
    } else {
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{result2}};
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

// =================== ORC timezones begin ===================
// ORC timezone uses java.util.TimeZone rules, which is different from java.time.ZoneId rules.

/**
 * @brief Get the transition index for the given time `time_ms` using binary search.
 * The transition array is sorted, each int64 value is composed of 44 bits
 * transition time in seconds and 20 bits offset in seconds.
 * @param begin the beginning of the transition array.
 * @param end the end of the transition array.
 * @param time_ms the input time in milliseconds to find the transition index for.
 */
__device__ static int get_transition_index(int64_t const* begin,
                                           int64_t const* end,
                                           int64_t time_ms)
{
  constexpr int OFFSET_SHIFT      = 20;
  constexpr int64_t MS_PER_SECOND = 1000L;

  if (begin == end) { return -1; }

  int low  = 0;
  int high = end - begin - 1;

  while (low <= high) {
    int mid = (low + high) / 2;
    // sign retained shift, then multiple 1000 to get milliseconds
    long midVal = (begin[mid] >> OFFSET_SHIFT) * MS_PER_SECOND;
    if (midVal < time_ms) {
      low = mid + 1;
    } else if (midVal > time_ms) {
      high = mid - 1;
    } else {
      return mid;
    }
  }

  // if beyond the transitions, returns that index.
  if (low >= (end - begin)) { return low; }
  return low - 1;
}

/**
 * @brief Get the int value from the least significant `num_bits` in a int64_t value.
 * Note: If the `num_bits` bit is 1, it returns negative value.
 * @param holder the int64_t value holds the bits.
 * @param num_bits the number of bits to get from the least significant bits.
 * @return the int value from the least significant `num_bits` in a int64_t value.
 */
__device__ static int64_t get_value_from_lowest_bits(int64_t holder, int num_bits)
{
  int64_t mask      = (1L << num_bits) - 1L;
  int64_t sign_mask = 1L << (num_bits - 1);

  if ((holder & sign_mask) != 0) {
    // the sign bit is 1, it's a negative value
    // set all the complement bits to 1
    return holder | (~mask);
  }

  // positive value
  return holder & mask;
}

/**
 * @brief Find the relative offset when moving between timezones at a particular point in time.
 * This is for ORC timezone support.
 *
 * This function implements `org.apache.orc.impl.SerializationUtils.convertBetweenTimezones`
 * Refer to link: https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/src/
 * java/org/apache/orc/impl/SerializationUtils.java#L1440
 *
 * If the input `trans_begin` == `trans_begin` == nullptr, it means the timezone is fixed offset,
 * then use the raw_offset directly.
 *
 * @param ts the input timestamp in UTC timezone to get the offset for.
 * @param writer_trans_begin The beginning of the writer timezone transition array, or 0 if using
 * fixed offset.
 * @param writer_trans_end The end of the writer timezone transition array, or 0 if using fixed
 * offset.
 * @param writer_raw_offset Writer timezone raw offset in seconds.
 * @param reader_trans_begin The beginning of the reader timezone transition array, or 0 if using
 * fixed offset.
 * @param reader_trans_end The end of the reader timezone transition array, or 0 if using fixed
 * offset.
 * @param reader_raw_offset Reader timezone raw offset in seconds.
 * @return the timestamp after apply the offsets between timezones.
 */
__device__ static cudf::timestamp_us convert_timestamp_between_timezones(
  cudf::timestamp_us ts,
  int64_t const* writer_trans_begin,
  int64_t const* writer_trans_end,
  cudf::size_type writer_raw_offset,
  int64_t const* reader_trans_begin,
  int64_t const* reader_trans_end,
  cudf::size_type reader_raw_offset)
{
  constexpr int32_t OFFSET_BITS   = 20;
  constexpr int64_t MS_PER_SECOND = 1000L;
  constexpr int64_t US_PER_SECOND = 1'000'000L;

  int64_t const epoch_milliseconds = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_ms>(ts.time_since_epoch()).count());

  int64_t writer_offset = [&] {
    int const writer_index =
      get_transition_index(writer_trans_begin, writer_trans_end, epoch_milliseconds);

    if (writer_index >= 0 && writer_index < (writer_trans_end - writer_trans_begin)) {
      return get_value_from_lowest_bits(writer_trans_begin[writer_index], OFFSET_BITS);
    } else {
      return static_cast<int64_t>(writer_raw_offset);
    }
  }();

  int64_t reader_offset = [&] {
    int const reader_index =
      get_transition_index(reader_trans_begin, reader_trans_end, epoch_milliseconds);
    if (reader_index >= 0 && reader_index < (reader_trans_end - reader_trans_begin)) {
      return get_value_from_lowest_bits(reader_trans_begin[reader_index], OFFSET_BITS);
    } else {
      return static_cast<int64_t>(reader_raw_offset);
    }
  }();

  int64_t reader_adjusted_offset = [&] {
    int64_t adjusted_milliseconds =
      epoch_milliseconds + (writer_offset - reader_offset) * MS_PER_SECOND;

    int const reader_adjusted_index =
      get_transition_index(reader_trans_begin, reader_trans_end, adjusted_milliseconds);

    if (reader_adjusted_index >= 0 &&
        reader_adjusted_index < (reader_trans_end - reader_trans_begin)) {
      return get_value_from_lowest_bits(reader_trans_begin[reader_adjusted_index], OFFSET_BITS);
    } else {
      return static_cast<int64_t>(reader_raw_offset);
    }
  }();

  int64_t final_offset_seconds =
    static_cast<int64_t>(writer_offset) - static_cast<int64_t>(reader_adjusted_offset);
  int64_t const epoch_us = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_us>(ts.time_since_epoch()).count());
  int64_t final_result = epoch_us + final_offset_seconds * US_PER_SECOND;
  return cudf::timestamp_us{cudf::duration_us{final_result}};
}

struct convert_timezones_functor {
  // writer timezone info
  int64_t const* writer_trans_begin;
  int64_t const* writer_trans_end;
  int32_t writer_raw_offset;

  // reader timezone info
  int64_t const* reader_trans_begin;
  int64_t const* reader_trans_end;
  int32_t reader_raw_offset;

  __device__ cudf::timestamp_us operator()(cudf::timestamp_us const& timestamp) const
  {
    return convert_timestamp_between_timezones(timestamp,
                                               writer_trans_begin,
                                               writer_trans_end,
                                               writer_raw_offset,
                                               reader_trans_begin,
                                               reader_trans_end,
                                               reader_raw_offset);
  }
};

std::unique_ptr<column> convert_timezones(cudf::column_view const& input,
                                          cudf::table_view const* writer_tz_info_table,
                                          cudf::size_type writer_raw_offset,
                                          cudf::table_view const* reader_tz_info_table,
                                          cudf::size_type reader_raw_offset,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  // Input is from Spark, so it should be TIMESTAMP_MICROSECONDS type
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "Input column must be of type TIMESTAMP_MICROSECONDS");
  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::detail::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  int64_t const* writer_trans_begin = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(0).begin<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int64_t const* writer_trans_end = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(0).end<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int64_t const* reader_trans_begin = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(0).begin<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int64_t const* reader_trans_end = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(0).end<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  thrust::transform(rmm::exec_policy_nosync(stream),
                    input.begin<cudf::timestamp_us>(),
                    input.end<cudf::timestamp_us>(),
                    results->mutable_view().begin<cudf::timestamp_us>(),
                    convert_timezones_functor{writer_trans_begin,
                                              writer_trans_end,
                                              writer_raw_offset,
                                              reader_trans_begin,
                                              reader_trans_end,
                                              reader_raw_offset});

  return results;
}

// =================== ORC timezones end ===================

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

std::unique_ptr<cudf::column> convert_orc_writer_reader_timezones(
  cudf::column_view const& input,
  cudf::table_view const* writer_tz_info_table,
  cudf::size_type writer_raw_offset,
  cudf::table_view const* reader_tz_info_table,
  cudf::size_type reader_raw_offset,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return convert_timezones(input,
                           writer_tz_info_table,
                           writer_raw_offset,
                           reader_tz_info_table,
                           reader_raw_offset,
                           stream,
                           mr);
}

}  // namespace spark_rapids_jni
