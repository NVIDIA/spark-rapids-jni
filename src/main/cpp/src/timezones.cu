/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/array>
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
                                             cudf::copy_bitmask(input, stream, mr),
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
  bool* output_mask;

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
      output_mask[row_idx] = false;
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
      output_mask[row_idx] = false;
    } else {
      // not overflowed
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{result}};
      output_mask[row_idx] = true;
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
  auto null_mask = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
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
                                               null_mask->mutable_view().begin<bool>()});

  auto [output_bitmask, null_count] = cudf::bools_to_mask(null_mask->view(), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask.release()), null_count); }

  return result;
}

// =================== ORC timezones begin ===================
// ORC timezone uses java.util.TimeZone rules, which is different from java.time.ZoneId rules.

// ---- Calendar helpers for DST computation on GPU ----
// Ported from java.util.SimpleTimeZone.getOffset logic.

constexpr int32_t MS_PER_SECOND = 1000;
constexpr int32_t MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int32_t MS_PER_HOUR   = 60 * MS_PER_MINUTE;
constexpr int64_t MS_PER_DAY    = 24LL * MS_PER_HOUR;

// Cumulative days before each month (non-leap). Index 0 = Jan.
// __device__ storage is required because days_in_month / days_before_month index
// this with non-compile-time values, which addresses elements at runtime.
__device__ constexpr cuda::std::array<int32_t, 12> DAYS_BEFORE_MONTH = {
  0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

__device__ static bool is_leap_year(int32_t year)
{
  return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}

__device__ static int32_t days_in_month(int32_t month, int32_t year)
{
  int32_t d =
    (month < 11) ? (DAYS_BEFORE_MONTH[month + 1] - DAYS_BEFORE_MONTH[month]) : 31;  // December
  if (month == 1 && is_leap_year(year)) { d++; }
  return d;
}

__device__ static int32_t days_before_month(int32_t month, int32_t year)
{
  int32_t d = DAYS_BEFORE_MONTH[month];
  if (month > 1 && is_leap_year(year)) { d++; }
  return d;
}

// Days from epoch (1970-01-01) to Jan 1 of the given year.
__device__ static int64_t days_from_epoch_to_year(int32_t year)
{
  int32_t y = year - 1;
  // Gregorian calendar days from epoch
  return 365LL * (year - 1970) + (y / 4 - 492) - (y / 100 - 19) + (y / 400 - 4);
}

// DST rule mode constants (same as SimpleTimeZone)
enum dst_rule_mode : int32_t {
  DOM_MODE          = 0,
  DOW_IN_MONTH_MODE = 1,
  DOW_GE_DOM_MODE   = 2,
  DOW_LE_DOM_MODE   = 3
};

// Time mode constants
enum dst_time_mode : int32_t { WALL_TIME = 0, STANDARD_TIME = 1, UTC_TIME = 2 };

/**
 * @brief Compute the day-of-month when a DST rule triggers for the given year and month.
 *
 * Implements the same logic as SimpleTimeZone's rule decoding:
 * - DOM_MODE: exact day of month
 * - DOW_IN_MONTH_MODE: nth occurrence of dayOfWeek (negative = from end)
 * - DOW_GE_DOM_MODE: first dayOfWeek on or after the given day
 * - DOW_LE_DOM_MODE: last dayOfWeek on or before the given day
 */
__device__ static int32_t compute_rule_day(
  int32_t rule_mode, int32_t rule_day, int32_t rule_dow, int32_t year, int32_t month)
{
  int32_t month_len = days_in_month(month, year);

  // Compute day-of-week of the 1st of the month
  int64_t first_of_month_epoch_days =
    days_from_epoch_to_year(year) + days_before_month(month, year);
  int64_t dow_raw = ((first_of_month_epoch_days + 4) % 7);
  if (dow_raw < 0) dow_raw += 7;
  int32_t first_dow = static_cast<int32_t>(dow_raw) + 1;  // 1=Sun..7=Sat

  switch (rule_mode) {
    case DOM_MODE: return rule_day;

    case DOW_IN_MONTH_MODE: {
      if (rule_day > 0) {
        // nth occurrence: 1st=first week, etc.
        int32_t diff = rule_dow - first_dow;
        if (diff < 0) diff += 7;
        return 1 + diff + (rule_day - 1) * 7;
      } else {
        // negative: from end of month. -1 = last occurrence.
        int32_t last_dow = ((first_dow - 1 + (month_len - 1)) % 7) + 1;
        int32_t diff     = last_dow - rule_dow;
        if (diff < 0) diff += 7;
        return month_len - diff + (rule_day + 1) * 7;
      }
    }

    case DOW_GE_DOM_MODE: {
      // First rule_dow on or after rule_day
      int64_t target_epoch = first_of_month_epoch_days + (rule_day - 1);
      int64_t target_raw   = ((target_epoch + 4) % 7);
      if (target_raw < 0) target_raw += 7;
      int32_t target_dow = static_cast<int32_t>(target_raw) + 1;
      int32_t diff       = rule_dow - target_dow;
      if (diff < 0) diff += 7;
      return rule_day + diff;
    }

    case DOW_LE_DOM_MODE: {
      // Last rule_dow on or before rule_day
      int64_t target_epoch = first_of_month_epoch_days + (rule_day - 1);
      int64_t target_raw   = ((target_epoch + 4) % 7);
      if (target_raw < 0) target_raw += 7;
      int32_t target_dow = static_cast<int32_t>(target_raw) + 1;
      int32_t diff       = target_dow - rule_dow;
      if (diff < 0) diff += 7;
      return rule_day - diff;
    }

    default: return rule_day;
  }
}

/**
 * @brief Compute the UTC millis of a DST transition for a given year.
 *
 * @param year The calendar year.
 * @param rule_month 0-based month.
 * @param rule_day Day parameter of the rule.
 * @param rule_dow Day-of-week parameter.
 * @param rule_time Time-of-day in ms.
 * @param rule_time_mode WALL_TIME / STANDARD_TIME / UTC_TIME.
 * @param rule_mode DOM / DOW_IN_MONTH / DOW_GE_DOM / DOW_LE_DOM.
 * @param raw_offset_ms The timezone raw offset in ms.
 * @param dst_savings_ms The DST savings in ms (needed for WALL_TIME conversion).
 * @param is_start_rule True for DST start (to determine WALL_TIME adjustment).
 */
__device__ static int64_t compute_transition_utc_ms(int32_t year,
                                                    int32_t rule_month,
                                                    int32_t rule_day,
                                                    int32_t rule_dow,
                                                    int32_t rule_time,
                                                    int32_t rule_time_mode,
                                                    int32_t rule_mode,
                                                    int32_t raw_offset_ms,
                                                    int32_t dst_savings_ms,
                                                    bool is_start_rule)
{
  int32_t actual_day = compute_rule_day(rule_mode, rule_day, rule_dow, year, rule_month);
  int64_t epoch_days =
    days_from_epoch_to_year(year) + days_before_month(rule_month, year) + (actual_day - 1);
  int64_t utc_ms = epoch_days * MS_PER_DAY + rule_time;

  // Convert from the specified time mode to UTC
  switch (rule_time_mode) {
    case WALL_TIME:
      utc_ms -= raw_offset_ms;
      // Wall time during DST-end means DST is still active, subtract savings.
      // Wall time during DST-start means DST is not yet active.
      if (!is_start_rule) { utc_ms -= dst_savings_ms; }
      break;
    case STANDARD_TIME: utc_ms -= raw_offset_ms; break;
    case UTC_TIME: break;
  }

  return utc_ms;
}

// Lightweight year extraction from epoch millis (no month/day computation).
__device__ static int32_t millis_to_year(int64_t epoch_ms)
{
  int64_t day_count =
    (epoch_ms >= 0) ? epoch_ms / MS_PER_DAY : (epoch_ms - MS_PER_DAY + 1) / MS_PER_DAY;
  int64_t days_since_1 = day_count + 719468;
  int32_t era =
    static_cast<int32_t>((days_since_1 >= 0 ? days_since_1 : days_since_1 - 146096) / 146097);
  int32_t doe   = static_cast<int32_t>(days_since_1 - static_cast<int64_t>(era) * 146097);
  int32_t yoe   = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int32_t year  = yoe + era * 400;
  int32_t doy   = doe - (365 * yoe + yoe / 4 - yoe / 100 + yoe / 400);
  int32_t mp    = (5 * doy + 2) / 153;
  int32_t month = mp < 10 ? mp + 2 : mp - 10;
  if (month <= 1) { year++; }
  return year;
}

/**
 * @brief Compute the total UTC offset (raw + DST) for a UTC timestamp using DST rules.
 *
 * This is the GPU equivalent of java.util.SimpleTimeZone.getOffset(long).
 * It computes the DST start and end transitions for the year containing the
 * timestamp, then checks if the timestamp falls within the DST window.
 *
 * Handles both Northern Hemisphere (start < end) and Southern Hemisphere
 * (start > end, i.e., DST spans year boundary).
 */
__device__ static int32_t compute_dst_offset(int64_t utc_ms,
                                             int32_t raw_offset_ms,
                                             spark_rapids_jni::dst_rule const& rule)
{
  if (!rule.has_dst) { return raw_offset_ms; }

  int32_t year = millis_to_year(utc_ms + raw_offset_ms);

  // Compute DST-on and DST-off transitions in UTC for this year
  int64_t dst_start = compute_transition_utc_ms(year,
                                                rule.start_month,
                                                rule.start_day,
                                                rule.start_dow,
                                                rule.start_time,
                                                rule.start_time_mode,
                                                rule.start_mode,
                                                raw_offset_ms,
                                                rule.dst_savings,
                                                true);

  int64_t dst_end = compute_transition_utc_ms(year,
                                              rule.end_month,
                                              rule.end_day,
                                              rule.end_dow,
                                              rule.end_time,
                                              rule.end_time_mode,
                                              rule.end_mode,
                                              raw_offset_ms,
                                              rule.dst_savings,
                                              false);

  bool in_dst;
  if (dst_start < dst_end) {
    // Northern Hemisphere: DST is [start, end)
    in_dst = (utc_ms >= dst_start && utc_ms < dst_end);
  } else {
    // Southern Hemisphere: DST is [start, year_end) ∪ [year_start, end)
    in_dst = (utc_ms >= dst_start || utc_ms < dst_end);
  }

  return in_dst ? raw_offset_ms + rule.dst_savings : raw_offset_ms;
}

/**
 * @brief Get the offset for a UTC time using the transition table + DST rule fallback.
 *
 * For timestamps within the transition table range, uses binary search.
 * For timestamps beyond the table, uses DST rule computation.
 * For timestamps before the first recorded transition, falls back to the
 * historical initial offset to match java.util.TimeZone behavior.
 */
__device__ static int32_t get_transition_index(int64_t const* begin,
                                               int64_t const* end,
                                               int64_t time_ms,
                                               int32_t const* offset_begin,
                                               int32_t initial_offset,
                                               int32_t raw_offset,
                                               spark_rapids_jni::dst_rule const& rule)
{
  if (begin == end) {
    // No transition table. Use DST rule if available, else fixed offset.
    return compute_dst_offset(time_ms, raw_offset, rule);
  }

  // upper_bound returns the first element strictly greater than time_ms, so
  // *iter > time_ms and the index we want is iter - 1.
  auto const iter = thrust::upper_bound(thrust::seq, begin, end, time_ms);
  if (iter == end) {
    // Beyond the transition table -- use DST rule for future dates
    return compute_dst_offset(time_ms, raw_offset, rule);
  }

  int32_t index = static_cast<int32_t>(cuda::std::distance(begin, iter));
  if (index == 0) {
    // Before the first recorded transition, java.util.TimeZone uses the
    // historical offset in effect before that transition, not the future rule.
    return initial_offset;
  }

  return offset_begin[index - 1];
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
 */
__device__ static cudf::timestamp_us convert_timestamp_between_timezones(
  cudf::timestamp_us ts,
  int64_t const* writer_trans_begin,
  int64_t const* writer_trans_end,
  int32_t const* writer_offsets_begin,
  int32_t const* writer_offsets_end,
  int32_t writer_raw_offset,
  int64_t const* reader_trans_begin,
  int64_t const* reader_trans_end,
  int32_t const* reader_offsets_begin,
  int32_t const* reader_offsets_end,
  int32_t reader_raw_offset)
{
  constexpr int64_t MICROS_PER_MILLI = 1000L;

  int64_t const epoch_millis = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_ms>(ts.time_since_epoch()).count());

  int32_t writer_offset_millis = get_transition_index(writer_trans_begin,
                                                      writer_trans_end,
                                                      epoch_millis,
                                                      writer_offsets_begin,
                                                      writer_offsets_end,
                                                      writer_raw_offset);

  int32_t reader_offset_millis = get_transition_index(reader_trans_begin,
                                                      reader_trans_end,
                                                      epoch_millis,
                                                      reader_offsets_begin,
                                                      reader_offsets_end,
                                                      reader_raw_offset);

  int64_t adjusted_milliseconds    = epoch_millis + (writer_offset_millis - reader_offset_millis);
  int const reader_adjusted_offset = get_transition_index(reader_trans_begin,
                                                          reader_trans_end,
                                                          adjusted_milliseconds,
                                                          reader_offsets_begin,
                                                          reader_offsets_end,
                                                          reader_raw_offset);

  int32_t final_offset_millis = writer_offset_millis - reader_adjusted_offset;
  int64_t const epoch_us      = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_us>(ts.time_since_epoch()).count());
  int64_t final_result = epoch_us + static_cast<int64_t>(final_offset_millis) * MICROS_PER_MILLI;
  return cudf::timestamp_us{cudf::duration_us{final_result}};
}

struct convert_timezones_functor {
  // writer timezone info
  int64_t const* writer_trans_begin;
  int64_t const* writer_trans_end;
  int32_t const* writer_offsets_begin;
  int32_t const* writer_offsets_end;
  int32_t writer_raw_offset;

  // reader timezone info
  int64_t const* reader_trans_begin;
  int64_t const* reader_trans_end;
  int32_t const* reader_offsets_begin;
  int32_t const* reader_offsets_end;
  int32_t reader_raw_offset;

  __device__ cudf::timestamp_us operator()(cudf::timestamp_us const& timestamp) const
  {
    return convert_timestamp_between_timezones(timestamp,
                                               writer_trans_begin,
                                               writer_trans_end,
                                               writer_offsets_begin,
                                               writer_offsets_end,
                                               writer_raw_offset,
                                               reader_trans_begin,
                                               reader_trans_end,
                                               reader_offsets_begin,
                                               reader_offsets_end,
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
                                             cudf::copy_bitmask(input, stream, mr),
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

  int32_t const* writer_offsets_begin = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(1).begin<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  int32_t const* writer_offsets_end = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(1).end<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
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

  int32_t const* reader_offsets_begin = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(1).begin<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  int32_t const* reader_offsets_end = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(1).end<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  thrust::transform(rmm::exec_policy_nosync(stream),
                    input.begin<cudf::timestamp_us>(),
                    input.end<cudf::timestamp_us>(),
                    results->mutable_view().begin<cudf::timestamp_us>(),
                    convert_timezones_functor{writer_trans_begin,
                                              writer_trans_end,
                                              writer_offsets_begin,
                                              writer_offsets_end,
                                              writer_raw_offset,
                                              reader_trans_begin,
                                              reader_trans_end,
                                              reader_offsets_begin,
                                              reader_offsets_end,
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
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      // Nanoseconds supported for users who need sub-microsecond precision
      // (e.g., when storing the resultant timestamp as string rather than
      // Spark TimestampType).
      return convert_timestamp_tz<cudf::timestamp_ns>(
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
