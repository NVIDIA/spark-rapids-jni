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

#pragma once

#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/std/limits>
#include <thrust/binary_search.h>

namespace spark_rapids_jni {

/**
 * @brief The utilities for date and time.
 */
struct date_time_utils {
  /**
   * @brief Is the year is leap year.
   */
  __device__ static bool is_leap_year(int year)
  {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
  }

  /**
   * @brief Get the number of days in a month.
   * @param year the year
   * @param month the month
   * @return the number of days in the month
   */
  __device__ static int days_in_month(int year, int month)
  {
    if (month == 2) { return is_leap_year(year) ? 29 : 28; }
    return (month == 4 || month == 6 || month == 9 || month == 11) ? 30 : 31;
  }

  /**
   * @brief Calculate the number of days since epoch 1970-01-01 from date.
   * Refer to https://howardhinnant.github.io/date_algorithms.html#days_from_civil.
   * Unlike cuda::std::chrono::year_month_day only supports year range [-32,767 , 32,767],
   * this implementation supports all int years.
   */
  __device__ static int64_t to_epoch_day(int year, int month, int day)
  {
    int32_t y          = year - (month <= 2);
    int32_t const era  = (y >= 0 ? y : y - 399) / 400;
    uint32_t const yoe = static_cast<uint32_t>(y - era * 400);                           // [0, 399]
    uint32_t const doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1;  // [0, 365]
    uint32_t const doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;  // [0, 146096]
    return era * 146097L + doe - 719468L;
  }

  /**
   * @brief Calculate year/month/day from the number of days since epoch 1970-01-01.
   * Refer to https://howardhinnant.github.io/date_algorithms.html#civil_from_days.
   * This implementation supports all int values as `epoch_day`.
   * @param epoch_day the number of days since 1970-01-01
   * @param year[out]      year output year
   * @param month[out]     month output month
   * @param day[out]       day output day
   */
  __device__ static void to_date(int32_t const epoch_day, int& year, int& month, int& day)
  {
    int64_t z = static_cast<int64_t>(epoch_day);
    z += 719468;
    int32_t const era  = static_cast<int32_t>((z >= 0 ? z : z - 146096) / 146097);
    uint32_t const doe = static_cast<uint32_t>(z - era * 146097);                // [0, 146096]
    uint32_t const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
    int32_t const y    = static_cast<uint32_t>(yoe) + era * 400;
    uint32_t const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                // [0, 365]
    uint32_t const mp  = (5 * doy + 2) / 153;                                    // [0, 11]
    day                = doy - (153 * mp + 2) / 5 + 1;                           // [1, 31]
    month              = mp < 10 ? mp + 3 : mp - 9;                              // [1, 12]
    year               = y + (month <= 2);
  }

  /**
   * @brief If the month/day is valid.
   */
  __device__ static bool is_valid_month_day(int year, int month, int day)
  {
    if (month < 1 || month > 12 || day < 1) {
      return false;  // Invalid month or day
    }

    // Check against the standard days
    return (day <= days_in_month(year, month));
  }

  /**
   * @brief If Spark date is valid.
   * Spark stores date as int with the value of days since epoch 1970-01-01.
   * Spark date: max year is 7 digits, so the rough check range is [-10,000,000, 10,000,000].
   * Note: Spark stores date as int, so the range is limited to 32-bit signed integer.
   */
  __device__ static bool is_valid_date_for_date(int year, int month, int day)
  {
    if (year < -10'000'000 || year > 10'000'000) { return false; }

    return is_valid_month_day(year, month, day);
  }

  /**
   * @brief If the date in Spark timestamp is valid.
   * Spark stores timestamp as long in microseconds.
   * Spark timestamp: max year is 6 digits, the rough check range is [-300'000, 300'000].
   */
  __device__ static bool is_valid_date_for_timestamp(int year, int month, int day)
  {
    if (year < -300'000 || year > 300'000) { return false; }

    return is_valid_month_day(year, month, day);
  }

  /**
   * @brief If the time is valid.
   * Spark timestamp: hour 0-23, minute 0-59, second 0-59, microseconds 0-999999.
   */
  __device__ static bool is_valid_time(int hour, int minute, int second, int microseconds)
  {
    return (hour >= 0 && hour < 24) && (minute >= 0 && minute < 60) &&
           (second >= 0 && second < 60) && (microseconds >= 0 && microseconds < 1'000'000);
  }
};

/**
 * @brief Represents local date in a timezone.
 * Spark stores date into Int as days since epoch 1970-01-01.
 * A Int is able to represent a date with max 7 digits of days.
 * The formula is: Int.MaxValue/days_per_year + 1970.
 */
struct date_segments {
  /**
   * @brief Constructor a default date segments.
   * By default, use epoch date: "1970-01-01".
   */
  __device__ date_segments() : year(1970), month(1), day(1) {}

  __device__ bool is_valid_date() const
  {
    return date_time_utils::is_valid_date_for_date(year, month, day);
  }

  /**
   * @brief Get days since epoch 1970-01-01.
   * Can handle all int years.
   */
  __device__ int64_t to_epoch_day() const
  {
    return date_time_utils::to_epoch_day(year, month, day);
  }

  int32_t year;

  // 1-12
  int32_t month;

  // 1-31; it is 29 for leap February, or 28 for regular February
  int32_t day;
};

struct overflow_checker {
  /**
   * Calculate the timestamp from epoch seconds and microseconds with checking overflow
   * @param seconds seconds from epoch
   * @param microseconds MUST be in range [0, 999999]
   * @param[out] result timestamp in microseconds
   * @return true if overflow occurred, flase otherwise
   */
  __device__ static bool get_timestamp_overflow(int64_t seconds,
                                                int32_t microseconds,
                                                int64_t& result)
  {
    constexpr int64_t micros_per_sec       = 1000000;
    constexpr int64_t max_v                = cuda::std::numeric_limits<int64_t>::max();
    constexpr int64_t min_v                = cuda::std::numeric_limits<int64_t>::min();
    constexpr int64_t max_positive_seconds = max_v / micros_per_sec;
    constexpr int64_t min_negative_seconds = min_v / micros_per_sec - 1;
    result                                 = seconds * micros_per_sec + microseconds;
    if (seconds > max_positive_seconds || seconds < min_negative_seconds) {
      return true;  // Overflow occurred
    }

    if (seconds > 0) { return microseconds > max_v - seconds * micros_per_sec; }

    if (seconds == min_negative_seconds) {
      // 224192L is calculated from 9999999999999999 / 1000000
      // BigDecimal(min_negative_seconds) * micros_per_sec - BigDecimal(min_v)
      return microseconds >= 224192L;
    }

    return false;
  }
};

// This device functor uses a binary search to find the instant of the transition
// to find the right offset to do the transition.
// To transition to UTC: do a binary search on the tzInstant child column and subtract
// the offset.
// To transition from UTC: do a binary search on the utcInstant child column and add
// the offset.
template <typename timestamp_type>
__device__ static timestamp_type convert_timestamp(
  timestamp_type const& timestamp,
  cudf::detail::lists_column_device_view const& transitions,
  cudf::size_type tz_index,
  bool to_utc)
{
  using duration_type = typename timestamp_type::duration;

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
  auto const idx = static_cast<cudf::size_type>(thrust::distance(transition_times.begin(), it));
  auto const list_offset = tz_transitions.element_offset(idx - 1);
  auto const utc_offset  = cuda::std::chrono::duration_cast<duration_type>(
    cudf::duration_s{static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset))});
  return to_utc ? timestamp - utc_offset : timestamp + utc_offset;
}

}  // namespace spark_rapids_jni
