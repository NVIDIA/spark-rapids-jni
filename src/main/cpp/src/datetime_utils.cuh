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

#include <cudf/types.hpp>

#include <cuda/std/limits>

namespace spark_rapids_jni {

/**
 * @brief The utilities for date and time.
 */
struct date_time_utils {
  /**
   * @brief Is the year is leap year.
   */
  __device__ static bool is_leap_year(int const year)
  {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
  }

  /**
   * @brief Get the number of days in a month.
   * @param year the year
   * @param month the month
   * @return the number of days in the month
   */
  __device__ static int days_in_month(int const year, int const month)
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
  __device__ static int64_t to_epoch_day(int const year, int const month, int const day)
  {
    int32_t y          = year - (month <= 2);
    const int32_t era  = (y >= 0 ? y : y - 399) / 400;
    const uint32_t yoe = static_cast<uint32_t>(y - era * 400);                           // [0, 399]
    const uint32_t doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1;  // [0, 365]
    const uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;  // [0, 146096]
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
    const int32_t era  = static_cast<int32_t>((z >= 0 ? z : z - 146096) / 146097);
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);                // [0, 146096]
    const uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
    const int32_t y    = static_cast<uint32_t>(yoe) + era * 400;
    const uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                // [0, 365]
    const uint32_t mp  = (5 * doy + 2) / 153;                                    // [0, 11]
    const uint32_t d   = doy - (153 * mp + 2) / 5 + 1;                           // [1, 31]
    const uint32_t m   = mp < 10 ? mp + 3 : mp - 9;                              // [1, 12]
    year               = y + (m <= 2);
    month              = static_cast<uint8_t>(m);
    day                = static_cast<uint8_t>(d);
  }

  /**
   * @brief If the month/day is valid.
   */
  __device__ static bool is_valid_month_day(int const year, int const month, int const day)
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
  __device__ static bool is_valid_date_for_date(int const year, int const month, int const day)
  {
    if (year < -10'000'000 || year > 10'000'000) { return false; }

    return is_valid_month_day(year, month, day);
  }

  /**
   * @brief If the date in Spark timestamp is valid.
   * Spark stores timestamp as long in microseconds.
   * Spark timestamp: max year is 6 digits, the rough check range is [-300'000, 300'000].
   */
  __device__ static bool is_valid_date_for_timestamp(int const year, int const month, int const day)
  {
    if (year < -300'000 || year > 300'000) { return false; }

    return is_valid_month_day(year, month, day);
  }

  /**
   * @brief If the time is valid.
   * Spark timestamp: hour 0-23, minute 0-59, second 0-59, microseconds 0-999999.
   */
  __device__ static bool is_valid_time(int const hour,
                                       int const minute,
                                       int const second,
                                       int const microseconds)
  {
    return (hour >= 0 && hour < 24) && (minute >= 0 && minute < 60) &&
           (second >= 0 && second < 60) && (microseconds >= 0 && microseconds < 1000000);
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

/**
 * @brief Represents local date time in a timezone with microsecond accuracy.
 * Spark stores timestamp into Long in microseconds.
 * A Long is able to represent a timestamp with max 6 digits of microseconds.
 * The formula is: Long.MaxValue/microseconds_per_year + 1970.
 */
struct ts_segments {
  /**
   * @brief Constructor a default timestamp segments.
   * By default, use epoch date with mid-night time: "1970-01-01 00:00:00.000000".
   */
  __device__ ts_segments()
    : year(1970), month(1), day(1), hour(0), minute(0), second(0), microseconds(0)
  {
  }

  /**
   * @brief Is this timestamp segments valid.
   */
  __device__ bool is_valid_ts() const
  {
    return date_time_utils::is_valid_date_for_timestamp(year, month, day) &&
           date_time_utils::is_valid_time(hour, minute, second, microseconds);
  }

  /**
   * @brief Get days since epoch 1970-01-01.
   * Can handle all int years.
   */
  __device__ int64_t to_epoch_day() const
  {
    return date_time_utils::to_epoch_day(year, month, day);
  }

  // max 6 digits for Spark timestamp
  int32_t year;

  // 1-12
  int32_t month;

  // 1-31; it is 29 for leap February, or 28 for regular February
  int32_t day;

  // 0-23
  int32_t hour;

  // 0-59
  int32_t minute;

  // 0-59
  int32_t second;

  // 0-999999, only parse 6 digits, ignore/truncate the rest digits
  int32_t microseconds;
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

}  // namespace spark_rapids_jni
