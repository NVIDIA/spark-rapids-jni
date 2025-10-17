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
#include <thrust/pair.h>

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

  /**
   * @brief Returns day of week in civil calendar for the specified epoch days.
   * Day of week is from 0(Monday) to 6(Sunday).
   * @param epoch_days the days from epoch time 1970-01-01
   * MUST be in range [Int32.MinValue - 8, Int64.MaxValue - Int32.MaxValue + 9],
   * In general, range [Int32.MinValue - 8, Int32.MaxValue + 8] is enough for Spark date.
   * @return the weekday for the days from epoch
   */
  __device__ static int64_t weekday_from_days(int64_t epoch_days) noexcept
  {
    // (Int32.MinValue - 8) days is Monday(value 0)
    int64_t const min_monday_days =
      static_cast<int64_t>(cuda::std::numeric_limits<int32_t>::min()) - 8L;
    return (epoch_days - min_monday_days) % 7L;
  }

  /**
   * @brief Calculate the days to move forward for a epoch days to become the expected weekday.
   * E.g: weekday of `epoch_days` = 0(Monday), `expected_weekday` = 6(Sunday), returns 6
   * E.g: weekday of `epoch_days` = 6(Sunday), `expected_weekday` = 0(Monday), returns 1
   * @param epoch_days The days from epoch time 1970-01-01
   * @param expected_weekday The to weekday of week, MUST be in range: from 0(Monday) to 6(Sunday)
   * @return difference to move forward in range [0, 6]
   */
  __device__ static int64_t weekday_difference(int64_t epoch_days, int64_t expected_weekday)
  {
    // int32_t forwards[7] = {6, 5, 4, 3, 2, 1, 0};
    // the forwards are for the difference from curr_weekday to Sunday(6)
    int64_t curr_weekday = weekday_from_days(epoch_days);
    int64_t index        = (curr_weekday + (6 - expected_weekday)) % 7;
    // 6 - index = forwards[index];
    return 6L - index;
  }

  /**
   * @brief Calculate the days to move backward for a epoch days to become the expected weekday.
   * E.g: weekday of `epoch_days` = 0(Monday), `expected_weekday` = 6(Sunday), returns 1
   * E.g: weekday of `epoch_days` = 6(Sunday), `expected_weekday` = 0(Monday), returns 6
   * @param epoch_days The days from epoch time 1970-01-01
   * @param expected_weekday The to weekday of week, MUST be in range: from 0(Monday) to 6(Sunday)
   * @return difference to move backward in range [0, 6]
   */
  __device__ static int64_t previous_weekday_difference(int64_t epoch_days,
                                                        int64_t expected_weekday)
  {
    // int32_t backwards[7] = {0, 1, 2, 3, 4, 5, 6};
    // the backwards are for the difference from curr_weekday to Monday(0)
    int64_t curr_weekday = weekday_from_days(epoch_days);
    int64_t index        = (curr_weekday + (7 - expected_weekday)) % 7;
    // index = backwards[index];
    return index;
  }

  /**
   * @brief Get the next or the same weekday from the specified `epoch_days` and expected weekday.
   * E.g.: if inputs are the days of 2025-10-10(Friday) and Sunday(6),
   * the next or the same Sunday is 2025-10-12.
   * @param epoch_days epoch days
   * @param expected_weekday expected weekday, 0 represents Monday, 6 represents Sunday.
   * @return the next or the same day that is `expected_weekday`
   */
  __device__ static int64_t next_or_same_weekday(int64_t const epoch_days,
                                                 int64_t const expected_weekday)
  {
    return epoch_days + weekday_difference(epoch_days, expected_weekday);
  }

  /**
   * @brief Get the previous or the same weekday from the specified `epoch_days` and expected
   * weekday. E.g.: if inputs are the days of 2025-10-13(Monday) and Sunday(6), the previous or the
   * same Sunday is 2025-10-12.
   * @param epoch_days epoch days
   * @param expected_weekday expected weekday, 0 represents Monday, 6 represents Sunday.
   * @return the previous or the same day that is `expected_weekday`
   */
  __device__ static int64_t previous_or_same_weekday(int64_t const epoch_days,
                                                     int64_t const expected_weekday)
  {
    return epoch_days - previous_weekday_difference(epoch_days, expected_weekday);
  }

  /**
   * @brief Get the year from epoch seconds.
   * @param seconds epoch seconds.
   * @return the year for the epoch seconds.
   */
  __device__ static int32_t get_year(int64_t seconds)
  {
    constexpr int64_t seconds_per_day = 86400L;
    int64_t days                      = seconds / seconds_per_day;
    int32_t year, month, day;
    to_date(days, year, month, day);
    return year;
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
    constexpr int64_t micros_per_sec       = 1'000'000;
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

namespace {

/**
 * Daylight Saving Time (DST) rule, it's decoded from the Java TZDB file.
 */
struct transition_rule {
  // the month of this transition
  int32_t month;

  // the day of month of this transition
  // if it's negative, count days from the end of the month
  int32_t dom;

  // weekday of week, [0, 6], 0 represents Monday, 6 represents Sunday,
  // if it's negative, this is ignored.
  int32_t dow;

  // time difference in seconds compared to the midnight
  int32_t time_diff_compared_to_midnight;

  // offset in seconds before this transition
  int32_t offset_before;

  // offset in seconds after this transition
  int32_t offset_after;

  __device__ transition_rule(int32_t month_,
                             int32_t dom_,
                             int32_t dow_,
                             int32_t time_diff_compared_to_midnight_,
                             int32_t offset_before_,
                             int32_t offset_after_)
    : month(month_),
      dom(dom_),
      dow(dow_),
      time_diff_compared_to_midnight(time_diff_compared_to_midnight_),
      offset_before(offset_before_),
      offset_after(offset_after_)
  {
  }
};

/**
 * Transition info
 */
struct transition_info {
  // UTC transition time point in seconds
  int64_t utc_seconds;

  int32_t offset_before;

  int32_t offset_after;

  __device__ bool is_gap() const { return offset_after > offset_before; }
};

struct daylight_saving_time_utils {
  /**
   * @brief Create transition info for the specified year and the transition rule.
   * @param year The year for the transition rule to apply
   * @param rule The transition rule
   * @param info[out] The transition info created
   */
  __device__ static void create_transition_info(int32_t const year,
                                                transition_rule const& rule,
                                                transition_info& info)
  {
    int64_t days;

    if (rule.dom > 0) {
      // day of month is positive
      days = date_time_utils::to_epoch_day(year, rule.month, rule.dom);
      if (rule.dow >= 0)  // 0~6, 0 represents Monday, 6 represents Sunday; -1 means ignore
      {
        // shift to the previous day of week
        days = date_time_utils::next_or_same_weekday(days, rule.dow);
      } else {
        // do nothing
        // also checked all the timezones in TimeZone.getAvailableIDs(), do not find dow is negative
      }
    } else {
      // day of month is negative, locate the day from the last day of month.
      // checked all the timezones in TimeZone.getAvailableIDs(), do not find any dom is negative.
      int32_t day_of_month = date_time_utils::days_in_month(year, rule.month) + 1 + rule.dom;
      days                 = date_time_utils::to_epoch_day(year, rule.month, day_of_month);
      if (rule.dow >= 0)  // 0~6, 0 represents Monday, 6 represents Sunday; -1 means ignore
      {
        // shift to the previous day of week
        days = date_time_utils::previous_or_same_weekday(days, rule.dow);
      } else {
        // do nothing
        // also checked all the timezones in TimeZone.getAvailableIDs(), do not find dow is negative
      }
    }
    constexpr int64_t seconds_per_day = 86400L;
    int64_t local_seconds = days * seconds_per_day + rule.time_diff_compared_to_midnight;
    info.utc_seconds      = local_seconds - rule.offset_before;
    info.offset_before    = rule.offset_before;
    info.offset_after     = rule.offset_after;
  }

  /**
   * @brief Get the offset in seconds for the specified epoch seconds in UTC.
   * @param seconds The epoch seconds in UTC
   * @param start_rule The DST start rule
   * @param end_rule The DST end rule
   * @return The offset in seconds
   */
  __device__ static int32_t get_offset_for_utc_time(int64_t seconds,
                                                    transition_rule start_rule,
                                                    transition_rule end_rule)
  {
    int32_t year = date_time_utils::get_year(seconds);
    transition_info infos[2];
    create_transition_info(year, start_rule, infos[0]);
    create_transition_info(year, end_rule, infos[1]);

    transition_info curr_info;
    for (int32_t i = 0; i < 2; ++i) {
      curr_info = infos[i];
      if (seconds < curr_info.utc_seconds) { return curr_info.offset_after; }
    }
    return curr_info.offset_after;
  }

  /**
   * @brief Get the offset in seconds for the specified epoch seconds in local time.
   * @param seconds The epoch seconds in local time
   * @param start_rule The DST start rule
   * @param end_rule The DST end rule
   * @return The offset in seconds
   */
  __device__ static int32_t get_offset_for_local_time(int64_t seconds,
                                                      transition_rule start_rule,
                                                      transition_rule end_rule)
  {
    int32_t year = date_time_utils::get_year(seconds);
    transition_info infos[2];
    create_transition_info(year, start_rule, infos[0]);
    create_transition_info(year, end_rule, infos[1]);
    if (infos[0].is_gap()) {
      // first rule is gap, second rule is overlap
      if (seconds < infos[0].utc_seconds + infos[0].offset_after) {
        return infos[0].offset_before;
      } else if (seconds >= infos[0].utc_seconds + infos[0].offset_after &&
                 seconds < infos[1].utc_seconds + infos[1].offset_before) {
        return infos[0].offset_after;
      } else {
        return infos[1].offset_after;
      }
    } else {
      // first rule is overlap, second rule is gap
      if (seconds < infos[0].utc_seconds + infos[0].offset_before) {
        return infos[0].offset_before;
      } else if (seconds >= infos[0].utc_seconds + infos[0].offset_before &&
                 seconds < infos[1].utc_seconds + infos[1].offset_after) {
        return infos[0].offset_after;
      } else {
        return infos[1].offset_after;
      }
    }
  }

  /**
   * @brief Create the DST rules from the list device view.
   * The list device view must contain 12 integers, each 6 integers represent a DST rule.
   * @param ldv The list device view
   * @return A pair of transition rules
   */
  __device__ static thrust::pair<transition_rule, transition_rule> create_dst_rules(
    cudf::list_device_view const& ldv)
  {
    return thrust::make_pair(transition_rule(ldv.element<int32_t>(0),
                                             ldv.element<int32_t>(1),
                                             ldv.element<int32_t>(2),
                                             ldv.element<int32_t>(3),
                                             ldv.element<int32_t>(4),
                                             ldv.element<int32_t>(5)),
                             transition_rule(ldv.element<int32_t>(6),
                                             ldv.element<int32_t>(7),
                                             ldv.element<int32_t>(8),
                                             ldv.element<int32_t>(9),
                                             ldv.element<int32_t>(10),
                                             ldv.element<int32_t>(11)));
  }
};

}  // namespace

/**
 * @brief Convert the timestamp from/to UTC for the specified timezone.
 * This function first apply DST rules if any, then apply the transitions.
 * For the timezone without DST rules, it only applies the transitions.
 * For the processing of apply the transitions:
 * - Uses a binary search to find the right offset to do the transition.
 * - For transition to UTC: do a binary search on the tzInstant.
 * - For transition from UTC: do a binary search on the utcInstant.
 * @param timestamp The timestamp to convert
 * @param transitions The transitions list column device view for the timezone
 * @param dsts The DST rules list column device view for the timezone
 * @param tz_index The timezone index
 * @param to_utc true to convert to UTC, false to convert from UTC
 * @return The converted timestamp
 */
template <typename timestamp_type>
__device__ static timestamp_type convert_timestamp(
  timestamp_type const& timestamp,
  cudf::detail::lists_column_device_view const& transitions,
  cudf::detail::lists_column_device_view const& dsts,
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

  auto const dst_integers = cudf::list_device_view{dsts, tz_index};
  // size of dst integers, MUST be 0 or 12, each 6 integers represent a DST rule
  auto const dst_integers_size = dst_integers.size();

  // DST processing begins
  if (dst_integers_size > 0) {
    // it's DST timezone
    int64_t last_transition_value = transition_times[list_size - 1];
    auto const transition_offset  = tz_transitions.element_offset(list_size - 1);
    auto const [rule1, rule2]     = daylight_saving_time_utils::create_dst_rules(dst_integers);
    if (epoch_seconds > last_transition_value) {
      int32_t offset_seconds;
      if (to_utc) {
        // search in local time
        int year = date_time_utils::get_year(epoch_seconds);
        offset_seconds =
          daylight_saving_time_utils::get_offset_for_local_time(epoch_seconds, rule1, rule2);
      } else {
        // search in UTC time
        int32_t last_offset = utc_offsets.element<int32_t>(transition_offset);
        int year            = date_time_utils::get_year(epoch_seconds + last_offset);
        offset_seconds =
          daylight_saving_time_utils::get_offset_for_utc_time(epoch_seconds, rule1, rule2);
      }
      auto const offset = cuda::std::chrono::duration_cast<duration_type>(
        cudf::duration_s{static_cast<int64_t>(offset_seconds)});
      return to_utc ? timestamp - offset : timestamp + offset;
    }
  }
  // DST processing ends

  auto const it = thrust::upper_bound(
    thrust::seq, transition_times.begin(), transition_times.end(), epoch_seconds);
  auto const idx = static_cast<cudf::size_type>(cuda::std::distance(transition_times.begin(), it));
  auto const list_offset = tz_transitions.element_offset(idx - 1);
  auto const utc_offset  = cuda::std::chrono::duration_cast<duration_type>(
    cudf::duration_s{static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset))});
  return to_utc ? timestamp - utc_offset : timestamp + utc_offset;
}

}  // namespace spark_rapids_jni
