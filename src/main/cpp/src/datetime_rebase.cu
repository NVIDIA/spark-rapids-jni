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

#include "datetime_rebase.hpp"

//
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

//
#include <rmm/exec_policy.hpp>

//
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace {

__device__ cuda::std::chrono::year_month_day get_ymd_from_days(int32_t days) {
  auto const days_since_epoch = cuda::std::chrono::sys_days(
      cuda::std::chrono::duration<int32_t, cuda::std::chrono::days::period>{days});
  return cuda::std::chrono::year_month_day(days_since_epoch);
}

__device__ cuda::std::chrono::year_month_day get_ymd_from_micros(int64_t micros) {
  auto const days_since_epoch = cuda::std::chrono::sys_days(
      static_cast<cuda::std::chrono::duration<int32_t, cuda::std::chrono::days::period>>(
          cuda::std::chrono::floor<cuda::std::chrono::days>(
              cuda::std::chrono::duration<int64_t, cuda::std::chrono::microseconds::period>(
                  micros))));

  return cuda::std::chrono::year_month_day(days_since_epoch);
}

auto __device__ days_from_julian(cuda::std::chrono::year_month_day const &ymd) {
  auto year = static_cast<int32_t>(ymd.year());
  auto const month = static_cast<uint32_t>(ymd.month());
  auto const day = static_cast<uint32_t>(ymd.day());

  // https://howardhinnant.github.io/date_algorithms.html#Example:%20Converting%20between%20the%20civil%20calendar%20and%20the%20Julian%20calendar
  // https://www.wikiwand.com/en/Julian_day#/Converting_Julian_calendar_date_to_Julian_Day_Number
  year -= (month <= 2);
  int32_t const era = (year >= 0 ? year : year - 3) / 4;
  uint32_t const year_of_era = static_cast<uint32_t>(year - era * 4);                    // [0, 3]
  uint32_t const day_of_year = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1; // [0, 365]
  uint32_t const day_of_era = year_of_era * 365 + day_of_year; // [0, 1460]
  return era * 1461 + static_cast<int32_t>(day_of_era) - 719470;
}

std::unique_ptr<cudf::column> gregorian_to_julian_days(cudf::column_view const &input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource *mr) {
  if (input.size() == 0) {
    return cudf::empty_like(input);
  }

  auto output = cudf::make_timestamp_column(input.type(), input.size(),
                                            cudf::detail::copy_bitmask(input, stream, mr),
                                            input.null_count(), stream, mr);

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.size()),
                    output->mutable_view().begin<cudf::timestamp_D>(),
                    [d_input = input.begin<cudf::timestamp_D>()] __device__(auto const idx) {
                      auto constexpr julian_end = cuda::std::chrono::year_month_day{
                          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10},
                          cuda::std::chrono::day{4}};
                      auto constexpr gregorian_start = cuda::std::chrono::year_month_day{
                          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10},
                          cuda::std::chrono::day{15}};

                      auto const ymd = get_ymd_from_days(d_input[idx].time_since_epoch().count());
                      if (ymd > julian_end && ymd < gregorian_start) {
                        // The same as rebasing from `ts = gregorian_start`.
                        // -141417 is the value of rebasing it.
                        return cudf::timestamp_D{cudf::timestamp_D::duration{-141427}};
                      }

                      // No change since this time.
                      if (ymd >= gregorian_start) {
                        return d_input[idx];
                      }

                      // Reinterpret year/month/day as in Julian calendar then compute the Julian
                      // days since epoch.
                      return cudf::timestamp_D{cudf::timestamp_D::duration{days_from_julian(ymd)}};
                    });

  return output;
}

/**
 * @brief Time components used by the date_time_formatter
 */
struct time_components {
  int32_t hour;
  int32_t minute;
  int32_t second;
  int32_t subsecond;
};

/**
 * @brief Specialized modulo expression that handles negative values.
 *
 * @code{.pseudo}
 * Examples:
 *     modulo(1,60)  ->  1
 *     modulo(-1,60) -> 59
 * @endcode
 */
__device__ int32_t modulo_time(int64_t time, int64_t base) {
  return static_cast<int32_t>(((time % base) + base) % base);
}

/**
 * @brief This function handles converting units by dividing and adjusting for negative values.
 *
 * @code{.pseudo}
 * Examples:
 *     scale(-61,60) -> -2
 *     scale(-60,60) -> -1
 *     scale(-59,60) -> -1
 *     scale( 59,60) ->  0
 *     scale( 60,60) ->  1
 *     scale( 61,60) ->  1
 * @endcode
 */
__device__ int64_t scale_time(int64_t time, int64_t base) {
  return (time - ((time < 0) * (base - 1L))) / base;
}

int64_t constexpr MICROS_PER_SECOND = 1'000'000L;

__device__ time_components get_time_components(int64_t micros) {

  auto const subsecond = modulo_time(micros, MICROS_PER_SECOND);
  micros = micros / MICROS_PER_SECOND - ((micros < 0) && (subsecond != 0));
  auto const hour = modulo_time(scale_time(micros, 3600), 24);
  auto const minute = modulo_time(scale_time(micros, 60), 60);
  auto const second = modulo_time(micros, 60);

  return time_components{hour, minute, second, subsecond};
}

std::unique_ptr<cudf::column> gregorian_to_julian_micros(cudf::column_view const &input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource *mr) {
  if (input.size() == 0) {
    return cudf::empty_like(input);
  }

  auto output = cudf::make_timestamp_column(input.type(), input.size(),
                                            cudf::detail::copy_bitmask(input, stream, mr),
                                            input.null_count(), stream, mr);

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.size()),
                    output->mutable_view().begin<cudf::timestamp_us>(),
                    [d_input = input.begin<cudf::timestamp_us>()] __device__(auto const idx) {
                      // October 15th, 1582 UTC.
                      // After this day, there is no difference in micros value between Gregorian
                      // and Julian calendars.
                      int64_t constexpr last_switch_gregorian_ts = -12219292800000000L;

                      auto const micros = d_input[idx].time_since_epoch().count();
                      if (micros >= last_switch_gregorian_ts) {
                        return d_input[idx];
                      }

                      // Reinterpret the input timestamp as in local Julian calendar and takes
                      // microseconds since the epoch from the Julian local date-time.
                      auto const ymd = get_ymd_from_micros(micros);
                      auto const days = days_from_julian(ymd);
                      auto const timeparts = get_time_components(micros);

                      int64_t timestamp = (days * 24L * 3600L) + (timeparts.hour * 3600L) +
                                          (timeparts.minute * 60L) + timeparts.second;
                      timestamp *= MICROS_PER_SECOND; // to microseconds
                      timestamp += timeparts.subsecond;

                      return cudf::timestamp_us{cudf::timestamp_us::duration{timestamp}};
                    });

  return output;
}

} // namespace

namespace cudf::jni {

std::unique_ptr<cudf::column> rebase_gregorian_to_julian(cudf::column_view const &input) {
  auto const type = input.type().id();
  CUDF_EXPECTS(type == cudf::type_id::TIMESTAMP_DAYS ||
                   type == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The input must be either day or microsecond timestamps to rebase.");

  if (input.size() == 0) {
    return cudf::empty_like(input);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr = rmm::mr::get_current_device_resource();

  if (type == cudf::type_id::TIMESTAMP_DAYS) {
    return gregorian_to_julian_days(input, stream, mr);
  }
  return gregorian_to_julian_micros(input, stream, mr);
}

} // namespace cudf::jni
