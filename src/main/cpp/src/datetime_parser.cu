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

#include <map>
#include <numeric>
#include <vector>

#include <iostream>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "datetime_parser.hpp"

namespace {

/**
 * Represents local date time in a time zone.
 */
struct timestamp_components {
  int32_t year;  // max 6 digits
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t microseconds;
};

/**
 * Convert a local time in a time zone to a UTC timestamp
 */
__device__ __host__ thrust::tuple<cudf::timestamp_us, bool> to_utc_timestamp(
  timestamp_components const& components, cudf::string_view const& time_zone)
{
  // TODO replace the following fake implementation
  long seconds = components.year * 365L * 86400L + components.month * 30L * 86400L +
                 components.day * 86400L + components.hour * 3600L + components.minute * 60L +
                 components.second;
  long us = seconds * 1000000L + components.microseconds;
  return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{us}}, true);
}

/**
 * Is white space
 */
__device__ __host__ inline bool is_whitespace(const char chr)
{
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

/**
 * Whether the given two strings are equal,
 * used to compare special timestamp strings ignoring case:
 *   "epoch", "now", "today", "yesterday", "tomorrow"
 * the expect string should be lower-case a-z chars
 */
__device__ __host__ inline bool equals_ascii_ignore_case(char const* actual_begin,
                                                         char const* actual_end,
                                                         char const* expect_begin,
                                                         char const* expect_end)
{
  if (actual_end - actual_begin != expect_end - expect_begin) { return false; }

  while (expect_begin < expect_end) {
    // the diff between upper case and lower case for a same char is 32
    if (*actual_begin != *expect_begin && *actual_begin != (*expect_begin - 32)) { return false; }
    actual_begin++;
    expect_begin++;
  }
  return true;
}

/**
 * Ported from Spark
 */
__device__ __host__ bool is_valid_digits(int segment, int digits)
{
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int constexpr maxDigitsYear = 6;
  // For the nanosecond part, more than 6 digits is allowed, but will be truncated.
  return segment == 6 || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
         // For the zoneId segment(7), it's could be zero digits when it's a region-based zone ID
         (segment == 7 && digits <= 2) ||
         (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}

/**
 * Ported from Spark:
 *   https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 *   org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * Parse a string with time zone to a timestamp.
 * The bool in the returned tuple is false if the parse failed.
 */
__device__ __host__ thrust::tuple<cudf::timestamp_us, bool> parse_string_to_timestamp_us(
  cudf::string_view const& timestamp_str,
  const char* default_time_zone,
  cudf::size_type default_time_zone_char_len,
  bool allow_time_zone,
  bool allow_special_expressions,
  cudf::timestamp_us epoch,
  cudf::timestamp_us now,
  cudf::timestamp_us today,
  cudf::timestamp_us tomorrow,
  cudf::timestamp_us yesterday)
{
  auto error_us = cudf::timestamp_us{cudf::duration_us{0}};

  if (timestamp_str.empty()) { return thrust::make_tuple(error_us, false); }

  const char* curr_ptr = timestamp_str.data();
  const char* end_ptr  = curr_ptr + timestamp_str.size_bytes();

  // trim left
  while (curr_ptr < end_ptr && is_whitespace(*curr_ptr)) {
    ++curr_ptr;
  }
  // trim right
  while (curr_ptr < end_ptr - 1 && is_whitespace(*(end_ptr - 1))) {
    --end_ptr;
  }

  // special strings: epoch, now, today, yesterday, tomorrow
  if (allow_special_expressions) {
    char const* begin_epoch = "epoch";
    char const* end_epoch   = begin_epoch + 5;

    char const* begin_now = "now";
    char const* end_now   = begin_now + 3;

    char const* begin_today = "today";
    char const* end_today   = begin_today + 5;

    char const* begin_tomorrow = "tomorrow";
    char const* end_tomorrow   = begin_tomorrow + 8;

    char const* begin_yesterday = "yesterday";
    char const* end_yesterday   = begin_yesterday + 9;

    if (equals_ascii_ignore_case(curr_ptr, end_ptr, begin_epoch, end_epoch)) {
      // epoch
      return thrust::make_tuple(epoch, true);
    } else if (equals_ascii_ignore_case(curr_ptr, end_ptr, begin_now, end_now)) {
      // now
      return thrust::make_tuple(now, true);
    } else if (equals_ascii_ignore_case(curr_ptr, end_ptr, begin_today, end_today)) {
      // today
      return thrust::make_tuple(today, true);
    } else if (equals_ascii_ignore_case(curr_ptr, end_ptr, begin_tomorrow, end_tomorrow)) {
      // tomorrow
      return thrust::make_tuple(tomorrow, true);
    } else if (equals_ascii_ignore_case(curr_ptr, end_ptr, begin_yesterday, end_yesterday)) {
      // yesterday
      return thrust::make_tuple(yesterday, true);
    }
  }

  if (curr_ptr == end_ptr) { return thrust::make_tuple(error_us, false); }

  const char* const bytes   = curr_ptr;
  const size_t bytes_length = end_ptr - curr_ptr;

  thrust::optional<cudf::string_view> tz;
  int segments[]             = {1, 1, 1, 0, 0, 0, 0, 0, 0};
  int segments_len           = 9;
  int i                      = 0;
  int current_segment_value  = 0;
  int current_segment_digits = 0;
  size_t j                   = 0;
  int digits_milli           = 0;
  bool just_time             = false;
  thrust::optional<int> year_sign;
  if ('-' == bytes[j] || '+' == bytes[j]) {
    if ('-' == bytes[j]) {
      year_sign = -1;
    } else {
      year_sign = 1;
    }
    j += 1;
  }

  while (j < bytes_length) {
    char b           = bytes[j];
    int parsed_value = static_cast<int32_t>(b - '0');
    if (parsed_value < 0 || parsed_value > 9) {
      if (0 == j && 'T' == b) {
        just_time = true;
        i += 3;
      } else if (i < 2) {
        if (b == '-') {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else if (0 == i && ':' == b && !year_sign.has_value()) {
          just_time = true;
          if (!is_valid_digits(3, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[3]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i                      = 4;
        } else {
          return thrust::make_tuple(error_us, false);
        }
      } else if (2 == i) {
        if (' ' == b || 'T' == b) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_tuple(error_us, false);
        }
      } else if (3 == i || 4 == i) {
        if (':' == b) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_tuple(error_us, false);
        }
      } else if (5 == i || 6 == i) {
        if ('.' == b && 5 == i) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
          tz = cudf::string_view(bytes + j, (bytes_length - j));
          j  = bytes_length - 1;
        }
        if (i == 6 && '.' != b) { i += 1; }
      } else {
        if (i < segments_len && (':' == b || ' ' == b)) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_tuple(error_us, false);
        }
      }
    } else {
      if (6 == i) { digits_milli += 1; }
      // We will truncate the nanosecond part if there are more than 6 digits, which results
      // in loss of precision
      if (6 != i || current_segment_digits < 6) {
        current_segment_value = current_segment_value * 10 + parsed_value;
      }
      current_segment_digits += 1;
    }
    j += 1;
  }

  if (!is_valid_digits(i, current_segment_digits)) { return thrust::make_tuple(error_us, false); }
  segments[i] = current_segment_value;

  while (digits_milli < 6) {
    segments[6] *= 10;
    digits_milli += 1;
  }

  cudf::string_view time_zone;
  if (tz.has_value()) {
    time_zone = tz.value();
  } else {
    time_zone = cudf::string_view(default_time_zone, default_time_zone_char_len);
  }

  segments[0] *= year_sign.value_or(1);
  // above is ported from Spark.

  if (default_time_zone_char_len == 0) {
    // invoke from `string_to_timestamp_without_time_zone`
    if (just_time || !allow_time_zone && tz.has_value()) {
      return thrust::make_tuple(error_us, false);
    }
  } else {
    // invoke from `string_to_timestamp`
    if (just_time) {
      // Update here to support the following format:
      //   `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
      //   `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
      // by set local date in a time zone: year-month-day.
      // Above 2 formats are time zone related, Spark uses LocalDate.now(zoneId)

      // do not support currently
      return thrust::make_tuple(error_us, false);
    }
  }

  // set components
  auto components = timestamp_components{segments[0],
                                         static_cast<int8_t>(segments[1]),
                                         static_cast<int8_t>(segments[2]),
                                         static_cast<int8_t>(segments[3]),
                                         static_cast<int8_t>(segments[4]),
                                         static_cast<int8_t>(segments[5]),
                                         segments[6]};

  return to_utc_timestamp(components, time_zone);
}

struct parse_timestamp_string_fn {
  cudf::column_device_view const d_strings;
  const char* default_time_zone;
  cudf::size_type default_time_zone_char_len;
  bool allow_time_zone;
  bool allow_special_expressions;
  // TODO the following should be passed in.
  // Note: today, tomorrow, yesterday are time zone related, should use time zone to generate.
  cudf::timestamp_us epoch     = cudf::timestamp_us{cudf::duration_us{111L}};
  cudf::timestamp_us now       = cudf::timestamp_us{cudf::duration_us{222L}};
  cudf::timestamp_us today     = cudf::timestamp_us{cudf::duration_us{333L}};
  cudf::timestamp_us tomorrow  = cudf::timestamp_us{cudf::duration_us{444L}};
  cudf::timestamp_us yesterday = cudf::timestamp_us{cudf::duration_us{555L}};

  __device__ thrust::tuple<cudf::timestamp_us, bool> operator()(const cudf::size_type& idx) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    return parse_string_to_timestamp_us(d_str,
                                        default_time_zone,
                                        default_time_zone_char_len,
                                        allow_time_zone,
                                        allow_special_expressions,
                                        epoch,
                                        now,
                                        today,
                                        tomorrow,
                                        yesterday);
  }
};

/**
 *
 * Trims and parses timestamp string column to a timestamp column and a is valid column
 *
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> to_timestamp(
  cudf::strings_column_view const& input,
  std::string_view const& default_time_zone,
  bool allow_time_zone,
  bool allow_special_expressions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto output_timestamp =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                input.size(),
                                cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                input.null_count(),
                                stream,
                                mr);
  // record which string is failed to parse.
  auto output_bool =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                  input.size(),
                                  cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                  input.null_count(),
                                  stream,
                                  mr);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    thrust::make_zip_iterator(
      thrust::make_tuple(output_timestamp->mutable_view().begin<cudf::timestamp_us>(),
                         output_bool->mutable_view().begin<bool>())),
    parse_timestamp_string_fn{*d_strings,
                              default_time_zone.data(),
                              static_cast<cudf::size_type>(default_time_zone.size()),
                              allow_time_zone,
                              allow_special_expressions});

  return std::make_pair(std::move(output_timestamp), std::move(output_bool));
}

/**
 * Set the null mask of timestamp column according to the valid column.
 */
void update_bitmask(cudf::column& timestamp_column,
                    cudf::column const& validity_column,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr)
{
  auto const& ts_view    = timestamp_column.view();
  auto const& valid_view = validity_column.view();
  std::vector<cudf::bitmask_type const*> masks;
  std::vector<cudf::size_type> offsets;
  if (timestamp_column.nullable()) {
    masks.push_back(ts_view.null_mask());
    offsets.push_back(ts_view.offset());
  }

  // generate bitmask from `validity_column`
  auto [valid_bitmask, valid_null_count] = cudf::detail::valid_if(
    valid_view.begin<bool>(), valid_view.end<bool>(), thrust::identity<bool>{}, stream, mr);

  masks.push_back(static_cast<cudf::bitmask_type*>(valid_bitmask.data()));
  offsets.push_back(0);

  // merge 2 bitmasks
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(masks, offsets, timestamp_column.size(), stream, mr);

  timestamp_column.set_null_mask(null_mask, null_count);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 */
std::pair<std::unique_ptr<cudf::column>, bool> parse_string_to_timestamp(
  cudf::strings_column_view const& input,
  std::string_view const& default_time_zone,
  bool allow_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  auto timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(timestamp_type.id()), true);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource();
  auto [timestamp_column, validity_column] =
    to_timestamp(input, default_time_zone, allow_time_zone, allow_special_expressions, stream, mr);

  if (ansi_mode) {
    // create scalar, value is false, is_valid is true
    cudf::numeric_scalar<bool> false_scalar{false, true, stream, mr};
    if (cudf::contains(*validity_column, false_scalar, stream)) {
      // has invalid value in validity column under ansi mode
      return std::make_pair(nullptr, false);
    } else {
      update_bitmask(*timestamp_column, *validity_column, stream, mr);
      return std::make_pair(std::move(timestamp_column), true);
    }
  } else {
    update_bitmask(*timestamp_column, *validity_column, stream, mr);
    return std::make_pair(std::move(timestamp_column), true);
  }
}

}  // namespace

namespace spark_rapids_jni {

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * If does not have time zone in string, use the default time zone.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp(
  cudf::strings_column_view const& input,
  std::string_view const& default_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  CUDF_EXPECTS(default_time_zone.size() > 0, "should specify default time zone");
  return parse_string_to_timestamp(
    input, default_time_zone, true, allow_special_expressions, ansi_mode);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * Do not use the time zone in string.
 * If allow_time_zone is false and string contains time zone, then the string is invalid.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp_without_time_zone(
  cudf::strings_column_view const& input,
  bool allow_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  return parse_string_to_timestamp(input,
                                   std::string_view(""),  // specify empty time zone
                                   allow_time_zone,
                                   allow_special_expressions,
                                   ansi_mode);
}

}  // namespace spark_rapids_jni
