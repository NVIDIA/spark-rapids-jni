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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

#include "datetime_parser.hpp"

namespace {

using timestamp_components = spark_rapids_jni::timestamp_components;

/**
 * Get the timestamp from epoch from a local date time in a specific time zone.
 * Note: local date time may be overlap or gap, refer to `ZonedDateTime.of`
 *
 */
__device__ cudf::timestamp_us
create_timestamp_from_components_and_zone(timestamp_components local_timestamp_components,
                                          cudf::string_view time_zone) {
  // TODO: implements:
  //   val localDateTime = LocalDateTime.of(localDate, localTime)
  //   val zonedDateTime = ZonedDateTime.of(localDateTime, zoneId)
  //   val instant = Instant.from(zonedDateTime) // main work
  //   instantToMicros(instant)
  // here just return a zero
  return cudf::timestamp_us{cudf::duration_us{0L}};
}

__device__ __host__ inline bool is_digit(const char chr) {
  return (chr >= '0' && chr <= '9');
}

__device__ __host__ inline bool is_whitespace(const char chr) {
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

/**
 * first trim the time zone,
 * then format (+|-)h:mm, (+|-)hh:m or (+|-)h:m to (+|-)hh:mm
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L39
 */
__device__ __host__ cudf::string_view format_zone_id(const cudf::string_view &time_zone_id) {
  const char *curr_ptr = time_zone_id.data();
  const char *end_ptr = curr_ptr + time_zone_id.size_bytes();

  // trim left
  int num_of_left_white_space = 0;
  while (curr_ptr < end_ptr && is_whitespace(*curr_ptr)) {
    ++curr_ptr;
    ++num_of_left_white_space;
  }
  // trim right
  while (curr_ptr < (end_ptr - 1) && is_whitespace(*(end_ptr - 1))) {
    --end_ptr;
  }

  const int length_after_trim = end_ptr - curr_ptr;
  int state = 0;
  char ret[] = "+00:00";     // save the formatted result
  bool is_valid_form = true; // is one form of: (+|-)h:mm$, (+|-)hh:m$, (+|-)h:m$, (+|-)hh:mm$
  int curr_digit_num = 0;
  while (curr_ptr <= end_ptr && is_valid_form) {
    char chr = *curr_ptr;
    if (0 == state) {                                           // expect '+' or '-'
      if (curr_ptr == end_ptr || !('+' == chr || '-' == chr)) { // get $
        is_valid_form = false;
      } else { // get '+' or '-'
        ret[0] = chr;
        state = 1;
      }
    } else if (1 == state) {     // exepct hour digits then ':'
      if (curr_ptr == end_ptr) { // get $
        is_valid_form = false;
      } else if (is_digit(chr) && curr_digit_num < 2) { // get digit
        ++curr_digit_num;
        // set hh part
        ret[1] = ret[2];
        ret[2] = chr;
      } else if (':' == chr && curr_digit_num > 0) { // get ':'
        curr_digit_num = 0;
        state = 2;
      } else {
        is_valid_form = false;
      }
    } else if (2 == state) {                            // expect minute digits then '$'
      if (curr_ptr == end_ptr && curr_digit_num > 0) {  // get $
        state = 3;                                      // success state
      } else if (is_digit(chr) && curr_digit_num < 2) { // get digit
        ++curr_digit_num;
        // set mm part
        ret[4] = ret[5];
        ret[5] = chr;
      } else {
        is_valid_form = false;
      }
    }
    ++curr_ptr;
  }

  if (3 == state) {
    // success
    return cudf::string_view(ret, 6);
  } else {
    // failed to format, just trim time zone id
    return cudf::string_view(time_zone_id.data() + num_of_left_white_space, length_after_trim);
  }
}

__device__ __host__ bool is_valid_digits(int segment, int digits) {
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int constexpr maxDigitsYear = 6;
  // For the nanosecond part, more than 6 digits is allowed, but will be truncated.
  return segment == 6 || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
         // For the zoneId segment(7), it's could be zero digits when it's a region-based zone ID
         (segment == 7 && digits <= 2) ||
         (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}

/**
 *
 * Try to parse timestamp string and get a tuple which contains:
 * - timestamp_components in timestamp string: (year, month, day, hour, minute, seconds,
 * microseconds). If timestamp string does not contain date and only contains time, then
 * (year,month,day) is a invalid value (-1, -1, -1). If timestamp string is invalid, then all the
 * components is -1.
 * - time zone in timestamp string, use default time zone if it's empty
 *
 * Note: the returned time zone is not validated
 *
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 */
__device__ __host__ thrust::pair<timestamp_components, cudf::string_view>
parse_string_to_timestamp_components_tz(cudf::string_view timestamp_str,
                                        cudf::string_view default_time_zone) {
  auto error_compoments = timestamp_components{-1, -1, -1, -1, -1, -1, -1};
  auto error_time_zone = cudf::string_view();

  if (timestamp_str.empty()) {
    return thrust::make_pair(error_compoments, error_time_zone);
  }

  const char *curr_ptr = timestamp_str.data();
  const char *end_ptr = curr_ptr + timestamp_str.size_bytes();

  // trim left
  while (curr_ptr < end_ptr && is_whitespace(*curr_ptr)) {
    ++curr_ptr;
  }
  // trim right
  while (curr_ptr < end_ptr - 1 && is_whitespace(*(end_ptr - 1))) {
    --end_ptr;
  }

  if (curr_ptr == end_ptr) {
    return thrust::make_pair(error_compoments, error_time_zone);
  }

  const char *const bytes = curr_ptr;
  const size_t bytes_length = end_ptr - curr_ptr;

  thrust::optional<cudf::string_view> tz;
  int segments[] = {1, 1, 1, 0, 0, 0, 0, 0, 0};
  int segments_len = 9;
  int i = 0;
  int current_segment_value = 0;
  int current_segment_digits = 0;
  size_t j = 0;
  int digits_milli = 0;
  bool just_time = false;
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
    char b = bytes[j];
    int parsed_value = static_cast<int32_t>(b - '0');
    if (parsed_value < 0 || parsed_value > 9) {
      if (0 == j && 'T' == b) {
        just_time = true;
        i += 3;
      } else if (i < 2) {
        if (b == '-') {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else if (0 == i && ':' == b && !year_sign.has_value()) {
          just_time = true;
          if (!is_valid_digits(3, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[3] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i = 4;
        } else {
          return thrust::make_pair(error_compoments, error_time_zone);
        }
      } else if (2 == i) {
        if (' ' == b || 'T' == b) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_pair(error_compoments, error_time_zone);
        }
      } else if (3 == i || 4 == i) {
        if (':' == b) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_pair(error_compoments, error_time_zone);
        }
      } else if (5 == i || 6 == i) {
        if ('.' == b && 5 == i) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
          tz = cudf::string_view(bytes + j, (bytes_length - j));
          j = bytes_length - 1;
        }
        if (i == 6 && '.' != b) {
          i += 1;
        }
      } else {
        if (i < segments_len && (':' == b || ' ' == b)) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_pair(error_compoments, error_time_zone);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_pair(error_compoments, error_time_zone);
        }
      }
    } else {
      if (6 == i) {
        digits_milli += 1;
      }
      // We will truncate the nanosecond part if there are more than 6 digits, which results
      // in loss of precision
      if (6 != i || current_segment_digits < 6) {
        current_segment_value = current_segment_value * 10 + parsed_value;
      }
      current_segment_digits += 1;
    }
    j += 1;
  }

  if (!is_valid_digits(i, current_segment_digits)) {
    return thrust::make_pair(error_compoments, error_time_zone);
  }
  segments[i] = current_segment_value;

  while (digits_milli < 6) {
    segments[6] *= 10;
    digits_milli += 1;
  }

  cudf::string_view timze_zone;
  if (tz.has_value()) {
    timze_zone = format_zone_id(tz.value());
  } else {
    timze_zone = default_time_zone;
  }

  segments[0] *= year_sign.value_or(1);
  // above is translated from Spark.

  // set components
  auto components = timestamp_components{segments[0],
                                         static_cast<int8_t>(segments[1]),
                                         static_cast<int8_t>(segments[2]),
                                         static_cast<int8_t>(segments[3]),
                                         static_cast<int8_t>(segments[4]),
                                         static_cast<int8_t>(segments[5]),
                                         segments[6]};
  if (just_time) {
    components.year = components.month = components.day = -1;
  }
  return thrust::make_pair(components, timze_zone);
}

struct parse_timestamp_string_fn {
  cudf::column_device_view const d_strings;
  cudf::string_view default_time_zone;

  __device__ cudf::timestamp_us operator()(const cudf::size_type &idx) const {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    auto components_tz = parse_string_to_timestamp_components_tz(d_str, default_time_zone);
    return create_timestamp_from_components_and_zone(components_tz.first, components_tz.second);
  }
};

/**
 *
 * Trims and parses timestamp string column to a timestamp column and a time zone
 * column
 *
 */
std::unique_ptr<cudf::column> parse_string_to_timestamp_and_time_zone(
    cudf::strings_column_view const &input, cudf::string_view default_time_zone,
    rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto output_timestamp = cudf::make_timestamp_column(
      cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, input.size(),
      cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count(), stream, mr);

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.size()),
                    output_timestamp->mutable_view().begin<cudf::timestamp_us>(),
                    parse_timestamp_string_fn{*d_strings, default_time_zone});

  return output_timestamp;
}

} // namespace

namespace spark_rapids_jni {

/**
 *
 * Trims and parses timestamp string column to a timestamp components column and a time zone
 * column, then create timestamp column
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * @param input input string column view.
 * @param default_time_zone if input string does not contain a time zone, use this time zone.
 * @returns timestamp components column and time zone string.
 * be empty.
 */
std::unique_ptr<cudf::column> parse_string_to_timestamp(cudf::strings_column_view const &input,
                                                        cudf::string_view default_time_zone) {
  auto timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  if (input.size() == 0) {
    return cudf::make_empty_column(timestamp_type.id());
  }

  auto const stream = cudf::get_default_stream();
  auto const mr = rmm::mr::get_current_device_resource();
  return parse_string_to_timestamp_and_time_zone(input, default_time_zone, stream, mr);
}

/**
 *
 * Refer to `SparkDateTimeUtils.stringToTimestampWithoutTimeZone`
 */
std::unique_ptr<cudf::column>
string_to_timestamp_without_time_zone(cudf::strings_column_view const &input,
                                      bool allow_time_zone) {
  // TODO
  throw std::runtime_error("Not implemented!!!");
}

/**
 *
 * Refer to `SparkDateTimeUtils.stringToTimestamp`
 */
std::unique_ptr<cudf::column> string_to_timestamp(cudf::strings_column_view const &input,
                                                  cudf::string_view time_zone) {
  // TODO
  throw std::runtime_error("Not implemented!!!");
}

} // namespace spark_rapids_jni
