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
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "datetime_parser.hpp"

namespace {

/**
 * represents local date time in a time zone.
 */
struct timestamp_components {
  int32_t year; // max 6 digits
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t microseconds;
};

/**
 * convert a local time in a time zone to UTC timestamp
 */
__device__ __host__ thrust::tuple<cudf::timestamp_us, bool>
to_utc_timestamp(timestamp_components components, cudf::string_view const &time_zone) {
  // TODO replace the temp implementation
  long v = 365L * 86400L * 1000000L * components.year + 30L * 86400L * 1000000L * components.month +
           86400L * 1000000L * components.day + 3600L * 1000000L * components.hour +
           60L * 1000000L * components.minute + 1000000L * components.second +
           components.microseconds;
  return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{v}}, true);
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

__device__ __host__ bool is_valid_digits(int segment, int digits) {
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int constexpr maxDigitsYear = 6;
  // For the nanosecond part, more than 6 digits is allowed, but will be truncated.
  return segment == 6 || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
         // For the zoneId segment(7), it's could be zero digits when it's a region-based zone ID
         (segment == 7 && digits <= 2) ||
         (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}

__device__ __host__ thrust::tuple<cudf::timestamp_us, bool>
parse_string_to_timestamp_us(cudf::string_view const &timestamp_str, const char *default_time_zone,
                             cudf::size_type default_time_zone_char_len, bool allow_time_zone,
                             bool allow_special_expressions, bool ansi_mode) {
  auto error_us = cudf::timestamp_us{cudf::duration_us{0}};

  if (timestamp_str.empty()) {
    return thrust::make_tuple(error_us, false);
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
    return thrust::make_tuple(error_us, false);
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
            return thrust::make_tuple(error_us, false);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else if (0 == i && ':' == b && !year_sign.has_value()) {
          just_time = true;
          if (!is_valid_digits(3, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[3] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i = 4;
        } else {
          return thrust::make_tuple(error_us, false);
        }
      } else if (2 == i) {
        if (' ' == b || 'T' == b) {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
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
          segments[i] = current_segment_value;
          current_segment_value = 0;
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
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          if (!is_valid_digits(i, current_segment_digits)) {
            return thrust::make_tuple(error_us, false);
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
            return thrust::make_tuple(error_us, false);
          }
          segments[i] = current_segment_value;
          current_segment_value = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return thrust::make_tuple(error_us, false);
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
    return thrust::make_tuple(error_us, false);
  }
  segments[i] = current_segment_value;

  while (digits_milli < 6) {
    segments[6] *= 10;
    digits_milli += 1;
  }

  cudf::string_view timze_zone;
  if (tz.has_value()) {
    timze_zone = tz.value();
  } else {
    timze_zone = cudf::string_view(default_time_zone, default_time_zone_char_len);
  }

  segments[0] *= year_sign.value_or(1);
  // above is ported from Spark.

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
  return to_utc_timestamp(components, timze_zone);
}

struct parse_timestamp_string_fn {
  cudf::column_device_view const d_strings;
  const char *default_time_zone;
  cudf::size_type default_time_zone_char_len;
  bool allow_time_zone;
  bool allow_special_expressions;
  bool ansi_mode;

  __device__ thrust::tuple<cudf::timestamp_us, bool> operator()(const cudf::size_type &idx) const {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    return parse_string_to_timestamp_us(d_str, default_time_zone, default_time_zone_char_len,
                                        allow_time_zone, allow_special_expressions, ansi_mode);
  }
};

/**
 *
 * Trims and parses timestamp string column to a timestamp column and a is valid column
 *
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
string_to_timestamp(cudf::strings_column_view const &input,
                    std::string_view const &default_time_zone, bool allow_time_zone,
                    bool allow_special_expressions, bool ansi_mode, rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource *mr) {
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto output_timestamp = cudf::make_timestamp_column(
      cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, input.size(),
      cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count(), stream, mr);
  auto output_bool = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8}, input.size(),
      cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count(), stream, mr);

  thrust::transform(
      rmm::exec_policy(stream), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input.size()),
      thrust::make_zip_iterator(
          thrust::make_tuple(output_timestamp->mutable_view().begin<cudf::timestamp_us>(),
                             output_bool->mutable_view().begin<bool>())),
      parse_timestamp_string_fn{*d_strings, default_time_zone.data(),
                                static_cast<cudf::size_type>(default_time_zone.size()),
                                allow_time_zone, allow_special_expressions, ansi_mode});

  return std::make_pair(std::move(output_timestamp), std::move(output_bool));
}

} // namespace

namespace spark_rapids_jni {

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
parse_string_to_timestamp(cudf::strings_column_view const &input,
                          std::string_view const &default_time_zone, bool allow_time_zone,
                          bool allow_special_expressions, bool ansi_mode) {
  auto timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(timestamp_type.id()),
                          cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8}));
  }

  auto const stream = cudf::get_default_stream();
  auto const mr = rmm::mr::get_current_device_resource();
  return string_to_timestamp(input, default_time_zone, allow_time_zone, allow_special_expressions,
                             ansi_mode, stream, mr);
}

} // namespace spark_rapids_jni
