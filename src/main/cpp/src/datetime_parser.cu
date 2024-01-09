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

#include "datetime_parser.hpp"

#include <iostream>
#include <vector>

#include <cuda/std/cassert>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>

#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>

#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using size_type                = cudf::size_type;
using string_view              = cudf::string_view;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

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
__device__ inline bool equals_ascii_ignore_case(char const *actual_begin,
                                                char const *actual_end,
                                                char const *expect_begin,
                                                char const *expect_end) {
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

enum ParseResult {
  OK = 0,
  INVALID = 1,
  UNSUPPORTED = 2
};

template <bool with_timezone>
struct parse_timestamp_string_fn {
  column_device_view const d_strings;
  column_device_view const special_datetime_names;
  size_type default_tz_index;
  bool allow_tz_in_date_str = true;
  // The list column of transitions to figure out the correct offset
  // to adjust the timestamp. The type of the values in this column is
  // LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32, looseTzInstant: int64>>.
  thrust::optional<lists_column_device_view const> transitions = thrust::nullopt;
  thrust::optional<column_device_view const> tz_indices = thrust::nullopt;

  __device__ thrust::tuple<cudf::timestamp_us, uint8_t> operator()(const cudf::size_type& idx) const
  {
    if (!d_strings.is_valid(idx)) {
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
    }

    auto const d_str = d_strings.element<cudf::string_view>(idx);

    timestamp_components ts_comp{};
    char const * tz_lit_ptr = nullptr;
    size_type tz_lit_len = 0;
    switch (parse_string_to_timestamp_us(&ts_comp, &tz_lit_ptr, &tz_lit_len, d_str)) {
      case ParseResult::INVALID:
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
      case ParseResult::UNSUPPORTED:
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::UNSUPPORTED);
      case ParseResult::OK:
      default:
        break;
    }

    if constexpr (!with_timezone) {
      // path without timezone, in which unix_timestamp is straightforwardly computed
      auto const ts_unaligned = compute_epoch_us(ts_comp);
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{ts_unaligned}}, ParseResult::OK);
    }
  
    // path with timezone, in which timezone offset has to be determined before computing unix_timestamp
    int64_t tz_offset;
    if (tz_lit_ptr == nullptr) {
      tz_offset = extract_timezone_offset(compute_loose_epoch_s(ts_comp), default_tz_index);
    } else {
      auto tz_view = string_view(tz_lit_ptr, tz_lit_len);
      if (auto [utc_offset, ret_code] = parse_utc_like_tz(tz_view); ret_code == 0) {
        tz_offset = utc_offset;
      } else if (ret_code == 1) {
        auto tz_index = query_index_from_tz_db(tz_view);
        if (tz_index > transitions->size()) {
          if (tz_index == tz_indices->size())
            return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
          return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::UNSUPPORTED);
        }
        tz_offset = extract_timezone_offset(compute_loose_epoch_s(ts_comp), tz_index);
      } else {
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
      }
    }

    auto const ts_unaligned = compute_epoch_us(ts_comp);

    return thrust::make_tuple(
        cudf::timestamp_us{cudf::duration_us{ts_unaligned - tz_offset * 1000000L}},
        ParseResult::OK);
  }

  // TODO: support CST/PST/AST
  __device__ inline thrust::pair<int64_t, uint8_t> parse_utc_like_tz(string_view const &tz_lit) const
  {
    size_type len = tz_lit.size_bytes();

    char const *ptr = tz_lit.data();

    if (*ptr == 'Z') {
      if (len > 1) return {0, 1};
      return {0, 0};
    }

    size_t char_offset = 0;

    if (len > 2
        && ((*ptr == 'G' && *(ptr + 1) == 'M' && *(ptr + 2) == 'T')
        || (*ptr == 'U' && *(ptr + 1) == 'T' && *(ptr + 2) == 'C'))) {
      char_offset = 3;
    }

    if (len == char_offset) return {0, 0};

    char const sign_char = *(ptr + char_offset++);
    int64_t sign;
    if (sign_char == '+') {
      sign = 1L;
    } else if (sign_char == '-') {
      sign = -1L;
    } else {
      return {0, char_offset < 3 ? 1 : 2};
    }

    int64_t hms[3] = {0L, 0L, 0L};
    bool has_colon = false;
    bool one_digit_mm = false;
    for (size_type i = 0; i < 3; i++) {
      if (i == 2 && one_digit_mm) return {0, 2};

      hms[i] = *(ptr + char_offset++) - '0';
      if (hms[i] < 0 || hms[i] > 9) return {0, 2};

      if (len == char_offset) {
        if (i > 0) {
          if (!has_colon) return {0, 2};
          one_digit_mm = true;
        }
        break;
      }

      if (*(ptr + char_offset) == ':') {
        if (len == ++char_offset) break;
        has_colon = true;
        continue;
      }

      auto digit = *(ptr + char_offset++) - '0';
      if (digit < 0 || digit > 9) return {0, 2};
      hms[i] = hms[i] * 10 + digit;

      if (len == char_offset) break;
      if (*(ptr + char_offset) == ':') {
        if (len == ++char_offset) break;
        has_colon = true;
        continue;
      }
      if (has_colon) return {0, 2};
    }

    if (hms[0] > 18 || hms[1] > 59 || hms[2] > 59) return {0, 2};
    if (hms[0] == 18 && hms[1] + hms[2] > 0) return {0, 2};

    return {sign * (hms[0] * 3600L + hms[1] * 60L + hms[2]), 0};
  }

  __device__ inline int query_index_from_tz_db(string_view const &tz_lit) const
  {
    // TODO: replace with more efficient approach (such as binary search or prefix tree)
    auto predicate = [tz = tz_indices, &tz_lit] __device__(auto const i) {
      return tz->element<string_view>(i) == tz_lit;
    };
    auto ret = thrust::find_if(thrust::seq,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(tz_indices->size()),
                               predicate);

    return *ret;
  }

  __device__ inline int64_t extract_timezone_offset(int64_t loose_epoch_second, size_type tz_index) const
  {
    auto const &utc_offsets = transitions->child().child(2);
    auto const &loose_instants = transitions->child().child(3);

    auto const local_transitions = cudf::list_device_view{*transitions, tz_index};
    auto const list_size = local_transitions.size();

    auto const transition_times = cudf::device_span<int64_t const>(
        loose_instants.data<int64_t>() + local_transitions.element_offset(0),
        static_cast<size_t>(list_size));

    auto const it = thrust::upper_bound(
        thrust::seq, transition_times.begin(), transition_times.end(), loose_epoch_second);
    auto const idx = static_cast<size_type>(thrust::distance(transition_times.begin(), it));
    auto const list_offset = local_transitions.element_offset(idx - 1);

    return static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset));
  }

  __device__ inline int64_t compute_loose_epoch_s(timestamp_components const& ts) const
  {
    return (ts.year * 400 + (ts.month - 1) * 31 + ts.day - 1) * 86400L + ts.hour * 3600L + ts.minute * 60L + ts.second;
  }

  __device__ inline int64_t compute_epoch_us(timestamp_components const& ts) const
  {
    auto const ymd =  // chrono class handles the leap year calculations for us
        cuda::std::chrono::year_month_day(
            cuda::std::chrono::year{ts.year},
            cuda::std::chrono::month{static_cast<uint32_t>(ts.month)},
            cuda::std::chrono::day{static_cast<uint32_t>(ts.day)});
    auto days = cuda::std::chrono::sys_days(ymd).time_since_epoch().count();

    int64_t timestamp_s = (days * 24L * 3600L) + (ts.hour * 3600L) + (ts.minute * 60L) + ts.second;

    return timestamp_s * 1000000L + ts.microseconds;
  }

  /**
   * Ported from Spark:
   *   https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   *   org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
   *
   * Parse a string with time zone to a timestamp.
   * The bool in the returned tuple is false if the parse failed.
   */
  __device__ inline ParseResult parse_string_to_timestamp_us(
      timestamp_components *ts_comp,
      char const **parsed_tz_ptr,
      size_type *parsed_tz_length,
      cudf::string_view const &timestamp_str) const {

    if (timestamp_str.empty()) { return ParseResult::INVALID; }

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

    // TODO: support special dates [epoch, now, today, yesterday, tomorrow]
    for (size_type i = 0; i < special_datetime_names.size(); i++) {
      auto const& ref = special_datetime_names.element<string_view>(i);
      if (equals_ascii_ignore_case(curr_ptr, end_ptr, ref.data(), ref.data() + ref.size_bytes())) {
        *parsed_tz_ptr = ref.data();
        *parsed_tz_length = ref.size_bytes();
        return ParseResult::UNSUPPORTED;
      }
    }

    if (curr_ptr == end_ptr) { return ParseResult::INVALID; }

    const char *const bytes = curr_ptr;
    const size_type bytes_length = end_ptr - curr_ptr;

    int segments[] = {1, 1, 1, 0, 0, 0, 0, 0, 0};
    int segments_len = 9;
    int i = 0;
    int current_segment_value = 0;
    int current_segment_digits = 0;
    size_t j = 0;
    int digits_milli = 0;
    // bool just_time = false;
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
          // just_time = true;
          i += 3;
        } else if (i < 2) {
          if (b == '-') {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else if (0 == i && ':' == b && !year_sign.has_value()) {
            // just_time = true;
            if (!is_valid_digits(3, current_segment_digits)) { return ParseResult::INVALID; }
            segments[3] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i = 4;
          } else {
            return ParseResult::INVALID;
          }
        } else if (2 == i) {
          if (' ' == b || 'T' == b) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
          }
        } else if (3 == i || 4 == i) {
          if (':' == b) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
          }
        } else if (5 == i || 6 == i) {
          if ('.' == b && 5 == i) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            if (!is_valid_digits(i, current_segment_digits) || !allow_tz_in_date_str) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
            *parsed_tz_ptr = bytes + j;
            // strip the whitespace between timestamp and timezone
            while (*parsed_tz_ptr < end_ptr && is_whitespace(**parsed_tz_ptr)) ++(*parsed_tz_ptr);
            *parsed_tz_length = end_ptr - *parsed_tz_ptr;
            break;
          }
          if (i == 6 && '.' != b) { i += 1; }
        } else {
          if (i < segments_len && (':' == b || ' ' == b)) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
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

    if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
    segments[i] = current_segment_value;

    while (digits_milli < 6) {
      segments[6] *= 10;
      digits_milli += 1;
    }

    segments[0] *= year_sign.value_or(1);
    // above is ported from Spark.

    // set components
    ts_comp->year = segments[0];
    ts_comp->month = static_cast<int8_t>(segments[1]);
    ts_comp->day = static_cast<int8_t>(segments[2]);
    ts_comp->hour = static_cast<int8_t>(segments[3]);
    ts_comp->minute = static_cast<int8_t>(segments[4]);
    ts_comp->second = static_cast<int8_t>(segments[5]);
    ts_comp->microseconds = segments[6];

    return ParseResult::OK;
  }
};

/**
 *
 * Trims and parses timestamp string column to a timestamp column and a is valid column
 *
 */
std::pair<std::unique_ptr<cudf::column>, bool> to_timestamp(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& special_datetime_lit,
  bool ansi_mode,
  bool allow_tz_in_date_str = true,
  size_type default_tz_index = 1000000000,
  cudf::column_view const *transitions = nullptr,
  cudf::strings_column_view const *tz_indices = nullptr)
{
  auto const stream = cudf::get_default_stream();
  auto const mr = rmm::mr::get_current_device_resource();

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto d_special_datetime_lit = cudf::column_device_view::create(special_datetime_lit.parent(), stream);

  auto result_col =
      cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                  input.size(),
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
  // record which string is failed to parse.
  auto result_valid_col =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::UINT8},
                                    input.size(),
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  if (transitions == nullptr || tz_indices == nullptr) {
    thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(input.size()),
        thrust::make_zip_iterator(
            thrust::make_tuple(result_col->mutable_view().begin<cudf::timestamp_us>(),
                               result_valid_col->mutable_view().begin<uint8_t>())),
        parse_timestamp_string_fn<false>{*d_strings,
                                         *d_special_datetime_lit,
                                         default_tz_index,
                                         allow_tz_in_date_str});
  } else {
    auto const ft_cdv_ptr = column_device_view::create(*transitions, stream);
    auto const d_transitions = lists_column_device_view{*ft_cdv_ptr};
    auto d_tz_indices = cudf::column_device_view::create(tz_indices->parent(), stream);

    thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(input.size()),
        thrust::make_zip_iterator(
            thrust::make_tuple(result_col->mutable_view().begin<cudf::timestamp_us>(),
                               result_valid_col->mutable_view().begin<uint8_t>())),
        parse_timestamp_string_fn<true>{*d_strings,
                                        *d_special_datetime_lit,
                                        default_tz_index,
                                        true,
                                        d_transitions,
                                        *d_tz_indices});
  }

  auto valid_view = result_valid_col->mutable_view();

  auto exception_exists = thrust::any_of(
    rmm::exec_policy(stream),
    valid_view.begin<uint8_t>(),
    valid_view.end<uint8_t>(),
    []__device__(uint8_t e) { return e == ParseResult::UNSUPPORTED; });
  if (exception_exists) {
    CUDF_FAIL("There exists unsupported timestamp schema!");
  }

  auto [valid_bitmask, valid_null_count] = cudf::detail::valid_if(
      valid_view.begin<uint8_t>(), valid_view.end<uint8_t>(),
      [] __device__(uint8_t e) { return e == 0; },
      stream, mr);

  if (ansi_mode && input.null_count() < valid_null_count) {
    // has invalid value in validity column under ansi mode
    return std::make_pair(nullptr, false);
  }

  result_col->set_null_mask(valid_bitmask, valid_null_count, stream);
  return std::make_pair(std::move(result_col), true);
}

}  // namespace

namespace spark_rapids_jni {

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * If does not have time zone in string, use the default time zone.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp_with_tz(
  cudf::strings_column_view const& input,
  cudf::column_view const& transitions,
  cudf::strings_column_view const& tz_indices,
  cudf::strings_column_view const& special_datetime_lit,
  cudf::size_type default_tz_index,
  bool ansi_mode)
{
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(cudf::type_id::TIMESTAMP_MICROSECONDS), true);
  }
  return to_timestamp(input, special_datetime_lit, ansi_mode, true, default_tz_index, &transitions, &tz_indices);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * Do not use the time zone in string.
 * If allow_time_zone is false and string contains time zone, then the string is invalid.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp_without_time_zone(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& special_datetime_lit,
  bool allow_time_zone,
  bool ansi_mode)
{
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(cudf::type_id::TIMESTAMP_MICROSECONDS), true);
  }
  return to_timestamp(input, special_datetime_lit, ansi_mode, allow_time_zone);
}

}  // namespace spark_rapids_jni
