/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "datetime_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace detail {

namespace {
/**
 * @brief Mark the date/time component to truncate.
 */
enum class truncate_component : uint8_t {
  YEAR,
  QUARTER,
  MONTH,
  WEEK,
  DAY,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND,
  INVALID
};

__device__ char to_upper(unsigned char const c) { return ('a' <= c && c <= 'z') ? c ^ 0x20 : c; }

// Parse the component to truncate from a string.
__device__ truncate_component parse_component(cudf::string_view const format)
{
  // This must be kept in sync with the `truncate_component` enum.
  char const* components[] = {"YEAR",
                              "QUARTER",
                              "MONTH",
                              "WEEK",
                              "DAY",
                              "HOUR",
                              "MINUTE",
                              "SECOND",
                              "MILLISECOND",
                              "MICROSECOND"};
  // Manually calculate sizes of the strings since `strlen` is not available in device code.
  cudf::size_type const comp_sizes[] = {4, 7, 5, 4, 3, 4, 6, 6, 11, 11};

  auto constexpr num_components = std::size(components);

  // auto const num_components = sizeof(components) / sizeof(components[0]);
  auto const* in_ptr = reinterpret_cast<unsigned char const*>(format.data());
  auto const in_len  = format.size_bytes();
  for (std::size_t comp_idx = 0; comp_idx < num_components; ++comp_idx) {
    if (in_len != comp_sizes[comp_idx]) { continue; }
    auto const* ptr = reinterpret_cast<unsigned char const*>(components[comp_idx]);
    bool equal      = true;
    for (cudf::size_type idx = 0; idx < in_len; ++idx) {
      if (to_upper(in_ptr[idx]) != ptr[idx]) {
        equal = false;
        break;
      }
    }
    if (equal) { return static_cast<truncate_component>(comp_idx); }
  }
  return truncate_component::INVALID;
}

__device__ inline uint32_t trunc_quarter_month(uint32_t month)
{
  auto const zero_based_month = month - 1u;
  return (zero_based_month / 3u) * 3u + 1u;
}

__device__ inline uint32_t trunc_to_monday(uint32_t days_since_epoch)
{
  // Since 1970/01/01 is Thursday, we have Thursday = 0, Friday = 1 and so on.
  auto constexpr MONDAY       = 4u;
  auto const day_of_week      = (days_since_epoch + MONDAY) % 7u;
  auto const days_to_subtract = day_of_week == 0 ? 6u : day_of_week - 1u;
  return days_since_epoch - days_to_subtract;
}

template <typename Timestamp>
__device__ inline thrust::optional<Timestamp> trunc_date(
  cuda::std::chrono::sys_days const days_since_epoch,
  cuda::std::chrono::year_month_day const ymd,
  truncate_component const trunc_comp)
{
  using namespace cuda::std::chrono;
  switch (trunc_comp) {
    case truncate_component::YEAR:
      return Timestamp{sys_days{year_month_day{ymd.year(), month{1}, day{1}}}};
    case truncate_component::QUARTER:
      return Timestamp{sys_days{year_month_day{
        ymd.year(), month{trunc_quarter_month(static_cast<uint32_t>(ymd.month()))}, day{1}}}};
    case truncate_component::MONTH:
      return Timestamp{sys_days{year_month_day{ymd.year(), ymd.month(), day{1}}}};
    case truncate_component::WEEK:
      return Timestamp{
        sys_days{days{trunc_to_monday(days_since_epoch.time_since_epoch().count())}}};
    default: return thrust::nullopt;
  }
}

struct truncate_date_fn {
  cudf::column_device_view datetime;
  cudf::column_device_view format;
  using Timestamp = cudf::timestamp_D;

  __device__ inline thrust::pair<Timestamp, bool> operator()(cudf::size_type const idx) const
  {
    auto const datetime_idx = datetime.size() > 1 ? idx : 0;
    auto const format_idx   = format.size() > 1 ? idx : 0;
    if (datetime.is_null(datetime_idx) || format.is_null(format_idx)) {
      return {Timestamp{}, false};
    }

    auto const fmt        = format.element<cudf::string_view>(format_idx);
    auto const trunc_comp = parse_component(fmt);
    if (trunc_comp == truncate_component::INVALID) { return {Timestamp{}, false}; }

    using namespace cuda::std::chrono;
    auto const ts               = datetime.element<Timestamp>(datetime_idx);
    auto const days_since_epoch = floor<days>(ts);
    auto const ymd              = year_month_day(days_since_epoch);

    auto const result = trunc_date<Timestamp>(days_since_epoch, ymd, trunc_comp);
    return {result.value_or(Timestamp{}), result.has_value()};
  }
};

struct truncate_timestamp_fn {
  cudf::column_device_view datetime;
  cudf::column_device_view format;
  using Timestamp = cudf::timestamp_us;

  __device__ inline thrust::pair<Timestamp, bool> operator()(cudf::size_type const idx) const
  {
    auto const datetime_idx = datetime.size() > 1 ? idx : 0;
    auto const format_idx   = format.size() > 1 ? idx : 0;
    if (datetime.is_null(datetime_idx) || format.is_null(format_idx)) {
      return {Timestamp{}, false};
    }

    auto const fmt        = format.element<cudf::string_view>(format_idx);
    auto const trunc_comp = parse_component(fmt);
    if (trunc_comp == truncate_component::INVALID) { return {Timestamp{}, false}; }

    using namespace cuda::std::chrono;
    auto const ts = datetime.element<Timestamp>(datetime_idx);

    // No truncation needed for microsecond timestamps.
    if (trunc_comp == truncate_component::MICROSECOND) { return {ts, true}; }

    // The components that are common for both date and timestamp: year, quarter, month, week.
    auto const days_since_epoch = floor<days>(ts);
    auto const ymd              = year_month_day(days_since_epoch);
    if (auto const try_trunc_date = trunc_date<Timestamp>(days_since_epoch, ymd, trunc_comp);
        try_trunc_date.has_value()) {
      return {try_trunc_date.value(), true};
    }

    auto time_since_midnight = ts - days_since_epoch;
    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    auto const hrs_  = [&] { return duration_cast<hours>(time_since_midnight); };
    auto const mins_ = [&] { return duration_cast<minutes>(time_since_midnight) - hrs_(); };
    auto const secs_ = [&] {
      return duration_cast<seconds>(time_since_midnight) - hrs_() - mins_();
    };
    auto const millisecs_ = [&] {
      return duration_cast<milliseconds>(time_since_midnight) - hrs_() - mins_() - secs_();
    };

    switch (trunc_comp) {
      case truncate_component::DAY: return {Timestamp{sys_days{ymd}}, true};
      case truncate_component::HOUR: return {Timestamp{sys_days{ymd} + hrs_()}, true};
      case truncate_component::MINUTE: return {Timestamp{sys_days{ymd} + mins_()}, true};
      case truncate_component::SECOND: return {Timestamp{sys_days{ymd} + secs_()}, true};
      case truncate_component::MILLISECOND: return {Timestamp{sys_days{ymd} + millisecs_()}, true};
      default: CUDF_UNREACHABLE("Unhandled truncating component.");
    }
  }
};

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       cudf::column_view const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const type_id = datetime.type().id();
  CUDF_EXPECTS(
    type_id == cudf::type_id::TIMESTAMP_DAYS || type_id == cudf::type_id::TIMESTAMP_MICROSECONDS,
    "The input must be either day or microsecond timestamps.");

  auto const size = std::max(datetime.size(), format.size());
  if (datetime.size() == 0 || format.size() == 0 || datetime.size() == datetime.null_count() ||
      format.size() == format.null_count()) {
    return cudf::make_fixed_width_column(
      datetime.type(), size, cudf::mask_state::ALL_NULL, stream, mr);
  }

  if (datetime.size() != format.size() && datetime.size() > 1 && format.size() > 1) {
    CUDF_FAIL(
      "If input columns have different number of rows,"
      " one of them must have exactly one row or empty.");
  }

  auto const d_datetime_ptr = cudf::column_device_view::create(datetime, stream);
  auto const d_format_ptr   = cudf::column_device_view::create(format, stream);
  auto const input_it       = thrust::make_counting_iterator(0);
  auto output =
    cudf::make_fixed_width_column(datetime.type(), size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(size, stream);

  if (type_id == cudf::type_id::TIMESTAMP_DAYS) {
    using Timestamp = cudf::timestamp_D;
    auto const output_it =
      thrust::make_zip_iterator(output->mutable_view().begin<Timestamp>(), validity.begin());
    thrust::transform(rmm::exec_policy_nosync(stream),
                      input_it,
                      input_it + size,
                      output_it,
                      truncate_date_fn{*d_datetime_ptr, *d_format_ptr});
  } else {
    using Timestamp = cudf::timestamp_us;
    auto const output_it =
      thrust::make_zip_iterator(output->mutable_view().begin<Timestamp>(), validity.begin());
    thrust::transform(rmm::exec_policy_nosync(stream),
                      input_it,
                      input_it + size,
                      output_it,
                      truncate_timestamp_fn{*d_datetime_ptr, *d_format_ptr});
  }

  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
  output->set_null_mask(null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr},
                        null_count);
  return output;
}

}  // namespace

}  // namespace detail

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       cudf::column_view const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::truncate(datetime, format, stream, mr);
}

}  // namespace spark_rapids_jni
