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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <thrust/binary_search.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace {

template <typename timestamp_type>
struct time_add {
  using duration_type = typename timestamp_type::duration;

  lists_column_device_view const transitions;

  size_type const tz_index;

  __device__ timestamp_type operator()(timestamp_type const& timestamp, int64_t duration) const
  {
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
    auto const idx         = static_cast<size_type>(thrust::distance(transition_times.begin(), it));
    auto const list_offset = tz_transitions.element_offset(idx - 1);
    auto const utc_offset  = cuda::std::chrono::duration_cast<duration_type>(
      cudf::duration_s{static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset))});

    return timestamp + utc_offset;
  }
};

}  // namespace

namespace spark_rapids_jni {

using namespace cudf;

namespace detail {

}  // namespace detail

}  // namespace spark_rapids_jni