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

#include "cookie_serializer.hpp"

#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/detail/codec.hpp>

#include <algorithm>
#include <future>
#include <numeric>
#include <utility>

namespace spark_rapids_jni {

namespace {

std::vector<std::vector<uint8_t>> compress(
  cudf::io::compression_type compression,
  cudf::host_span<cudf::host_span<uint8_t const> const> inputs)
{
  auto const num_chunks = inputs.size();
  std::vector<std::vector<uint8_t>> output(num_chunks);

  // Generate order vector to submit largest tasks first.
  std::vector<size_t> task_order(num_chunks);
  std::iota(task_order.begin(), task_order.end(), 0);
  std::ranges::sort(task_order,
                    std::ranges::greater{},  // Sort descending
                    [&](size_t a) { return inputs[a].size(); });

  std::vector<std::future<std::pair<size_t, std::vector<uint8_t>>>> tasks;
  for (auto const idx : task_order) {
    auto task = [inp = inputs[idx], compression, idx] {
      return std::pair{idx, cudf::io::detail::compress(compression, inp)};
    };
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(std::move(task)));
  }

  for (auto& task : tasks) {
    auto [idx, out_bytes] = task.get();
    output[idx]           = std::move(out_bytes);
  }
  return output;
}

std::vector<std::vector<uint8_t>> decompress(
  cudf::io::compression_type compression,
  cudf::host_span<cudf::host_span<uint8_t const> const> inputs)
{
  auto const num_chunks = inputs.size();
  std::vector<std::vector<uint8_t>> output(num_chunks);

  // Generate order vector to submit largest tasks first.
  std::vector<size_t> task_order(num_chunks);
  std::iota(task_order.begin(), task_order.end(), 0);
  std::ranges::sort(task_order,
                    std::ranges::greater{},  // Sort descending
                    [&](size_t a) { return inputs[a].size(); });

  std::vector<std::future<std::pair<size_t, std::vector<uint8_t>>>> tasks;
  for (auto const idx : task_order) {
    auto task = [inp = inputs[idx], compression, idx] {
      return std::pair{idx, cudf::io::detail::decompress(compression, inp)};
    };
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(std::move(task)));
  }

  for (auto& task : tasks) {
    auto [idx, out_bytes] = task.get();
    output[idx]           = std::move(out_bytes);
  }
  return output;
}

}  // namespace

}  // namespace spark_rapids_jni