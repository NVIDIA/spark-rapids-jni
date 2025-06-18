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

#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/detail/codec.hpp>

#include <algorithm>
#include <future>
#include <numeric>
#include <utility>

namespace spark_rapids_jni {

namespace {

constexpr char MAGIC[]           = "COOKIE";
constexpr uint32_t VERSION_MAJOR = 1;
constexpr uint32_t VERSION_MINOR = 0;

template <typename CodecFunc>
std::vector<std::future<std::pair<size_t, std::vector<uint8_t>>>> codec(
  cudf::host_span<cudf::host_span<uint8_t const> const> inputs, CodecFunc&& codec_fn)
{
  // Generate order vector to submit largest tasks first.
  std::vector<size_t> task_order(inputs.size());
  std::iota(task_order.begin(), task_order.end(), 0);
  std::ranges::sort(task_order,
                    std::ranges::greater{},  // Sort descending
                    [&](size_t a) { return inputs[a].size(); });

  std::vector<std::future<std::pair<size_t, std::vector<uint8_t>>>> tasks;
  for (auto const idx : task_order) {
    auto task = [inp = inputs[idx], idx, codec_fn] { return std::pair{idx, codec_fn(inp)}; };
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(std::move(task)));
  }
  return tasks;
}

}  // namespace

std::vector<uint8_t> serialize_cookie(cudf::host_span<cudf::host_span<uint8_t const> const> inputs,
                                      cudf::io::compression_type compression)
{
  // Compress the input buffers first to detect any issues with the compression early on.
  auto codec_tasks = codec(inputs, [compression](cudf::host_span<uint8_t const> input) {
    return cudf::io::detail::compress(compression, input);
  });

  // Must be 32-bit unsigned integer for serialization.
  auto const num_chunks = static_cast<uint32_t>(inputs.size());

  std::vector<std::vector<uint8_t>> compressed_inputs(num_chunks);
  for (auto& task : codec_tasks) {
    auto [idx, compressed] = task.get();
    compressed_inputs[idx] = std::move(compressed);
  }

  std::size_t out_data_size = 0;
  std::vector<uint64_t> offsets(num_chunks + 1, 0);
  std::vector<cudf::io::compression_type> buff_compressions(num_chunks);

  for (std::size_t idx = 0; idx < num_chunks; ++idx) {
    // For each input buffer, add two extra integers for checksum and compression type.
    out_data_size += 2 * sizeof(uint32_t);

    auto const output_compressed = compressed_inputs[idx].size() < inputs[idx].size();
    buff_compressions[idx] = output_compressed ? compression : cudf::io::compression_type::NONE;
    out_data_size += output_compressed ? compressed_inputs[idx].size() : inputs[idx].size();
    offsets[idx + 1] = out_data_size;
  }

  // Header is the extra data that is stored before the compressed input data.
  std::size_t header_size = 0;
  header_size += 6ul;                                  // magic "COOKIE"
  header_size += 2ul * sizeof(uint32_t);               // version major and minor
  header_size += sizeof(uint32_t);                     // number of input buffers
  header_size += sizeof(uint64_t) * (num_chunks + 1);  // offsets

  std::vector<uint8_t> output;
  output.reserve(header_size + out_data_size);

  // Write the output.
  auto ptr = output.data();

  auto const magic_size = strlen(MAGIC);
  CUDF_EXPECTS(magic_size == 6, "Internal error: Invalid Cookie MAGIC.");
  std::memcpy(ptr, MAGIC, magic_size);
  ptr += magic_size;

  std::memcpy(ptr, &VERSION_MAJOR, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, &VERSION_MINOR, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  std::memcpy(ptr, &num_chunks, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  std::memcpy(ptr, offsets.data(), offsets.size() * sizeof(uint64_t));
  ptr += offsets.size() * sizeof(uint64_t);

  for (std::size_t idx = 0; idx < num_chunks; ++idx) {
    uint32_t const check_sum = 0;  // dummy checksum, currently not implemented.
    std::memcpy(ptr, &check_sum, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    std::memcpy(ptr, &buff_compressions[idx], sizeof(int32_t));
    ptr += sizeof(int32_t);

    cudf::host_span<uint8_t const> out_buff =
      buff_compressions[idx] == cudf::io::compression_type::NONE ? inputs[idx]
                                                                 : compressed_inputs[idx];
    std::memcpy(ptr, out_buff.data(), out_buff.size());
    ptr += out_buff.size();
  }

  return output;
}

}  // namespace spark_rapids_jni