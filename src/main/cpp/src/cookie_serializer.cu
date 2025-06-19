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
#include <string_view>
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

// The header is defined as containing the following fields:
// - Magic string "COOKIE" (6 bytes)
// - Version major (4 bytes)
// - Version minor (4 bytes)
// - Number of input buffers (4 bytes)
// - Offsets for the input buffers (num_chunks + 1) * 8 bytes
std::size_t compute_header_size(uint32_t num_chunks)
{
  std::size_t header_size = 0;
  header_size += 6ul;                                  // magic "COOKIE"
  header_size += 2ul * sizeof(uint32_t);               // version major and minor
  header_size += sizeof(uint32_t);                     // number of input buffers
  header_size += sizeof(uint64_t) * (num_chunks + 1);  // offsets
  return header_size;
}

template <typename T, typename U>
[[nodiscard]] std::size_t copy(U* dst, T const* src, std::size_t size)
{
  std::memcpy(reinterpret_cast<uint8_t*>(dst), reinterpret_cast<uint8_t const*>(src), size);
  return size;
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

  std::vector<uint8_t> output;
  output.resize(compute_header_size(num_chunks) + out_data_size);

  // Write the output.
  auto ptr = output.data();

  auto const magic_size = strlen(MAGIC);
  CUDF_EXPECTS(magic_size == 6ul, "Internal error: Invalid Cookie MAGIC.");
  ptr += copy(ptr, MAGIC, magic_size);

  ptr += copy(ptr, &VERSION_MAJOR, sizeof(uint32_t));
  ptr += copy(ptr, &VERSION_MINOR, sizeof(uint32_t));
  ptr += copy(ptr, &num_chunks, sizeof(uint32_t));
  ptr += copy(ptr, offsets.data(), offsets.size() * sizeof(uint64_t));

  for (std::size_t idx = 0; idx < num_chunks; ++idx) {
    uint32_t const check_sum = 0;  // dummy checksum, currently not implemented.
    ptr += copy(ptr, &check_sum, sizeof(uint32_t));
    ptr += copy(ptr, &buff_compressions[idx], sizeof(int32_t));

    cudf::host_span<uint8_t const> out_buff =
      buff_compressions[idx] == cudf::io::compression_type::NONE ? inputs[idx]
                                                                 : compressed_inputs[idx];
    ptr += copy(ptr, out_buff.data(), out_buff.size());
  }

  return output;
}

std::vector<std::vector<uint8_t>> deserialize_cookie(cudf::host_span<uint8_t const> input)
{
  CUDF_EXPECTS(input.size() > 6ul, "Input data for Cookie deserialization is too short.");

  auto ptr         = input.data();
  auto const magic = std::string_view{reinterpret_cast<char const*>(ptr), 6};
  CUDF_EXPECTS(magic == MAGIC, "Invalid input for Cookie deserialization: invalid MAGIC.");
  ptr += 6ul;  // Skip the magic string.

  uint32_t version = 0;
  ptr += copy(&version, ptr, sizeof(uint32_t));
  CUDF_EXPECTS(version == VERSION_MAJOR,
               "Invalid input for Cookie deserialization: incompatible data versions.");
  ptr += copy(&version, ptr, sizeof(uint32_t));  // minor version is unused for now

  uint32_t num_chunks = 0;
  ptr += copy(&num_chunks, ptr, sizeof(uint32_t));

  std::vector<uint64_t> offsets(num_chunks + 1, 0);
  ptr += copy(offsets.data(), ptr, (num_chunks + 1) * sizeof(uint64_t));

  std::vector<std::vector<uint8_t>> output(num_chunks);
  uint32_t check_sum                          = 0;
  cudf::io::compression_type compression_type = cudf::io::compression_type::NONE;

  for (std::size_t idx = 0; idx < num_chunks; ++idx) {
    ptr += copy(&check_sum, ptr, sizeof(uint32_t));  // unused for now
    ptr += copy(&compression_type, ptr, sizeof(int32_t));
  }

  return output;
}

}  // namespace spark_rapids_jni