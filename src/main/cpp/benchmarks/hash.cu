/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_utilities.hpp>

#include <cudf/io/types.hpp>

#include <hash.hpp>
#include <nvbench/nvbench.cuh>

constexpr auto min_width = 10;
constexpr auto max_width = 10;

static void hash(nvbench::state& state)
{
  auto const size_bytes = static_cast<std::size_t>(state.get_int64("size_bytes"));
  auto const max_depth  = static_cast<cudf::size_type>(state.get_int64("max_depth"));

  auto const bench_structs = true;
  auto const input_table   = [&] {
    if (bench_structs) {
      data_profile const table_profile =
        data_profile_builder().no_validity().struct_depth(max_depth).struct_types(
          std::vector<cudf::type_id>{
            cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::STRING});
      return create_random_table(
        {cudf::type_id::STRUCT}, table_size_bytes{size_bytes}, table_profile);
    } else {
      data_profile const table_profile =
        data_profile_builder()
          .no_validity()
          .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
          .distribution(cudf::type_id::LIST, distribution_id::NORMAL, min_width, max_width)
          .list_depth(max_depth)
          .list_type(cudf::type_id::STRING);
      return create_random_table(
        {cudf::type_id::LIST}, table_size_bytes{size_bytes}, table_profile);
    }
  }();

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               // `hive_hash` can be substituted with other hash functions
               spark_rapids_jni::hive_hash(*input_table);
               timer.stop();
             });

  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

NVBENCH_BENCH(hash)
  .set_name("hash")
  .add_int64_axis("size_bytes",
                  {50'000'000, 100'000'000, 500'000'000, 1'000'000'000})  // 50MB, 100MB, 500MB, 1GB
  .add_int64_axis("max_depth", {1, 2, 4, 8});
