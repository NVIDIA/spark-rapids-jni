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

#include <bloom_filter.hpp>
#include <hash.cuh>

#include <benchmarks/common/generate_input.hpp>

#include <nvbench/nvbench.cuh>

#include <cudf_test/column_utilities.hpp>

static void bloom_filter_put(nvbench::state& state)
{
  constexpr int num_rows   = 150'000'000;
  constexpr int num_hashes = 3;

  // create the bloom filter
  cudf::size_type const bloom_filter_bytes = state.get_int64("bloom_filter_bytes");
  cudf::size_type const bloom_filter_bits  = bloom_filter_bytes * CHAR_BIT;
  auto bloom_filter = spark_rapids_jni::bloom_filter_create(bloom_filter_bits);

  // create a column of hashed values
  data_profile_builder builder;
  builder.no_validity();
  auto const src   = create_random_table({{cudf::type_id::INT64}}, row_count{num_rows}, builder);
  auto const input = spark_rapids_jni::xxhash64(*src);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      spark_rapids_jni::bloom_filter_put(
        {reinterpret_cast<cudf::bitmask_type*>(bloom_filter->data()), bloom_filter->size()},
        bloom_filter_bits,
        *input,
        num_hashes);
      stream.synchronize();
      timer.stop();

      // clear the bloom filter
      cudaMemset(bloom_filter->data(), 0, bloom_filter->size());
    });

  size_t const bytes_read    = num_rows * sizeof(int64_t);
  size_t const bytes_written = num_rows * sizeof(cudf::bitmask_type) * num_hashes;
  auto const time            = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(std::size_t{num_rows}, "Rows Inserted");
  state.add_global_memory_reads(bytes_read, "Bytes read");
  state.add_global_memory_writes(bytes_written, "Bytes written");
  state.add_element_count(static_cast<double>(bytes_written) / time, "Write bytes/sec");
}

NVBENCH_BENCH(bloom_filter_put)
  .set_name("Bloom Filter Put")
  .add_int64_axis("bloom_filter_bytes",
                  {512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024});
