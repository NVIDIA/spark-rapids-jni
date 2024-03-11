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

#include "bloom_filter.hpp"
#include "utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

class BloomFilterTest : public cudf::test::BaseFixture {};

struct is_zero {
  __device__ bool operator()(cudf::bitmask_type w) { return w == 0; }
};

TEST_F(BloomFilterTest, Initialization)
{
  constexpr int num_hashes = 3;
  std::vector<int> expected{1, 2, 3};

  for (size_t idx = 0; idx < expected.size(); idx++) {
    auto bloom_filter =
      spark_rapids_jni::bloom_filter_create(num_hashes, expected[idx], cudf::get_default_stream());

    auto const bloom_filter_size = expected[idx] * sizeof(int64_t);
    CUDF_EXPECTS(
      bloom_filter->view().size() == spark_rapids_jni::bloom_filter_header_size + bloom_filter_size,
      "Bloom filter not of expected size");

    auto bytes = (bloom_filter->view().data<int8_t>()) + spark_rapids_jni::bloom_filter_header_size;
    CUDF_EXPECTS(
      thrust::all_of(
        rmm::exec_policy(cudf::get_default_stream()), bytes, bytes + bloom_filter_size, is_zero{}),
      "Bloom filter not initialized to 0");
  }
}

TEST_F(BloomFilterTest, BuildAndProbe)
{
  auto stream                      = cudf::get_default_stream();
  constexpr int bloom_filter_longs = (1024 * 1024);
  constexpr int num_hashes         = 3;

  auto bloom_filter = spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);

  cudf::test::fixed_width_column_wrapper<int64_t> input{20, 80, 100, 99, 47, -9, 234000000};
  spark_rapids_jni::bloom_filter_put(*bloom_filter, input, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<bool> expected{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
  auto result = spark_rapids_jni::bloom_filter_probe(probe, *bloom_filter, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(BloomFilterTest, BuildWithNullsAndProbe)
{
  auto stream                      = cudf::get_default_stream();
  constexpr int bloom_filter_longs = (1024 * 1024);
  constexpr int num_hashes         = 3;

  auto bloom_filter = spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);
  cudf::test::fixed_width_column_wrapper<int64_t> input{{20, 80, 100, 99, 47, -9, 234000000},
                                                        {0, 1, 1, 1, 0, 1, 1}};

  spark_rapids_jni::bloom_filter_put(*bloom_filter, input, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<bool> expected{0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0};
  auto result = spark_rapids_jni::bloom_filter_probe(probe, *bloom_filter, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(BloomFilterTest, BuildAndProbeWithNulls)
{
  auto stream                      = cudf::get_default_stream();
  constexpr int bloom_filter_longs = (1024 * 1024);
  constexpr int num_hashes         = 3;

  cudf::test::fixed_width_column_wrapper<int64_t> input{20, 80, 100, 99, 47, -9, 234000000};
  auto bloom_filter = spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);

  spark_rapids_jni::bloom_filter_put(*bloom_filter, input, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    {20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3}, {0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<bool> expected{{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                                                        {0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto result = spark_rapids_jni::bloom_filter_probe(probe, *bloom_filter, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

struct bloom_filter_stride_transform {
  int const stride;

  cudf::size_type __device__ operator()(cudf::size_type i) { return i * stride; }
};

TEST_F(BloomFilterTest, ProbeMerged)
{
  auto stream                      = cudf::get_default_stream();
  constexpr int bloom_filter_longs = (1024 * 1024);
  constexpr int num_hashes         = 3;

  // column a
  cudf::test::fixed_width_column_wrapper<int64_t> col_a{20, 80, 100, 99, 47, -9, 234000000};
  auto bloom_filter_a =
    spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);
  spark_rapids_jni::bloom_filter_put(*bloom_filter_a, col_a, stream);

  // column b
  cudf::test::fixed_width_column_wrapper<int64_t> col_b{100, 200, 300, 400};
  auto bloom_filter_b =
    spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);
  spark_rapids_jni::bloom_filter_put(*bloom_filter_b, col_b, stream);

  // column c
  cudf::test::fixed_width_column_wrapper<int64_t> col_c{-100, -200, -300, -400};
  auto bloom_filter_c =
    spark_rapids_jni::bloom_filter_create(num_hashes, bloom_filter_longs, stream);
  spark_rapids_jni::bloom_filter_put(*bloom_filter_c, col_c, stream);

  // pre-merge the individual bloom filters. the merge function expects the inputs to be a single
  // list column, with each row representing a bloom filter.
  std::vector<cudf::column_view> cols(
    {bloom_filter_a->view(), bloom_filter_b->view(), bloom_filter_c->view()});
  auto premerge_children = cudf::concatenate(cols);
  auto premerge_offsets  = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, 4);
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + 4,
                    premerge_offsets->mutable_view().begin<cudf::size_type>(),
                    bloom_filter_stride_transform{bloom_filter_a->view().size()});
  auto premerged = cudf::make_lists_column(
    3, std::move(premerge_offsets), std::move(premerge_children), 0, rmm::device_buffer{});

  // merged bloom filter
  auto bloom_filter_merged = spark_rapids_jni::bloom_filter_merge(*premerged);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    -9, 200, 300, 6000, -2546, 99, 65535, 0, -100, -200, -300, -400};
  cudf::test::fixed_width_column_wrapper<bool> expected{1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  auto result = spark_rapids_jni::bloom_filter_probe(probe, *bloom_filter_merged, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
