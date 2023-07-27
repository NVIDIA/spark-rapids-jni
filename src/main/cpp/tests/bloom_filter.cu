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
  std::vector<std::pair<int, int>> expected({{1, 1}, {32, 1}, {33, 2}, {64, 2}});

  for (size_t idx = 0; idx < expected.size(); idx++) {
    auto bloom_filter =
      spark_rapids_jni::bloom_filter_create(expected[idx].first, cudf::get_default_stream());
    CUDF_EXPECTS(bloom_filter->size() == expected[idx].second * sizeof(cudf::bitmask_type),
                 "Bloom filter not of expected size");
    auto bytes = static_cast<uint8_t const*>(bloom_filter->data());
    CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(cudf::get_default_stream()),
                                bytes,
                                bytes + bloom_filter->size(),
                                is_zero{}),
                 "Bloom filter not initialized to 0");
  }
}

TEST_F(BloomFilterTest, BuildAndProbe)
{
  auto stream                     = cudf::get_default_stream();
  constexpr int bloom_filter_bits = (1024 * 1024) * 8;
  constexpr int num_hashes        = 3;

  cudf::test::fixed_width_column_wrapper<int64_t> input{20, 80, 100, 99, 47, -9, 234000000};
  auto _bloom_filter = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  cudf::device_span<cudf::bitmask_type> bloom_filter{
    static_cast<cudf::bitmask_type*>(_bloom_filter->data()),
    _bloom_filter->size() / sizeof(cudf::bitmask_type)};

  spark_rapids_jni::bloom_filter_put(bloom_filter, bloom_filter_bits, input, 3, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<bool> expected{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
  auto result = spark_rapids_jni::bloom_filter_probe(
    probe, bloom_filter, bloom_filter_bits, num_hashes, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(BloomFilterTest, BuildWithNullsAndProbe)
{
  auto stream                     = cudf::get_default_stream();
  constexpr int bloom_filter_bits = (1024 * 1024) * 8;
  constexpr int num_hashes        = 3;

  auto _bloom_filter = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  cudf::test::fixed_width_column_wrapper<int64_t> input{{20, 80, 100, 99, 47, -9, 234000000},
                                                        {0, 1, 1, 1, 0, 1, 1}};
  cudf::device_span<cudf::bitmask_type> bloom_filter{
    static_cast<cudf::bitmask_type*>(_bloom_filter->data()),
    _bloom_filter->size() / sizeof(cudf::bitmask_type)};

  spark_rapids_jni::bloom_filter_put(bloom_filter, bloom_filter_bits, input, 3, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<bool> expected{0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0};
  auto result = spark_rapids_jni::bloom_filter_probe(
    probe, bloom_filter, bloom_filter_bits, num_hashes, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(BloomFilterTest, BuildAndProbeWithNulls)
{
  auto stream                     = cudf::get_default_stream();
  constexpr int bloom_filter_bits = (1024 * 1024) * 8;
  constexpr int num_hashes        = 3;

  cudf::test::fixed_width_column_wrapper<int64_t> input{20, 80, 100, 99, 47, -9, 234000000};
  auto _bloom_filter = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  cudf::device_span<cudf::bitmask_type> bloom_filter{
    static_cast<cudf::bitmask_type*>(_bloom_filter->data()),
    _bloom_filter->size() / sizeof(cudf::bitmask_type)};

  spark_rapids_jni::bloom_filter_put(bloom_filter, bloom_filter_bits, input, 3, stream);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    {20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3}, {0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<bool> expected{{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                                                        {0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto result = spark_rapids_jni::bloom_filter_probe(
    probe, bloom_filter, bloom_filter_bits, num_hashes, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(BloomFilterTest, ProbeMerged)
{
  auto stream                     = cudf::get_default_stream();
  constexpr int bloom_filter_bits = (1024 * 1024) * 8;
  constexpr int num_hashes        = 3;

  // column a
  cudf::test::fixed_width_column_wrapper<int64_t> col_a{20, 80, 100, 99, 47, -9, 234000000};
  auto _bloom_filter_a = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  auto bloom_filter_a  = spark_rapids_jni::bloom_filter_to_span(*_bloom_filter_a);
  spark_rapids_jni::bloom_filter_put(bloom_filter_a, bloom_filter_bits, col_a, 3, stream);

  // column b
  cudf::test::fixed_width_column_wrapper<int64_t> col_b{100, 200, 300, 400};
  auto _bloom_filter_b = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  auto bloom_filter_b  = spark_rapids_jni::bloom_filter_to_span(*_bloom_filter_b);
  spark_rapids_jni::bloom_filter_put(bloom_filter_b, bloom_filter_bits, col_b, 3, stream);

  // column c
  cudf::test::fixed_width_column_wrapper<int64_t> col_c{-100, -200, -300, -400};
  auto _bloom_filter_c = spark_rapids_jni::bloom_filter_create(bloom_filter_bits, stream);
  auto bloom_filter_c  = spark_rapids_jni::bloom_filter_to_span(*_bloom_filter_c);
  spark_rapids_jni::bloom_filter_put(bloom_filter_c, bloom_filter_bits, col_c, 3, stream);

  // merged bloom filter
  auto _bloom_filter_merged =
    spark_rapids_jni::bitmask_bitwise_or({bloom_filter_a, bloom_filter_b, bloom_filter_c}, stream);
  auto bloom_filter_merged = spark_rapids_jni::bloom_filter_to_span(*_bloom_filter_merged);

  // probe
  cudf::test::fixed_width_column_wrapper<int64_t> probe{
    -9, 200, 300, 6000, -2546, 99, 65535, 0, -100, -200, -300, -400};
  cudf::test::fixed_width_column_wrapper<bool> expected{1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  auto result = spark_rapids_jni::bloom_filter_probe(
    probe, bloom_filter_merged, bloom_filter_bits, num_hashes, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
