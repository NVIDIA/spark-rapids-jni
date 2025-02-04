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

#include "shuffle_split.hpp"

#include <cuda/functional>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cub/device/device_memcpy.cuh>

struct ShuffleSplitTests : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::table> reshape_table(cudf::table_view const& tbl, std::vector<int> const& splits, std::vector<int> const& remaps)
{
  auto split_result = cudf::split(tbl, splits);
  std::vector<cudf::table_view> remapped;
  remapped.reserve(split_result.size());
  std::transform(remaps.begin(), remaps.end(), std::back_inserter(remapped), [&](int i){
    return split_result[i];
  });
  return cudf::concatenate(remapped);
}

spark_rapids_jni::shuffle_split_result reshape_partitions(cudf::device_span<uint8_t const> partitions,
                                                          cudf::device_span<size_t const> partition_offsets,
                                                          std::vector<int> const& remaps,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(remaps.size() == partition_offsets.size() - 1, "Invaid remaps vector size");
  auto temp_mr = cudf::get_current_device_resource_ref();

  // generate new partition offsets
  auto d_remaps = cudf::detail::make_device_uvector_async(remaps, stream, temp_mr);
  rmm::device_uvector<size_t> remapped_offsets(partition_offsets.size(), stream, mr);
  auto const num_partitions = partition_offsets.size() - 1;
  auto remapped_size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([partition_offsets = partition_offsets.begin(), remaps = d_remaps.begin(), num_partitions] __device__(size_t i){
    auto const ri = remaps[i];
    return i >= num_partitions ? 0 : partition_offsets[ri+1] - partition_offsets[ri];
  }));
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         remapped_size_iter,
                         remapped_size_iter + num_partitions + 1,
                         remapped_offsets.begin());

  // swizzle the data
  rmm::device_buffer remapped_partitions(partitions.size(), stream, mr);
  auto input_iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<void*>([partitions = partitions.data(), partition_offsets = partition_offsets.begin(), remaps = d_remaps.begin()] __device__(size_t i) {
      return reinterpret_cast<void*>(const_cast<uint8_t*>(partitions) + partition_offsets[remaps[i]]);
    }));
  auto size_iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([partition_offsets = partition_offsets.begin(), remaps = d_remaps.begin()] __device__(size_t i) {
      auto const ri = remaps[i];
      return partition_offsets[ri + 1] - partition_offsets[ri];
    }));
  auto output_iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<void*>([remapped_partitions = remapped_partitions.data(), remapped_offsets = remapped_offsets.begin()] __device__(size_t i) {
      return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(remapped_partitions) + remapped_offsets[i]);
    }));  

  size_t temp_storage_bytes;
  cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, num_partitions, stream);
  rmm::device_buffer temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceMemcpy::Batched(
    temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, num_partitions, stream);

  return {std::make_unique<rmm::device_buffer>(std::move(remapped_partitions)), std::move(remapped_offsets)};
}

void run_split(cudf::table_view const& tbl, std::vector<cudf::size_type> const& splits, std::vector<cudf::size_type> const& remaps = {})
{
  auto [split_data, split_metadata] = spark_rapids_jni::shuffle_split(tbl,
                                                                      splits,
                                                                      cudf::get_default_stream(),
                                                                      rmm::mr::get_current_device_resource());

  // maybe reshape the results
  if(remaps.size() > 0){    
    CUDF_EXPECTS(remaps.size() == splits.size() + 1, "Invalid remap vector size");
    CUDF_EXPECTS(remaps.size() == split_data.offsets.size() - 1, "Invaid remaps vector size");
    auto reshaped_table = reshape_table(tbl, splits, remaps);
    auto reshaped_data = reshape_partitions({static_cast<uint8_t*>(split_data.partitions->data()), split_data.partitions->size()}, 
                                            split_data.offsets, 
                                            remaps, 
                                            cudf::get_default_stream(), 
                                            rmm::mr::get_current_device_resource());

    auto result = spark_rapids_jni::shuffle_assemble(split_metadata,
                                                     {static_cast<uint8_t*>(reshaped_data.partitions->data()), reshaped_data.partitions->size()},
                                                     reshaped_data.offsets,
                                                     cudf::get_default_stream(),
                                                     rmm::mr::get_current_device_resource());
    
    CUDF_TEST_EXPECT_TABLES_EQUAL(*reshaped_table, *result);
  } else {  
    auto result = spark_rapids_jni::shuffle_assemble(split_metadata,
                                                     {static_cast<uint8_t*>(split_data.partitions->data()), split_data.partitions->size()},
                                                     split_data.offsets,
                                                     cudf::get_default_stream(),
                                                     rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_TABLES_EQUAL(tbl, *result);
  }  
}

TEST_F(ShuffleSplitTests, Simple)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  // 4 columns split once
  {
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};
    run_split(tbl, {10});
  }

  // 4 columns split multiple times
  {
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};
    run_split(tbl, {10, 100, 2756, 7777});
  }
}

TEST_F(ShuffleSplitTests, Strings)
{
  {
    cudf::test::strings_column_wrapper col{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};
    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }

  {
    cudf::test::strings_column_wrapper col0{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};
    cudf::test::strings_column_wrapper col1{{"blue",
                                             "green",
                                             "yellow",
                                             "red",
                                             "black",
                                             "white",
                                             "gray",
                                             "aquamarine",
                                             "mauve",
                                             "ultraviolet"}};
    cudf::test::strings_column_wrapper col2{
      {"left", "up", "right", "down", "", "space", "", "delete", "end", "insert"}};
    cudf::test::strings_column_wrapper col3{
      {"a", "b", "c", "de", "fg", "h", "i", "jk", "lmn", "opq"}};
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }
}

TEST_F(ShuffleSplitTests, Lists)
{
  // list<uint64_t>
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }

  // list<list<uint64_t>
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{{9, 8}, {7, 6, 5}},
             {lcw{}, {4}},
             {{3, 2, 1, 0}, {20, 21, 22, 23, 24}},
             {lcw{}, {66, 666}},
             {{123, 7}, {100, 101}},
             {{1, 2, 4}, {8, 6, 5}}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};

    run_split(tbl, {});
    run_split(tbl, {4});
    run_split(tbl, {1, 4});
  }
}

TEST_F(ShuffleSplitTests, Struct)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);

  cudf::test::fixed_width_column_wrapper<float> child0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> child1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> child2(iter, iter + num_rows);
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(child0.release());
  struct_children.push_back(child1.release());
  struct_children.push_back(child2.release());
  cudf::test::structs_column_wrapper col1(std::move(struct_children));

  // 4 columns split once
  {
    cudf::table_view tbl{
      {static_cast<cudf::column_view>(col0), static_cast<cudf::column_view>(col1)}};
    run_split(tbl, {10});
  }

  // 4 columns split multiple times
  {
    cudf::table_view tbl{
      {static_cast<cudf::column_view>(col0), static_cast<cudf::column_view>(col1)}};
    run_split(tbl, {10, 100, 2756, 7777});
  }
}

TEST_F(ShuffleSplitTests, Nulls)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 3; });

  cudf::test::fixed_width_column_wrapper<int> col(iter, iter + num_rows, validity);
  cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};
  run_split(tbl, {3});
}

TEST_F(ShuffleSplitTests, ShortNulls)
{
  // one full 32 bit validity word
  {
    cudf::test::fixed_width_column_wrapper<int> col(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
      {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1});

    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    // try each bit seperately
    for (int idx = 0; idx < static_cast<cudf::column_view>(col).size(); idx++) {
      run_split(tbl, {idx});
    }
    // all bits in the first byte
    run_split(tbl, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    // all bits in the entire word
    run_split(tbl, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

    // misc
    run_split(tbl, {});
    run_split(tbl, {2, 5, 18, 30});
    run_split(tbl, {8, 16, 24});
    run_split(tbl, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});
  }

  // less than one word
  {
    cudf::test::fixed_width_column_wrapper<int> col({0, 1, 2, 3, 4, 5, 6}, {1, 1, 1, 0, 0, 0, 0});

    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    for (int idx = 0; idx < static_cast<cudf::column_view>(col).size(); idx++) {
      run_split(tbl, {idx});
    }

    run_split(tbl, {});
    run_split(tbl, {0, 1, 2, 3, 4, 5});
    run_split(tbl, {2, 5});
    run_split(tbl, {0, 2, 4});
  }
}

TEST_F(ShuffleSplitTests, EmptySplits)
{
  cudf::size_type const num_rows = 100;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                        static_cast<cudf::column_view>(col1),
                        static_cast<cudf::column_view>(col2),
                        static_cast<cudf::column_view>(col3)}};
  run_split(tbl, {});
}

TEST_F(ShuffleSplitTests, EmptyInputs)
{
  cudf::size_type const num_rows = 0;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                        static_cast<cudf::column_view>(col1),
                        static_cast<cudf::column_view>(col2),
                        static_cast<cudf::column_view>(col3)}};
  run_split(tbl, {});
}

TEST_F(ShuffleSplitTests, NestedTypes)
{
  // struct<list, list, string>
  {
    using lcw = cudf::test::lists_column_wrapper<int64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101}};
    lcw col1{{1, 2, 3},
             {7},
             {99, 100},
             {4, 5, 6},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101, -1, -2, -3, -4 - 5, -6}};

    cudf::test::strings_column_wrapper col2{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};

    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(col0.release());
    struct_children.push_back(col1.release());
    struct_children.push_back(col2.release());
    cudf::test::structs_column_wrapper struct_col(std::move(struct_children));

    {
      cudf::table_view tbl{{static_cast<cudf::column_view>(struct_col)}};
      run_split(tbl, {});
      run_split(tbl, {2});
      run_split(tbl, {2, 4});
    }
  }

  // list<struct<list, list>>
  // this test specifically triggers an important case in the code: branching row counts caused
  // by structs
  {
    // struct<list, list>
    using lcw = cudf::test::lists_column_wrapper<int64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101},
             {1, 1, 1},
             {2},
             {0},
             {2256, 12, 224, 5},
             {9, 9, 9, 9, 9},
             {-1, -2}};
    lcw col1{{1, 2, 3},
             {7},
             {99, 100},
             {4, 5, 6},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {1},
             {66, 666},
             {123, 7},
             {100, 101, -1, -2, -3, -4 - 5, -6},
             {},
             {6, 5, 4, 3, 2, 1, 0},
             {-10, 0, 1},
             {0, 0, 0, 0},
             {},
             {0, 1, 0, 1, 0, 1}};
    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(col0.release());
    struct_children.push_back(col1.release());
    cudf::test::structs_column_wrapper struct_col(std::move(struct_children));

    // list<struct<list, list>>
    cudf::test::fixed_width_column_wrapper<int> offsets{0, 2, 4, 6, 7, 9, 9, 12, 16};
    auto list_col = cudf::make_lists_column(8, offsets.release(), struct_col.release(), 0, {});

    cudf::table_view tbl{{*list_col}};
    run_split(tbl, {});
    run_split(tbl, {2});
    run_split(tbl, {2, 4});
  }
}

TEST_F(ShuffleSplitTests, Reshaping)
{
  // a key feature of the kud0 format is being able to reassemble arbitrary partitions. 
  // so for example, if we had a shuffle_split() call that produced partitions ABCD, we might 
  // want to reassemble that as CDAB. or we may have multiple shuffle_split calls producing multiple
  // sets of partitions that need to be stitched together, such as:
  // - shuffle_split()    -> ABCD
  // - shuffle_split()    -> XYZW
  // - shuffle_assemble(AXBYCDWZ)
  {
    cudf::test::fixed_width_column_wrapper<int> col0{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
                                                     {1, 1, 1, 1, 0, 0, 1, 0, 1, 0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  0}};
    cudf::table_view tbl{{col0}};
    run_split(tbl, {2, 10}, {2, 0, 1});
  }
}