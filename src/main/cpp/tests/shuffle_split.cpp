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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

struct ShuffleSplitTests : public cudf::test::BaseFixture {};

void run_split(cudf::table_view const& tbl, std::vector<cudf::size_type> const& splits)
{
  auto [split_data, split_metadata] = spark_rapids_jni::shuffle_split(
    tbl, splits, cudf::get_default_stream(), cudf::get_current_device_resource());
  auto result = spark_rapids_jni::shuffle_assemble(
    split_metadata,
    {static_cast<uint8_t*>(split_data.partitions->data()), split_data.partitions->size()},
    split_data.offsets,
    cudf::get_default_stream(),
    cudf::get_current_device_resource());
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl, *result);
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
