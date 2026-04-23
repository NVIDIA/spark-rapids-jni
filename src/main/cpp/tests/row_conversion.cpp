/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

#include <row_conversion.hpp>
#include <utilities/iterator.cuh>

#include <cstddef>
#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>
#include <random>

struct ColumnToRowTests : public cudf::test::BaseFixture {};
struct RowToColumnTests : public cudf::test::BaseFixture {};

TEST_F(ColumnToRowTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SimpleString)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1, 0, -1});
  cudf::test::strings_column_wrapper b(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
}

TEST_F(ColumnToRowTests, DoubleString)
{
  cudf::test::strings_column_wrapper a(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::test::fixed_width_column_wrapper<int32_t> b({0, 1, 2, 3, 4});
  cudf::test::strings_column_wrapper c({"world",
                                        "hello",
                                        "this string isn't as long",
                                        "this one isn't so short though when you think about it",
                                        "dlrow"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
}

TEST_F(ColumnToRowTests, BigStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 50;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);
}

TEST_F(ColumnToRowTests, ManyStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working",
    "some strings",
    "are split into multiple strings",
    "some strings have all their data",
    "lots of choices of strings and sizes is sure to test the offset calculation code to ensure "
    "that even a really long string ends up in the correct spot for the final destination allowing "
    "for even crazy run-on sentences to be inserted into the data"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 1'000'000;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);
}

TEST_F(ColumnToRowTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Tall)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({rand()}));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());

    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Non2Power)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Big)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Bigger)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Biggest)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

// Reproducer for https://github.com/NVIDIA/spark-rapids/issues/10062.
//
// Root cause: detail::determine_tiles() drops the trailing column when adding
// it would exceed shmem_limit_per_tile AND it is the very last column in the
// table. When the threshold fires, the function closes the previous tile but
// resets current_tile_width to 0 before restarting, treating `col` as part of
// a new tile. If `col` is already the last column, the loop exits and the
// `if (current_tile_width > 0)` tail never fires, so the tile containing only
// `col` is never emitted to the kernel. That column's output bytes are left
// at whatever the device allocator handed back (often 0, sometimes stale),
// producing the non-deterministic `b = 0` observed in the pivot test.
//
// The schema below mirrors the packed layout captured from a real failing
// pivot batch: 191 INT64 columns followed by 1 INT32 column `b`. With
// shmem_limit_per_tile ~= 49136 and tile_height = 32, the 191 INT64s fit in
// a single tile (1528 * 32 = 48896 bytes), and adding the trailing INT32
// tips the estimate to 1536 * 32 = 49152 > 49136, triggering the threshold
// exactly on the last column -- the shape required to hit the bug.
//
// We avoid convert_from_rows here because it uses the same layout code and
// would mask the bug; instead we dump the raw JCUDF bytes and check the
// INT32 location explicitly.
TEST_F(ColumnToRowTests, PivotLikeLayout)
{
  constexpr int num_longs      = 191;
  constexpr int num_rows       = 100;
  constexpr int num_iterations = 100;  // catch non-deterministic failures

  // Build sparse pivot-like values and validity for the 191 INT64 columns.
  // Column 0 -> `a` (always valid).
  // Columns 1..190 -> 95 pairs (count[i], max[i]); exactly one pair per row
  // is valid, all others are null with the backing data slot set to 0.
  std::vector<std::vector<int64_t>> long_vals(num_longs, std::vector<int64_t>(num_rows, 0));
  std::vector<std::vector<uint8_t>> long_valid(num_longs, std::vector<uint8_t>(num_rows, 0));

  constexpr int num_pairs = (num_longs - 1) / 2;  // 95 pairs
  for (int r = 0; r < num_rows; ++r) {
    long_vals[0][r]  = static_cast<int64_t>(0xA000) + r;
    long_valid[0][r] = 1;
  }
  for (int r = 0; r < num_rows; ++r) {
    int const k                  = r % num_pairs;
    int const count_col_idx      = 1 + 2 * k;
    int const max_col_idx        = count_col_idx + 1;
    long_vals[count_col_idx][r]  = static_cast<int64_t>(0xC000) + r;
    long_valid[count_col_idx][r] = 1;
    long_vals[max_col_idx][r]    = static_cast<int64_t>(0xD000) + r;
    long_valid[max_col_idx][r]   = 1;
  }

  // `b` (INT32, always valid). This is the column that gets corrupted to 0.
  std::vector<int32_t> expected_ints(num_rows);
  for (int r = 0; r < num_rows; ++r) {
    expected_ints[r] = 0x11223344 + r;
  }

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> long_cols;
  long_cols.reserve(num_longs);
  for (int i = 0; i < num_longs; ++i) {
    long_cols.emplace_back(long_vals[i].begin(), long_vals[i].end(), long_valid[i].begin());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> int_col(expected_ints.begin(),
                                                          expected_ints.end());

  std::vector<cudf::column_view> views;
  views.reserve(num_longs + 1);
  for (auto& c : long_cols) {
    views.emplace_back(c);
  }
  views.emplace_back(int_col);
  cudf::table_view in(views);

  // JCUDF layout derived from compute_column_information():
  //   - num_longs Longs occupy [0, 8*num_longs)
  //   - INT32 at [8*num_longs, 8*num_longs + 4)
  //   - validity starts at 8*num_longs + 4, size = ceil((num_longs + 1)/8)
  //   - row stride = round_up_8(data_end + validity_bytes)
  constexpr std::size_t int_offset     = static_cast<std::size_t>(num_longs) * sizeof(int64_t);
  constexpr std::size_t data_end       = int_offset + sizeof(int32_t);
  constexpr std::size_t validity_bytes = (num_longs + 1 + 7) / 8;
  constexpr std::size_t row_stride     = (data_end + validity_bytes + 7) & ~std::size_t{7};

  int fail_count = 0;
  for (int iter = 0; iter < num_iterations; ++iter) {
    auto rows = spark_rapids_jni::convert_to_rows(in);
    ASSERT_EQ(rows.size(), 1u);

    auto child      = cudf::lists_column_view(*rows[0]).child();
    auto host_bytes = cudf::test::to_host<int8_t>(child).first;
    ASSERT_GE(host_bytes.size(), row_stride * num_rows);

    for (int r = 0; r < num_rows; ++r) {
      int32_t actual = 0;
      std::memcpy(&actual, host_bytes.data() + r * row_stride + int_offset, sizeof(int32_t));
      if (actual != expected_ints[r]) {
        if (fail_count < 5) {
          ADD_FAILURE() << "iter=" << iter << " row=" << r << " expected=0x" << std::hex
                        << std::setw(8) << std::setfill('0') << expected_ints[r] << " actual=0x"
                        << std::setw(8) << std::setfill('0') << actual << std::dec;
        }
        ++fail_count;
      }
    }
  }
  EXPECT_EQ(fail_count, 0) << "total mismatches over " << num_iterations << " iterations * "
                           << num_rows << " rows";
}

TEST_F(RowToColumnTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Tall)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({i}));  // rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypes)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::FLOAT64},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      cudf::data_type{cudf::type_id::FLOAT32},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT64}};

  cudf::test::fixed_width_column_wrapper<int64_t> c0({3, 9, 4, 2, 20, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> c1({5.0, 9.5, 0.9, 7.23, 2.8, 0.0},
                                                    {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c2({5, 1, 0, 2, 7, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> c3({true, false, false, true, false, false},
                                                  {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> c4({1.0f, 3.5f, 5.9f, 7.1f, 9.8f, 0.0f},
                                                   {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c5({2, 3, 4, 5, 9, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_point_column_wrapper<int32_t> c6(
    {-300, 500, 950, 90, 723, 0}, {1, 1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<int64_t> c7(
    {-80, 30, 90, 20, 200, 0}, {1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-1});

  cudf::table_view in({c0, c1, c2, c3, c4, c5, c6, c7});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypesLarge)
{
  std::vector<cudf::column> cols;
  std::vector<cudf::data_type> schema{};

  // 15 columns of each type with 1 million entries
  constexpr int num_rows{1024 * 1024 * 1};

  std::default_random_engine re;
  std::uniform_real_distribution<double> rand_double(std::numeric_limits<double>::min(),
                                                     std::numeric_limits<double>::max());
  std::uniform_int_distribution<int64_t> rand_int64(std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::max());
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](auto i) -> int64_t { return rand_int64(re); });
  auto d = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](auto i) -> double { return rand_double(re); });

  auto all_valid =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto i) { return 1; });
  auto none_valid =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto i) { return 0; });
  auto most_valid = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 2 == 0 ? 0 : 1; });
  auto few_valid = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 13 == 0 ? 1 : 0; });

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int8_t>(r, r + num_rows, all_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int16_t>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT16});
  }

  for (int i = 0; i < 15; ++i) {
    if (i < 5) {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, few_valid)
                        .release()
                        .release());
    } else {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, none_valid)
                        .release()
                        .release());
    }
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<float>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<double>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT64});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<bool>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::BOOL8});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
         r, r + num_rows, all_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>(
         r, r + num_rows, most_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int32_t>(
                      r, r + num_rows, all_valid, numeric::scale_type{-2})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int64_t>(
                      r, r + num_rows, most_valid, numeric::scale_type{-1})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL64});
  }

  std::vector<cudf::column_view> views(cols.begin(), cols.end());
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Non2Power)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Big)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Bigger)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Biggest)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SimpleString)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1, 0, -1});
  cudf::test::strings_column_wrapper b(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32},
                                         cudf::data_type{cudf::type_id::STRING}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);
  EXPECT_EQ(new_rows.size(), 1);
  for (auto& row : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*row), schema);
    EXPECT_EQ(row->size(), 5);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *new_cols);
  }
}

TEST_F(RowToColumnTests, DoubleString)
{
  cudf::test::strings_column_wrapper a(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::test::fixed_width_column_wrapper<int32_t> b({0, 1, 2, 3, 4});
  cudf::test::strings_column_wrapper c({"world",
                                        "hello",
                                        "this string isn't as long",
                                        "this one isn't so short though when you think about it",
                                        "dlrow"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::STRING},
                                         cudf::data_type{cudf::type_id::INT32},
                                         cudf::data_type{cudf::type_id::STRING}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < new_rows.size(); ++i) {
    auto new_cols =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    EXPECT_EQ(new_rows[0]->size(), 5);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *new_cols);
  }
}

TEST_F(RowToColumnTests, BigStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 50;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (auto& i : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*i), schema);

    auto in_view = cudf::slice(in, {0, new_cols->num_rows()});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_view[0], *new_cols);
  }
}

TEST_F(RowToColumnTests, ManyStrings)
{
  // The sizing of this test is very sensitive to the state of the random number generator,
  // i.e., depending on the order of execution, the number of times the largest string is
  // selected will lead to out-of-memory exceptions. Seeding the RNG here helps prevent that.
  srand(1);
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine ",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working",
    "some strings",
    "are split into multiple strings",
    "some strings have all their data",
    "lots of choices of strings and sizes is sure to test the offset calculation code to ensure "
    "that even a really long string ends up in the correct spot for the final destination allowing "
    "for even crazy run-on sentences to be inserted into the data"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 300'000;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (auto& i : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*i), schema);

    auto in_view = cudf::slice(in, {0, new_cols->num_rows()});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_view[0], *new_cols);
  }
}
