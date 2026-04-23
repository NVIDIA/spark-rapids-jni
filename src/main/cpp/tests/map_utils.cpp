/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include <map_utils.hpp>

using size_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
using int_col  = cudf::test::fixed_width_column_wrapper<int32_t>;

struct MapUtilsTests : public cudf::test::BaseFixture {};

// ---------------------------------------------------------------------------
// Input-validation branches: covers the three CUDF_EXPECTS at map_utils.cpp:
// non-LIST input, LIST child not STRUCT, STRUCT not 2 children.
// ---------------------------------------------------------------------------

TEST_F(MapUtilsTests, NonListInputThrows)
{
  auto const input = int_col{1, 2, 3};
  EXPECT_THROW(spark_rapids_jni::map_from_entries(input, /*throw_on_null_key=*/true),
               cudf::logic_error);
}

TEST_F(MapUtilsTests, ListOfNonStructThrows)
{
  // LIST<INT32> — child is not STRUCT.
  auto offsets  = size_col{0, 2, 3}.release();
  auto children = int_col{1, 2, 3}.release();
  auto list =
    cudf::make_lists_column(2, std::move(offsets), std::move(children), 0, rmm::device_buffer{});
  EXPECT_THROW(spark_rapids_jni::map_from_entries(list->view(), true), cudf::logic_error);
}

TEST_F(MapUtilsTests, StructWithWrongArityThrows)
{
  // LIST<STRUCT<key>> — single-child struct (needs 2).
  auto keys    = int_col{1, 2, 3};
  auto structs = cudf::test::structs_column_wrapper({keys}).release();
  auto offsets = size_col{0, 2, 3}.release();
  auto list =
    cudf::make_lists_column(2, std::move(offsets), std::move(structs), 0, rmm::device_buffer{});
  EXPECT_THROW(spark_rapids_jni::map_from_entries(list->view(), true), cudf::logic_error);
}

// Contract check also fires for empty inputs — the nested-child validation runs
// before the zero-row early return, matching the header @throws documentation.
TEST_F(MapUtilsTests, EmptyNonListInputStillThrows)
{
  auto const input = int_col{};
  EXPECT_THROW(spark_rapids_jni::map_from_entries(input, true), cudf::logic_error);
}

// ---------------------------------------------------------------------------
// Non-INT32 keys: strings-keyed path. The implementation is type-generic
// (delegates to cudf::is_null / cudf::lists::contains_nulls), but strings have
// non-trivial child structure (offsets + chars), so an explicit test pins the
// `structs.child(0)` / `keys.nullable()` / `keys.null_count()` path for a
// compound key type.
// ---------------------------------------------------------------------------

TEST_F(MapUtilsTests, StringKeyNullThrows)
{
  // row 0: {"a", 10}, {null_key, 20}  →  must throw (null key in valid struct).
  auto keys    = cudf::test::strings_column_wrapper({"a", "x"}, {1, 0});
  auto values  = int_col{10, 20};
  auto structs = cudf::test::structs_column_wrapper({keys, values}).release();
  auto offsets = size_col{0, 2}.release();
  auto list_col =
    cudf::make_lists_column(1, std::move(offsets), std::move(structs), 0, rmm::device_buffer{});
  EXPECT_THROW(spark_rapids_jni::map_from_entries(list_col->view(), true), cudf::logic_error);
}

TEST_F(MapUtilsTests, StringKeyNonNullSucceeds)
{
  // All-valid string keys: should pass through without throwing and without masking.
  auto keys    = cudf::test::strings_column_wrapper({"a", "b", "c"});
  auto values  = int_col{10, 20, 30};
  auto structs = cudf::test::structs_column_wrapper({keys, values}).release();
  auto offsets = size_col{0, 2, 3}.release();
  auto list_col =
    cudf::make_lists_column(2, std::move(offsets), std::move(structs), 0, rmm::device_buffer{});
  std::unique_ptr<cudf::column> result;
  EXPECT_NO_THROW(result = spark_rapids_jni::map_from_entries(list_col->view(), true));
  EXPECT_EQ(result->size(), 2);
  EXPECT_EQ(result->null_count(), 0);
}
