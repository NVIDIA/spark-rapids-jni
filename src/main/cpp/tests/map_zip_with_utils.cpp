#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/lists/lists_column_view.hpp>

#include <map_zip_with_utils.hpp>

using namespace cudf;
using indices_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

constexpr test::debug_output_level verbosity{test::debug_output_level::ALL_ERRORS};

struct MapZipWithUtilsTests : public test::BaseFixture {};

TEST_F(MapZipWithUtilsTests, BasicMapZipTest)
{
  {
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col1 = cudf::column(*struct_col1);
    auto list_offsets_column =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows = list_offsets_column->size() - 1;
    auto list_col1     = cudf::make_lists_column(
      num_list_rows, std::move(list_offsets_column), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{7, 8, 9, 10, 11, 12};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{{12, 35, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col2 = cudf::column(*struct_col2);
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows2 = list_offsets_column2->size() - 1;
    auto list_col2      = cudf::make_lists_column(
      num_list_rows2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto const k =
      cudf::test::fixed_width_column_wrapper<size_type>{1, 2, 7, 8, 3, 9, 4, 5, 10, 11, 6, 12};
    auto const v1 = cudf::test::fixed_width_column_wrapper<size_type>{
      {48, 27, 3, 4, 25, 6, 31, 351, 9, 10, 351, 12}, {1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<size_type>{
      {5, 6, 12, 35, 1, 25, 3, 3, 31, 351, 2, 351}, {0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1}};
    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, OneEmptyMapTest)
{
  {
    // Test with one empty map and one non-empty map
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2                = cudf::test::fixed_width_column_wrapper<int32_t>{};
    auto vals2                = cudf::test::fixed_width_column_wrapper<int32_t>{};
    auto struct_col2          = cudf::test::structs_column_wrapper({keys2, vals2}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 0}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    // First result should contain values from map1, second should be empty
    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0}, {0, 0, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, OverlappingKeysTest)
{
  {
    // Test with overlapping keys between maps
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300, 400}, {1, 1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 4}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 5, 6};
    auto vals2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{250, 350, 500, 600}, {1, 1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 4}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300, 400, 0, 0},
                                                                    {1, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 250, 350, 0, 500, 600},
                                                                    {0, 1, 1, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, NonOverlappingKeysTest)
{
  {
    // Test with completely non-overlapping keys
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2       = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5, 6};
    auto vals2       = cudf::test::fixed_width_column_wrapper<int32_t>{{40, 50, 60}, {1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto const v1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 0, 0, 0}, {1, 1, 1, 0, 0, 0}};
    auto const v2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0, 40, 50, 60}, {0, 0, 0, 1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, MultipleRowsTest)
{
  {
    // Test with multiple rows (multiple maps)
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 40, 50}, {1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 5}
                                  .release();  // 2 rows: [1,2], [3,4,5]
    auto list_col1 =
      cudf::make_lists_column(2, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 4, 6, 7};
    auto vals2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{25, 35, 45, 65, 75}, {1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3, 5}
                                  .release();  // 2 rows: [2,3,4], [6,7]
    auto list_col2 =
      cudf::make_lists_column(2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 3, 4, 5, 6, 7};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {10, 20, 0, 0, 30, 40, 50, 0, 0}, {1, 1, 0, 0, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {0, 25, 35, 45, 0, 0, 0, 65, 75}, {0, 1, 1, 1, 0, 0, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, SingleElementMapsTest)
{
  {
    // Test with single element maps
    auto keys1                = cudf::test::fixed_width_column_wrapper<int32_t>{1};
    auto vals1                = cudf::test::fixed_width_column_wrapper<int32_t>{{100}, {1}};
    auto struct_col1          = cudf::test::structs_column_wrapper({keys1, vals1}, {1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 1}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2                = cudf::test::fixed_width_column_wrapper<int32_t>{2};
    auto vals2                = cudf::test::fixed_width_column_wrapper<int32_t>{{200}, {1}};
    auto struct_col2          = cudf::test::structs_column_wrapper({keys2, vals2}, {1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 1}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 0}, {1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 200}, {0, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, IdenticalKeysTest)
{
  {
    // Test with identical keys in both maps
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals2       = cudf::test::fixed_width_column_wrapper<int32_t>{{150, 250, 350}, {1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300}, {1, 1, 1}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{150, 250, 350}, {1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, LargeMapsTest)
{
  {
    // Test with larger maps
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}).release();
    auto list_offsets_column1 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 3, 4, 6, 6, 10}.release();
    auto list_col1 =
      cudf::make_lists_column(5, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{3, 11, 12, 13, 4, 5, 6, 7, 9, 8, 14, 15};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 163, 189}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}).release();
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 4, 5, 8, 8, 12}.release();
    auto list_col2 =
      cudf::make_lists_column(5, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k = cudf::test::fixed_width_column_wrapper<int32_t>{
      1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 7, 8, 9, 10, 14, 15};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {10, 20, 30, 0, 0, 0, 40, 50, 60, 0, 70, 80, 90, 100, 0, 0},
      {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {0, 0, 55, 66, 77, 88, 99, 110, 121, 132, 0, 154, 143, 0, 163, 189},
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0).child(1), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1).child(1), verbosity);
  }
}

// TEST_F(MapZipWithUtilsTests, NullValuesTest)
// {
//   {
//     // Test with null values in the maps
//     auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
//     auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 0, 1}}; //
//     null at index 1 auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1,
//     1}).release(); auto list_offsets_column1 =
//     cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release(); auto list_col1 =
//     cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

//     auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 4};
//     auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{{25, 35, 45}, {0, 1, 1}}; //
//     null at index 0 auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1,
//     1}).release(); auto list_offsets_column2 =
//     cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release(); auto list_col2 =
//     cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

//     auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
//     cudf::lists_column_view(*list_col2));

//     // Expected: union of keys {1, 2, 3, 4}
//     // Values from map1: [10, null, 30, null]
//     // Values from map2: [null, null, 35, 45]
//     auto const expected_vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 0, 30, 0},
//     {1, 0, 1, 0}}; auto const expected_vals2 =
//     cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 35, 45}, {0, 0, 1, 1}};
//     CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals1,
//     cudf::lists_column_view(*results.first).child(), verbosity);
//     CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals2,
//     cudf::lists_column_view(*results.second).child(), verbosity);
//   }
// }