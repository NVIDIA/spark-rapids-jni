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

#include "mixed_sort_merge_join.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>
#include <rmm/device_uvector.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

// Helper function to convert device_uvector to column_view for testing
cudf::column_view to_column_view(rmm::device_uvector<cudf::size_type> const& vec)
{
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32}, 
                           vec.size(), 
                           vec.data(),
                           nullptr,  // null_mask
                           0,        // null_count
                           0,        // offset
                           {});      // children
}

// Helper function to extract validity mask from a column view
std::vector<bool> extract_validity_mask(cudf::column_view const& col)
{
  std::vector<bool> valid(col.size(), true);
  
  if (col.null_mask() != nullptr) {
    size_t mask_size_bytes = cudf::bitmask_allocation_size_bytes(col.size());
    std::vector<cudf::bitmask_type> mask_host(mask_size_bytes / sizeof(cudf::bitmask_type));
    CUDF_CUDA_TRY(cudaMemcpy(mask_host.data(),
                             col.null_mask(),
                             mask_size_bytes,
                             cudaMemcpyDeviceToHost));
    for (cudf::size_type i = 0; i < col.size(); i++) {
      valid[i] = cudf::bit_is_set(mask_host.data(), i + col.offset());
    }
  }
  
  return valid;
}

class MixedSortMergeJoinTest : public cudf::test::BaseFixture {};

TEST_F(MixedSortMergeJoinTest, InnerJoinBasic)
{
  // Test basic inner join with equality keys and conditional expression
  // Left equality:  {0, 1, 2}
  // Right equality: {1, 2, 3}
  // Left conditional:  {4, 4, 4}
  // Right conditional: {3, 4, 5}
  // Condition: left_col > right_col
  // Expected result: row 1 from left matches with row 0 from right (1==1 and 4>3)

  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: {1}, {0} - row 1 from left matches row 0 from right
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinNoMatches)
{
  // Test inner join where equality keys match but conditional expression never true
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({10, 20});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col (will always be false)
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no matches
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftJoinBasic)
{
  // Test basic left join
  // Left equality:  {0, 1, 2}
  // Right equality: {1, 2, 3}
  // Left conditional:  {4, 4, 4}
  // Right conditional: {3, 4, 5}
  // Condition: left_col > right_col

  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: all left rows (0, 1, 2), with row 1 matching right row 0, others null
  EXPECT_EQ(3, left_result->size());
  EXPECT_EQ(3, right_result->size());

  // Check that we have row 1 from left matching with row 0 from right
  std::vector<cudf::size_type> left_host(left_result->size());
  std::vector<cudf::size_type> right_host(right_result->size());
  CUDF_CUDA_TRY(cudaMemcpy(left_host.data(),
                           left_result->data(),
                           left_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_host.data(),
                           right_result->data(),
                           right_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));

  // Find the matching pair
  bool found_match = false;
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (left_host[i] == 1 && right_host[i] == 0) {
      found_match = true;
      break;
    }
  }
  EXPECT_TRUE(found_match);

  // Unmatched left rows should have right index == right_num_rows (OOB sentinel for NULLIFY gather)
  auto const right_num_rows = right_eq_table.num_rows();
  int unmatched_seen = 0;
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (!(left_host[i] == 1 && right_host[i] == 0)) {
      EXPECT_EQ(right_num_rows, right_host[i]);
      unmatched_seen++;
    }
  }
  EXPECT_EQ(2, unmatched_seen);
}

TEST_F(MixedSortMergeJoinTest, LeftJoinEmptyRight)
{
  // Test left join with empty right table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: all left rows with null right indices, which are marked as out-of-bounds = right_num_rows (0)
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, LeftJoinGatherNullify)
{
  // Verify that using the right gather map with NULLIFY policy produces nulls for unmatched rows
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table    = cudf::table_view({left_equality});
  auto right_eq_table   = cudf::table_view({right_equality});
  auto left_cond_table  = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_map, right_map] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Gather from right conditional using right_map with NULLIFY policy
  auto gathered_right_tbl = cudf::gather(
    right_cond_table,
    to_column_view(*right_map),
    cudf::out_of_bounds_policy::NULLIFY);

  // Locate the matched pair position (left==1 with right==0)
  std::vector<cudf::size_type> left_host(left_map->size());
  std::vector<cudf::size_type> right_host(right_map->size());
  CUDF_CUDA_TRY(cudaMemcpy(left_host.data(),
                           left_map->data(),
                           left_map->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_host.data(),
                           right_map->data(),
                           right_map->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));

  size_t match_pos = left_host.size();
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (left_host[i] == 1 && right_host[i] == 0) {
      match_pos = i;
      break;
    }
  }
  ASSERT_LT(match_pos, left_host.size());

  // Expected gathered-right column: value 3 at match_pos, null elsewhere
  std::vector<int32_t> expected_vals(left_host.size(), 0);
  std::vector<bool> expected_valid(left_host.size(), false);
  expected_vals[match_pos]  = 3;
  expected_valid[match_pos] = true;

  cudf::test::fixed_width_column_wrapper<int32_t> expected(
    expected_vals.begin(), expected_vals.end(), expected_valid.begin());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    expected,
    gathered_right_tbl->view().column(0));
}

TEST_F(MixedSortMergeJoinTest, LeftSemiJoinBasic)
{
  // Test basic left semi join
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: {1} - only row 1 from left has a match
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, to_column_view(*result));
}

TEST_F(MixedSortMergeJoinTest, LeftSemiJoinEmptyRight)
{
  // Test left semi join with empty right table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: empty result - no rows match
  EXPECT_EQ(0, result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftAntiJoinBasic)
{
  // Test basic left anti join
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_anti_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: {0, 2} - rows 0 and 2 from left have no matches
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({0, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, to_column_view(*result));
}

TEST_F(MixedSortMergeJoinTest, LeftAntiJoinEmptyRight)
{
  // Test left anti join with empty right table - all left rows should be returned
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_anti_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: {0, 1, 2} - all left rows have no matches
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({0, 1, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, to_column_view(*result));
}

TEST_F(MixedSortMergeJoinTest, MultipleEqualityKeys)
{
  // Test with multiple equality key columns
  cudf::test::fixed_width_column_wrapper<int32_t> left_eq1({1, 1, 2, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_eq2({1, 2, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_eq1({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_eq2({2, 1});

  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({10, 20, 30, 40});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({15, 25});

  auto left_eq_table   = cudf::table_view({left_eq1, left_eq2});
  auto right_eq_table  = cudf::table_view({right_eq1, right_eq2});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: rows 1 and 2 both match
  // Row 1: (1,2)==(1,2) and 20>15
  // Row 2: (2,1)==(2,1) and 30>25
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 2});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, to_column_view(*result));
}

TEST_F(MixedSortMergeJoinTest, LeftJoinMultipleMatchesPerLeftRow)
{
  // Test left join where a single left row matches multiple right rows
  // Left:  {1, 2} with conditional {10, 20}
  // Right: {1, 1, 2} with conditional {5, 7, 15}
  // Condition: left > right
  // Expected: Left row 0 (1,10) matches right rows 0 and 1 (1,5 and 1,7)
  //           Left row 1 (2,20) matches right row 2 (2,15)
  //           Total: 3 output rows
  
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({10, 20});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({5, 7, 15});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: 3 rows total (left row 0 appears twice, left row 1 appears once)
  EXPECT_EQ(3, left_result->size());
  EXPECT_EQ(3, right_result->size());

  // Copy results to host for verification
  std::vector<cudf::size_type> left_host(left_result->size());
  std::vector<cudf::size_type> right_host(right_result->size());
  CUDF_CUDA_TRY(cudaMemcpy(left_host.data(),
                           left_result->data(),
                           left_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_host.data(),
                           right_result->data(),
                           right_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));

  // Count how many times each left row appears
  int left_0_count = 0;
  int left_1_count = 0;
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (left_host[i] == 0) {
      left_0_count++;
      // Left row 0 should match right rows 0 or 1
      EXPECT_TRUE(right_host[i] == 0 || right_host[i] == 1) 
        << "Left row 0 matched unexpected right row " << right_host[i];
    } else if (left_host[i] == 1) {
      left_1_count++;
      // Left row 1 should match right row 2
      EXPECT_EQ(2, right_host[i]) << "Left row 1 should match right row 2";
    }
  }
  
  // Left row 0 should appear twice (matches right rows 0 and 1)
  EXPECT_EQ(2, left_0_count) << "Left row 0 should appear twice";
  // Left row 1 should appear once (matches right row 2)
  EXPECT_EQ(1, left_1_count) << "Left row 1 should appear once";
}

TEST_F(MixedSortMergeJoinTest, InnerJoinMultipleMatchesPerLeftRow)
{
  // Test inner join where a single left row matches multiple right rows
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({10, 20});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({5, 7, 25});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: Left row 0 (1,10) matches right rows 0 and 1 (both 10 > 5 and 10 > 7)
  //           Left row 1 (2,20) does NOT match right row 2 (20 < 25)
  //           Total: 2 output rows
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({0, 0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, PreSortedTables)
{
  // Test with pre-sorted equality tables
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});  // already sorted
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3}); // already sorted
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition,
    cudf::sorted::YES,  // left is sorted
    cudf::sorted::YES,  // right is sorted
    cudf::null_equality::EQUAL);

  // Expected: {1}, {0} - row 1 from left matches row 0 from right
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinComplexCondition)
{
  // Test with more complex conditional expression (AND of two comparisons)
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({0, 1, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> left_cond1({3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> left_cond2({10, 20, 30, 40});
  cudf::test::fixed_width_column_wrapper<int32_t> right_cond1({5, 4, 5, 7});
  cudf::test::fixed_width_column_wrapper<int32_t> right_cond2({30, 40, 50, 60});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_cond1, left_cond2});
  auto right_cond_table = cudf::table_view({right_cond1, right_cond2});

  // Build AST: left_cond1 < right_cond1 AND literal(35) < right_cond2
  auto left_ref_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref_1 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto right_ref_2 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  
  auto scalar_35 = cudf::numeric_scalar<int32_t>(35);
  auto literal_35 = cudf::ast::literal(scalar_35);
  
  auto op1 = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_ref_1, right_ref_1);
  auto op2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, literal_35, right_ref_2);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, op1, op2);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: row 3 from left (eq=4, cond1=6) matches row 3 from right (eq=4, cond1=7, cond2=60)
  // because 4==4 AND 6<7 AND 35<60
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({3});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({3});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinAsymmetricSizes)
{
  // Test with asymmetric table sizes (left larger than right)
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 2, 1, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({0, 1, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({3, 5, 4, 10});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({5, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col == right_col (testing equality in conditional)
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: row 2 from left (eq=1, cond=4) matches row 1 from right (eq=1, cond=4)
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({2});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinDuplicateKeysWithDifferentConditionalResults)
{
  // Test with duplicate equality keys that produce different conditional results
  // This ensures we're properly evaluating conditions for all equality matches
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({5, 10, 15});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({7, 8});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: (1,10) matches (1,7) since 10>7 AND (2,15) matches (2,8) since 15>8
  // But (1,5) does NOT match (1,7) since 5<7
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({1, 2});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinNullsInConditionalColumns)
{
  // Test with nulls in conditional columns (not equality columns)
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({10, 20, 30}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({5, 15, 25}, {true, true, false});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: Only (1,10) > (1,5) = true should match
  // (2,null) > (2,15) = null/false, (3,30) > (3,null) = null/false
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinWithGatherVerification)
{
  // Test inner join and verify gather produces correct results
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_data({100, 200, 300});
  cudf::test::fixed_width_column_wrapper<int32_t> right_data({10, 20, 30});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_full_table = cudf::table_view({left_equality, left_data});
  auto right_full_table = cudf::table_view({right_equality, right_data});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_map, right_map] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Perform gather on both tables
  auto gathered_left = cudf::gather(left_full_table, to_column_view(*left_map));
  auto gathered_right = cudf::gather(right_full_table, to_column_view(*right_map));

  // Expected: Only row 1 from left (200) matches with row 0 from right (10)
  cudf::test::fixed_width_column_wrapper<int32_t> expected_left_eq({1});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_left_data({200});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_right_eq({1});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_right_data({10});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left_eq, gathered_left->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left_data, gathered_left->view().column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right_eq, gathered_right->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right_data, gathered_right->view().column(1));
}

TEST_F(MixedSortMergeJoinTest, LeftJoinWithMultipleDataColumns)
{
  // Test left join with multiple data columns and gather to verify results
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_data1({100, 200, 300});
  cudf::test::fixed_width_column_wrapper<int32_t> left_data2({10, 20, 30});
  cudf::test::fixed_width_column_wrapper<int32_t> right_data1({1000, 2000});
  cudf::test::fixed_width_column_wrapper<int32_t> right_data2({50, 60});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_full_table = cudf::table_view({left_equality, left_data1, left_data2});
  auto right_full_table = cudf::table_view({right_equality, right_data1, right_data2});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_map, right_map] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Perform gather with NULLIFY for right table
  auto gathered_left = cudf::gather(left_full_table, to_column_view(*left_map));
  auto gathered_right = cudf::gather(right_full_table, 
                                     to_column_view(*right_map), 
                                     cudf::out_of_bounds_policy::NULLIFY);

  // All left rows should be present
  EXPECT_EQ(3, gathered_left->num_rows());
  EXPECT_EQ(3, gathered_right->num_rows());

  // Copy results to host to verify (order may vary)
  std::vector<int32_t> left_eq_host(3);
  std::vector<int32_t> left_data1_host(3);
  std::vector<int32_t> left_data2_host(3);
  std::vector<int32_t> right_data1_host(3);
  std::vector<int32_t> right_data2_host(3);
  std::vector<bool> right_data1_valid(3);
  std::vector<bool> right_data2_valid(3);

  auto left_view = gathered_left->view();
  CUDF_CUDA_TRY(cudaMemcpy(left_eq_host.data(),
                           left_view.column(0).data<int32_t>(),
                           3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(left_data1_host.data(),
                           left_view.column(1).data<int32_t>(),
                           3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(left_data2_host.data(),
                           left_view.column(2).data<int32_t>(),
                           3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));

  auto right_view = gathered_right->view();
  CUDF_CUDA_TRY(cudaMemcpy(right_data1_host.data(),
                           right_view.column(1).data<int32_t>(),
                           3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_data2_host.data(),
                           right_view.column(2).data<int32_t>(),
                           3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));

  // Extract validity masks using helper function
  auto const& right_col1 = right_view.column(1);
  auto const& right_col2 = right_view.column(2);
  
  right_data1_valid = extract_validity_mask(right_col1);
  right_data2_valid = extract_validity_mask(right_col2);

  // Verify: All left rows present with correct data
  std::set<int32_t> left_eq_set(left_eq_host.begin(), left_eq_host.end());
  EXPECT_EQ(std::set<int32_t>({0, 1, 2}), left_eq_set);

  // Find position of each left row and verify corresponding data
  for (int i = 0; i < 3; i++) {
    int left_row = left_eq_host[i];
    
    // Verify left data matches original row
    EXPECT_EQ(100 + left_row * 100, left_data1_host[i]) 
      << "Left row " << left_row << " data1 mismatch at position " << i;
    EXPECT_EQ(10 + left_row * 10, left_data2_host[i])
      << "Left row " << left_row << " data2 mismatch at position " << i;

    // Verify right data: only left row 1 should match (eq=1, 4>3)
    if (left_row == 1) {
      // Row 1 matches right row 0: should have values 1000, 50
      EXPECT_TRUE(right_data1_valid[i]) << "Right data1 should be valid for matched row 1";
      EXPECT_TRUE(right_data2_valid[i]) << "Right data2 should be valid for matched row 1";
      if (right_data1_valid[i]) {
        EXPECT_EQ(1000, right_data1_host[i]);
      }
      if (right_data2_valid[i]) {
        EXPECT_EQ(50, right_data2_host[i]);
      }
    } else {
      // Rows 0 and 2 don't match: should be null
      EXPECT_FALSE(right_data1_valid[i]) << "Right data1 should be null for unmatched row " << left_row;
      EXPECT_FALSE(right_data2_valid[i]) << "Right data2 should be null for unmatched row " << left_row;
    }
  }
}

TEST_F(MixedSortMergeJoinTest, LeftSemiJoinDuplicateKeys)
{
  // Test left semi join with duplicate equality keys
  // Ensure each left row appears at most once even if it has multiple matches
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({0, 1, 3, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({5, 4, 5, 6});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col == right_col (equality in conditional)
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: rows 1 and 3 from left both match (eq=1 with cond=4 and cond=6)
  // But semi-join should return unique indices: {1, 3}
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, to_column_view(*result));
}

TEST_F(MixedSortMergeJoinTest, LeftAntiJoinWithGatherVerification)
{
  // Test left anti join and verify gather produces correct unmatched rows
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_data({100, 200, 300});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_full_table = cudf::table_view({left_equality, left_data});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result_map = spark_rapids_jni::mixed_sort_merge_left_anti_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Perform gather
  auto gathered = cudf::gather(left_full_table, to_column_view(*result_map));

  // Expected: rows 0 and 2 from left don't match (row 1 matches with 4>3)
  cudf::test::fixed_width_column_wrapper<int32_t> expected_eq({0, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_data({100, 300});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_eq, gathered->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, gathered->view().column(1));
}

TEST_F(MixedSortMergeJoinTest, NullsInEqualityKeysWithNullEqual)
{
  // Test with nulls in equality keys where nulls should match
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2, 3}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3}, {true, true, false});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 6, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition,
    cudf::sorted::NO,
    cudf::sorted::NO,
    cudf::null_equality::EQUAL);

  // Expected: (1,1) passes 4>3 AND (null,null) passes 6>5 when nulls are equal
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, NullsInEqualityKeysWithNullUnequal)
{
  // Test with nulls in equality keys where nulls should NOT match
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({1, 2, 3}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3}, {true, true, false});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 6, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition,
    cudf::sorted::NO,
    cudf::sorted::NO,
    cudf::null_equality::UNEQUAL);

  // Expected: Only (1,1) passes 4>3. Nulls don't match when UNEQUAL.
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

// ===== Empty Table Tests =====

TEST_F(MixedSortMergeJoinTest, InnerJoinEmptyLeftTable)
{
  // Test inner join with empty left table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, InnerJoinEmptyRightTable)
{
  // Test inner join with empty right table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({4, 4, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, InnerJoinBothTablesEmpty)
{
  // Test inner join with both tables empty
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftJoinEmptyLeftTable)
{
  // Test left join with empty left table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results (no left rows to preserve)
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftSemiJoinEmptyLeftTable)
{
  // Test left semi join with empty left table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results
  EXPECT_EQ(0, result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftAntiJoinEmptyLeftTable)
{
  // Test left anti join with empty left table
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional({});
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional({3, 4, 5});

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_anti_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: no results
  EXPECT_EQ(0, result->size());
}

// ===== Literal-Only Expression Tests =====

TEST_F(MixedSortMergeJoinTest, InnerJoinLiteralOnlyExpressionAlwaysTrue)
{
  // Test with empty conditional tables and a literal-only expression that's always true
  // This should match all equality pairs
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  
  // Create empty conditional tables (0 columns, but matching row counts)
  std::vector<cudf::column_view> left_cond_cols;
  std::vector<cudf::column_view> right_cond_cols;
  auto left_cond_table = cudf::table_view(left_cond_cols);
  auto right_cond_table = cudf::table_view(right_cond_cols);

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});

  // Build AST: literal(5) > literal(3) - always true, no column references
  auto scalar_5 = cudf::numeric_scalar<int32_t>(5);
  auto scalar_3 = cudf::numeric_scalar<int32_t>(3);
  auto literal_5 = cudf::ast::literal(scalar_5);
  auto literal_3 = cudf::ast::literal(scalar_3);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal_5, literal_3);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: All equality matches since condition is always true
  // Left {0, 1, 2} matches Right {1, 2, 3} on equality:
  // Row 1 from left matches row 0 from right (both are 1)
  // Row 2 from left matches row 1 from right (both are 2)
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left({1, 2});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right({0, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_left, to_column_view(*left_result));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_right, to_column_view(*right_result));
}

TEST_F(MixedSortMergeJoinTest, InnerJoinLiteralOnlyExpressionAlwaysFalse)
{
  // Test with empty conditional tables and a literal-only expression that's always false
  // This should match no pairs
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  
  // Create empty conditional tables (0 columns, but matching row counts)
  std::vector<cudf::column_view> left_cond_cols;
  std::vector<cudf::column_view> right_cond_cols;
  auto left_cond_table = cudf::table_view(left_cond_cols);
  auto right_cond_table = cudf::table_view(right_cond_cols);

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});

  // Build AST: literal(3) > literal(5) - always false, no column references
  auto scalar_3 = cudf::numeric_scalar<int32_t>(3);
  auto scalar_5 = cudf::numeric_scalar<int32_t>(5);
  auto literal_3 = cudf::ast::literal(scalar_3);
  auto literal_5 = cudf::ast::literal(scalar_5);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal_3, literal_5);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: No matches since condition is always false
  EXPECT_EQ(0, left_result->size());
  EXPECT_EQ(0, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LeftJoinLiteralOnlyExpression)
{
  // Test left join with literal-only expression
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality({1, 2, 3});
  
  // Create empty conditional tables
  std::vector<cudf::column_view> left_cond_cols;
  std::vector<cudf::column_view> right_cond_cols;
  auto left_cond_table = cudf::table_view(left_cond_cols);
  auto right_cond_table = cudf::table_view(right_cond_cols);

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});

  // Build AST: literal(5) > literal(3) - always true
  auto scalar_5 = cudf::numeric_scalar<int32_t>(5);
  auto scalar_3 = cudf::numeric_scalar<int32_t>(3);
  auto literal_5 = cudf::ast::literal(scalar_5);
  auto literal_3 = cudf::ast::literal(scalar_3);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal_5, literal_3);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: All left rows present
  // Rows 1 and 2 match on equality with condition true
  // Row 0 has no equality match, so unmatched (right index = 3 = right_num_rows)
  EXPECT_EQ(3, left_result->size());
  EXPECT_EQ(3, right_result->size());

  std::vector<cudf::size_type> left_host(left_result->size());
  std::vector<cudf::size_type> right_host(right_result->size());
  CUDF_CUDA_TRY(cudaMemcpy(left_host.data(),
                           left_result->data(),
                           left_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_host.data(),
                           right_result->data(),
                           right_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));

  // Verify all left rows are present
  std::set<cudf::size_type> left_rows(left_host.begin(), left_host.end());
  EXPECT_EQ(std::set<cudf::size_type>({0, 1, 2}), left_rows);
}

// ===== Large Data Test =====

TEST_F(MixedSortMergeJoinTest, LargeRandomDataSemiJoin)
{
  // Test with larger dataset to verify scalability and correctness
  // Create 10,000 rows with repeated values
  constexpr int N = 10000;
  constexpr int num_repeats = 100;
  constexpr int num_unique = N / num_repeats;
  
  std::vector<int32_t> left_eq_data(N);
  std::vector<int32_t> right_eq_data(N);
  std::vector<int32_t> left_cond_data(N);
  std::vector<int32_t> right_cond_data(N);
  
  // Generate repeated sequences
  for (int i = 0; i < num_repeats; ++i) {
    for (int j = 0; j < num_unique; ++j) {
      left_eq_data[i * num_unique + j] = j;
      right_eq_data[i * num_unique + j] = j;
      left_cond_data[i * num_unique + j] = j + 1;  // left is always greater
      right_cond_data[i * num_unique + j] = j;
    }
  }
  
  // Shuffle to test unsorted performance
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left_eq_data.begin(), left_eq_data.end(), gen);
  std::shuffle(right_eq_data.begin(), right_eq_data.end(), gen);
  
  // Create column wrappers
  cudf::test::fixed_width_column_wrapper<int32_t> left_equality(left_eq_data.begin(), left_eq_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> right_equality(right_eq_data.begin(), right_eq_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> left_conditional(left_cond_data.begin(), left_cond_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> right_conditional(right_cond_data.begin(), right_cond_data.end());

  auto left_eq_table   = cudf::table_view({left_equality});
  auto right_eq_table  = cudf::table_view({right_equality});
  auto left_cond_table = cudf::table_view({left_conditional});
  auto right_cond_table = cudf::table_view({right_conditional});

  // Build AST: left_col > right_col
  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto condition = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table,
    right_eq_table,
    left_cond_table,
    right_cond_table,
    condition);

  // Expected: All unique values should have matches since left_cond > right_cond for all pairs
  // Each left row should match at least one right row with the same equality key
  EXPECT_GT(result->size(), 0);
  
  // Verify that all returned indices are valid
  std::vector<cudf::size_type> result_host(result->size());
  CUDF_CUDA_TRY(cudaMemcpy(result_host.data(),
                           result->data(),
                           result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  
  for (auto idx : result_host) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, N);
  }
  
  // Verify no duplicates in semi-join result
  std::set<cudf::size_type> unique_indices(result_host.begin(), result_host.end());
  EXPECT_EQ(unique_indices.size(), result_host.size()) << "Semi-join result should have no duplicates";
}

// =============================================================================
// EQUALITY-ONLY SORT-MERGE JOIN TESTS
// =============================================================================

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftJoinBasic)
{
  // Test basic left join without conditional expression
  // Left:  {0, 1, 2}
  // Right: {1, 2, 3}
  // Expected: all left rows, with rows 1 and 2 matching, row 0 unmatched

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2, 3});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto [left_result, right_result] = spark_rapids_jni::sort_merge_left_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: all 3 left rows
  EXPECT_EQ(3, left_result->size());
  EXPECT_EQ(3, right_result->size());

  // Copy results to host for verification
  std::vector<cudf::size_type> left_host(left_result->size());
  std::vector<cudf::size_type> right_host(right_result->size());
  CUDF_CUDA_TRY(cudaMemcpy(left_host.data(),
                           left_result->data(),
                           left_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(right_host.data(),
                           right_result->data(),
                           right_result->size() * sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost));

  // Verify we have the expected matching pairs: (1,0) and (2,1)
  std::set<std::pair<cudf::size_type, cudf::size_type>> matches;
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (right_host[i] != right_table.num_rows()) {  // Not OOB
      matches.insert({left_host[i], right_host[i]});
    }
  }
  EXPECT_EQ(2, matches.size());
  EXPECT_TRUE(matches.count({1, 0}) > 0);  // left 1 matches right 0 (both have value 1)
  EXPECT_TRUE(matches.count({2, 1}) > 0);  // left 2 matches right 1 (both have value 2)

  // Verify unmatched row 0 from left has OOB sentinel
  bool found_unmatched = false;
  for (size_t i = 0; i < left_host.size(); ++i) {
    if (left_host[i] == 0) {
      EXPECT_EQ(right_table.num_rows(), right_host[i]);
      found_unmatched = true;
    }
  }
  EXPECT_TRUE(found_unmatched);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftJoinEmptyRight)
{
  // Test left join with empty right table

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto [left_result, right_result] = spark_rapids_jni::sort_merge_left_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: all left rows with OOB right indices
  EXPECT_EQ(3, left_result->size());
  EXPECT_EQ(3, right_result->size());

  // With empty right table, special code path returns left indices in order
  auto left_col  = to_column_view(*left_result);
  auto right_col = to_column_view(*right_result);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_left({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_right({0, 0, 0});  // All OOB (right_num_rows=0)

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left, left_col);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right, right_col);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftJoinNullKeys)
{
  // Test left join with nulls in keys (nulls equal)

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({1, 2}, {1, 0});  // 1, null
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2}, {0, 1});  // null, 2

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto [left_result, right_result] = spark_rapids_jni::sort_merge_left_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: both left rows should match (1==1 at indices 0,na and null==null at indices 1,0)
  EXPECT_EQ(left_result->size(), 2);
  EXPECT_EQ(right_result->size(), 2);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftSemiJoinBasic)
{
  // Test basic left semi join without conditional expression
  // Left:  {0, 1, 2}
  // Right: {1, 2, 3}
  // Expected: {1, 2} from left (rows with matches)

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2, 3});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: indices 1 and 2 from left
  EXPECT_EQ(result->size(), 2);

  auto result_col = to_column_view(*result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftSemiJoinEmptyRight)
{
  // Test left semi join with empty right table - no matches

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  EXPECT_EQ(result->size(), 0);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftSemiJoinDuplicateKeys)
{
  // Test left semi join with duplicate keys - each left row appears once
  // Left:  {1, 1, 2}
  // Right: {1, 2}
  // Expected: {0, 1, 2} or {0, 2} (unique indices 0,1 map to same key, then 2)

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({1, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: 3 left rows all match (indices 0, 1, 2)
  EXPECT_EQ(result->size(), 3);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftAntiJoinBasic)
{
  // Test basic left anti join without conditional expression
  // Left:  {0, 1, 2}
  // Right: {1, 2, 3}
  // Expected: {0} from left (rows without matches)

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2, 3});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_anti_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: index 0 from left (value 0 has no match)
  EXPECT_EQ(result->size(), 1);

  auto result_col = to_column_view(*result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftAntiJoinEmptyRight)
{
  // Test left anti join with empty right table - all left rows returned

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_anti_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: all left rows
  EXPECT_EQ(result->size(), 3);

  auto result_col = to_column_view(*result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyLeftAntiJoinAllMatch)
{
  // Test left anti join where all rows match - empty result

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2, 3});

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_anti_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  EXPECT_EQ(result->size(), 0);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyPreSortedTables)
{
  // Test with pre-sorted tables (performance optimization)

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys({0, 1, 2});  // sorted
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys({1, 2, 3});  // sorted

  auto left_table  = cudf::table_view({left_keys});
  auto right_table = cudf::table_view({right_keys});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::YES, cudf::sorted::YES, cudf::null_equality::EQUAL);

  // Expected: indices 1 and 2 from left
  EXPECT_EQ(result->size(), 2);

  auto result_col = to_column_view(*result);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col);
}

TEST_F(MixedSortMergeJoinTest, EqualityOnlyMultipleColumns)
{
  // Test with multiple key columns

  cudf::test::fixed_width_column_wrapper<int32_t> left_col1({1, 1, 2, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> left_col2({1, 2, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_col1({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> right_col2({2, 1});

  auto left_table  = cudf::table_view({left_col1, left_col2});
  auto right_table = cudf::table_view({right_col1, right_col2});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Expected: (1,2) matches at index 1, (2,1) matches at index 2
  EXPECT_EQ(result->size(), 2);
}

// =============================================================================
// LARGE SCALE TESTS (Demonstrating sort-merge join advantages)
// =============================================================================

// Helper function to create a strings column from host data
std::unique_ptr<cudf::column> make_strings_column_from_host_vector(
  std::vector<std::string> const& data, rmm::cuda_stream_view stream)
{
  if (data.empty()) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  // Build offsets and chars buffer
  std::vector<cudf::size_type> offsets;
  offsets.reserve(data.size() + 1);
  offsets.push_back(0);
  
  std::string chars;
  for (auto const& s : data) {
    chars.append(s);
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }

  // Copy to device
  rmm::device_uvector<cudf::size_type> d_offsets(offsets.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                                offsets.data(),
                                offsets.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  rmm::device_buffer d_chars(chars.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_chars.data(),
                                chars.data(),
                                chars.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  return cudf::make_strings_column(
    data.size(),
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   offsets.size(),
                                   d_offsets.release(),
                                   rmm::device_buffer{},
                                   0),
    std::move(d_chars),
    0,
    rmm::device_buffer{});
}

TEST_F(MixedSortMergeJoinTest, LargeScaleConditionalLeftJoin)
{
  // This test demonstrates sort-merge join handling join explosion efficiently
  // Hash-based joins would hang or be extremely slow with this many duplicates
  // Left: 1 row with IMSI, hour, and timestamp
  // Right: 25M+ rows with same IMSI and hour, all matching the time range
  // Condition: left.time >= right.start_time AND left.time <= right.end_time
  // Result: 25M+ output pairs (join explosion!)

  auto stream = cudf::get_default_stream();
  
  // Create left table (1 row)
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  cudf::test::fixed_width_column_wrapper<int64_t> left_time({1759115400L});
  
  auto left_eq_table = cudf::table_view({left_imsi->view(), left_hour});
  auto left_cond_table = cudf::table_view({left_time});

  // Create right table with millions of duplicate keys that all match
  constexpr cudf::size_type num_rows = 25445819;  // Original size - demonstrates join explosion
  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  std::vector<int64_t> right_start_time_data(num_rows, 1759113600L);
  std::vector<int64_t> right_end_time_data(num_rows, 1759117199L);
  
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_start_time(right_start_time_data.begin(), right_start_time_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_end_time(right_end_time_data.begin(), right_end_time_data.end());
  
  auto right_eq_table = cudf::table_view({right_imsi->view(), right_hour});
  auto right_cond_table = cudf::table_view({right_start_time, right_end_time});

  // Build AST: left_time >= right_start_time AND left_time <= right_end_time
  auto left_time_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_start_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto right_end_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  
  auto ge_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, left_time_ref, right_start_ref);
  auto le_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, left_time_ref, right_end_ref);
  auto and_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);

  // Execute join - should handle many duplicates efficiently
  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_left_join(
    left_eq_table, right_eq_table, left_cond_table, right_cond_table, and_expr);

  // All right rows should match (condition is satisfied for all)
  EXPECT_EQ(num_rows, left_result->size());
  EXPECT_EQ(num_rows, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleConditionalInnerJoin)
{
  // Same join explosion scenario but for inner join
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  cudf::test::fixed_width_column_wrapper<int64_t> left_time({1759115400L});
  
  auto left_eq_table = cudf::table_view({left_imsi->view(), left_hour});
  auto left_cond_table = cudf::table_view({left_time});

  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  std::vector<int64_t> right_start_time_data(num_rows, 1759113600L);
  std::vector<int64_t> right_end_time_data(num_rows, 1759117199L);
  
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_start_time(right_start_time_data.begin(), right_start_time_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_end_time(right_end_time_data.begin(), right_end_time_data.end());
  
  auto right_eq_table = cudf::table_view({right_imsi->view(), right_hour});
  auto right_cond_table = cudf::table_view({right_start_time, right_end_time});

  auto left_time_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_start_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto right_end_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  
  auto ge_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, left_time_ref, right_start_ref);
  auto le_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, left_time_ref, right_end_ref);
  auto and_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);

  auto [left_result, right_result] = spark_rapids_jni::mixed_sort_merge_inner_join(
    left_eq_table, right_eq_table, left_cond_table, right_cond_table, and_expr);

  EXPECT_EQ(num_rows, left_result->size());
  EXPECT_EQ(num_rows, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleConditionalLeftSemiJoin)
{
  // Semi join is more efficient - it finds matches then deduplicates
  // But still needs to scan through millions of potential matches
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  cudf::test::fixed_width_column_wrapper<int64_t> left_time({1759115400L});
  
  auto left_eq_table = cudf::table_view({left_imsi->view(), left_hour});
  auto left_cond_table = cudf::table_view({left_time});

  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  std::vector<int64_t> right_start_time_data(num_rows, 1759113600L);
  std::vector<int64_t> right_end_time_data(num_rows, 1759117199L);
  
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_start_time(right_start_time_data.begin(), right_start_time_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_end_time(right_end_time_data.begin(), right_end_time_data.end());
  
  auto right_eq_table = cudf::table_view({right_imsi->view(), right_hour});
  auto right_cond_table = cudf::table_view({right_start_time, right_end_time});

  auto left_time_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_start_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto right_end_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  
  auto ge_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, left_time_ref, right_start_ref);
  auto le_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, left_time_ref, right_end_ref);
  auto and_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);

  auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(
    left_eq_table, right_eq_table, left_cond_table, right_cond_table, and_expr);

  // Semi join returns unique left indices, so should be 1
  EXPECT_EQ(1, result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleConditionalLeftAntiJoin)
{
  // Anti join needs to check all potential matches to confirm no matches exist
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  cudf::test::fixed_width_column_wrapper<int64_t> left_time({1759115400L});
  
  auto left_eq_table = cudf::table_view({left_imsi->view(), left_hour});
  auto left_cond_table = cudf::table_view({left_time});

  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  std::vector<int64_t> right_start_time_data(num_rows, 1759113600L);
  std::vector<int64_t> right_end_time_data(num_rows, 1759117199L);
  
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_start_time(right_start_time_data.begin(), right_start_time_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_end_time(right_end_time_data.begin(), right_end_time_data.end());
  
  auto right_eq_table = cudf::table_view({right_imsi->view(), right_hour});
  auto right_cond_table = cudf::table_view({right_start_time, right_end_time});

  auto left_time_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_start_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto right_end_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  
  auto ge_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, left_time_ref, right_start_ref);
  auto le_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, left_time_ref, right_end_ref);
  auto and_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);

  auto result = spark_rapids_jni::mixed_sort_merge_left_anti_join(
    left_eq_table, right_eq_table, left_cond_table, right_cond_table, and_expr);

  // Anti join should return 0 (left row matches)
  EXPECT_EQ(0, result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleEqualityOnlyLeftJoin)
{
  // Equality-only join explosion: 1 left row matches 25M+ right rows
  // No conditional filtering, so ALL matches are in the result
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  // Create left table (1 row)
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  
  auto left_table = cudf::table_view({left_imsi->view(), left_hour});

  // Create right table with many matching keys
  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  
  auto right_table = cudf::table_view({right_imsi->view(), right_hour});

  auto [left_result, right_result] = spark_rapids_jni::sort_merge_left_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // All right rows should match with the single left row
  EXPECT_EQ(num_rows, left_result->size());
  EXPECT_EQ(num_rows, right_result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleEqualityOnlyLeftSemiJoin)
{
  // Semi join only needs to find that a match exists, not enumerate all matches
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  
  auto left_table = cudf::table_view({left_imsi->view(), left_hour});

  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  
  auto right_table = cudf::table_view({right_imsi->view(), right_hour});

  auto result = spark_rapids_jni::sort_merge_left_semi_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Semi join returns unique left indices, so should be 1
  EXPECT_EQ(1, result->size());
}

TEST_F(MixedSortMergeJoinTest, LargeScaleEqualityOnlyLeftAntiJoin)
{
  // Anti join needs to confirm no matches exist among millions of candidates
  auto stream = cudf::get_default_stream();
  
  constexpr cudf::size_type num_rows = 25445819;
  
  std::vector<std::string> left_imsi_data{"310260250298289"};
  auto left_imsi = make_strings_column_from_host_vector(left_imsi_data, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> left_hour({0});
  
  auto left_table = cudf::table_view({left_imsi->view(), left_hour});

  std::vector<std::string> right_imsi_data(num_rows, "310260250298289");
  auto right_imsi = make_strings_column_from_host_vector(right_imsi_data, stream);
  
  std::vector<int32_t> right_hour_data(num_rows, 0);
  cudf::test::fixed_width_column_wrapper<int32_t> right_hour(right_hour_data.begin(), right_hour_data.end());
  
  auto right_table = cudf::table_view({right_imsi->view(), right_hour});

  auto result = spark_rapids_jni::sort_merge_left_anti_join(
    left_table, right_table, cudf::sorted::NO, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Anti join should return 0 (left row matches)
  EXPECT_EQ(0, result->size());
}

