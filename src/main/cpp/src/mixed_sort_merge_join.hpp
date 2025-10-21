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

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace spark_rapids_jni {

/**
 * @file
 * @brief Sort-merge join implementations with optional conditional filtering
 *
 * This file provides both equality-only and conditional (mixed) sort-merge joins.
 * The equality-only joins are more efficient when no conditional filtering is needed.
 */

// =============================================================================
// EQUALITY-ONLY SORT-MERGE JOINS (Non-conditional)
// =============================================================================

/**
 * @brief Returns a pair of row index vectors corresponding to an inner join
 * on equality keys only.
 *
 * This performs a sort-merge inner join without any conditional filtering.
 * Only returns rows that have matches in both tables.
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param is_left_sorted Enum to indicate if left table is pre-sorted
 * @param is_right_sorted Enum to indicate if right table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] for the inner join result
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      cudf::sorted is_left_sorted       = cudf::sorted::NO,
                      cudf::sorted is_right_sorted      = cudf::sorted::NO,
                      cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                      rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a left outer join
 * on equality keys only.
 *
 * This performs a sort-merge left outer join without any conditional filtering.
 * All rows from the left table are preserved. Rows with no match in the right
 * table will have their right index set to an out-of-bounds value
 * (right_keys.num_rows()).
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param is_left_sorted Enum to indicate if left table is pre-sorted
 * @param is_right_sorted Enum to indicate if right table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] for the left join result
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_left_join(cudf::table_view const& left_keys,
                     cudf::table_view const& right_keys,
                     cudf::sorted is_left_sorted       = cudf::sorted::NO,
                     cudf::sorted is_right_sorted      = cudf::sorted::NO,
                     cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to a left semi join on equality keys only.
 *
 * This performs a sort-merge left semi join without any conditional filtering.
 * Returns indices of left table rows that have at least one match in the right table.
 * Each left row appears at most once in the result.
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param is_left_sorted Enum to indicate if left table is pre-sorted
 * @param is_right_sorted Enum to indicate if right table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A vector of indices from the left table that have matches in the right table
 */
rmm::device_uvector<cudf::size_type> sort_merge_left_semi_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to a left anti join on equality keys only.
 *
 * This performs a sort-merge left anti join without any conditional filtering.
 * Returns indices of left table rows that have NO matches in the right table.
 * Each left row appears at most once in the result.
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param is_left_sorted Enum to indicate if left table is pre-sorted
 * @param is_right_sorted Enum to indicate if right table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A vector of indices from the left table that do not have matches in the right table
 */
rmm::device_uvector<cudf::size_type> sort_merge_left_anti_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// =============================================================================
// CONDITIONAL (MIXED) SORT-MERGE JOINS
// =============================================================================

/**
 * @brief Returns a pair of row index vectors corresponding to an inner join
 * where rows match both on equality keys and conditional expression.
 *
 * This function performs a sort-merge join on the equality keys, then filters
 * the results using the conditional expression evaluated on the conditional tables.
 *
 * The first returned vector contains the row indices from the left table that
 * have a match in the right table (in unspecified order). The corresponding
 * values in the second returned vector are the matched row indices from the
 * right table.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * is_right_sorted: NO
 * is_left_sorted: NO
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional
 * do not match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional
 * do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The AST expression condition on which to join
 * @param is_left_sorted Enum to indicate if left equality table is pre-sorted
 * @param is_right_sorted Enum to indicate if right equality table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to
 * construct the result of performing a mixed sort-merge inner join
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
mixed_sort_merge_inner_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a left join
 * where rows match both on equality keys and conditional expression.
 *
 * This function performs a sort-merge left join on the equality keys, then filters
 * matching pairs using the conditional expression. Rows from the left table that
 * have no matches after filtering will have a null (out-of-bounds) right index.
 *
 * The first returned vector contains the row indices from the left table.
 * The corresponding value in the second returned vector is either (1) the row
 * index of the matched row from the right table, or (2) an unspecified
 * out-of-bounds value for non-matching rows.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{0, 1, 2}, {3, 0, 3}}
 * @endcode
 *
 * Note: Unmatched left rows will have their right index set to `right_equality.num_rows()`.
 * This out-of-bounds value can be used with `cudf::gather` and
 * `cudf::out_of_bounds_policy::NULLIFY` to produce null values for unmatched rows.
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional
 * do not match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional
 * do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The AST expression condition on which to join
 * @param is_left_sorted Enum to indicate if left equality table is pre-sorted
 * @param is_right_sorted Enum to indicate if right equality table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to
 * construct the result of performing a mixed sort-merge left join
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
mixed_sort_merge_left_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to a left semi join where rows
 * match both on equality keys and conditional expression.
 *
 * This function performs a sort-merge join on the equality keys, then filters
 * to include only left rows that have at least one match satisfying the
 * conditional expression. Each left row appears at most once in the result.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {1}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional
 * do not match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional
 * do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The AST expression condition on which to join
 * @param is_left_sorted Enum to indicate if left equality table is pre-sorted
 * @param is_right_sorted Enum to indicate if right equality table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A vector of indices from the left table that have matches in the right table
 */
rmm::device_uvector<cudf::size_type> mixed_sort_merge_left_semi_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to a left anti join where rows
 * do NOT match on both equality keys and conditional expression.
 *
 * This function performs a sort-merge join on the equality keys, then filters
 * to include only left rows that have NO matches satisfying the conditional
 * expression. Each left row appears at most once in the result.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {0, 2}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional
 * do not match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional
 * do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The AST expression condition on which to join
 * @param is_left_sorted Enum to indicate if left equality table is pre-sorted
 * @param is_right_sorted Enum to indicate if right equality table is pre-sorted
 * @param compare_nulls Whether or not null values in equality keys join to each other
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A vector of indices from the left table that do not have matches in the right table
 */
rmm::device_uvector<cudf::size_type> mixed_sort_merge_left_anti_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted       = cudf::sorted::NO,
  cudf::sorted is_right_sorted      = cudf::sorted::NO,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
