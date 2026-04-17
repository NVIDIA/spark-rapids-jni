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

#include "map_utils.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> map_from_entries(cudf::column_view const& input,
                                               bool throw_on_null_key,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "map_from_entries: input must be a LIST column");

  if (input.size() == 0) { return cudf::make_empty_column(input.type()); }

  auto const lists_cv = cudf::lists_column_view(input);
  auto const structs  = lists_cv.child();
  CUDF_EXPECTS(structs.type().id() == cudf::type_id::STRUCT,
               "map_from_entries: list child must be a STRUCT column");
  CUDF_EXPECTS(structs.num_children() >= 1,
               "map_from_entries: struct must have at least one child column (KEY)");

  // Step 1: Per-row flag — does row i contain any null struct entry?
  // contains_nulls returns BOOL8, size = input.size().
  // A null outer row itself yields null in has_null_entry; bools_to_mask treats null as false
  // (bit=0), so outer-null rows are correctly kept null in the final mask.
  auto has_null_entry = cudf::lists::contains_nulls(lists_cv, stream, mr);

  // Fast path: no null struct entries anywhere — simple global null-key check.
  auto any_null_entry_scalar = cudf::reduce(*has_null_entry,
                                            *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                            cudf::data_type{cudf::type_id::BOOL8},
                                            stream,
                                            mr);
  bool const any_null_entry =
    any_null_entry_scalar->is_valid(stream) &&
    static_cast<cudf::numeric_scalar<bool>*>(any_null_entry_scalar.get())->value(stream);

  if (!any_null_entry) {
    // All struct entries are valid.  Any null key in the flat key column is a real null key.
    auto const keys = structs.child(0);
    if (throw_on_null_key && keys.null_count() > 0) {
      throw cudf::logic_error("Cannot use null as map key.");
    }
    return std::make_unique<cudf::column>(input, stream, mr);
  }

  // Slow path: at least one row contains a null struct entry.
  //
  // CPU semantics: if a row's array has any null struct entry the entire output row is null,
  // regardless of whether another entry in that row also has a null key.  We must therefore
  // throw "Cannot use null as map key" only for rows that satisfy BOTH:
  //   (a) the row has NO null struct entry  (has_null_entry = false), AND
  //   (b) at least one entry's key is null inside a valid (non-null) struct.
  //
  // Per-entry boolean: null_key_in_valid[j] = key_is_null[j] AND struct_is_valid[j]
  auto const keys     = structs.child(0);
  auto key_is_null    = cudf::is_null(keys, stream, mr);     // flat BOOL8
  auto struct_is_null = cudf::is_null(structs, stream, mr);  // flat BOOL8
  auto struct_is_valid =
    cudf::unary_operation(*struct_is_null, cudf::unary_operator::NOT, stream, mr);  // flat BOOL8
  auto null_key_in_valid = cudf::binary_operation(*key_is_null,
                                                  *struct_is_valid,
                                                  cudf::binary_operator::BITWISE_AND,
                                                  cudf::data_type{cudf::type_id::BOOL8},
                                                  stream,
                                                  mr);

  // Reduce per-list: does this row contain any entry where null_key_in_valid = true?
  // segmented_reduce(max) over the flat boolean values using the list offsets as boundaries.
  // is_null() always returns a fully-valid boolean column, so null_key_in_valid has no nulls;
  // EXCLUDE null_policy only affects empty-list rows (which yield a null result, safely
  // treated as false in the AND below).
  auto const offsets_col  = lists_cv.offsets();
  auto const offsets_span = cudf::device_span<cudf::size_type const>(
    offsets_col.data<cudf::size_type>(), offsets_col.size());

  auto row_has_null_key =
    cudf::segmented_reduce(*null_key_in_valid,
                           offsets_span,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::EXCLUDE,
                           stream,
                           mr);

  // Throw only when: row has no null struct entry AND row has a null key in a valid struct.
  // For rows with null struct entries (has_null_entry = true), the whole output row is masked
  // to null below, so their null keys are irrelevant — no exception should be thrown for them.
  if (throw_on_null_key) {
    auto no_null_entry =
      cudf::unary_operation(*has_null_entry, cudf::unary_operator::NOT, stream, mr);
    // NULL AND anything = NULL; reduce(any) skips nulls, so null rows are safely ignored.
    auto should_throw     = cudf::binary_operation(*no_null_entry,
                                               *row_has_null_key,
                                               cudf::binary_operator::BITWISE_AND,
                                               cudf::data_type{cudf::type_id::BOOL8},
                                               stream,
                                               mr);
    auto any_throw_scalar = cudf::reduce(*should_throw,
                                         *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);
    bool const any_throw =
      any_throw_scalar->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool>*>(any_throw_scalar.get())->value(stream);
    if (any_throw) { throw cudf::logic_error("Cannot use null as map key."); }
  }

  // Null-mask rows that contain null struct entries, then purge their child data.
  // cudf::make_default_constructed_scalar does not support LIST type, so we build the
  // null mask directly and use purge_nonempty_nulls to clear child data for null rows
  // (cudf requires null LIST rows to have empty offset spans).
  //   keep[i] = NOT has_null_entry[i]:  false/null → bit=0 (null), true → bit=1 (valid)
  // bools_to_mask treats NULL inputs as false, so already-null outer rows (where
  // has_null_entry is null) are also masked to null.
  auto keep_valid_col =
    cudf::unary_operation(*has_null_entry, cudf::unary_operator::NOT, stream, mr);
  auto [entry_mask_uptr, entry_nc] = cudf::bools_to_mask(*keep_valid_col, stream, mr);

  auto result = std::make_unique<cudf::column>(input, stream, mr);
  if (entry_nc > 0) {
    // Build a dummy column carrying entry_mask as its null mask so we can use
    // bitmask_and(table_view) to safely AND with the input's existing null mask.
    // Use LIST type (compound) so the column_view constructor accepts nullptr data.
    auto const entry_mask_ptr = static_cast<cudf::bitmask_type const*>(entry_mask_uptr->data());
    auto const dummy          = cudf::column_view(cudf::data_type{cudf::type_id::LIST},
                                         input.size(),
                                         nullptr,
                                         entry_mask_ptr,
                                         static_cast<cudf::size_type>(entry_nc));
    // bitmask_and treats non-nullable columns as all-valid, handling both nullable and
    // non-nullable inputs correctly.
    auto [combined_mask, combined_nc] =
      cudf::bitmask_and(cudf::table_view{std::vector<cudf::column_view>{input, dummy}}, stream, mr);
    result->set_null_mask(std::move(combined_mask), combined_nc);
  }
  // purge_nonempty_nulls adjusts list offsets so each null outer row has an empty span,
  // satisfying cudf's invariant that null rows in nested columns are empty.
  return cudf::purge_nonempty_nulls(result->view(), stream, mr);
}

}  // namespace spark_rapids_jni
