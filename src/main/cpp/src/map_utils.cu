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

#include "map_utils.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <vector>

namespace spark_rapids_jni {

namespace {
// Extract a bool from a BOOL8 reduce scalar; returns false if scalar is null (empty-column case).
// Each of is_valid(stream) and value(stream) performs a blocking device→host copy.
// Two syncs per call; acceptable because the scalar is small and syncs are rare.
auto bool_scalar_value(std::unique_ptr<cudf::scalar> const& s, rmm::cuda_stream_view stream) -> bool
{
  return s->is_valid(stream) &&
         static_cast<cudf::numeric_scalar<bool> const*>(s.get())->value(stream);
}

// Reduce a BOOL8 column to a single bool via ANY aggregation.
// Returns false when the column is empty (scalar is null).
auto reduce_any(cudf::column_view const& col,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr) -> bool
{
  auto s = cudf::reduce(col,
                        *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                        cudf::data_type{cudf::type_id::BOOL8},
                        stream,
                        mr);
  return bool_scalar_value(s, stream);
}

// Error message used by both fast-path and slow-path null-key checks.
constexpr char kNullKeyError[] = "Cannot use null as map key.";
}  // namespace

std::unique_ptr<cudf::column> map_from_entries(cudf::column_view const& input,
                                               bool throw_on_null_key,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "map_from_entries: input must be a LIST column");

  if (input.size() == 0) { return cudf::empty_like(input); }

  auto const lists_cv = cudf::lists_column_view(input);
  auto const structs  = lists_cv.child();
  CUDF_EXPECTS(structs.type().id() == cudf::type_id::STRUCT,
               "map_from_entries: list child must be a STRUCT column");
  CUDF_EXPECTS(structs.num_children() == 2,
               "map_from_entries: struct must have exactly 2 children (KEY, VALUE)");

  // Use the default resource for all temporaries; only the output column uses mr.
  auto const temp_mr = cudf::get_current_device_resource_ref();

  // Per-row flag: does row i contain any null struct entry?
  // contains_nulls returns BOOL8 of size input.size(); null for outer-null rows.
  auto has_null_entry = cudf::lists::contains_nulls(lists_cv, stream, temp_mr);

  // Slice-correct offsets span: covers exactly the visible rows of this (potentially sliced)
  // input. offsets_begin() accounts for input.offset(), so the span is valid for sliced columns.
  auto const offsets_span = cudf::device_span<cudf::size_type const>(
    lists_cv.offsets_begin(), static_cast<std::size_t>(input.size()) + 1);

  // Fast path: no null struct entries anywhere — simple global null-key check.
  bool const any_null_entry = reduce_any(*has_null_entry, stream, temp_mr);

  if (!any_null_entry) {
    // All struct entries are valid. Use segmented_reduce over the slice-correct offsets to
    // check for null keys — structs.child(0) is the un-sliced underlying column, so
    // keys.null_count() would count nulls outside the visible slice range.
    auto const keys = structs.child(0);
    if (throw_on_null_key && keys.nullable() && keys.null_count() > 0) {
      auto key_is_null_fp = cudf::is_null(keys, stream, temp_mr);
      auto row_hnk_fp =
        cudf::segmented_reduce(*key_is_null_fp,
                               offsets_span,
                               *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::EXCLUDE,
                               stream,
                               temp_mr);
      // Outer-null rows must not trigger a throw; AND with is_valid(input) before reducing.
      auto input_is_valid = cudf::is_valid(input, stream, temp_mr);
      auto row_throw      = cudf::binary_operation(*row_hnk_fp,
                                              *input_is_valid,
                                              cudf::binary_operator::BITWISE_AND,
                                              cudf::data_type{cudf::type_id::BOOL8},
                                              stream,
                                              temp_mr);
      if (reduce_any(*row_throw, stream, temp_mr)) { throw cudf::logic_error(kNullKeyError); }
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
  auto const keys      = structs.child(0);
  auto key_is_null     = cudf::is_null(keys, stream, temp_mr);
  auto struct_is_valid = cudf::is_valid(structs, stream, temp_mr);  // Single kernel vs. is_null+NOT
  auto null_key_in_valid = cudf::binary_operation(*key_is_null,
                                                  *struct_is_valid,
                                                  cudf::binary_operator::BITWISE_AND,
                                                  cudf::data_type{cudf::type_id::BOOL8},
                                                  stream,
                                                  temp_mr);

  // Reduce per-list: does this row contain any entry where null_key_in_valid = true?
  auto row_has_null_key =
    cudf::segmented_reduce(*null_key_in_valid,
                           offsets_span,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::EXCLUDE,
                           stream,
                           temp_mr);

  // NOT(has_null_entry): true for rows whose struct entries are all valid.
  // Computed once and reused for both the throw check and the output null mask.
  // bools_to_mask treats NULL inputs as false, so outer-null rows (where has_null_entry
  // is null) are correctly masked to null in the output.
  auto no_null_entry =
    cudf::unary_operation(*has_null_entry, cudf::unary_operator::NOT, stream, temp_mr);

  // Throw only when: row has no null struct entry AND row has a null key in a valid struct.
  // For rows with null struct entries (has_null_entry = true), the whole output row is masked
  // to null below, so their null keys are irrelevant — no exception is thrown for them.
  if (throw_on_null_key) {
    // NULL AND anything = NULL; reduce(any) skips nulls, so null rows are safely ignored.
    auto should_throw = cudf::binary_operation(*no_null_entry,
                                               *row_has_null_key,
                                               cudf::binary_operator::BITWISE_AND,
                                               cudf::data_type{cudf::type_id::BOOL8},
                                               stream,
                                               temp_mr);
    if (reduce_any(*should_throw, stream, temp_mr)) { throw cudf::logic_error(kNullKeyError); }
  }

  // Build null mask from no_null_entry: false/null → bit=0 (null), true → bit=1 (valid).
  auto [entry_mask_uptr, entry_nc] = cudf::bools_to_mask(*no_null_entry, stream, temp_mr);

  // Since any_null_entry == true, no_null_entry has at least one false/null bit, so entry_nc > 0.
  CUDF_EXPECTS(
    entry_nc > 0,
    "map_from_entries: reached slow path with entry_nc == 0 — fast path invariant broken");

  // Combine input's existing null mask with entry_mask via the offset-aware raw-bitmask overload
  // so input.null_mask() is consumed from input.offset() while entry_mask starts at bit 0.
  // The resulting combined_mask_buf is always bit-0-aligned for input.size() rows.
  auto const entry_mask_ptr = static_cast<cudf::bitmask_type const*>(entry_mask_uptr->data());
  std::vector<cudf::bitmask_type const*> masks;
  std::vector<cudf::size_type> begin_bits;
  if (input.nullable()) {
    masks.push_back(input.null_mask());
    begin_bits.push_back(input.offset());
  }
  masks.push_back(entry_mask_ptr);
  begin_bits.push_back(0);
  auto [combined_mask_buf, combined_nc] =
    cudf::bitmask_and(cudf::host_span<cudf::bitmask_type const* const>{masks.data(), masks.size()},
                      cudf::host_span<cudf::size_type const>{begin_bits.data(), begin_bits.size()},
                      input.size(),
                      stream,
                      temp_mr);
  auto const* mask_ptr = static_cast<cudf::bitmask_type const*>(combined_mask_buf.data());

  // combined_mask_buf stores valid bits starting at bit 0 for input.size() rows, so
  // result_view must use offset=0. For sliced input, slice the offsets child to start at
  // input.offset() so list row ranges remain correct with the new zero offset.
  auto const raw_offsets    = input.child(cudf::lists_column_view::offsets_column_index);
  auto const sliced_offsets = cudf::column_view(raw_offsets.type(),
                                                input.size() + 1,
                                                raw_offsets.head<void>(),
                                                nullptr,
                                                0,
                                                raw_offsets.offset() + input.offset());

  auto const result_view =
    cudf::column_view(input.type(),
                      input.size(),
                      nullptr,
                      mask_ptr,
                      combined_nc,
                      /*offset=*/0,
                      {sliced_offsets, input.child(cudf::lists_column_view::child_column_index)});
  // purge_nonempty_nulls adjusts list offsets so each null outer row has an empty span,
  // satisfying cudf's invariant that null rows in nested columns are empty.
  return cudf::purge_nonempty_nulls(result_view, stream, mr);
}

}  // namespace spark_rapids_jni
