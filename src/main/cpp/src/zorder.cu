/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "zorder.hpp"

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> interleave_bits(
  cudf::table_view const& tbl,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) {

  auto num_columns = tbl.num_columns();
  CUDF_EXPECTS(num_columns > 0, "The input table must have at least one column.");


  CUDF_EXPECTS(is_fixed_width(tbl.begin()->type()), "Only fixed width columns can be used");

  auto const type_id = tbl.begin()->type().id();
  auto const base_size = cudf::size_of(tbl.begin()->type());
  CUDF_EXPECTS(
    std::all_of(tbl.begin(),
                tbl.end(),
                [type_id](cudf::column_view const& col) { return col.type().id() == type_id; }),
    "All columns of the input table must be the same type.");

  // Because the input is a table we know that they all have the same length.
  auto num_rows = tbl.num_rows();

  const cudf::size_type max_bytes_allowed = std::numeric_limits<cudf::size_type>::max();

  int64_t tmp_column_data_size = num_rows * base_size;
  if (tmp_column_data_size > max_bytes_allowed) {
    throw std::logic_error("Input is too large to process");
  }

  int64_t total_output_size = tmp_column_data_size * num_columns;
  if (total_output_size > max_bytes_allowed) {
    throw std::logic_error("Input is too large to process");
  }

  cudf::size_type output_size = static_cast<cudf::size_type>(total_output_size);

  auto input_dv = cudf::table_device_view::create(tbl, stream);

  auto output_data_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT8}, output_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto output_dv_ptr = cudf::mutable_column_device_view::create(*output_data_col, stream);

  thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::make_counting_iterator<cudf::size_type>(0),
          output_size,
          [col = *output_dv_ptr, 
           num_columns,
           base_size,
           input = *input_dv] __device__ (cudf::size_type ret_idx) {
             // Flip the "endianness" of the output based off of the number of columns
             cudf::size_type flipped_start_byte_index = (ret_idx / num_columns) * num_columns;
             cudf::size_type flipped_ret_idx = flipped_start_byte_index + (num_columns - 1 - (ret_idx - flipped_start_byte_index));

             // Start with the highest bit for output
             cudf::size_type ret_bit = 7;
             uint8_t ret_byte = 0;
             while (ret_bit >= 0) {
               int64_t total_output_bit = flipped_ret_idx * 8L + ret_bit;

               // The order of the columns needs to be [0 to N] for the highest bit, so flip them too
               cudf::size_type column_idx = num_columns - 1 - (total_output_bit % num_columns);
               auto column = input.column(column_idx);

               // Also we need tp convert the endian byte order when we read the bytes.
               int64_t bit_within_column = total_output_bit / num_columns;
               cudf::size_type le_read_byte_index = bit_within_column / 8;
               cudf::size_type bit_offset = bit_within_column % 8;
               cudf::size_type input_row_number = le_read_byte_index / base_size;
               cudf::size_type start_item_byte_index = input_row_number * base_size;
               cudf::size_type read_byte_index = start_item_byte_index + (base_size - 1 - (le_read_byte_index - start_item_byte_index));

               uint32_t byte_data = column.data<uint8_t>()[read_byte_index];
               bool is_valid = (column.null_mask() ? cudf::bit_is_set(column.null_mask(), input_row_number + column.offset()): true);
               if (!is_valid) {
                 byte_data = 0;
               }
               uint32_t tmp = ((byte_data >> bit_offset) & 1) << ret_bit;
               ret_byte = static_cast<uint8_t>(ret_byte | tmp);

               ret_bit -= 1;
             }
             col.data<uint8_t>()[ret_idx] = ret_byte;
           });
  
  auto offset_begin = thrust::make_constant_iterator(base_size * num_columns);
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offset_begin, offset_begin + num_rows, stream, mr);

  return cudf::make_lists_column(num_rows,
          std::move(offsets_column),
          std::move(output_data_col),
          0,
          rmm::device_buffer(),
          stream,
          mr);
}

} // namespace spark_rapids_jni
