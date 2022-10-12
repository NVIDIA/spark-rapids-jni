/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "zorder.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> interleave_bits(
  cudf::table_view const& tbl,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) {

  auto num_columns = tbl.num_columns();
  CUDF_EXPECTS(num_columns > 0, "The input table must have at least one column.");
  CUDF_EXPECTS(is_fixed_width(tbl.begin()->type()), "Only fixed width columns can be used");

  auto const type_id = tbl.begin()->type().id();
  auto const data_type_size = cudf::size_of(tbl.begin()->type());
  CUDF_EXPECTS(
    std::all_of(tbl.begin(),
                tbl.end(),
                [type_id](cudf::column_view const& col) { return col.type().id() == type_id; }),
    "All columns of the input table must be the same type.");

  // Because the input is a table we know that they all have the same length.
  auto num_rows = tbl.num_rows();

  const cudf::size_type max_bytes_allowed = std::numeric_limits<cudf::size_type>::max();

  int64_t total_output_size = static_cast<int64_t>(num_rows) * data_type_size * num_columns;
  CUDF_EXPECTS (total_output_size <= max_bytes_allowed, "Input is too large to process");

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
     data_type_size,
     input = *input_dv] __device__ (cudf::size_type ret_idx) {
       // The most significant byte needs to come from the most significant column, so we switch the order of the output
       // bytes to match that
       cudf::size_type const flipped_start_byte_index = (ret_idx / num_columns) * num_columns;
       cudf::size_type const flipped_ret_idx = flipped_start_byte_index + (num_columns - 1 - (ret_idx - flipped_start_byte_index));

       uint8_t ret_byte = 0;
       for (cudf::size_type output_bit_offset = 7; output_bit_offset >= 0; output_bit_offset--) {
         // The index (in bits) of the output bit we are computing right now
         int64_t const output_bit_index = flipped_ret_idx * 8L + output_bit_offset;

         // The most significant bit should come from the most significant column, but 0 is
         // our most significant column, so switch the order of the columns.
         cudf::size_type const column_idx = num_columns - 1 - (output_bit_index % num_columns);
         auto column = input.column(column_idx);

         // Also we need to convert the endian byte order when we read the bytes.
         int64_t const bit_index_within_column = output_bit_index / num_columns;
         cudf::size_type const little_endian_read_byte_index = bit_index_within_column / 8;
         cudf::size_type const read_bit_offset = bit_index_within_column % 8;
         cudf::size_type const input_row_number = little_endian_read_byte_index / data_type_size;
         cudf::size_type const start_row_byte_index = input_row_number * data_type_size;
         cudf::size_type const read_byte_index = start_row_byte_index + (data_type_size - 1 - (little_endian_read_byte_index - start_row_byte_index));

         uint32_t const byte_data = column.is_valid(input_row_number) ? column.data<uint8_t>()[read_byte_index] : 0;
         uint32_t const tmp = ((byte_data >> read_bit_offset) & 1) << output_bit_offset;
         ret_byte = static_cast<uint8_t>(ret_byte | tmp);
       }
       col.data<uint8_t>()[ret_idx] = ret_byte;
     });
  
  auto offset_begin = thrust::make_constant_iterator(data_type_size * num_columns);
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
