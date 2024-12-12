/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <numeric>

#include "shuffle_split.hpp"
//#include "db_test_workspace.cuh"
//#include "db_test.cuh"

namespace spark_rapids_jni {

using namespace cudf;

namespace {

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr std::size_t split_align = 64;

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a separate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               const int* _offsets,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               size_type _column_offset)
    : type(_type),
      offsets(_offsets),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  cudf::type_id type;
  const int* offsets;        // a pointer to device memory offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
  size_type column_offset;   // offset in the case of a sliced column
};

enum class buffer_type {
  VALIDITY,
  OFFSETS,
  DATA
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  /*
  // constant across all copy commands for this buffer
  std::size_t buf_size;  // total size of buffer in bytes
  int num_elements;      // # of elements to be copied
  int element_size;      // size of each element in bytes
  int num_rows;  // # of rows to be copied(which may be different from num_elements in the case of
                 // validity or offset buffers)

  int src_element_index;   // element index to start reading from from my associated source buffer
  // std::size_t dst_offset;  // my offset into the per-partition allocation
  //int value_shift;         // amount to shift values down by (for offset buffers)
  //int bit_shift;           // # of bits to shift right by (for validity buffers)
  //size_type valid_count;   // validity count for this block of work
  buffer_type type;

  int src_buf_index;       // source buffer index
  int dst_buf_index;       // destination buffer index
  */
  
  std::size_t buf_size;    // total size of buffer
  buffer_type type;

  int src_buf_index;
  std::size_t src_offset;
  std::size_t dst_offset;
};

/**
 * @brief Copy a single buffer of column data, shifting values (for offset columns),
 * and validity (for validity buffers) as necessary.
 *
 * Copies a single partition of a source column buffer to a destination buffer. Shifts
 * element values by value_shift in the case of a buffer of offsets (value_shift will
 * only ever be > 0 in that case).  Shifts elements bitwise by bit_shift in the case of
 * a validity buffer (bif_shift will only ever be > 0 in that case).  This function assumes
 * value_shift and bit_shift will never be > 0 at the same time.
 *
 * This function expects:
 * - src may be a misaligned address
 * - dst must be an aligned address
 *
 * This function always does the ALU work related to value_shift and bit_shift because it is
 * entirely memory-bandwidth bound.
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param num_elements Number of elements to copy
 * @param element_size Size of each element in bytes
 * @param src_element_index Element index to start copying at
 * @param stride Size of the kernel block
 * @param value_shift Shift incoming 4-byte offset values down by this amount
 * @param bit_shift Shift incoming data right by this many bits
 * @param num_rows Number of rows being copied
 * @param valid_count Optional pointer to a value to store count of set bits
 */
template <int block_size>
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t const* __restrict__ src,
                            int t,
                            std::size_t num_elements,
                            std::size_t element_size,
                            std::size_t src_element_index,
                            uint32_t stride,
                            int value_shift,
                            int bit_shift,
                            std::size_t num_rows,
                            size_type* valid_count)
{
  src += (src_element_index * element_size);

  size_type thread_valid_count = 0;

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  std::size_t const num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  uint32_t const ofs = reinterpret_cast<uintptr_t>(src) % 4;
  std::size_t pos    = t * 16;
  stride *= 16;
  while (pos + 20 <= num_bytes) {
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t*>((src + pos) - ofs);
    uint4 v              = uint4{in32[0], in32[1], in32[2], in32[3]};
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs * 8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs * 8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs * 8 + bit_shift);
      v.w = __funnelshift_r(v.w, in32[4], ofs * 8 + bit_shift);
    }
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos / 16] = v;
    if (valid_count) {
      thread_valid_count += (__popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w));
    }
    pos += stride;
  }

  // copy trailing bytes
  if (t == 0) {
    std::size_t remainder;
    if (num_bytes < 16) {
      remainder = num_bytes;
    } else {
      std::size_t const last_bracket = (num_bytes / 16) * 16;
      remainder                      = num_bytes - last_bracket;
      if (remainder < 4) {
        // we had less than 20 bytes for the last possible 16 byte copy, so copy 16 + the extra
        remainder += 16;
      }
    }

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and
    // alignment must be a multiple of 4. value shifting and bit shifting are mutually exclusive
    // and will never both be true at the same time.
    if (value_shift || bit_shift) {
      std::size_t idx = (num_bytes - remainder) / 4;
      uint32_t v = remainder > 0 ? (reinterpret_cast<uint32_t const*>(src)[idx] - value_shift) : 0;

      constexpr size_type rows_per_element = 32;
      auto const have_trailing_bits = ((num_elements * rows_per_element) - num_rows) < bit_shift;
      while (remainder) {
        // if we're at the very last word of a validity copy, we do not always need to read the next
        // word to get the final trailing bits.
        auto const read_trailing_bits = bit_shift > 0 && remainder == 4 && have_trailing_bits;
        uint32_t const next           = (read_trailing_bits || remainder > 4)
                                          ? (reinterpret_cast<uint32_t const*>(src)[idx + 1] - value_shift)
                                          : 0;

        uint32_t const val = (v >> bit_shift) | (next << (32 - bit_shift));
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint32_t*>(dst)[idx] = val;
        v                                     = next;
        idx++;
        remainder -= 4;
      }
    } else {
      while (remainder) {
        std::size_t const idx = num_bytes - remainder--;
        uint32_t const val    = reinterpret_cast<uint8_t const*>(src)[idx];
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint8_t*>(dst)[idx] = val;
      }
    }
  }

  if (valid_count) {
    if (num_bytes == 0) {
      if (!t) { *valid_count = 0; }
    } else {
      using BlockReduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      size_type block_valid_count{BlockReduce(temp_storage).Sum(thread_valid_count)};
      if (!t) {
        // we may have copied more bits than there are actual rows in the output.
        // so we need to subtract off the count of any bits that shouldn't have been
        // considered during the copy step.
        std::size_t const max_row    = (num_bytes * 8);
        std::size_t const slack_bits = max_row > num_rows ? max_row - num_rows : 0;
        auto const slack_mask        = set_most_significant_bits(slack_bits);
        if (slack_mask > 0) {
          uint32_t const last_word = reinterpret_cast<uint32_t*>(dst + (num_bytes - 4))[0];
          block_valid_count -= __popc(last_word & slack_mask);
        }
        *valid_count = block_valid_count;
      }
    }
  }
}

// The block of functions below are all related:
//
// compute_offset_stack_size()
// setup_src_buf_data()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.  For example, setup_src_buf_info() produces a series of information
// structs that must appear in the same order that setup_src_buf_data() produces
// buffers.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns.
 */
template <typename InputIter>
std::size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (col.nullable() ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
}

/**
 * @brief Retrieve all buffers for a range of source columns.
 *
 * Retrieve the individual buffers that make up a range of input columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param out_buf Iterator into output buffer infos
 *
 * @returns next output buffer iterator
 */
template <typename InputIter, typename OutputIter>
OutputIter setup_src_buf_data(InputIter begin, InputIter end, OutputIter out_buf)
{
  std::for_each(begin, end, [&out_buf](column_view const& col) {
    if (col.nullable()) {
      *out_buf = reinterpret_cast<uint8_t const*>(col.null_mask());
      out_buf++;
    }
    // NOTE: we're always returning the base pointer here.  column-level offset is accounted
    // for later. Also, for some column types (string, list, struct) this pointer will be null
    // because there is no associated data with the root column.
    *out_buf = col.head<uint8_t>();
    out_buf++;

    out_buf = setup_src_buf_data(col.child_begin(), col.child_end(), out_buf);
  });
  return out_buf;
}

/**
 * @brief Count the total number of source buffers, broken down by type (validity, offset, data) 
 * we will be copying from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (data, validity).
 * A string column with validity would be 3 buffers (chars, offsets, validity).
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffer per type for this range of columns
 */
struct src_buf_count {
  size_t validity_buf_count;
  size_t offset_buf_count;
  size_t data_buf_count;
};
struct src_buf_count_add {
  src_buf_count operator()(src_buf_count const& lhs, src_buf_count const& rhs)
  {
    return {lhs.validity_buf_count + rhs.validity_buf_count,
            lhs.offset_buf_count + rhs.offset_buf_count,
            lhs.data_buf_count + rhs.data_buf_count};
  }
};
template <typename InputIter>
src_buf_count count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    auto const has_offsets_child = col.type().id() == cudf::type_id::LIST ||
                                   (col.type().id() == cudf::type_id::STRING && col.num_children() > 0);
    src_buf_count const counts{static_cast<size_t>(col.nullable() ? 1 : 0),
                               static_cast<size_t>(has_offsets_child ? 1 : 0),
                               size_t{1}};
    src_buf_count const child_counts = count_src_bufs(col.child_begin(), col.child_end());
    return src_buf_count_add{}(counts, child_counts);
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), src_buf_count{0, 0, 0}, src_buf_count_add{});
}

/**
 * @brief Computes source buffer information for the copy kernel.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param current Current source buffer info to be written to
 * @param offset_stack_pos Integer representing our current offset nesting depth
 * (how many list or string levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
void setup_source_buf_info(InputIter begin,
                           InputIter end,
                           src_buf_info* head,
                           src_buf_info *& validity_cur,
                           src_buf_info *& offset_cur,
                           src_buf_info *& data_cur,
                           std::vector<int8_t>& flattened_col_has_validity,
                           int& offset_stack_pos,
                           rmm::cuda_stream_view stream,
                           int parent_offset_index = -1,
                           int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  void operator()(column_view const& col,
                  src_buf_info *& validity_cur,
                  src_buf_info *& offset_cur,
                  src_buf_info *& data_cur,
                  std::vector<int8_t>& flattened_col_has_validity,
                  int& offset_stack_pos,
                  int parent_offset_index,
                  int offset_depth,
                  rmm::cuda_stream_view)
  {
    flattened_col_has_validity.push_back(col.nullable());
    if (col.nullable()) {
      add_null_buffer(col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth);
    }

    // info for the data buffer
    *data_cur = src_buf_info(
      col.type().id(), nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
    data_cur++;
    
    offset_stack_pos += offset_depth;
  }

  template <typename T, typename... Args>
  std::enable_if_t<std::is_same_v<T, cudf::dictionary32>, void>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type");
  }

 private:
  void add_null_buffer(column_view const& col,
                       src_buf_info* validity_cur,
                       int offset_stack_pos,
                       int parent_offset_index,
                       int offset_depth)
  {
    // info for the validity buffer
    *validity_cur = src_buf_info(
      type_id::INT32, nullptr, offset_stack_pos, parent_offset_index, true, col.offset());
    validity_cur++;

    offset_stack_pos += offset_depth;
  }
};

template <>
void buf_info_functor::operator()<cudf::string_view>(
  column_view const& col,
  src_buf_info *& validity_cur,
  src_buf_info *& offset_cur,
  src_buf_info *& data_cur,
  std::vector<int8_t>& flattened_col_has_validity,
  int &offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view)
{
  flattened_col_has_validity.push_back(col.nullable());
  if (col.nullable()) {
    add_null_buffer(col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // the way strings are arranged, the strings column itself contains char data, but our child
  // offsets column actually contains our offsets. So our parent_offset_index is actually our child.

  // string columns don't necessarily have children if they are empty
  auto const has_offsets_child = col.num_children() > 0;

  // string columns contain the underlying chars data.
  *data_cur = src_buf_info(type_id::STRING,
                          nullptr,
                          offset_stack_pos,
                          // if I have offsets, it's index will be the current offset buffer, otherwise
                          // it's whatever my parent's was.
                          has_offsets_child ? (offset_cur - head) : parent_offset_index,
                          false,
                          col.offset());
  data_cur++;
  // if I have offsets, I need to include that in the stack size
  offset_stack_pos += has_offsets_child ? offset_depth + 1 : offset_depth;

  if (has_offsets_child) {
    CUDF_EXPECTS(col.num_children() == 1, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    CUDF_EXPECTS(not scv.offsets().nullable(), "Encountered nullable string offsets column");
    *offset_cur = src_buf_info(type_id::INT32,
                               // note: offsets can be null in the case where the string column
                               // has been created with empty_like().
                               scv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                               offset_stack_pos,
                               parent_offset_index,
                               false,
                               col.offset());

    offset_cur++;
    offset_stack_pos += offset_depth;
  }
}

template <>
void buf_info_functor::operator()<cudf::list_view>(
  column_view const& col,
  src_buf_info *& validity_cur,
  src_buf_info *& offset_cur,
  src_buf_info *& data_cur,
  std::vector<int8_t>& flattened_col_has_validity,
  int &offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view stream)
{
  lists_column_view lcv(col);

  flattened_col_has_validity.push_back(col.nullable());
  if (col.nullable()) {
    add_null_buffer(col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *data_cur = src_buf_info(
    type_id::LIST, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  data_cur++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  *offset_cur        = src_buf_info(type_id::INT32,
                                    // note: offsets can be null in the case where the lists column
                                    // has been created with empty_like().
                                    lcv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                                    offset_stack_pos,
                                    parent_offset_index,
                                    false,
                                    col.offset());
  // since we are crossing an offset boundary, calculate our new depth and parent offset index.  
  parent_offset_index = offset_cur - head;
  offset_cur++;
  offset_stack_pos += offset_depth;
  offset_depth++;

  setup_source_buf_info(col.child_begin() + 1,
                        col.child_end(),
                        head,
                        validity_cur,
                        offset_cur,
                        data_cur,
                        flattened_col_has_validity,
                        offset_stack_pos,
                        stream,
                        parent_offset_index,
                        offset_depth);
}

template <>
void buf_info_functor::operator()<cudf::struct_view>(
  column_view const& col,
  src_buf_info *& validity_cur,
  src_buf_info *& offset_cur,
  src_buf_info *& data_cur,
  std::vector<int8_t>& flattened_col_has_validity,
  int &offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  rmm::cuda_stream_view stream)
{
  flattened_col_has_validity.push_back(col.nullable());
  if (col.nullable()) {
    add_null_buffer(col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *data_cur = src_buf_info(
    type_id::STRUCT, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  data_cur++;
  offset_stack_pos += offset_depth;

  // recurse on children
  cudf::structs_column_view scv(col);
  std::vector<column_view> sliced_children;
  sliced_children.reserve(scv.num_children());
  std::transform(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(scv.num_children()),
    std::back_inserter(sliced_children),
    [&scv, &stream](size_type child_index) { return scv.get_sliced_child(child_index, stream); });
  setup_source_buf_info(sliced_children.begin(),
                        sliced_children.end(),
                        head,
                        validity_cur,
                        offset_cur,
                        data_cur,
                        flattened_col_has_validity,
                        offset_stack_pos,
                        stream,
                        parent_offset_index,
                        offset_depth);
}

template <typename InputIter>
void setup_source_buf_info(InputIter begin,
                           InputIter end,
                           src_buf_info* head,
                           src_buf_info*& validity_cur,
                           src_buf_info*& offset_cur,
                           src_buf_info*& data_cur,
                           std::vector<int8_t>& flattened_col_has_validity,
                           int &offset_stack_pos,
                           rmm::cuda_stream_view stream,
                           int parent_offset_index,
                           int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    cudf::type_dispatcher(col.type(),
                          buf_info_functor{head},
                          col,
                          validity_cur,
                          offset_cur,
                          data_cur,
                          flattened_col_has_validity,
                          offset_stack_pos,
                          parent_offset_index,
                          offset_depth,
                          stream);
  });
}

/**
 * @brief Given a set of input columns and processed split buffers, produce
 * output columns.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param out_begin Output iterator of column views
 * @param base_ptr Pointer to the base address of copied data for the working partition
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
template <typename InputIter, typename BufInfo, typename Output>
BufInfo build_output_columns(InputIter begin,
                             InputIter end,
                             BufInfo info_begin,
                             Output out_begin,
                             uint8_t const* const base_ptr)
{
  auto current_info = info_begin;
  std::transform(begin, end, out_begin, [&current_info, base_ptr](column_view const& src) {
    auto [bitmask_ptr, null_count] = [&]() {
      if (src.nullable()) {
        auto const ptr =
          current_info->num_elements == 0
            ? nullptr
            : reinterpret_cast<bitmask_type const*>(base_ptr + current_info->dst_offset);
        auto const null_count = current_info->num_elements == 0
                                  ? 0
                                  : (current_info->num_rows - current_info->valid_count);
        ++current_info;
        return std::pair(ptr, null_count);
      }
      return std::pair(static_cast<bitmask_type const*>(nullptr), 0);
    }();

    // size/data pointer for the column
    auto const size = current_info->num_elements;
    uint8_t const* data_ptr =
      size == 0 || src.head() == nullptr ? nullptr : base_ptr + current_info->dst_offset;
    ++current_info;

    // children
    auto children = std::vector<column_view>{};
    children.reserve(src.num_children());

    current_info = build_output_columns(
      src.child_begin(), src.child_end(), current_info, std::back_inserter(children), base_ptr);

    return column_view{src.type(), size, data_ptr, bitmask_ptr, null_count, 0, std::move(children)};
  });

  return current_info;
}

struct partition_size_info {
  size_t validity_size;
  size_t offset_size;
  size_t data_size;
};

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = std::size_t;
  using difference_type   = std::size_t;
  using pointer           = std::size_t*;
  using reference         = std::size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Output iterator for writing values to the valid_count field of the
 * dst_buf_info struct
 */
/*
struct dst_valid_count_output_iterator {
  dst_buf_info* c;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_valid_count_output_iterator operator+ __host__ __device__(int i)
  {
    return dst_valid_count_output_iterator{c + i};
  }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->valid_count; }
};
*/

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct size_of_helper {
  template <typename T>
  constexpr std::enable_if_t<!is_fixed_width<T>() && !std::is_same_v<T, cudf::string_view>, size_t>
    __device__ operator()() const
  {    
    return 0;
  }

  template <typename T>
  constexpr std::enable_if_t<!is_fixed_width<T>() && std::is_same_v<T, cudf::string_view>, size_t>
    __device__ operator()() const
  {
    return sizeof(cudf::device_storage_type_t<int8_t>);
  }

  template <typename T>
  constexpr std::enable_if_t<is_fixed_width<T>(), size_t> __device__ operator()() const noexcept
  {
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

template <typename InputIter>
std::pair<size_t, size_type> count_flattened_columns(InputIter begin, InputIter end, int depth = 0)
{
  auto child_count = [&](column_view const& col, int depth) -> std::pair<size_type, size_type> {
    if(col.type().id() == cudf::type_id::STRUCT){
      return count_flattened_columns(col.child_begin(), col.child_end(), depth+1);
    } else if(col.type().id() == cudf::type_id::LIST){
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      return count_flattened_columns(children.begin(), children.end(), depth+1);
    }
    return {0, depth};
  };

  size_type col_count = 0;
  size_type max_depth = 0;
  std::for_each(begin, end, [&](column_view const& col){
    auto const cc = child_count(col, depth);
    col_count += (1 + cc.first);
    max_depth = std::max(max_depth, cc.second);
  });

  return {col_count, max_depth};
}

struct partition_header {
  uint32_t    magic_number;
  uint32_t    version;
  uint64_t    offset;
  uint64_t    num_rows;
  uint64_t    validity_size;
  uint64_t    offset_size;
  uint64_t    data_size;
};
constexpr size_t validity_pad = 4;
constexpr size_t offset_pad = 8;
constexpr size_t data_pad = 8;

size_t compute_per_partition_metadata_size(size_t total_columns)
{
  auto const has_validity_length = (total_columns + 7) / 8; // has-validity bit per column
  return sizeof(partition_header) + has_validity_length;
}

__global__ void pack_per_partition_data_kernel(uint8_t* out_buffer,
                                               size_t num_partitions,
                                               size_t columns_per_partition,
                                               size_t bufs_per_partition,
                                               src_buf_info const* src_buf_info,
                                               size_type const* split_indices,
                                               int8_t const* flattened_col_has_validity,
                                               size_t const* out_buffer_offsets,
                                               partition_size_info const* partition_sizes)
{
  constexpr uint32_t magic = 'ODUK';
  constexpr uint32_t kudo_version = 1;

  int const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto const threads_per_partition = cudf::util::round_up_safe(columns_per_partition, static_cast<size_t>(cudf::detail::warp_size));
  auto const partition_index = tid / threads_per_partition;
  if(partition_index >= num_partitions){
    return;
  }  
  auto const col_index = tid % threads_per_partition;

  // start of the metadata buffer for this partition
  uint8_t* buf_start = out_buffer + out_buffer_offsets[partition_index];
  partition_header* pheader = reinterpret_cast<partition_header*>(buf_start);

  // first thread in each partition stores constant stuff
  if(col_index == 0){
    pheader->magic_number = magic;
    pheader->version = kudo_version;

    pheader->offset = 0;  // TODO

    // it is possible to get in here with no columns -or- no rows.
    size_type partition_num_rows = 0;
    if(col_index < columns_per_partition){
      partition_num_rows = split_indices[partition_index+1] - split_indices[partition_index];
      // printf("CBI: %d %d %d\n", (int)partition_index, (int)col_index, (int)partition_num_rows);
    }
    pheader->num_rows = partition_num_rows;

    auto const& psize = partition_sizes[partition_index];
    pheader->validity_size = psize.validity_size;
    pheader->offset_size = psize.offset_size;
    pheader->data_size = psize.data_size;
  }

  bitmask_type* has_validity = reinterpret_cast<bitmask_type*>(buf_start + sizeof(partition_header));

  // store has-validity bits. note that the kudo format only aligns to byte boundaries here, but we are guaranteed that the overall buffer is
  // padded out to >= 4 bytes.  
  bitmask_type mask = __ballot_sync(0xffffffff, col_index < columns_per_partition ? flattened_col_has_validity[col_index] : 0);
  if((col_index % cudf::detail::warp_size == 0) && col_index < columns_per_partition){
    // printf("HV: %d : %d, %d, %d\n", (int)(col_index / cudf::detail::warp_size), (int)mask, (int)col_index, (int)tid);
    has_validity[col_index / cudf::detail::warp_size] = mask;
  }
}

// perform the copy.
void copy_data(size_t num_src_bufs, uint8_t const **src_bufs, size_t num_bufs, uint8_t* dst_buf, dst_buf_info* d_dst_buf_info, rmm::cuda_stream_view stream)
{
  auto input_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([src_bufs, d_dst_buf_info] __device__ (size_t i){
    auto const& cinfo = d_dst_buf_info[i];
    //printf("Split src (%d %lu): %lu\n", cinfo.src_buf_index, i, (uint64_t)(src_bufs[cinfo.src_buf_index] + cinfo.src_offset));
    return reinterpret_cast<void*>(const_cast<uint8_t*>(src_bufs[cinfo.src_buf_index] + cinfo.src_offset));
  }));
  auto output_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([dst_buf, d_dst_buf_info] __device__ (size_t i){
    auto const& cinfo = d_dst_buf_info[i];
    //printf("Split dst (%d %lu): %lu\n", cinfo.src_buf_index, i,(uint64_t)(dst_buf + cinfo.dst_offset));
    return reinterpret_cast<void*>(dst_buf + cinfo.dst_offset);
  }));
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([d_dst_buf_info] __device__ (size_t i){
    auto const& cinfo = d_dst_buf_info[i];
    //printf("Split size (%d %lu): %lu\n", cinfo.src_buf_index, i, cinfo.buf_size);
    return cinfo.buf_size;
  }));

  size_t temp_storage_bytes;
  cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, num_bufs, stream);
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceMemcpy::Batched(temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, num_bufs, stream);

  // debug.
  stream.synchronize();
  int whee = 10;
  whee++;
}

template <typename InputIter>
void populate_column_data(shuffle_split_metadata& meta, InputIter begin, InputIter end)
{
  std::for_each(begin, end, [&meta](column_view const& col){
    switch(col.type().id()){
    case cudf::type_id::STRUCT:
      meta.col_info.push_back({col.type().id(), col.num_children()});
      populate_column_data(meta, col.child_begin(), col.child_end());
      break;
    
    case cudf::type_id::LIST: {
      meta.col_info.push_back({col.type().id(), 1});
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      populate_column_data(meta, children.begin(), children.end());
      } break;

    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      // TODO: scale.
      meta.col_info.push_back({col.type().id(), 0});
      break;

    default:
      meta.col_info.push_back({col.type().id(), 0});
      break;
    }
  });
}

// returns global metadata describing the table and the size of the
// internal per-partition data
shuffle_split_metadata compute_metadata(cudf::table_view const& input, size_t total_flattened_columns)
{
  // compute the metadata
  shuffle_split_metadata ret;
  ret.col_info.reserve(total_flattened_columns);
  populate_column_data(ret, input.begin(), input.end());
  return ret;
}

};  // anonymous namespace

std::pair<shuffle_split_result, shuffle_split_metadata> shuffle_split(cudf::table_view const& input,
                                                                      std::vector<size_type> const& splits,
                                                                      rmm::cuda_stream_view stream,
                                                                      rmm::device_async_resource_ref mr)
{
  // empty inputs
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    rmm::device_uvector<size_t> empty_offsets(1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), empty_offsets.begin(), empty_offsets.end(), 0);
    return {shuffle_split_result{std::make_unique<rmm::device_buffer>(0, stream, mr), std::move(empty_offsets)},
            shuffle_split_metadata{compute_metadata(input, 0)}};
  }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns");
  }
  {
    size_type begin = 0;
    for (std::size_t i = 0; i < splits.size(); i++) {
      size_type end = splits[i];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
      begin = end;
    }
  }

  auto temp_mr = cudf::get_current_device_resource_ref();

  size_t const num_partitions   = splits.size() + 1;
  size_t const num_root_columns = input.num_columns();

  // if inputs are empty, just return num_partitions empty tables
  /*
  if (input.column(0).size() == 0) {
    // sanitize the inputs (to handle corner cases like sliced tables)
    std::vector<std::unique_ptr<column>> empty_columns;
    empty_columns.reserve(input.num_columns());
    std::transform(
      input.begin(), input.end(), std::back_inserter(empty_columns), [](column_view const& col) {
        return cudf::empty_like(col);
      });
    std::vector<cudf::column_view> empty_column_views;
    empty_column_views.reserve(input.num_columns());
    std::transform(empty_columns.begin(),
                   empty_columns.end(),
                   std::back_inserter(empty_column_views),
                   [](std::unique_ptr<column> const& col) { return col->view(); });
    table_view empty_inputs(empty_column_views);

    // build the empty results
    std::vector<packed_table> result;
    result.reserve(num_partitions);
    auto iter = thrust::make_counting_iterator(0);
    std::transform(iter,
                   iter + num_partitions,
                   std::back_inserter(result),
                   [&empty_inputs](int partition_index) {
                     return packed_table{
                       empty_inputs,
                       packed_columns{std::make_unique<std::vector<uint8_t>>(pack_metadata(
                                        empty_inputs, static_cast<uint8_t const*>(nullptr), 0)),
                                      std::make_unique<rmm::device_buffer>()}};
                   });

    return result;
  }
  */

  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers
  src_buf_count const num_src_bufs_by_type = count_src_bufs(input.begin(), input.end());
  size_t const num_src_bufs = num_src_bufs_by_type.validity_buf_count +
                              num_src_bufs_by_type.offset_buf_count +
                              num_src_bufs_by_type.data_buf_count;
  size_t const num_bufs = num_src_bufs * num_partitions;
  auto const bufs_per_partition = num_src_bufs;

  // packed block of memory 1. split indices and src_buf_info structs
  std::size_t const indices_size =
    cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), split_align);
  std::size_t const src_buf_info_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align);
  // host-side
  std::vector<uint8_t> h_indices_and_source_info(indices_size + src_buf_info_size);
  size_type* h_indices = reinterpret_cast<size_type*>(h_indices_and_source_info.data());
  src_buf_info* h_src_buf_head =  
    reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size);
  src_buf_info* h_validity_buf_info = h_src_buf_head;
  src_buf_info* h_offset_buf_info = h_validity_buf_info + num_src_bufs_by_type.validity_buf_count;
  src_buf_info* h_data_buf_info = h_offset_buf_info + num_src_bufs_by_type.offset_buf_count;
  // device-side
  // gpu-only : stack space needed for nested list offset calculation
  int const offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
  std::size_t const offset_stack_size =
    offset_stack_partition_size * num_partitions * sizeof(size_type);
  rmm::device_buffer d_indices_and_source_info(indices_size + src_buf_info_size + offset_stack_size,
                                               stream,
                                               rmm::mr::get_current_device_resource());
  auto* d_indices              = reinterpret_cast<size_type*>(d_indices_and_source_info.data());
  src_buf_info* d_src_buf_info = reinterpret_cast<src_buf_info*>(
    reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) + indices_size);
  size_type* d_offset_stack =
    reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) +
                                 indices_size + src_buf_info_size);

  // compute splits -> indices.
  h_indices[0]              = 0;
  h_indices[num_partitions] = input.column(0).size();
  std::copy(splits.begin(), splits.end(), std::next(h_indices));

  // setup source buf info
  auto const total_flattened_columns = count_flattened_columns(input.begin(), input.end()).first;
  std::vector<int8_t> flattened_col_has_validity;
  flattened_col_has_validity.reserve(total_flattened_columns);
  int offset_stack_pos = 0;
  setup_source_buf_info(input.begin(), input.end(), h_src_buf_head, h_validity_buf_info, h_offset_buf_info, h_data_buf_info, flattened_col_has_validity, offset_stack_pos, stream);
  auto d_flattened_col_has_validity = cudf::detail::make_device_uvector_async(flattened_col_has_validity, stream.value(), temp_mr);

  // HtoD indices and source buf info to device
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyDefault, stream.value()));

  // packed block of memory 2. partition buffer sizes and dst_buf_info structs
  std::size_t const partition_sizes_size =
    cudf::util::round_up_safe(num_partitions * sizeof(partition_size_info), split_align);
  std::size_t const dst_buf_info_size =
    cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align);
  // host-side
  std::vector<uint8_t> h_buf_sizes_and_dst_info(partition_sizes_size + dst_buf_info_size);
  std::size_t* h_buf_sizes = reinterpret_cast<std::size_t*>(h_buf_sizes_and_dst_info.data());
  dst_buf_info* h_dst_buf_info =
    reinterpret_cast<dst_buf_info*>(h_buf_sizes_and_dst_info.data() + partition_sizes_size);
  // device-side  
  rmm::device_buffer d_buf_sizes_and_dst_info(
    partition_sizes_size + dst_buf_info_size, stream, temp_mr);
  partition_size_info* d_partition_sizes     = reinterpret_cast<partition_size_info*>(d_buf_sizes_and_dst_info.data());
  dst_buf_info* d_dst_buf_info = reinterpret_cast<dst_buf_info*>(
    static_cast<uint8_t*>(d_buf_sizes_and_dst_info.data()) + partition_sizes_size);

  // this has to be a separate allocation because it gets returned.
  rmm::device_uvector<size_t> d_partition_offsets(num_partitions + 1, stream, mr);

  // compute sizes of each buffer in each partition, including alignment.
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<std::size_t>(0),
    thrust::make_counting_iterator<std::size_t>(num_bufs),
    d_dst_buf_info,
    [bufs_per_partition,
     d_indices,
     d_src_buf_info,
     d_offset_stack,
     offset_stack_partition_size] __device__(std::size_t t) {
      int const partition_index   = t / bufs_per_partition;
      int const src_buf_index = t % bufs_per_partition;
      auto const& src_info    = d_src_buf_info[src_buf_index];

      // apply nested offsets (for lists and string columns).
      //
      // We can't just use the incoming row indices to figure out where to read from in a
      // nested list situation.  We have to apply offsets every time we cross a boundary
      // (list or string).  This loop applies those offsets so that our incoming row_index_start
      // and row_index_end get transformed to our final values.
      //
      int const stack_pos = src_info.offset_stack_pos + (partition_index * offset_stack_partition_size);
      size_type* offset_stack  = &d_offset_stack[stack_pos];
      int parent_offsets_index = src_info.parent_offsets_index;
      int stack_size           = 0;
      int root_column_offset   = src_info.column_offset;
      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        root_column_offset         = d_src_buf_info[parent_offsets_index].column_offset;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      // make sure to include the -column- offset on the root column in our calculation.
      int row_start = d_indices[partition_index] + root_column_offset;
      int row_end   = d_indices[partition_index + 1] + root_column_offset;
      while (stack_size > 0) {
        stack_size--;
        auto const offsets = d_src_buf_info[offset_stack[stack_size]].offsets;
        // this case can happen when you have empty string or list columns constructed with
        // empty_like()
        if (offsets != nullptr) {
          row_start = offsets[row_start];
          row_end   = offsets[row_end];
        }
      }

      // final element indices and row count
      size_t const src_element_index = src_info.is_validity ? row_start / 8 : row_start;
      int const num_rows          = row_end - row_start;
      // if I am an offsets column, all my values need to be shifted
      //int const value_shift = src_info.offsets == nullptr ? 0 : src_info.offsets[row_start];
      // if I am a validity column, we may need to shift bits
      //int const bit_shift = src_info.is_validity ? row_start % 32 : 0;
      // # of rows isn't necessarily the same as # of elements to be copied.
      auto const num_elements = [&]() {
        if (src_info.offsets != nullptr && num_rows > 0) {
          return num_rows + 1;
        } else if (src_info.is_validity) {
          return (num_rows + 7) / 8;
        }
        return num_rows;
      }();
      buffer_type const btype = [&]() {
        if (src_info.offsets != nullptr) {
          return buffer_type::OFFSETS;
        } else if (src_info.is_validity) {
          return buffer_type::VALIDITY;
        }
        return buffer_type::DATA;
      }();
      int const element_size = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      std::size_t const bytes =
        static_cast<std::size_t>(num_elements) * static_cast<std::size_t>(element_size);
      
      // printf("P: %d %d %lu\n", partition_index, src_buf_index, bytes);

      return dst_buf_info{bytes,
                          btype,
                          src_buf_index,
                          src_element_index * element_size,
                          0};   // dst_offset is computed later
                          //num_elements,
                          //element_size,
                          //num_rows,
                          //src_element_index,
                          // 0,
                          //btype,
                          //src_info.is_validity ? 1 : 0,
                          //value_shift,
                          //bit_shift,
                          //src_info.is_validity ? 1 : 0,
                          //src_buf_index,
                          //split_index};
    });

  // compute per-partition metadata size
  auto const per_partition_metadata_size = compute_per_partition_metadata_size(total_flattened_columns);

  auto partition_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([bufs_per_partition] __device__ (size_t buf_index){
    return buf_index / bufs_per_partition;
  }));

  // - compute: size of all validity buffers, size of all offset buffers, size of all data buffers
  auto buf_sizes_by_type =
    cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<partition_size_info>([d_dst_buf_info] __device__ (int index){ 
      switch(d_dst_buf_info[index].type){
      case buffer_type::VALIDITY: return partition_size_info{d_dst_buf_info[index].buf_size, 0, 0};
      case buffer_type::OFFSETS: return partition_size_info{0, d_dst_buf_info[index].buf_size, 0};
      case buffer_type::DATA: return partition_size_info{0, 0, d_dst_buf_info[index].buf_size};
      default: break;
      }
      return partition_size_info{0, 0, 0};
    }));
  auto buf_size_reduce = cuda::proclaim_return_type<partition_size_info>([] __device__ (partition_size_info const& lhs, partition_size_info const& rhs){
                           auto const validity_size = lhs.validity_size + rhs.validity_size;
                           auto const offset_size = lhs.offset_size + rhs.offset_size;
                           auto const data_size = lhs.data_size + rhs.data_size;
                           return partition_size_info{validity_size, offset_size, data_size};
                        });
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        partition_keys,
                        partition_keys + num_bufs,
                        buf_sizes_by_type,
                        thrust::make_discard_iterator(),
                        d_partition_sizes,
                        thrust::equal_to{}, // key equality check
                        buf_size_reduce);  

  // - compute partition start offsets and total output buffer size overall
  auto partition_size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([num_partitions, d_partition_sizes, per_partition_metadata_size] __device__ (size_t i){
    return i >= num_partitions ? 0 :
      cudf::util::round_up_safe(cudf::util::round_up_safe(per_partition_metadata_size + d_partition_sizes[i].validity_size, validity_pad) +
                                cudf::util::round_up_safe(d_partition_sizes[i].offset_size, offset_pad) +
                                d_partition_sizes[i].data_size,
                                data_pad);
  }));
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         partition_size_iter,
                         partition_size_iter + num_partitions + 1,
                         d_partition_offsets.begin());

  size_t dst_buf_total_size;
  cudaMemcpyAsync(&dst_buf_total_size, d_partition_offsets.begin() + num_partitions, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
  
  /*
  {
    std::vector<partition_size_info> h_partition_sizes(num_partitions);
    cudaMemcpy(h_partition_sizes.data(), d_partition_sizes, sizeof(partition_size_info) * num_partitions, cudaMemcpyDeviceToHost);
    std::vector<size_t> h_partition_offsets(num_partitions + 1);
    cudaMemcpy(h_partition_offsets.data(), d_partition_offsets.data(), sizeof(size_t) * (num_partitions + 1), cudaMemcpyDeviceToHost);
    printf("Per partition metadata size : %lu\n", per_partition_metadata_size);
    for(size_t idx=0; idx<num_partitions; idx++){
      size_t const partition_total = h_partition_offsets[idx+1] - h_partition_offsets[idx];
      printf("HBS(%lu): %lu, %lu, %lu, %lu, %lu\n", idx, h_partition_sizes[idx].validity_size, 
                                               h_partition_sizes[idx].offset_size, 
                                               h_partition_sizes[idx].data_size,
                                               partition_total,
                                               h_partition_offsets[idx]);
    }
  }
  */
  
  // generate individual buffer offsets
  auto buf_sizes = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([d_dst_buf_info] __device__ (size_t i){
    return d_dst_buf_info[i].buf_size;
  }));  
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                partition_keys,
                                partition_keys + num_bufs,
                                buf_sizes,
                                dst_offset_output_iterator{d_dst_buf_info});
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy(stream, temp_mr),
                  iter,
                  iter + num_bufs,
                  [per_partition_metadata_size,
                   bufs_per_partition,
                   d_dst_buf_info,
                   d_partition_sizes,
                   d_partition_offsets = d_partition_offsets.begin()]  __device__ (size_type i){

    auto const partition_index = i / bufs_per_partition;
    auto const section_offset = cuda::proclaim_return_type<size_t>([&] __device__ (){
      auto const& ps = d_partition_sizes[partition_index];
      switch(d_dst_buf_info[i].type){
      case buffer_type::OFFSETS: return cudf::util::round_up_safe(per_partition_metadata_size + ps.validity_size, validity_pad);
      case buffer_type::DATA: return cudf::util::round_up_safe(cudf::util::round_up_safe(per_partition_metadata_size + ps.validity_size, validity_pad) + ps.offset_size,
                                                               offset_pad);
      default: return per_partition_metadata_size;
      }
    })();    
    size_t const pre = d_dst_buf_info[i].dst_offset;
    d_dst_buf_info[i].dst_offset += d_partition_offsets[partition_index] + section_offset;
    auto const& ps = d_partition_sizes[partition_index];
    // printf("dst(p:%d %d (%d)): %lu, %lu, %lu %lu (%lu %lu)\n", (int)i, (int)partition_index, (int)d_dst_buf_info[i].type, d_partition_offsets[partition_index], section_offset, d_dst_buf_info[i].dst_offset, pre, ps.validity_size, ps.offset_size);
  });

  // packed block of memory 3. pointers to source and destination buffers (and stack space on the
  // gpu for offset computation)
  std::size_t const src_bufs_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align);
  // host-side
  std::vector<uint8_t> h_src_and_dst_buffers(src_bufs_size/* + dst_bufs_size*/);
  uint8_t const** h_src_bufs = reinterpret_cast<uint8_t const**>(h_src_and_dst_buffers.data());
  rmm::device_buffer d_src_and_dst_buffers(src_bufs_size,
                                           stream,
                                           temp_mr);
  auto const** d_src_bufs = reinterpret_cast<uint8_t const**>(d_src_and_dst_buffers.data());

  // setup src buffers
  setup_src_buf_data(input.begin(), input.end(), h_src_bufs);
  
  // HtoD src buffers
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_src_bufs, h_src_bufs, src_bufs_size, cudaMemcpyDefault, stream.value()));

  // allocate output buffer
  rmm::device_buffer dst_buf(dst_buf_total_size, stream, mr);

  // pack per-partition data. one thread per (flattened) column.
  size_type const thread_count_per_partition = cudf::util::round_up_safe(total_flattened_columns, static_cast<size_t>(cudf::detail::warp_size));
  cudf::detail::grid_1d const grid{thread_count_per_partition * static_cast<size_type>(num_partitions), 128};
  pack_per_partition_data_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    reinterpret_cast<uint8_t*>(dst_buf.data()),
    num_partitions,
    total_flattened_columns,
    num_src_bufs,
    d_src_buf_info,
    d_indices,
    d_flattened_col_has_validity.data(),
    d_partition_offsets.data(),
    d_partition_sizes);

  /*
  {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_dst_buf_info, d_dst_buf_info, dst_buf_info_size, cudaMemcpyDefault, stream.value()));    
    stream.synchronize();
  }
  */

  stream.synchronize();

  // perform the copy.
  copy_data(num_src_bufs, d_src_bufs, num_bufs, reinterpret_cast<uint8_t*>(dst_buf.data()), d_dst_buf_info, stream);

  stream.synchronize();
 
  // return result;
  return {shuffle_split_result{std::make_unique<rmm::device_buffer>(std::move(dst_buf)), std::move(d_partition_offsets)},
          compute_metadata(input, total_flattened_columns)};
}

namespace {

#define OUTPUT_ITERATOR(__name, __T, __field_name)                                                  \
  template<typename __T>                                                                            \
  struct __name##generic_output_iter {                                                              \
    __T* c;                                                                                         \
    using value_type        = decltype(__T::__field_name);                                          \
    using difference_type   = size_t;                                                               \
    using pointer           = decltype(__T::__field_name)*;                                         \
    using reference         = decltype(__T::__field_name)&;                                         \
    using iterator_category = thrust::output_device_iterator_tag;                                   \
                                                                                                    \
    __name##generic_output_iter operator+ __host__ __device__(int i) { return {c + i}; }            \
                                                                                                    \
    __name##generic_output_iter& operator++ __host__ __device__()                                   \
    {                                                                                               \
      c++;                                                                                          \
      return *this;                                                                                 \
    }                                                                                               \
                                                                                                    \
    reference operator[] __device__(int i) { return dereference(c + i); }                           \
    reference operator* __device__() { return dereference(c); }                                     \
                                                                                                    \
  private:                                                                                          \
    reference __device__ dereference(__T* c) { return c->__field_name; }                            \
  };                                                                                                \
  using __name = __name##generic_output_iter<__T>

// per-flattened-column information
struct assemble_column_info {
  cudf::type_id         type;
  bool                  has_validity;
  size_type             num_rows, num_chars;
  size_type             null_count;
  size_type             num_children;
};
OUTPUT_ITERATOR(assemble_column_info_num_rows_output_iter, assemble_column_info, num_rows);
OUTPUT_ITERATOR(assemble_column_info_has_validity_output_iter, assemble_column_info, has_validity);

constexpr size_t bitmask_allocation_size_bytes(size_type number_of_bits, int pad = 1)
{
  return cudf::util::round_up_safe((number_of_bits + 7) / 8, pad);
}

// a copy batch. 1 per block.
struct assemble_batch {
  __device__ assemble_batch(int8_t const* _src, int8_t* _dst, size_t _size, bool _validity, int _value_shift, int _bit_shift):
    src(_src), dst(_dst), size(_size), validity(_validity), value_shift(_value_shift), bit_shift(_bit_shift){}

  int8_t const* src;
  int8_t* dst;
  size_t              size;     // bytes
  bool                validity; // whether or not this is a validity buffer
  int value_shift;              // amount to shift values down by (for offset buffers)
  int bit_shift;                // # of bits to shift left by (for validity buffers)
  size_type valid_count = 0;    // (output) validity count for this block of work
};

struct assemble_buffer_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename BufIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  {
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // no offsets for fixed width types
    
    // data
    auto const data_size = cudf::util::round_up_safe(cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows, split_align);
    *data_out = rmm::device_buffer(data_size, stream, mr);
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);    

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), split_align);
    *offsets_out = rmm::device_buffer(offsets_size, stream, mr);

    // no data for lists
  } 

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // no offsets or data for structs
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // chars TODO
    // auto const chars_size = cudf::util::round_up_safe(sizeof(int8_t) * (col.num_chars + 1), shuffle_split_partition_data_align);
    // out.push_back(rmm::device_buffer(chars_size, stream, mr));
    *data_out = rmm::device_buffer(0, stream, mr);

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), split_align);
    *offsets_out = rmm::device_buffer(offsets_size, stream, mr);    
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                                         !std::is_same_v<T, cudf::list_view> && 
                                                         !std::is_same_v<T, cudf::string_view> && 
                                                         !cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, BufIter& validity_out, BufIter& offsets_out, BufIter& data_out)
  { 
    CUDF_FAIL("Unsupported type in assemble_buffer_functor");
  }
 
private:
  rmm::device_buffer alloc_validity(size_type num_rows)
  {
    return rmm::device_buffer(bitmask_allocation_size_bytes(num_rows, split_align), stream, mr);
  }
};

struct assemble_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    // no offsets
    auto const data = buffer + 2;

    out.push_back(std::make_unique<cudf::column>(cudf::data_type{col->type},
                  col->num_rows,
                  std::move(*data),
                  col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                  col->null_count));
    
    return {col + 1, buffer + 3};
  }

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    buffer += 3;

    // build children
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(col->num_children);
    auto next = col + 1;
    for(size_type i=0; i<col->num_children; i++){
      std::tie(next, buffer) = cudf::type_dispatcher(cudf::data_type{next->type},
                                                     assemble_column_functor{stream, mr},
                                                     next,
                                                     buffer,
                                                     children);
    }    

    out.push_back(cudf::make_structs_column(col->num_rows,
                                            std::move(children),
                                            col->null_count,
                                            col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                                            stream,
                                            mr));
    return {next, buffer};
  }
    
    /*
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  size_t operator()(size_t cur, host_span<assemble_column_info const> assemble_data, host_span<rmm::device_buffer> buffers, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto col = assemble_data[cur];
    auto validity = cur;
    auto offsets = col.has_validity ? cur + 1 : cur;
    cur = offsets + 1;

    // build offsets
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      col.num_rows + 1,
                                                      std::move(buffers[offsets]),
                                                      rmm::device_buffer{},
                                                      0);

    // build the child
    std::vector<std::unique_ptr<cudf::column>> child_col;
    cur = cudf::type_dispatcher(cudf::data_type{col.type},
                                *this,
                                cur,
                                assemble_data,
                                buffers,
                                child_col);
    
    // build the final column
    out.push_back(cudf::make_lists_column(col.num_rows,
                                          std::move(offsets_col),
                                          std::move(child_col.back()),
                                          col.null_count,
                                          col.has_validity ? std::move(buffers[validity]) : rmm::device_buffer{},
                                          stream,
                                          mr));
    return cur;
  }  
  */

  // template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::list_view> and !std::is_same_v<T, cudf::struct_view>)>
  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    CUDF_FAIL("Unsupported type in shuffle_assemble");
  }
};

struct assemble_empty_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {    
    out.push_back(std::make_unique<cudf::column>(cudf::data_type{col->type},
                  0,
                  rmm::device_buffer{},
                  rmm::device_buffer{},
                  0));
    
    return {col + 1};
  }

  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    // build children
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(col->num_children);
    auto next = col + 1;
    for(size_type i=0; i<col->num_children; i++){
      next = cudf::type_dispatcher(cudf::data_type{next->type},
                                   assemble_empty_column_functor{stream, mr},
                                   next,
                                   children);
    }    

    out.push_back(cudf::make_structs_column(0,
                                            std::move(children),
                                            0,
                                            rmm::device_buffer{},
                                            stream,
                                            mr));
    return next;
  }
    
    /*
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  size_t operator()(size_t cur, host_span<assemble_column_info const> assemble_data, host_span<rmm::device_buffer> buffers, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto col = assemble_data[cur];
    auto validity = cur;
    auto offsets = col.has_validity ? cur + 1 : cur;
    cur = offsets + 1;

    // build offsets
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      col.num_rows + 1,
                                                      std::move(buffers[offsets]),
                                                      rmm::device_buffer{},
                                                      0);

    // build the child
    std::vector<std::unique_ptr<cudf::column>> child_col;
    cur = cudf::type_dispatcher(cudf::data_type{col.type},
                                *this,
                                cur,
                                assemble_data,
                                buffers,
                                child_col);
    
    // build the final column
    out.push_back(cudf::make_lists_column(col.num_rows,
                                          std::move(offsets_col),
                                          std::move(child_col.back()),
                                          col.null_count,
                                          col.has_validity ? std::move(buffers[validity]) : rmm::device_buffer{},
                                          stream,
                                          mr));
    return cur;
  }
  */

  // template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::list_view> and !std::is_same_v<T, cudf::struct_view>)>
  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view>)>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    CUDF_FAIL("Unsupported type in shuffle_assemble");
  }
};

// The size that contiguous split uses internally as the GPU unit of work.
// The number of `desired_batch_size` batches equals the number of CUDA blocks
// that will be used for the main kernel launch (`copy_partitions`).
constexpr std::size_t desired_assemble_batch_size = 1 * 1024 * 1024;


// returns:
// - a vector of assemble_column_info structs representing the destination column data.
//   the vector is of length global_metadata.col_info.size()  that is, the flattened list of columns in the table.
//
// - the same vector as above, but in host memory. 
//
// - a vector of assemble_column_info structs, representing the source column data.
//   the vector is of length global_metadata.col_info.size() * the # of partitions. 
//
std::tuple<rmm::device_uvector<assemble_column_info>,
           std::vector<assemble_column_info>,
           rmm::device_uvector<assemble_column_info>,
           size_t>
assemble_build_column_info(shuffle_split_metadata const& h_global_metadata,
                           cudf::device_span<int8_t const> partitions, 
                           cudf::device_span<size_t const> partition_offsets,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<shuffle_split_col_data> global_metadata = cudf::detail::make_device_uvector_async(h_global_metadata.col_info, stream, temp_mr);

  // "columns" here means the number of flattened columns in the entire source table, not just the
  // number of columns at the top level
  auto const num_columns = global_metadata.size();
  size_type const num_partitions = partition_offsets.size() - 1;
  auto const num_column_instances = num_columns * num_partitions;

  // generate per-column data ------------------------------------------------------
  rmm::device_uvector<assemble_column_info> column_info(num_columns, stream, temp_mr);

  // compute:
  //  - indices into the char count data for string columns
  //  - offset into the partition data where has-validity begins
  /*
  rmm::device_uvector<size_type> char_count_indices(num_columns + 1, stream, temp_mr);
  auto cc_index_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([global_metadata = global_metadata.begin(), num_columns] __device__ (size_type i) {
    return i >= num_columns ? 0 : (global_metadata[i].type == cudf::type_id::STRING ? 1 : 0);
  }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr), cc_index_iter, cc_index_iter + num_columns + 1, char_count_indices.begin());
  size_type const per_partition_num_char_counts = char_count_indices.back_element(stream);
  // the +1 is for the per-partition overall row count at the very beginning
  auto const has_validity_offset = (per_partition_num_char_counts + 1) * sizeof(size_type);
  
  {
    auto h_char_count_indices = cudf::detail::make_std_vector_sync(char_count_indices, stream);
    printf("per_partition_num_char_counts : %d\n", per_partition_num_char_counts);
    printf("has_validity_offset : %lu\n", has_validity_offset);
    for(size_t idx=0; idx<h_char_count_indices.size(); idx++){
      printf("h_char_count_indices(%lu): %d\n", idx, h_char_count_indices[idx]);
    }
  } 
  */

  // compute per-partition metadata size
  auto const per_partition_metadata_size = compute_per_partition_metadata_size(h_global_metadata.col_info.size());

  // compute has-validity
  // note that we are iterating vertically -> horizontally here, with each column's individual piece per partition first.
  auto column_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([num_partitions] __device__ (size_type i){
    return i / num_partitions;
  }));  
  auto has_validity_values = cudf::detail::make_counting_transform_iterator(0, 
    cuda::proclaim_return_type<bool>([num_partitions,
                                      partitions = partitions.data(),
                                      partition_offsets = partition_offsets.begin()]
                                      __device__ (int i) -> bool {
      auto const partition_index = i % num_partitions;
      bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(partitions + partition_offsets[partition_index] + sizeof(partition_header));
      auto const col_index = i / num_partitions;
      
      // int has_validity = has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
      // printf("HVV: %d, %d, %d, %d, %d\n", (int)partition_index, (int)partition_offsets[partition_index], (int)sizeof(partition_header), (int)col_index, (int)has_validity);
      
      return has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    })
  );
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        column_keys,
                        column_keys +  num_column_instances,
                        has_validity_values,
                        thrust::make_discard_iterator(),
                        assemble_column_info_has_validity_output_iter{column_info.begin()},
                        thrust::equal_to<size_type>{},
                        thrust::logical_or<bool>{});
  /*
  {
    auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);
    for(size_t idx=0; idx<h_column_info.size(); idx++){
      printf("h_column_info(%lu): has_validity = %d\n", idx, (int)(h_column_info[idx].has_validity ? 1 : 0));
    }
  }
  */

  //print_span(cudf::device_span<size_t const>(partition_offsets));

  // compute overall row count
  auto row_count_values = cudf::detail::make_counting_transform_iterator(0,
    cuda::proclaim_return_type<size_t>([num_partitions,
                                                 partitions = partitions.data(),
                                                 partition_offsets = partition_offsets.begin()]
                                                 __device__ (int i){
                                                  partition_header const*const pheader = reinterpret_cast<partition_header const*>(partitions + partition_offsets[i]);
                                                  return pheader->num_rows;
                                                 }));
  size_t const row_count =  thrust::reduce(rmm::exec_policy_nosync(stream, temp_mr),
                                            row_count_values,
                                            row_count_values + num_partitions);
  
  // compute char counts for strings
  // note that we are iterating vertically -> horizontally here, with each column's individual piece per partition first.
  // TODO: use an output iterator and write directly to the outgoing assembly_info structs
  /*
  auto cc_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<cudf::size_type>([num_partitions] __device__ (int i){
    return i / num_partitions;
  }));
  auto char_count_values = cudf::detail::make_counting_transform_iterator(0,
    cuda::proclaim_return_type<cudf::size_type>([num_partitions,
                                                 partitions = partitions.data(),
                                                 partition_offsets = partition_offsets.begin(),
                                                 global_metadata = global_metadata.begin()]
                                                 __device__ (int i){
      auto const partition_index = i % num_partitions;
      auto const col_index = i / num_partitions;

      // non-string columns don't have a char count
      auto const column_type = global_metadata[col_index].type;
      if(column_type != cudf::type_id::STRING){
        return 0;
      }

      // string columns
      size_type const*const char_counts = reinterpret_cast<size_type const*>(partitions + partition_offsets[partition_index] + 4);
      // printf("RCI %d : %d, partition_index = %d\n", (int)col_index, char_counts[col_index], (int)partition_index);
      return char_counts[col_index];
    })
  );
  rmm::device_uvector<size_type> char_counts(num_columns, stream, temp_mr);
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        cc_keys, 
                        cc_keys + num_column_instances,
                        char_count_values,
                        thrust::make_discard_iterator(),
                        char_counts.begin());
  print_span(static_cast<cudf::device_span<size_type const>>(char_counts));
  */
  
  // copy type and summed row counts
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_columns, [row_count,
                                                                                        column_info = column_info.begin(),
                                                                                        global_metadata = global_metadata.begin()
                                                                                        //char_count_indices = char_count_indices.begin(),
                                                                                        //char_counts = char_counts.begin()]
                                                                                        ]
                                                                                        __device__ (size_type col_index){
    auto const& metadata = global_metadata[col_index];
    auto& cinfo = column_info[col_index];
    
    cinfo.type = metadata.type;
    cinfo.null_count = 0; // TODO
    cinfo.num_children = metadata.num_children;
    
    cinfo.num_rows = row_count;
    
    // string columns store the char count separately
    // cinfo.num_chars = cinfo.type == cudf::type_id::STRING ? char_counts[char_count_indices[col_index]] : 0;
  });
    
  /*
  {
    auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);
    for(size_t idx=0; idx<h_column_info.size(); idx++){
      printf("col_info[%lu]: type = %d has_validity = %d num_rows = %d num_chars = %d null_count = %d\n", idx,
        (int)h_column_info[idx].type, h_column_info[idx].has_validity ? 1 : 0, h_column_info[idx].num_rows, h_column_info[idx].num_chars, h_column_info[idx].null_count);
    }
  }
  */

  // generate per-column-instance data ------------------------------------------------------

  // has-validity, type, row count
  rmm::device_uvector<assemble_column_info> column_instance_info(num_column_instances, stream, temp_mr);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_column_instances, [//char_count_indices = char_count_indices.begin(),
                                                                                                 column_instance_info = column_instance_info.begin(),
                                                                                                 global_metadata = global_metadata.begin(),
                                                                                                 partitions = partitions.data(),
                                                                                                 partition_offsets = partition_offsets.begin(),
                                                                                                 num_columns]
                                                                                                 __device__ (size_type i){
    auto const partition_index = i / num_columns;
    auto const col_index = i % num_columns;
    auto const col_instance_index = (partition_index * num_columns) + col_index;

    auto const& metadata = global_metadata[col_index];
    auto& cinstance_info = column_instance_info[col_instance_index];

    uint8_t const*const buf_start = reinterpret_cast<uint8_t const*>(partitions + partition_offsets[partition_index]);
    partition_header const*const pheader = reinterpret_cast<partition_header const*>(buf_start);

    bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(buf_start + sizeof(partition_header));
    cinstance_info.has_validity = has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    
    cinstance_info.type = metadata.type;
    cinstance_info.null_count = 0; // TODO
    cinstance_info.num_children = metadata.num_children;
    
    cinstance_info.num_rows = pheader->num_rows;
    
    // string columns store the char count separately
    /*
    if(metadata.type == cudf::type_id::STRING){
      size_type const*const char_counts = reinterpret_cast<size_type const*>(pheader + 4);
      cinstance_info.num_chars = char_counts[char_count_indices[col_index]];
    }
    */
  });
    
  /*
  {
    auto h_column_instance_info = cudf::detail::make_std_vector_sync(column_instance_info, stream);
    for(size_t idx=0; idx<h_column_instance_info.size(); idx++){
      size_type const partition_index = idx / num_columns;
      size_type const col_index = idx % num_columns;
      size_type const col_instance_index = (partition_index * num_columns) + col_index;

      printf("col_info[%d, %d, %d]: type = %d has_validity = %d num_rows = %d num_chars = %d null_count = %d\n",
        partition_index, col_index, col_instance_index,
        (int)h_column_instance_info[idx].type, h_column_instance_info[idx].has_validity ? 1 : 0, h_column_instance_info[idx].num_rows, h_column_instance_info[idx].num_chars, h_column_instance_info[idx].null_count);
    }
  } 
  */

  // compute per-partition metadata size
  // size_t const metadata_rc_size = ((per_partition_num_char_counts + 1) * sizeof(size_type));  
  // size_t const per_partition_metadata_size = cudf::util::round_up_safe(metadata_rc_size + metadata_has_validity_size, shuffle_split_partition_data_align);
  // size_t const metadata_has_validity_size = per_partition_metadata_size - header_size;

  return {std::move(column_info), cudf::detail::make_std_vector_sync(column_info, stream), std::move(column_instance_info), per_partition_metadata_size};
}

// Important: this returns the size of the buffer -without- padding. just the size of
// the raw bytes containing the actual data.
struct assemble_buffer_size_functor {
  template <typename T, typename OutputIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  {
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows) : 0;

    // no offsets for fixed width types
    *offsets_out = 0;

    // data
    *data_out = cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows) : 0;

    // offsets
    *offsets_out = sizeof(size_type) * (col.num_rows + 1);

    // no data for lists
    *data_out = 0;
  } 

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows) : 0;

    // no offsets or data for structs
    *offsets_out = 0;
    *data_out = 0;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows) : 0;

    // chars
    // TODO:
    // *out++ = sizeof(int8_t) * (col.num_chars + 1);
    *data_out = 0;

    // offsets
    *offsets_out = sizeof(size_type) * (col.num_rows + 1);
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                       !std::is_same_v<T, cudf::list_view> && 
                                       !std::is_same_v<T, cudf::string_view> && 
                                       !cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  {
  }
};

constexpr size_t size_to_batch_count(size_t bytes)
{
  return std::max(std::size_t{1}, util::round_up_unsafe(bytes, desired_assemble_batch_size) / desired_assemble_batch_size);
}

template<typename SizeIterator, typename GroupFunction>
rmm::device_uvector<std::invoke_result_t<GroupFunction>> transform_expand(SizeIterator first,
                                                                          SizeIterator last,
                                                                          GroupFunction op,
                                                                          rmm::cuda_stream_view stream,
                                                                          rmm::device_async_resource_ref mr)
{ 
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto value_count = std::distance(first, last);
  auto size_wrapper = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([value_count, first] __device__ (size_t i){
    return i >= value_count ? 0 : first[i];
  }));
  rmm::device_uvector<size_t> group_offsets(value_count + 1, stream, temp_mr);
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         size_wrapper,
                         size_wrapper + group_offsets.size(),
                         group_offsets.begin());
  size_t total_size = group_offsets.back_element(stream); // note memcpy and device sync
  
  using OutputType = std::invoke_result_t<GroupFunction>;
  rmm::device_uvector<OutputType> result(total_size, stream, mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + total_size,
                    result.begin(),
                    cuda::proclaim_return_type<OutputType>([op, group_offsets_begin = group_offsets.begin(), group_offsets_end = group_offsets.end()] __device__ (size_t i){
                      auto const group_index = thrust::lower_bound(thrust::seq, group_offsets_begin, group_offsets_end, i) - group_offsets_begin;
                      auto const intra_group_index = i - group_offsets_begin[group_index];
                      return op(group_index, intra_group_index);
                    }));

  return result;
}

// returns destination buffers
std::pair<std::vector<rmm::device_buffer>, rmm::device_uvector<assemble_batch>> assemble_build_buffers(rmm::device_uvector<assemble_column_info> const& column_info,
                                                                                                       rmm::device_uvector<assemble_column_info> const& column_instance_info,
                                                                                                       cudf::device_span<int8_t const> partitions,
                                                                                                       cudf::device_span<size_t const> partition_offsets,
                                                                                                       size_t per_partition_metadata_size,
                                                                                                       rmm::cuda_stream_view stream,
                                                                                                       rmm::device_async_resource_ref mr)
{  
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto h_column_info = cudf::detail::make_std_vector_async(column_info, stream);
  auto const num_columns = column_info.size();
  auto const num_partitions = partition_offsets.size() - 1;

/*
  // # of validity buffers, offset buffers, data buffers, per partition.
  // NOTE: to simplify things, we are going to keep a validity buffer for every column, even if it ends up being empty.
  //       this keeps the source<->dst buffer mapping simpler.
  auto buf_count_by_type =
    thrust::make_transform_iterator(column_info.begin(), cuda::proclaim_return_type<partition_size_info>([] __device__ (assemble_column_info const& cinfo){
      return partition_size_info{1,
                                 cinfo.type == type_id::STRING || cinfo.type == type_id::LIST ? size_t{1} : size_t{0},
                                 cinfo.type != type_id::STRING && cinfo.type != type_id::LIST && cinfo.type != type_id::STRUCT ? size_t{1} : size_t{0}};
    }));
  auto buf_count_reduce = cuda::proclaim_return_type<partition_size_info>([per_partition_metadata_size] __device__ (partition_size_info const& lhs, partition_size_info const& rhs){
                            auto const validity_size = lhs.validity_size + rhs.validity_size;
                            auto const offset_size = lhs.offset_size + rhs.offset_size;
                            auto const data_size = lhs.data_size + rhs.data_size;
                            return partition_size_info{validity_size, offset_size, data_size};
                          });
  partition_size_info buf_type_counts = thrust::reduce(rmm::exec_policy(stream, temp_mr),
                                                       buf_count_by_type,
                                                       buf_count_by_type + num_columns,
                                                       partition_size_info{0, 0, 0},
                                                       buf_count_reduce);
  auto const dst_buf_count = buf_type_counts.validity_size + buf_type_counts.offset_size + buf_type_counts.data_size;
  */
  // to simplify things, we will reserve 3 buffers for each column. validity, data, offsets. not every column will use all of them, so those
  // buffers will remain unallocated/zero size. 
  auto const dst_buf_count = num_columns * 3;

  // allocate output buffers. ordered in the array as (validity, offsets, data) per column.
  // TODO: potentially add an option to do this contiguous-split style where we allocate a single
  // huge buffer and wrap it with column_views.
  // std::vector<rmm::device_buffer> assemble_buffers(buf_type_counts.validity_size + buf_type_counts.offset_size + buf_type_counts.data_size);
  std::vector<rmm::device_buffer> assemble_buffers(dst_buf_count);
  auto dst_validity_iter = assemble_buffers.begin();
  auto dst_offsets_iter = assemble_buffers.begin() + 1;
  auto dst_data_iter = assemble_buffers.begin() + 2;
  
  // for each column, a mapping to it's corresponding validity, offset and data buffer
  stream.synchronize(); // for h_column_info
  
  for(size_t idx=0; idx<h_column_info.size(); idx++){
    cudf::type_dispatcher(cudf::data_type{h_column_info[idx].type},
                          assemble_buffer_functor{stream, mr},
                          h_column_info[idx],
                          dst_validity_iter,
                          dst_offsets_iter,
                          dst_data_iter);
    dst_validity_iter += 3;
    dst_offsets_iter += 3;
    dst_data_iter += 3;
  }
  std::vector<int8_t*> h_dst_buffers(assemble_buffers.size());
  std::transform(assemble_buffers.begin(), assemble_buffers.end(), h_dst_buffers.begin(), [](rmm::device_buffer& buf){
    return reinterpret_cast<int8_t*>(buf.data());
  });
  auto dst_buffers = cudf::detail::make_device_uvector_async(h_dst_buffers, stream, temp_mr);
  //auto column_to_buffer_map = cudf::detail::make_device_uvector_async(h_column_to_buffer_map, stream, temp_mr);

  int whee = 10;
  whee++;

  // compute:
  // - row indices by partition
  // - unpadded sizes of the source buffers
  // - offsets into the partition data where each source buffer starts
  size_t const buffers_per_partition = assemble_buffers.size();
  size_t const num_src_buffers = buffers_per_partition * num_partitions;  
  rmm::device_uvector<size_type> partition_row_indices(num_partitions, stream, temp_mr);
  // ordered by partition, with each column containing (validity, offsets, data)
  rmm::device_uvector<size_t> src_sizes_unpadded(num_src_buffers, stream, mr);
  rmm::device_uvector<size_t> src_offsets(num_src_buffers, stream, mr);
  // arranged in destination buffer order, each column containinf (validity, offsets, data)
  rmm::device_uvector<size_t> dst_offsets(num_src_buffers, stream, mr);  
  {
    // generate partition row indices
    auto row_count_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([column_instance_info = column_instance_info.begin(),
                                                                                                                  columns_per_partition = column_info.size()] __device__ (size_type i){
      auto col = column_instance_info[i * columns_per_partition];
      return col.num_rows;
    }));
    thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                           row_count_iter,
                           row_count_iter + num_partitions,
                           partition_row_indices.begin());
    //print_span(cudf::device_span<size_type const>{partition_row_indices});

    // generate unpadded sizes of the source buffers
    auto const num_column_instances = column_instance_info.size();
    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [buffers_per_partition,
                     num_columns,
                     column_instance_info = column_instance_info.begin(),
                     src_sizes_unpadded = src_sizes_unpadded.begin()] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;

      auto const& cinfo_instance = column_instance_info[col_instance_index];
      auto const validity_buf_index = (col_index * 3) + (partition_index * buffers_per_partition);
      auto const offset_buf_index = ((col_index * 3) + 1) + (partition_index * buffers_per_partition);
      auto const data_buf_index = ((col_index * 3) + 2) + (partition_index * buffers_per_partition);
      cudf::type_dispatcher(cudf::data_type{cinfo_instance.type},
                            assemble_buffer_size_functor{},
                            cinfo_instance,
                            &src_sizes_unpadded[validity_buf_index],
                            &src_sizes_unpadded[offset_buf_index],
                            &src_sizes_unpadded[data_buf_index]);
      //printf("SSU: %d %d (%d %d %d) (%d %d %d) (%lu %lu %lu)\n", (int)partition_index, (int)col_index, 
        //                                                (int)column_to_buffer_map[validity_buf_index], (int)column_to_buffer_map[offset_buf_index], (int)column_to_buffer_map[data_buf_index],
          //                                              (int)validity_buf_index, (int)offset_buf_index, (int)data_buf_index,
            //                                            src_sizes_unpadded[validity_buf_index], src_sizes_unpadded[offset_buf_index], src_sizes_unpadded[data_buf_index]);
    });
    //print_span(cudf::device_span<size_t const>{src_sizes_unpadded});
        
    // scan to source offsets, by partition
    auto partition_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([buffers_per_partition] __device__ (size_t i){
      return i / buffers_per_partition;
    }));
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  partition_keys,
                                  partition_keys + num_src_buffers,
                                  src_sizes_unpadded.begin(),
                                  src_offsets.begin());
    //print_span(cudf::device_span<size_t const>{src_offsets});
    
    // adjust the source offsets:
    // - add metadata offset
    // - take padding into account 
    // - add partition offset
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [num_columns,
                     buffers_per_partition,
                     column_instance_info = column_instance_info.begin(),
                     src_offsets = src_offsets.begin(),
                     partition_offsets = partition_offsets.begin(),
                     partitions = partitions.data(),
                     per_partition_metadata_size] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const partition_offset = partition_offsets[partition_index];
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;

      partition_header const*const pheader = reinterpret_cast<partition_header const*>(partitions + partition_offset);

      auto const validity_section_offset = partition_offset + per_partition_metadata_size;
      auto const validity_buf_index = (col_index * 3) + (partition_index * buffers_per_partition);
      src_offsets[validity_buf_index] += validity_section_offset;

      auto const offset_section_offset = cudf::util::round_up_safe(validity_section_offset + pheader->validity_size, validity_pad);
      auto const offset_buf_index = ((col_index * 3) + 1) + (partition_index * buffers_per_partition);
      src_offsets[offset_buf_index] += offset_section_offset;
      
      auto const data_section_offset = cudf::util::round_up_safe(offset_section_offset + pheader->offset_size, offset_pad);
      auto const data_buf_index = ((col_index * 3) + 2) + (partition_index * buffers_per_partition);
      src_offsets[data_buf_index] += data_section_offset;

      //printf("MHO: %d, partition_index = %d, partition_offset = %lu, col_index = %d, col_instance_index = %d, validity_offset = %lu, offset_offset = %lu, data_offset = %lu\n", 
        //    i, (int)partition_index, partition_offset, (int)col_index, (int)col_instance_index, validity_section_offset, offset_section_offset, data_section_offset);
    });
    //print_span(cudf::device_span<size_t const>{src_offsets});

    // compute: generate destination buffer offsets
    // NOTE: dst_offsets is arranged in destination buffer order, not source buffer order.
    // We're wasting a little work here as the validity computation has to be redone later.    
    auto dst_buf_key = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([num_partitions] __device__ (size_t i){
      return i / num_partitions;
    }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin(), num_partitions, buffers_per_partition] __device__ (size_t i){
      auto const dst_buf_index = i / num_partitions;
      auto const partition_index = i % num_partitions;
      auto const src_buf_index = (partition_index * buffers_per_partition) + dst_buf_index;
      return src_sizes_unpadded[src_buf_index];
    }));
    // dst_offsets is arranged     
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  dst_buf_key,
                                  dst_buf_key + num_src_buffers,
                                  size_iter,
                                  dst_offsets.begin());
    //print_span(cudf::device_span<size_t const>{dst_offsets});

    // for validity, we need to do a little more work. our destination positions are defined by bit position,
    // not byte position. so round down into the nearest starting bitmask word.    
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                     iter,
                     iter + num_column_instances,
                     [column_info = column_info.begin(),
                      num_columns,
                      buffers_per_partition,
                      partition_row_indices = partition_row_indices.begin(),
                      dst_offsets = dst_offsets.begin()] __device__ (size_t i){
      auto const col_index = i / num_columns;
      auto const partition_index = i % num_columns;
      auto const& cinfo = column_info[col_index];
      if(cinfo.has_validity){
        auto const validity_buf_index = (col_index * 3) + (partition_index * buffers_per_partition);
        dst_offsets[validity_buf_index] = (partition_row_indices[partition_index] / 32) * sizeof(bitmask_type);
      }
    });
    //print_span(cudf::device_span<size_t const>{dst_offsets});
  }
  
  // generate copy batches ------------------------------------

  // generate batches.
  // - validity and offsets will be copied by custom kernels, so we will subdivide them them into batches of 1 MB
  // - data is copied by cub, so we will do no subdivision of the batches under the assumption that cub will make it's
  //   own smart internal decisions
  auto batch_count_iter = cudf::detail::make_counting_transform_iterator(0, 
                                                                         cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin()] __device__ (size_t i){
                                                                           return i % 3 == 2 ? 1 : size_to_batch_count(src_sizes_unpadded[i]);
                                                                         }));
  auto copy_batches = transform_expand(batch_count_iter, 
                                       batch_count_iter + src_sizes_unpadded.size(),
                                       cuda::proclaim_return_type<assemble_batch>([dst_buffers = dst_buffers.begin(),
                                                                                   dst_offsets = dst_offsets.begin(),
                                                                                   partitions = partitions.data(),
                                                                                   buffers_per_partition,
                                                                                   num_partitions,
                                                                                   src_sizes_unpadded = src_sizes_unpadded.begin(),
                                                                                   src_offsets = src_offsets.begin(),
                                                                                   desired_assemble_batch_size = desired_assemble_batch_size,
                                                                                   partition_row_indices = partition_row_indices.begin()] __device__ (size_t src_buf_index, size_t batch_index){
                                         auto const batch_offset = batch_index * desired_assemble_batch_size;
                                         auto const partition_index = src_buf_index / buffers_per_partition;
                                         
                                         auto const src_offset = src_offsets[src_buf_index];
                                        
                                         auto const dst_buf_index = src_buf_index % buffers_per_partition;
                                         auto const dst_offset_index = (dst_buf_index * num_partitions) + partition_index;
                                         auto const dst_offset = dst_offsets[dst_offset_index];

                                         auto const bytes = (src_buf_index % 3 == 2) ? src_sizes_unpadded[src_buf_index] : std::min(src_sizes_unpadded[src_buf_index] - batch_offset, desired_assemble_batch_size);
                                         
                                         /*
                                         printf("ET: partition_index=%lu, src_buf_index=%lu, dst_buf_index=%lu, batch_index=%lu, src_offset=%lu, dst_offset=%lu bytes=%lu bit_shift = %d\n", 
                                           partition_index,
                                           src_buf_index,
                                           dst_buf_index,
                                           batch_index,
                                           src_offset + batch_offset,
                                           dst_offset + batch_offset,
                                           bytes,
                                           partition_row_indices[partition_index] % 32);
                                           */

                                         return assemble_batch {
                                          partitions + src_offset + batch_offset,
                                          dst_buffers[dst_buf_index] + dst_offset + batch_offset,
                                          bytes,
                                          0,  // TODO: handle offsets
                                          partition_row_indices[partition_index] % 32,  // bit shift for the validity copy step
                                          0};
                                         }),
                                       stream,
                                       mr);

  return {std::move(assemble_buffers), std::move(copy_batches)};
}

void assemble_copy(rmm::device_uvector<assemble_batch> const& batches, rmm::cuda_stream_view stream)
{
  // main data copy. everything except validity and offsets
  {
    auto input_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([batches = batches.begin()] __device__ (size_t i){
      //printf("SRC: %lu\n", (uint64_t)(batches[(i * 3) + 2].src));
      return reinterpret_cast<void*>(const_cast<int8_t*>(batches[(i * 3) + 2].src));
    }));
    auto output_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([batches = batches.begin()] __device__ (size_t i){
      //printf("DST: %lu\n", (uint64_t)(batches[(i * 3) + 2].dst));
      return reinterpret_cast<void*>(const_cast<int8_t*>(batches[(i * 3) + 2].dst));
    }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([batches = batches.begin()] __device__ (size_t i){
      //printf("SIZE: %lu\n", (uint64_t)(batches[(i * 3) + 2].size));
      return batches[(i * 3) + 2].size;
    }));

    size_t temp_storage_bytes;
    cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, batches.size() / 3, stream);
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
    cub::DeviceMemcpy::Batched(temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, batches.size() / 3, stream);
  }

#if 0
  // copy validity
  /*
  constexpr int block_size = 256;
  cudf::detail::grid_1d const grid{static_cast<cudf::thread_index_type>(batches.size()), block_size};
  copy_validity<block_size><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(batches);
  */  
  #endif

  stream.synchronize();
}

// assemble all the columns and the final table from the intermediate buffers
std::unique_ptr<cudf::table> build_table(std::vector<assemble_column_info> const& assembly_data,
                                         std::vector<rmm::device_buffer>& assembly_buffers,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = assembly_data.begin();
  auto buffer = assembly_buffers.begin();
  while(column != assembly_data.end()){
    std::tie(column, buffer) = cudf::type_dispatcher(cudf::data_type{column->type},
                                                     assemble_column_functor{stream, mr},
                                                     column,
                                                     buffer,
                                                     columns);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// assemble all the columns and the final table from the intermediate buffers
std::unique_ptr<cudf::table> build_empty_table(std::vector<shuffle_split_col_data> const& col_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = col_info.begin();
  while(column != col_info.end()){
    column = cudf::type_dispatcher(cudf::data_type{column->type},
                                   assemble_empty_column_functor{stream, mr},
                                   column,
                                   columns);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}; // anonymous namespace

std::unique_ptr<cudf::table> shuffle_assemble(shuffle_split_metadata const& metadata,
                                              cudf::device_span<int8_t const> partitions,
                                              cudf::device_span<size_t const> partition_offsets,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  // if the input is empty, just generate an empty table
  if(partition_offsets.size() == 1){
    return build_empty_table(metadata.col_info, stream, mr);
  }

  // generate the info structs representing the flattened column hierarchy. the total number of assembled rows, null counts, etc
  auto [column_info, h_column_info, column_instance_info, per_partition_metadata_size] = assemble_build_column_info(metadata, partitions, partition_offsets, stream, mr);

  // generate the (empty) output buffers based on the column info. note that is not a 1:1 mapping between column info
  // and buffers, since some columns will have validity and some will not.
  auto [dst_buffers, batches] = assemble_build_buffers(column_info, column_instance_info, partitions, partition_offsets, per_partition_metadata_size, stream, mr);  

  // copy the data. note that this does not sync.
  assemble_copy(batches, stream);

  // build the final table while the gpu is performing the copy
  auto ret = build_table(h_column_info, dst_buffers, stream, mr);
  stream.synchronize();
  return ret;
}

};  // namespace spark_rapids_jni
