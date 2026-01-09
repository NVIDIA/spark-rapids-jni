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

#include "protobuf.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <thrust/iterator/counting_iterator.h>

namespace {

constexpr int WT_VARINT = 0;
constexpr int WT_64BIT  = 1;
constexpr int WT_LEN    = 2;
constexpr int WT_32BIT  = 5;

}  // namespace

namespace spark_rapids_jni {

constexpr int ENC_DEFAULT = 0;
constexpr int ENC_FIXED   = 1;
constexpr int ENC_ZIGZAG  = 2;

}  // namespace spark_rapids_jni

namespace {

__device__ inline bool read_varint(uint8_t const* cur,
                                   uint8_t const* end,
                                   uint64_t& out,
                                   int& bytes)
{
  out       = 0;
  bytes     = 0;
  int shift = 0;
  while (cur < end && bytes < 10) {
    uint8_t b = *cur++;
    // For the 10th byte (bytes == 9, shift == 63), only the lowest bit is valid
    // since we can only fit 1 more bit into uint64_t
    if (bytes == 9 && (b & 0xFE) != 0) {
      return false;  // Invalid: 10th byte has more than 1 significant bit
    }
    out |= (static_cast<uint64_t>(b & 0x7Fu) << shift);
    bytes++;
    if ((b & 0x80u) == 0) { return true; }
    shift += 7;
  }
  return false;
}

__device__ inline bool skip_field(uint8_t const* cur,
                                  uint8_t const* end,
                                  int wt,
                                  uint8_t const*& out_cur)
{
  out_cur = cur;
  switch (wt) {
    case WT_VARINT: {
      uint64_t tmp;
      int n;
      if (!read_varint(out_cur, end, tmp, n)) return false;
      out_cur += n;
      return true;
    }
    case WT_64BIT:
      if (end - out_cur < 8) return false;
      out_cur += 8;
      return true;
    case WT_32BIT:
      if (end - out_cur < 4) return false;
      out_cur += 4;
      return true;
    case WT_LEN: {
      uint64_t len64;
      int n;
      if (!read_varint(out_cur, end, len64, n)) return false;
      out_cur += n;
      // Check for both buffer overflow and int overflow
      if (len64 > static_cast<uint64_t>(end - out_cur) || len64 > static_cast<uint64_t>(INT_MAX))
        return false;
      out_cur += static_cast<cudf::size_type>(len64);
      return true;
    }
    default: return false;
  }
}

template <typename T>
__device__ inline T load_le(uint8_t const* p);

template <>
__device__ inline uint32_t load_le<uint32_t>(uint8_t const* p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

template <>
__device__ inline uint64_t load_le<uint64_t>(uint8_t const* p)
{
  uint64_t v = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<uint64_t>(p[i]) << (8 * i));
  }
  return v;
}

template <typename OutT, bool ZigZag = false>
__global__ void extract_varint_kernel(
  cudf::column_device_view const d_in, int field_number, OutT* out, bool* valid, int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;
  if (in.nullable() && in.is_null(row)) {
    valid[row] = false;
    return;
  }

  // Use sliced child + offsets normalized to the slice start to correctly handle
  // list columns with non-zero row offsets (and any child offsets).
  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;
  // Defensive bounds checks: if offsets are inconsistent, avoid illegal memory access.
  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    valid[row] = false;
    return;
  }
  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

  bool found = false;
  OutT value{};
  while (cur < stop) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, stop, key, key_bytes)) {
      atomicExch(error_flag, 1);
      break;
    }
    cur += key_bytes;
    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);
    if (fn == 0) {
      atomicExch(error_flag, 1);
      break;
    }
    if (fn == field_number) {
      if (wt != WT_VARINT) {
        atomicExch(error_flag, 1);
        break;
      }
      uint64_t v;
      int n;
      if (!read_varint(cur, stop, v, n)) {
        atomicExch(error_flag, 1);
        break;
      }
      cur += n;
      if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
      value = static_cast<OutT>(v);
      found = true;
      // Continue scanning to allow "last one wins" semantics.
    } else {
      uint8_t const* next;
      if (!skip_field(cur, stop, wt, next)) {
        atomicExch(error_flag, 1);
        break;
      }
      cur = next;
    }
  }

  if (found) {
    out[row]   = value;
    valid[row] = true;
  } else {
    valid[row] = false;
  }
}

template <typename OutT, int WT>
__global__ void extract_fixed_kernel(
  cudf::column_device_view const d_in, int field_number, OutT* out, bool* valid, int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;
  if (in.nullable() && in.is_null(row)) {
    valid[row] = false;
    return;
  }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;
  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    valid[row] = false;
    return;
  }
  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

  bool found = false;
  OutT value{};
  while (cur < stop) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, stop, key, key_bytes)) {
      atomicExch(error_flag, 1);
      break;
    }
    cur += key_bytes;
    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);
    if (fn == 0) {
      atomicExch(error_flag, 1);
      break;
    }
    if (fn == field_number) {
      if (wt != WT) {
        atomicExch(error_flag, 1);
        break;
      }
      if constexpr (WT == WT_32BIT) {
        if (stop - cur < 4) {
          atomicExch(error_flag, 1);
          break;
        }
        uint32_t raw = load_le<uint32_t>(cur);
        cur += 4;
        // Use memcpy to avoid undefined behavior from type punning
        memcpy(&value, &raw, sizeof(value));
      } else {
        if (stop - cur < 8) {
          atomicExch(error_flag, 1);
          break;
        }
        uint64_t raw = load_le<uint64_t>(cur);
        cur += 8;
        // Use memcpy to avoid undefined behavior from type punning
        memcpy(&value, &raw, sizeof(value));
      }
      found = true;
    } else {
      uint8_t const* next;
      if (!skip_field(cur, stop, wt, next)) {
        atomicExch(error_flag, 1);
        break;
      }
      cur = next;
    }
  }

  if (found) {
    out[row]   = value;
    valid[row] = true;
  } else {
    valid[row] = false;
  }
}

__global__ void extract_string_kernel(cudf::column_device_view const d_in,
                                      int field_number,
                                      cudf::strings::detail::string_index_pair* out_pairs,
                                      int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;
  if (in.nullable() && in.is_null(row)) {
    out_pairs[row] = cudf::strings::detail::string_index_pair{nullptr, 0};
    return;
  }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;
  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    out_pairs[row] = cudf::strings::detail::string_index_pair{nullptr, 0};
    return;
  }
  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

  cudf::strings::detail::string_index_pair pair{nullptr, 0};
  while (cur < stop) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, stop, key, key_bytes)) {
      atomicExch(error_flag, 1);
      break;
    }
    cur += key_bytes;
    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);
    if (fn == 0) {
      atomicExch(error_flag, 1);
      break;
    }
    if (fn == field_number) {
      if (wt != WT_LEN) {
        atomicExch(error_flag, 1);
        break;
      }
      uint64_t len64;
      int n;
      if (!read_varint(cur, stop, len64, n)) {
        atomicExch(error_flag, 1);
        break;
      }
      cur += n;
      // Check for both buffer overflow and int overflow
      if (len64 > static_cast<uint64_t>(stop - cur) || len64 > static_cast<uint64_t>(INT_MAX)) {
        atomicExch(error_flag, 1);
        break;
      }
      pair.first  = reinterpret_cast<char const*>(cur);
      pair.second = static_cast<cudf::size_type>(len64);
      cur += static_cast<cudf::size_type>(len64);
      // Continue scanning to allow "last one wins".
    } else {
      uint8_t const* next;
      if (!skip_field(cur, stop, wt, next)) {
        atomicExch(error_flag, 1);
        break;
      }
      cur = next;
    }
  }

  out_pairs[row] = pair;
}

inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<bool> const& valid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + valid.size();
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) { return ptr[i]; };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> decode_protobuf_to_struct(
  cudf::column_view const& binary_input,
  std::vector<int> const& field_numbers,
  std::vector<cudf::data_type> const& out_types,
  std::vector<int> const& encodings,
  bool fail_on_errors)
{
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");
  CUDF_EXPECTS(field_numbers.size() == out_types.size(),
               "field_numbers and out_types must have the same length");
  CUDF_EXPECTS(encodings.size() == out_types.size(),
               "encodings and out_types must have the same length");

  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();

  auto d_in = cudf::column_device_view::create(binary_input, stream);
  auto rows = binary_input.size();

  // Track parse errors across kernels.
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(out_types.size());

  auto const threads = 256;
  auto const blocks  = static_cast<int>((rows + threads - 1) / threads);

  for (std::size_t i = 0; i < out_types.size(); ++i) {
    auto const fn  = field_numbers[i];
    auto const dt  = out_types[i];
    auto const enc = encodings[i];
    switch (dt.id()) {
      case cudf::type_id::BOOL8: {
        rmm::device_uvector<uint8_t> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_DEFAULT) {
          extract_varint_kernel<uint8_t><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for BOOL8 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::INT32: {
        rmm::device_uvector<int32_t> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_ZIGZAG) {
          extract_varint_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_FIXED) {
          extract_fixed_kernel<int32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_DEFAULT) {
          extract_varint_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for INT32 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::UINT32: {
        rmm::device_uvector<uint32_t> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_FIXED) {
          extract_fixed_kernel<uint32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_DEFAULT) {
          extract_varint_kernel<uint32_t><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for UINT32 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::INT64: {
        rmm::device_uvector<int64_t> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_ZIGZAG) {
          extract_varint_kernel<int64_t, true><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_FIXED) {
          extract_fixed_kernel<int64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_DEFAULT) {
          extract_varint_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for INT64 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::UINT64: {
        rmm::device_uvector<uint64_t> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_FIXED) {
          extract_fixed_kernel<uint64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else if (enc == ENC_DEFAULT) {
          extract_varint_kernel<uint64_t><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for UINT64 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::FLOAT32: {
        rmm::device_uvector<float> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_DEFAULT) {
          extract_fixed_kernel<float, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for FLOAT32 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::FLOAT64: {
        rmm::device_uvector<double> out(rows, stream, mr);
        rmm::device_uvector<bool> valid(rows, stream, mr);
        if (enc == ENC_DEFAULT) {
          extract_fixed_kernel<double, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, out.data(), valid.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for FLOAT64 protobuf field");
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        children.push_back(
          std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::STRING: {
        rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(rows, stream, mr);
        if (enc == ENC_DEFAULT) {
          extract_string_kernel<<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, pairs.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for STRING protobuf field");
        }
        children.push_back(
          cudf::strings::detail::make_strings_column(pairs.begin(), pairs.end(), stream, mr));
        break;
      }
      case cudf::type_id::LIST: {
        // For protobuf `bytes` fields: we reuse the string extraction kernel to get the
        // length-delimited raw bytes. The resulting strings column is then re-interpreted as
        // LIST<INT8> by extracting its internal offsets and char data (which is just raw bytes).
        rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(rows, stream, mr);
        if (enc == ENC_DEFAULT) {
          extract_string_kernel<<<blocks, threads, 0, stream.value()>>>(
            *d_in, fn, pairs.data(), d_error.data());
        } else {
          CUDF_FAIL("Unsupported encoding for LIST protobuf field");
        }
        auto strings =
          cudf::strings::detail::make_strings_column(pairs.begin(), pairs.end(), stream, mr);

        // Use strings_column_view to get the underlying data
        cudf::strings_column_view scv(*strings);
        auto const null_count = strings->null_count();

        // Get offsets - need to copy since we can't take ownership from a view
        auto offsets_col = std::make_unique<cudf::column>(scv.offsets(), stream, mr);

        // Get chars data as INT8 column
        auto chars_data = scv.chars_begin(stream);
        auto chars_size = static_cast<cudf::size_type>(scv.chars_size(stream));
        rmm::device_uvector<int8_t> chars_vec(chars_size, stream, mr);
        if (chars_size > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(chars_vec.data(),
                                        chars_data,
                                        chars_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream.value()));
        }
        // Create INT8 column from chars data
        auto child_col = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT8},
          chars_size,
          chars_vec.release(),
          rmm::device_buffer{},  // no null mask for chars
          0);                    // no nulls

        // Get null mask
        auto null_mask = cudf::copy_bitmask(*strings, stream, mr);

        children.push_back(cudf::make_lists_column(rows,
                                                   std::move(offsets_col),
                                                   std::move(child_col),
                                                   null_count,
                                                   std::move(null_mask),
                                                   stream,
                                                   mr));
        break;
      }
      default: CUDF_FAIL("Unsupported output type for protobuf decoder");
    }
  }

  // Check for kernel launch errors
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  // Check for any parse errors.
  int h_error = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0, "Malformed protobuf message or unsupported wire type");
  }

  // Note: We intentionally do NOT propagate input nulls to the output STRUCT validity.
  // The expected semantics for this low-level helper (see ProtobufTest) are:
  // - The STRUCT row is always valid (non-null)
  // - Individual children are null if the input message is null or the field is missing
  //
  // Higher-level Spark expressions can still apply their own null semantics if desired.
  rmm::device_buffer struct_mask{0, stream, mr};
  auto const struct_null_count = 0;

  return cudf::make_structs_column(
    rows, std::move(children), struct_null_count, std::move(struct_mask), stream, mr);
}

}  // namespace spark_rapids_jni
