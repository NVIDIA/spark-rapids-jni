/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "bloom_filter.hpp"
#include "murmur_hash.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <byteswap.h>

namespace spark_rapids_jni {

namespace {

using bloom_hash_type = spark_rapids_jni::murmur_hash_value_type;

__device__ inline std::pair<cudf::size_type, cudf::bitmask_type> gpu_get_hash_mask(
  bloom_hash_type h, cudf::size_type bloom_filter_bits)
{
  // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L94
  auto const index = (h < 0 ? ~h : h) % static_cast<bloom_hash_type>(bloom_filter_bits);

  // spark expects serialized bloom filters to be big endian (64 bit longs),
  // so we will produce a big endian buffer. if spark CPU ends up consuming it, it can do so
  // directly. the gpu bloom filter implementation will always be handed the same serialized buffer.
  auto const word_index = cudf::word_index(index) ^ 0x1;  // word-swizzle within 64 bit long
  auto const bit_index =
    cudf::intra_word_index(index) ^ 0x18;                 // byte swizzle within the 32 bit word

  return {word_index, (1 << bit_index)};
}

template <bool nullable>
__global__ void gpu_bloom_filter_put(cudf::bitmask_type* const bloom_filter,
                                     cudf::size_type bloom_filter_bits,
                                     cudf::column_device_view input,
                                     cudf::size_type num_hashes)
{
  size_t const tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= input.size()) { return; }

  if constexpr (nullable) {
    if (!input.is_valid(tid)) { return; }
  }

  // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L87
  auto const el            = input.element<int64_t>(tid);
  bloom_hash_type const h1 = MurmurHash3_32<int64_t>(0)(el);
  bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(el);

  // set a bit in the bloom filter for each hashed value
  for (auto idx = 1; idx <= num_hashes; idx++) {
    bloom_hash_type combined_hash = h1 + (idx * h2);

    auto const [word_index, mask] = gpu_get_hash_mask(combined_hash, bloom_filter_bits);
    atomicOr(bloom_filter + word_index, mask);
  }
}

struct bloom_probe_functor {
  cudf::bitmask_type const* const bloom_filter;
  cudf::size_type const bloom_filter_bits;
  cudf::size_type const num_hashes;

  __device__ bool operator()(int64_t input) const
  {
    // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L110
    // this code could be combined with the very similar code in gpu_bloom_filter_put. i've
    // left it this way since the expectation is that we will early out fairly often, whereas
    // in the build case we never early out so doing the additional if() return check is pointless.
    bloom_hash_type const h1 = MurmurHash3_32<int64_t>(0)(input);
    bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(input);

    // set a bit in the bloom filter for each hashed value
    for (auto idx = 1; idx <= num_hashes; idx++) {
      bloom_hash_type combined_hash = h1 + (idx * h2);
      auto const [word_index, mask] = gpu_get_hash_mask(combined_hash, bloom_filter_bits);
      if (!(bloom_filter[word_index] & mask)) { return false; }
    }
    return true;
  }
};

constexpr int spark_bloom_filter_version = 1;

/*
  Pack a bloom_filter_header (passed as little endian) into a bloom filter buffer.
*/
void pack_bloom_filter_header(cudf::device_span<int8_t> buf,
                              bloom_filter_header const& header,
                              rmm::cuda_stream_view stream)
{
  // swizzle to big endian
  bloom_filter_header header_swizzled{
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.version))),
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_hashes))),
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_longs)))};

  // header goes at the top of the buffer
  cudaMemcpyAsync(
    buf.data(), &header_swizzled, bloom_filter_header_size, cudaMemcpyHostToDevice, stream);
}

/*
  Unpack bloom filter information from a bloom filter buffer. returns the header, a span
  representing the bloom filter bits and the number of bloom filter bits.
*/
std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type>, int> unpack_bloom_filter(
  cudf::device_span<int8_t> bloom_filter, rmm::cuda_stream_view stream)
{
  bloom_filter_header header_swizzled;
  cudaMemcpyAsync(&header_swizzled,
                  bloom_filter.data(),
                  bloom_filter_header_size,
                  cudaMemcpyDeviceToHost,
                  stream);
  stream.synchronize();

  // swizzle to little endian.
  bloom_filter_header header{
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header_swizzled.version))),
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header_swizzled.num_hashes))),
    static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header_swizzled.num_longs)))};
  return {header,
          {reinterpret_cast<cudf::bitmask_type*>(bloom_filter.data() + bloom_filter_header_size),
           static_cast<size_t>(header.num_longs) * 2},
          header.num_longs * 64};
}

/*
  Unpack bloom filter information from column_view that wraps a bloom filter buffer. returns the
  header, a span representing the bloom filter bits and the number of bloom filter bits.
*/
std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type>, int> unpack_bloom_filter(
  cudf::column_view const& bloom_filter, rmm::cuda_stream_view stream)
{
  // the const_cast is necessary because list_scalar does not provide a mutable_view() function.
  return unpack_bloom_filter(
    cudf::device_span<int8_t>{const_cast<int8_t*>(bloom_filter.data<int8_t>()),
                              static_cast<size_t>(bloom_filter.size())},
    stream);
}

}  // anonymous namespace

std::unique_ptr<cudf::list_scalar> bloom_filter_create(int num_hashes,
                                                       int bloom_filter_longs,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  auto const bloom_filter_size = (bloom_filter_longs * sizeof(int64_t));
  auto const buf_size          = bloom_filter_header_size + bloom_filter_size;

  // build the packed bloom filter buffer ------------------
  rmm::device_buffer buf{buf_size, stream, mr};

  // pack the header
  bloom_filter_header header{spark_bloom_filter_version, num_hashes, bloom_filter_longs};
  pack_bloom_filter_header({reinterpret_cast<int8_t*>(buf.data()), buf_size}, header, stream);
  // memset the bloom filter bits to 0.

  CUDF_CUDA_TRY(cudaMemsetAsync(reinterpret_cast<int8_t*>(buf.data()) + bloom_filter_header_size,
                                0,
                                bloom_filter_size,
                                stream));

  // create the 1-row list column and move it into a scalar.
  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::INT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

void bloom_filter_put(cudf::list_scalar& bloom_filter,
                      cudf::column_view const& input,
                      rmm::cuda_stream_view stream)
{
  // unpack the bloom filter
  auto [header, buffer, bloom_filter_bits] = unpack_bloom_filter(bloom_filter.view(), stream);

  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter size");
  CUDF_EXPECTS(buffer.size() == cudf::num_bitmask_words(bloom_filter_bits),
               "Bloom filter bit/length mismatch");
  CUDF_EXPECTS(buffer.size() % 8 == 0, "Bloom is not a whole number of 64 bit longs");

  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{input.size(), block_size, 1};
  auto d_input             = cudf::column_device_view::create(input);

  if (input.has_nulls()) {
    gpu_bloom_filter_put<true><<<grid.num_blocks, block_size, 0, stream.value()>>>(
      buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
  } else {
    gpu_bloom_filter_put<false><<<grid.num_blocks, block_size, 0, stream.value()>>>(
      buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
  }
}

std::unique_ptr<cudf::list_scalar> bloom_filter_merge(cudf::column_view const& bloom_filters,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  // unpack the bloom filter
  cudf::lists_column_view lcv(bloom_filters);
  // since the list child column is just a bunch of packed bloom filter buffers one after another,
  // we can just pass the base data pointer to unpack the first one.
  auto [header, _, bloom_filter_bits] = unpack_bloom_filter(lcv.child(), stream);

  auto const bloom_filter_size = (header.num_longs * sizeof(int64_t));
  auto const buf_size          = bloom_filter_header_size + bloom_filter_size;

  // build the packed bloom filter buffer ------------------
  rmm::device_buffer buf{buf_size, stream, mr};
  pack_bloom_filter_header({reinterpret_cast<int8_t*>(buf.data()), buf_size}, header, stream);

  auto src = lcv.child().data<int8_t>() + bloom_filter_header_size;
  auto dst = reinterpret_cast<cudf::bitmask_type*>(reinterpret_cast<int8_t*>(buf.data()) +
                                                   bloom_filter_header_size);

  // bitwise-or all the bloom filters together
  cudf::size_type num_words = header.num_longs * 2;
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + num_words,
    dst,
    [src, num_buffers = bloom_filters.size(), stride = buf_size] __device__(
      cudf::size_type word_index) {
      cudf::bitmask_type out = (reinterpret_cast<cudf::bitmask_type const*>(src))[word_index];
      for (auto idx = 1; idx < num_buffers; idx++) {
        out |= (reinterpret_cast<cudf::bitmask_type const*>(src + idx * stride))[word_index];
      }
      return out;
    });

  // create the 1-row list column and move it into a scalar.
  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::INT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

std::unique_ptr<cudf::column> bloom_filter_probe(cudf::column_view const& input,
                                                 cudf::list_scalar& bloom_filter,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  // unpack the bloom filter
  auto [header, buffer, bloom_filter_bits] = unpack_bloom_filter(bloom_filter.view(), stream);

  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter");
  CUDF_EXPECTS(buffer.size() == cudf::num_bitmask_words(bloom_filter_bits),
               "Bloom filter bit/length mismatch");
  CUDF_EXPECTS(buffer.size() % 8 == 0, "Bloom is not a whole number of 64 bit longs");

  // duplicate input mask
  auto out = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                           input.size(),
                                           cudf::copy_bitmask(input),
                                           input.null_count(),
                                           stream,
                                           mr);

  thrust::transform(
    rmm::exec_policy(stream),
    input.begin<int64_t>(),
    input.end<int64_t>(),
    out->mutable_view().begin<bool>(),
    bloom_probe_functor{
      buffer.data(), static_cast<cudf::size_type>(bloom_filter_bits), header.num_hashes});

  return out;
}

}  // namespace spark_rapids_jni
