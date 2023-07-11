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

#include "murmur_hash.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace {

using bloom_hash_type = spark_rapids_jni::murmur_hash_value_type;

__device__ inline std::pair<cudf::size_type, cudf::bitmask_type> gpu_get_hash_mask(
  bloom_hash_type h, cudf::size_type bloom_filter_bits)
{
  // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L94
  auto const index      = (h < 0 ? ~h : h) % static_cast<bloom_hash_type>(bloom_filter_bits);
  auto const word_index = cudf::word_index(index);
  auto const bit_index  = cudf::intra_word_index(index);
  return {word_index, (1 << bit_index)};
}

__global__ void gpu_bloom_filter_build(cudf::bitmask_type* const bloom_filter,
                                       cudf::size_type bloom_filter_bits,
                                       cudf::device_span<int64_t const> input,
                                       cudf::size_type num_hashes)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= input.size()) { return; }

  // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L87
  bloom_hash_type const h1 = MurmurHash3_32<int64_t>(0)(input[tid]);
  bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(input[tid]);

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

  __device__ bool operator()(int64_t input)
  {
    // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L110
    // this code could be combined with the very similar code in gpu_bloom_filter_build. i've
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

}  // anonymous namespace

rmm::device_uvector<cudf::bitmask_type> bloom_filter_create(cudf::size_type bloom_filter_bits,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<cudf::bitmask_type> out(
    cudf::num_bitmask_words(bloom_filter_bits), stream, mr);
  cudaMemsetAsync(out.data(), 0, out.size() * sizeof(cudf::bitmask_type), stream);
  return out;
}

void bloom_filter_build(rmm::device_uvector<cudf::bitmask_type>& bloom_filter,
                        cudf::size_type bloom_filter_bits,
                        cudf::column_view const& input,
                        cudf::size_type num_hashes,
                        rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(input.type() == cudf::data_type{cudf::type_id::INT64} && !input.nullable(),
               "bloom filter input expects a non-nullable column of int64s");
  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter size specified");

  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{input.size(), block_size, 1};
  gpu_bloom_filter_build<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    bloom_filter.data(), bloom_filter_bits, input, num_hashes);
}

rmm::device_uvector<cudf::bitmask_type> bloom_filter_merge(
  rmm::device_uvector<cudf::bitmask_type> const& a,
  rmm::device_uvector<cudf::bitmask_type> const& b,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(a.size() == b.size(),
               "bloom_filter_merge encountered mismatched input filter sizes");

  rmm::device_uvector<cudf::bitmask_type> out(a.size(), stream, mr);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + a.size(),
    out.begin(),
    [a = a.begin(), b = b.begin()] __device__(cudf::size_type i) { return a[i] | b[i]; });

  return out;
}

std::unique_ptr<cudf::column> bloom_filter_probe(
  cudf::column_view const& input,
  rmm::device_uvector<cudf::bitmask_type> const& bloom_filter,
  cudf::size_type bloom_filter_bits,
  cudf::size_type num_hashes,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == cudf::data_type{cudf::type_id::INT64} && !input.nullable(),
               "bloom filter input expects a non-nullable column of int64s");
  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter");

  auto out = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    input.begin<int64_t>(),
                    input.end<int64_t>(),
                    out->mutable_view().begin<bool>(),
                    bloom_probe_functor{bloom_filter.data(), bloom_filter_bits, num_hashes});
  return out;
}

}  // namespace spark_rapids_jni
