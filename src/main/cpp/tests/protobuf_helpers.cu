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

#include "protobuf/protobuf_kernels.cuh"

#include <cudf_test/base_fixture.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace {

// Populate a device_uvector<bool> from a host pattern. We can't memcpy std::vector<bool>
// directly because it's bit-packed, so we stage through uint8_t.
rmm::device_uvector<bool> make_valid(std::vector<bool> const& host,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  std::vector<uint8_t> tmp(host.size());
  for (size_t i = 0; i < host.size(); i++) {
    tmp[i] = host[i] ? 1 : 0;
  }
  rmm::device_uvector<bool> dvec(host.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    dvec.data(), tmp.data(), host.size() * sizeof(bool), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();
  return dvec;
}

}  // namespace

struct ProtobufHelperTests : public cudf::test::BaseFixture {};

TEST_F(ProtobufHelperTests, NullMaskFromValidExactSize)
{
  // valid.size() == num_rows: the typical caller path.
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  auto valid = make_valid({true, false, true, true, false}, stream, mr);
  auto [mask, null_count] =
    spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(valid, 5, stream, mr);

  EXPECT_EQ(2, null_count);
}

TEST_F(ProtobufHelperTests, NullMaskFromValidPaddedBuffer)
{
  // valid.size() > num_rows: builders pad `valid` to max(1, num_rows) to avoid 0-sized
  // device_uvectors. The resulting mask must reflect `num_rows`, not the padded buffer size.
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  // Padded buffer of 3 bools, but logical row count is 0.
  auto valid = make_valid({false, false, false}, stream, mr);
  auto [mask, null_count] =
    spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(valid, 0, stream, mr);

  EXPECT_EQ(0, null_count);
}

TEST_F(ProtobufHelperTests, NullMaskFromValidPaddedTwoRows)
{
  // valid.size() > num_rows with non-trivial num_rows: only the first num_rows entries should
  // contribute to the mask; trailing padding bytes must not be observed.
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  // First 2 entries are the logical rows; trailing bytes are padding garbage.
  auto valid = make_valid({true, false, true, true, false}, stream, mr);
  auto [mask, null_count] =
    spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(valid, 2, stream, mr);

  EXPECT_EQ(1, null_count);
}

TEST_F(ProtobufHelperTests, NullMaskFromValidThrowsWhenBufferSmaller)
{
  // valid.size() < num_rows: misuse caught by CUDF_EXPECTS. Without the check the helper
  // would feed an out-of-range counting iterator into valid_if and produce a mask backed by
  // out-of-bounds reads.
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  auto valid = make_valid({true, true}, stream, mr);
  EXPECT_THROW(spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(valid, 5, stream, mr),
               cudf::logic_error);
}
