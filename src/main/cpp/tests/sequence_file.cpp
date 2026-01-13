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

#include "sequence_file.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace {

/**
 * @brief Write a big-endian int32 to a buffer at the given offset.
 */
void write_int32_be(std::vector<uint8_t>& buffer, size_t offset, int32_t value)
{
  buffer[offset + 0] = static_cast<uint8_t>((value >> 24) & 0xFF);
  buffer[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
  buffer[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFF);
  buffer[offset + 3] = static_cast<uint8_t>(value & 0xFF);
}

/**
 * @brief Generate test sync marker.
 */
std::vector<uint8_t> generate_sync_marker()
{
  return {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};
}

/**
 * @brief Build SequenceFile record data (no header, just records).
 *
 * Record format:
 *   recordLength (int32 BE) = keyLen + valueLen
 *   keyLength (int32 BE)
 *   keyBytes
 *   valueBytes
 *
 * Optionally insert sync markers.
 */
std::vector<uint8_t> build_test_data(
  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> const& records,
  std::vector<uint8_t> const& sync_marker,
  int insert_sync_every = 0)
{
  std::vector<uint8_t> data;
  int record_count = 0;

  for (auto const& [key, value] : records) {
    // Optionally insert sync marker
    if (insert_sync_every > 0 && record_count > 0 && (record_count % insert_sync_every) == 0) {
      // Sync marker indicator: -1 (0xFFFFFFFF)
      size_t sync_start = data.size();
      data.resize(data.size() + 4 + 16);
      write_int32_be(data, sync_start, -1);
      std::copy(sync_marker.begin(), sync_marker.end(), data.begin() + sync_start + 4);
    }

    int32_t key_len    = static_cast<int32_t>(key.size());
    int32_t value_len  = static_cast<int32_t>(value.size());
    int32_t record_len = key_len + value_len;

    size_t record_start = data.size();
    data.resize(data.size() + 8 + record_len);

    write_int32_be(data, record_start, record_len);
    write_int32_be(data, record_start + 4, key_len);
    std::copy(key.begin(), key.end(), data.begin() + record_start + 8);
    std::copy(value.begin(), value.end(), data.begin() + record_start + 8 + key_len);

    record_count++;
  }

  return data;
}

/**
 * @brief Copy host data to device buffer.
 */
rmm::device_uvector<uint8_t> to_device(std::vector<uint8_t> const& host_data,
                                       rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint8_t> device_data(host_data.size(), stream);
  cudaMemcpyAsync(
    device_data.data(), host_data.data(), host_data.size(), cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();
  return device_data;
}

/**
 * @brief Extract data from a LIST<UINT8> column as vectors of byte vectors.
 */
std::vector<std::vector<uint8_t>> extract_list_data(cudf::column_view const& col,
                                                    rmm::cuda_stream_view stream)
{
  auto lists_view = cudf::lists_column_view(col);
  auto offsets    = lists_view.offsets();
  auto child      = lists_view.child();

  // Copy offsets to host
  std::vector<int32_t> h_offsets(offsets.size());
  cudaMemcpyAsync(h_offsets.data(),
                  offsets.data<int32_t>(),
                  offsets.size() * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());

  // Copy child data to host
  std::vector<uint8_t> h_child(child.size());
  if (child.size() > 0) {
    cudaMemcpyAsync(
      h_child.data(), child.data<uint8_t>(), child.size(), cudaMemcpyDeviceToHost, stream.value());
  }

  stream.synchronize();

  // Build result
  std::vector<std::vector<uint8_t>> result;
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    int32_t start = h_offsets[i];
    int32_t end   = h_offsets[i + 1];
    result.emplace_back(h_child.begin() + start, h_child.begin() + end);
  }

  return result;
}

}  // namespace

class SequenceFileTest : public cudf::test::BaseFixture {};

TEST_F(SequenceFileTest, EmptyData)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  auto result = spark_rapids_jni::parse_sequence_file(nullptr, 0, sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_TRUE(result.key_column != nullptr);
  EXPECT_TRUE(result.value_column != nullptr);
  EXPECT_EQ(result.key_column->size(), 0);
  EXPECT_EQ(result.value_column->size(), 0);
}

TEST_F(SequenceFileTest, SingleRecord)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<uint8_t> key   = {'k', 'e', 'y', '1'};
  std::vector<uint8_t> value = {'v', 'a', 'l', 'u', 'e', '1'};

  auto host_data   = build_test_data({{key, value}}, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 1);

  auto keys   = extract_list_data(result.key_column->view(), stream);
  auto values = extract_list_data(result.value_column->view(), stream);

  ASSERT_EQ(keys.size(), 1);
  ASSERT_EQ(values.size(), 1);
  EXPECT_EQ(keys[0], key);
  EXPECT_EQ(values[0], value);
}

TEST_F(SequenceFileTest, MultipleRecords)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records = {
    {{'k', '1'}, {'v', '1'}},
    {{'k', '2', '2'}, {'v', '2', '2', '2'}},
    {{'k', '3', '3', '3'}, {'v', '3'}},
  };

  auto host_data   = build_test_data(records, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 3);

  auto keys   = extract_list_data(result.key_column->view(), stream);
  auto values = extract_list_data(result.value_column->view(), stream);

  ASSERT_EQ(keys.size(), 3);
  ASSERT_EQ(values.size(), 3);

  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(keys[i], records[i].first) << "Key mismatch at index " << i;
    EXPECT_EQ(values[i], records[i].second) << "Value mismatch at index " << i;
  }
}

TEST_F(SequenceFileTest, RecordsWithSyncMarkers)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  // Create 10 records with sync markers every 3 records
  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records;
  for (int i = 0; i < 10; ++i) {
    std::vector<uint8_t> key   = {'k', static_cast<uint8_t>('0' + i)};
    std::vector<uint8_t> value = {'v', static_cast<uint8_t>('0' + i), 'x', 'y', 'z'};
    records.emplace_back(key, value);
  }

  auto host_data   = build_test_data(records, sync_marker, 3);  // Sync every 3 records
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 10);

  auto keys   = extract_list_data(result.key_column->view(), stream);
  auto values = extract_list_data(result.value_column->view(), stream);

  ASSERT_EQ(keys.size(), 10);
  ASSERT_EQ(values.size(), 10);

  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(keys[i], records[i].first) << "Key mismatch at index " << i;
    EXPECT_EQ(values[i], records[i].second) << "Value mismatch at index " << i;
  }
}

TEST_F(SequenceFileTest, KeyOnly)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records = {
    {{'k', '1'}, {'v', '1'}},
    {{'k', '2'}, {'v', '2'}},
  };

  auto host_data   = build_test_data(records, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, false, stream);

  EXPECT_EQ(result.num_rows, 2);
  EXPECT_TRUE(result.key_column != nullptr);
  EXPECT_TRUE(result.value_column == nullptr);

  auto keys = extract_list_data(result.key_column->view(), stream);
  ASSERT_EQ(keys.size(), 2);
  EXPECT_EQ(keys[0], records[0].first);
  EXPECT_EQ(keys[1], records[1].first);
}

TEST_F(SequenceFileTest, ValueOnly)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records = {
    {{'k', '1'}, {'v', '1'}},
    {{'k', '2'}, {'v', '2'}},
  };

  auto host_data   = build_test_data(records, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, false, true, stream);

  EXPECT_EQ(result.num_rows, 2);
  EXPECT_TRUE(result.key_column == nullptr);
  EXPECT_TRUE(result.value_column != nullptr);

  auto values = extract_list_data(result.value_column->view(), stream);
  ASSERT_EQ(values.size(), 2);
  EXPECT_EQ(values[0], records[0].second);
  EXPECT_EQ(values[1], records[1].second);
}

TEST_F(SequenceFileTest, EmptyKeyAndValue)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records = {
    {{}, {}},                  // Both empty
    {{'k'}, {}},               // Value empty
    {{}, {'v'}},               // Key empty
    {{'k', '2'}, {'v', '2'}},  // Normal
  };

  auto host_data   = build_test_data(records, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 4);

  auto keys   = extract_list_data(result.key_column->view(), stream);
  auto values = extract_list_data(result.value_column->view(), stream);

  ASSERT_EQ(keys.size(), 4);
  ASSERT_EQ(values.size(), 4);

  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(keys[i], records[i].first) << "Key mismatch at index " << i;
    EXPECT_EQ(values[i], records[i].second) << "Value mismatch at index " << i;
  }
}

TEST_F(SequenceFileTest, LargeRecords)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  // Create a large key and value
  std::vector<uint8_t> large_key(10000);
  std::vector<uint8_t> large_value(50000);
  std::mt19937 rng(42);
  for (auto& b : large_key)
    b = static_cast<uint8_t>(rng() % 256);
  for (auto& b : large_value)
    b = static_cast<uint8_t>(rng() % 256);

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records = {
    {large_key, large_value},
    {{'s', 'm', 'a', 'l', 'l'}, {'v', 'a', 'l'}},
    {large_key, {'t', 'i', 'n', 'y'}},
  };

  auto host_data   = build_test_data(records, sync_marker);
  auto device_data = to_device(host_data, stream);

  auto result = spark_rapids_jni::parse_sequence_file(
    device_data.data(), device_data.size(), sync_marker, true, true, stream);

  EXPECT_EQ(result.num_rows, 3);

  auto keys   = extract_list_data(result.key_column->view(), stream);
  auto values = extract_list_data(result.value_column->view(), stream);

  ASSERT_EQ(keys.size(), 3);
  ASSERT_EQ(values.size(), 3);

  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(keys[i], records[i].first) << "Key mismatch at index " << i;
    EXPECT_EQ(values[i], records[i].second) << "Value mismatch at index " << i;
  }
}

TEST_F(SequenceFileTest, CountRecords)
{
  auto sync_marker = generate_sync_marker();
  auto stream      = cudf::get_default_stream();

  std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> records;
  for (int i = 0; i < 100; ++i) {
    records.emplace_back(std::vector<uint8_t>{'k', static_cast<uint8_t>(i)},
                         std::vector<uint8_t>{'v', static_cast<uint8_t>(i)});
  }

  auto host_data   = build_test_data(records, sync_marker, 10);  // Sync every 10 records
  auto device_data = to_device(host_data, stream);

  auto count =
    spark_rapids_jni::count_records(device_data.data(), device_data.size(), sync_marker, stream);

  EXPECT_EQ(count, 100);
}

TEST_F(SequenceFileTest, InvalidSyncMarkerSize)
{
  auto stream = cudf::get_default_stream();

  std::vector<uint8_t> invalid_sync = {0x01, 0x02, 0x03};  // Only 3 bytes

  std::vector<uint8_t> dummy_data = {0x00,
                                     0x00,
                                     0x00,
                                     0x08,  // recordLen = 8
                                     0x00,
                                     0x00,
                                     0x00,
                                     0x04,  // keyLen = 4
                                     't',
                                     'e',
                                     's',
                                     't',  // key
                                     'd',
                                     'a',
                                     't',
                                     'a'};  // value

  auto device_data = to_device(dummy_data, stream);

  EXPECT_THROW(spark_rapids_jni::parse_sequence_file(
                 device_data.data(), device_data.size(), invalid_sync, true, true, stream),
               cudf::logic_error);
}
