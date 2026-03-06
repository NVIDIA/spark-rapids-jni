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

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>
#include <protobuf.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Protobuf wire-format encoding helpers (host side, for generating test data)
// ---------------------------------------------------------------------------

void encode_varint(std::vector<uint8_t>& buf, uint64_t value)
{
  while (value > 0x7F) {
    buf.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(value));
}

void encode_tag(std::vector<uint8_t>& buf, int field_number, int wire_type)
{
  encode_varint(buf, (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type));
}

void encode_varint_field(std::vector<uint8_t>& buf, int field_number, int64_t value)
{
  encode_tag(buf, field_number, /*WT_VARINT=*/0);
  encode_varint(buf, static_cast<uint64_t>(value));
}

void encode_fixed32_field(std::vector<uint8_t>& buf, int field_number, float value)
{
  encode_tag(buf, field_number, /*WT_32BIT=*/5);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_fixed64_field(std::vector<uint8_t>& buf, int field_number, double value)
{
  encode_tag(buf, field_number, /*WT_64BIT=*/1);
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_len_field(std::vector<uint8_t>& buf, int field_number, void const* data, size_t len)
{
  encode_tag(buf, field_number, /*WT_LEN=*/2);
  encode_varint(buf, len);
  auto const* p = static_cast<uint8_t const*>(data);
  buf.insert(buf.end(), p, p + len);
}

void encode_string_field(std::vector<uint8_t>& buf, int field_number, std::string const& s)
{
  encode_len_field(buf, field_number, s.data(), s.size());
}

// Encode a nested message: write its content into a temporary buffer, then emit as WT_LEN.
template <typename Fn>
void encode_nested_message(std::vector<uint8_t>& buf, int field_number, Fn&& content_fn)
{
  std::vector<uint8_t> inner;
  content_fn(inner);
  encode_len_field(buf, field_number, inner.data(), inner.size());
}

// Encode a packed repeated int32 field.
void encode_packed_repeated_int32(std::vector<uint8_t>& buf,
                                  int field_number,
                                  std::vector<int32_t> const& values)
{
  std::vector<uint8_t> packed;
  for (auto v : values) {
    encode_varint(packed, static_cast<uint64_t>(static_cast<uint32_t>(v)));
  }
  encode_len_field(buf, field_number, packed.data(), packed.size());
}

// ---------------------------------------------------------------------------
// Build a cuDF LIST<UINT8> column from host message buffers
// ---------------------------------------------------------------------------

std::unique_ptr<cudf::column> make_binary_column(std::vector<std::vector<uint8_t>> const& messages)
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();

  std::vector<int32_t> h_offsets(messages.size() + 1);
  h_offsets[0] = 0;
  for (size_t i = 0; i < messages.size(); i++) {
    h_offsets[i + 1] = h_offsets[i] + static_cast<int32_t>(messages[i].size());
  }
  int32_t total_bytes = h_offsets.back();

  std::vector<uint8_t> h_data;
  h_data.reserve(total_bytes);
  for (auto const& m : messages) {
    h_data.insert(h_data.end(), m.begin(), m.end());
  }

  rmm::device_buffer d_data(h_data.data(), h_data.size(), stream, mr);
  rmm::device_buffer d_offsets(h_offsets.data(), h_offsets.size() * sizeof(int32_t), stream, mr);
  stream.synchronize();

  auto child_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::UINT8}, total_bytes, std::move(d_data), rmm::device_buffer{}, 0);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(h_offsets.size()),
                                                    std::move(d_offsets),
                                                    rmm::device_buffer{},
                                                    0);

  return cudf::make_lists_column(static_cast<cudf::size_type>(messages.size()),
                                 std::move(offsets_col),
                                 std::move(child_col),
                                 0,
                                 rmm::device_buffer{});
}

// ---------------------------------------------------------------------------
// Schema + message generators for different benchmark scenarios
// ---------------------------------------------------------------------------

using nfd = spark_rapids_jni::nested_field_descriptor;

// Case 1: Flat scalars only — many top-level scalar fields.
//   message FlatMessage {
//     int32  f1 = 1;
//     int64  f2 = 2;
//     ...
//     float  f_k   = k;     (cycling through int32, int64, float, double, bool)
//     string s_k+1 = k+1;   (a few string fields)
//   }
struct FlatScalarCase {
  int num_int_fields;
  int num_string_fields;

  spark_rapids_jni::ProtobufDecodeContext build_context() const
  {
    spark_rapids_jni::ProtobufDecodeContext ctx;
    ctx.fail_on_errors = true;

    // type_id cycle for integer-like fields
    cudf::type_id int_types[] = {cudf::type_id::INT32,
                                 cudf::type_id::INT64,
                                 cudf::type_id::FLOAT32,
                                 cudf::type_id::FLOAT64,
                                 cudf::type_id::BOOL8};
    int wt_for_type[]         = {0 /*WT_VARINT*/, 0, 5 /*WT_32BIT*/, 1 /*WT_64BIT*/, 0};

    int fn = 1;
    for (int i = 0; i < num_int_fields; i++, fn++) {
      int ti  = i % 5;
      auto ty = int_types[ti];
      int wt  = wt_for_type[ti];
      int enc = spark_rapids_jni::ENC_DEFAULT;
      if (ty == cudf::type_id::FLOAT32) enc = spark_rapids_jni::ENC_FIXED;
      if (ty == cudf::type_id::FLOAT64) enc = spark_rapids_jni::ENC_FIXED;
      ctx.schema.push_back({fn, -1, 0, wt, ty, enc, false, false, false});
    }
    for (int i = 0; i < num_string_fields; i++, fn++) {
      ctx.schema.push_back(
        {fn, -1, 0, 2 /*WT_LEN*/, cudf::type_id::STRING, 0, false, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(5, 50);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      int fn    = 1;
      for (int i = 0; i < num_int_fields; i++, fn++) {
        int ti = i % 5;
        switch (ti) {
          case 0: encode_varint_field(buf, fn, int_dist(rng)); break;
          case 1: encode_varint_field(buf, fn, int_dist(rng)); break;
          case 2: encode_fixed32_field(buf, fn, static_cast<float>(int_dist(rng))); break;
          case 3: encode_fixed64_field(buf, fn, static_cast<double>(int_dist(rng))); break;
          case 4: encode_varint_field(buf, fn, rng() % 2); break;
        }
      }
      for (int i = 0; i < num_string_fields; i++, fn++) {
        int len = str_len_dist(rng);
        std::string s(len, ' ');
        for (int c = 0; c < len; c++) {
          s[c] = alphabet[rng() % alphabet.size()];
        }
        encode_string_field(buf, fn, s);
      }
    }
    return messages;
  }
};

// Case 2: Nested message — a top-level message with a nested struct child.
//   message OuterMessage {
//     int32  id = 1;
//     string name = 2;
//     InnerMessage inner = 3;
//   }
//   message InnerMessage {
//     int32  x = 1;
//     int64  y = 2;
//     string data = 3;
//     ... (num_inner_fields fields)
//   }
struct NestedMessageCase {
  int num_inner_fields;  // scalar fields inside InnerMessage

  spark_rapids_jni::ProtobufDecodeContext build_context() const
  {
    spark_rapids_jni::ProtobufDecodeContext ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, top-level)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: name (string, top-level)
    ctx.schema.push_back({2, -1, 0, 2, cudf::type_id::STRING, 0, false, false, false});
    // idx 2: inner (STRUCT, top-level)
    ctx.schema.push_back({3, -1, 0, 2, cudf::type_id::STRUCT, 0, false, false, false});

    // Inner message children (parent_idx=2, depth=1)
    cudf::type_id inner_types[] = {
      cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::STRING};
    int inner_wt[] = {0, 0, 2};

    for (int i = 0; i < num_inner_fields; i++) {
      int ti = i % 3;
      ctx.schema.push_back({i + 1, 2, 1, inner_wt[ti], inner_types[ti], 0, false, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(5, 30);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));
      encode_string_field(buf, 2, random_string(str_len_dist(rng)));

      encode_nested_message(buf, 3, [&](std::vector<uint8_t>& inner) {
        for (int i = 0; i < num_inner_fields; i++) {
          int ti = i % 3;
          switch (ti) {
            case 0: encode_varint_field(inner, i + 1, int_dist(rng)); break;
            case 1: encode_varint_field(inner, i + 1, int_dist(rng)); break;
            case 2: encode_string_field(inner, i + 1, random_string(str_len_dist(rng))); break;
          }
        }
      });
    }
    return messages;
  }
};

// Case 3: Repeated fields — top-level repeated scalars and a repeated nested message.
//   message RepeatedMessage {
//     int32           id = 1;
//     repeated int32  tags = 2;
//     repeated string labels = 3;
//     repeated Item   items = 4;
//   }
//   message Item {
//     int32  item_id = 1;
//     string item_name = 2;
//     int64  value = 3;
//   }
struct RepeatedFieldCase {
  int avg_tags_per_row;
  int avg_labels_per_row;
  int avg_items_per_row;

  spark_rapids_jni::ProtobufDecodeContext build_context() const
  {
    spark_rapids_jni::ProtobufDecodeContext ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, scalar)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: tags (repeated int32, packed)
    ctx.schema.push_back({2, -1, 0, 0, cudf::type_id::INT32, 0, true, false, false});
    // idx 2: labels (repeated string)
    ctx.schema.push_back({3, -1, 0, 2, cudf::type_id::STRING, 0, true, false, false});
    // idx 3: items (repeated STRUCT)
    ctx.schema.push_back({4, -1, 0, 2, cudf::type_id::STRUCT, 0, true, false, false});
    // idx 4: Item.item_id (int32, child of idx 3)
    ctx.schema.push_back({1, 3, 1, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 5: Item.item_name (string, child of idx 3)
    ctx.schema.push_back({2, 3, 1, 2, cudf::type_id::STRING, 0, false, false, false});
    // idx 6: Item.value (int64, child of idx 3)
    ctx.schema.push_back({3, 3, 1, 0, cudf::type_id::INT64, 0, false, false, false});

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(3, 20);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    // Vary count per row around the average (±50%)
    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];

      // id
      encode_varint_field(buf, 1, int_dist(rng));

      // tags (packed repeated int32)
      {
        int n = vary(avg_tags_per_row);
        std::vector<int32_t> tags(n);
        for (auto& t : tags)
          t = int_dist(rng);
        if (n > 0) encode_packed_repeated_int32(buf, 2, tags);
      }

      // labels (unpacked repeated string)
      {
        int n = vary(avg_labels_per_row);
        for (int i = 0; i < n; i++) {
          encode_string_field(buf, 3, random_string(str_len_dist(rng)));
        }
      }

      // items (repeated nested message)
      {
        int n = vary(avg_items_per_row);
        for (int i = 0; i < n; i++) {
          encode_nested_message(buf, 4, [&](std::vector<uint8_t>& inner) {
            encode_varint_field(inner, 1, int_dist(rng));
            encode_string_field(inner, 2, random_string(str_len_dist(rng)));
            encode_varint_field(inner, 3, int_dist(rng));
          });
        }
      }
    }
    return messages;
  }
};

// Case 4: Wide repeated message — stress-tests repeated struct child scanning.
//   message WideRepeatedMessage {
//     int32         id = 1;
//     repeated Item items = 2;
//   }
//   message Item {
//     int32 / int64 / float / double / bool / string child fields ...
//     ... (num_child_fields fields)
//   }
//
// This case is intentionally generic and contains no customer schema details.
// It is designed to exercise `scan_repeated_message_children_kernel` with a
// wide repeated STRUCT payload similar in shape to real-world schema-projection
// workloads.
struct WideRepeatedMessageCase {
  int num_child_fields;
  int avg_items_per_row;

  spark_rapids_jni::ProtobufDecodeContext build_context() const
  {
    spark_rapids_jni::ProtobufDecodeContext ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (scalar)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: items (repeated STRUCT)
    ctx.schema.push_back({2, -1, 0, 2, cudf::type_id::STRUCT, 0, true, false, false});

    cudf::type_id child_types[] = {cudf::type_id::INT32,
                                   cudf::type_id::INT64,
                                   cudf::type_id::FLOAT32,
                                   cudf::type_id::FLOAT64,
                                   cudf::type_id::BOOL8,
                                   cudf::type_id::STRING};
    int child_wt[]              = {0, 0, 5, 1, 0, 2};
    int child_enc[]             = {spark_rapids_jni::ENC_DEFAULT,
                                   spark_rapids_jni::ENC_DEFAULT,
                                   spark_rapids_jni::ENC_FIXED,
                                   spark_rapids_jni::ENC_FIXED,
                                   spark_rapids_jni::ENC_DEFAULT,
                                   spark_rapids_jni::ENC_DEFAULT};

    // Keep strings sparse so the case remains dominated by wide child scanning
    // rather than varlen copy traffic.
    for (int i = 0; i < num_child_fields; i++) {
      int ti = (i % 10 == 9) ? 5 : (i % 5);
      ctx.schema.push_back(
        {i + 1, 1, 1, child_wt[ti], child_types[ti], child_enc[ti], false, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(6, 18);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));

      int n = vary(avg_items_per_row);
      for (int item_idx = 0; item_idx < n; item_idx++) {
        encode_nested_message(buf, 2, [&](std::vector<uint8_t>& inner) {
          for (int i = 0; i < num_child_fields; i++) {
            int ti = (i % 10 == 9) ? 5 : (i % 5);
            int fn = i + 1;
            switch (ti) {
              case 0: encode_varint_field(inner, fn, int_dist(rng)); break;
              case 1: encode_varint_field(inner, fn, int_dist(rng)); break;
              case 2: encode_fixed32_field(inner, fn, static_cast<float>(int_dist(rng))); break;
              case 3: encode_fixed64_field(inner, fn, static_cast<double>(int_dist(rng))); break;
              case 4: encode_varint_field(inner, fn, rng() % 2); break;
              case 5: encode_string_field(inner, fn, random_string(str_len_dist(rng))); break;
            }
          }
        });
      }
    }
    return messages;
  }
};

// Case 4: Many repeated fields — stress-tests per-repeated-field sync overhead.
//   message WideRepeatedMessage {
//     int32              id = 1;
//     repeated int32     r_int_1 = 2;
//     repeated int32     r_int_2 = 3;
//     ...
//     repeated string    r_str_1 = N;
//     repeated string    r_str_2 = N+1;
//     ...
//   }
struct ManyRepeatedFieldsCase {
  int num_repeated_int;
  int num_repeated_str;

  spark_rapids_jni::ProtobufDecodeContext build_context() const
  {
    spark_rapids_jni::ProtobufDecodeContext ctx;
    ctx.fail_on_errors = true;

    int fn = 1;
    // idx 0: id (scalar)
    ctx.schema.push_back({fn++, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});

    for (int i = 0; i < num_repeated_int; i++) {
      ctx.schema.push_back({fn++, -1, 0, 0, cudf::type_id::INT32, 0, true, false, false});
    }
    for (int i = 0; i < num_repeated_str; i++) {
      ctx.schema.push_back({fn++, -1, 0, 2, cudf::type_id::STRING, 0, true, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows,
                                                      int avg_elems_per_field,
                                                      std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(3, 15);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };
    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      int fn    = 1;

      encode_varint_field(buf, fn++, int_dist(rng));

      for (int i = 0; i < num_repeated_int; i++) {
        int cur_fn = fn++;
        int n      = vary(avg_elems_per_field);
        if (n > 0) {
          std::vector<int32_t> vals(n);
          for (auto& v : vals)
            v = int_dist(rng);
          encode_packed_repeated_int32(buf, cur_fn, vals);
        }
      }
      for (int i = 0; i < num_repeated_str; i++) {
        int cur_fn = fn++;
        int n      = vary(avg_elems_per_field);
        for (int j = 0; j < n; j++) {
          encode_string_field(buf, cur_fn, random_string(str_len_dist(rng)));
        }
      }
    }
    return messages;
  }
};

}  // anonymous namespace

// ===========================================================================
// Benchmark 1: Flat scalars — measures per-field extraction overhead
// ===========================================================================
static void BM_protobuf_flat_scalars(nvbench::state& state)
{
  auto const num_rows   = static_cast<int>(state.get_int64("num_rows"));
  auto const num_fields = static_cast<int>(state.get_int64("num_fields"));
  int const num_str     = std::max(1, num_fields / 10);
  int const num_int     = num_fields - num_str;

  FlatScalarCase flat_case{num_int, num_str};
  auto ctx = flat_case.build_context();

  std::mt19937 rng(42);
  auto messages   = flat_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_flat_scalars)
  .set_name("Protobuf Flat Scalars")
  .add_int64_axis("num_rows", {10'000, 100'000, 500'000})
  .add_int64_axis("num_fields", {10, 50, 200});

// ===========================================================================
// Benchmark 2: Nested messages — measures nested struct build overhead
// ===========================================================================
static void BM_protobuf_nested(nvbench::state& state)
{
  auto const num_rows     = static_cast<int>(state.get_int64("num_rows"));
  auto const inner_fields = static_cast<int>(state.get_int64("inner_fields"));

  NestedMessageCase nested_case{inner_fields};
  auto ctx = nested_case.build_context();

  std::mt19937 rng(42);
  auto messages   = nested_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_nested)
  .set_name("Protobuf Nested Message")
  .add_int64_axis("num_rows", {10'000, 100'000, 500'000})
  .add_int64_axis("inner_fields", {5, 20, 100});

// ===========================================================================
// Benchmark 3: Repeated fields — measures repeated field pipeline overhead
// ===========================================================================
static void BM_protobuf_repeated(nvbench::state& state)
{
  auto const num_rows  = static_cast<int>(state.get_int64("num_rows"));
  auto const avg_items = static_cast<int>(state.get_int64("avg_items"));

  RepeatedFieldCase rep_case{/*avg_tags=*/5, /*avg_labels=*/3, /*avg_items=*/avg_items};
  auto ctx = rep_case.build_context();

  std::mt19937 rng(42);
  auto messages   = rep_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated)
  .set_name("Protobuf Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("avg_items", {1, 5, 20});

// ===========================================================================
// Benchmark 4: Wide repeated message — measures repeated struct child scan cost
// ===========================================================================
static void BM_protobuf_wide_repeated_message(nvbench::state& state)
{
  auto const num_rows         = static_cast<int>(state.get_int64("num_rows"));
  auto const num_child_fields = static_cast<int>(state.get_int64("num_child_fields"));
  auto const avg_items        = static_cast<int>(state.get_int64("avg_items"));

  WideRepeatedMessageCase wide_case{num_child_fields, avg_items};
  auto ctx = wide_case.build_context();

  std::mt19937 rng(42);
  auto messages   = wide_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_wide_repeated_message)
  .set_name("Protobuf Wide Repeated Message")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_child_fields", {20, 100, 200})
  .add_int64_axis("avg_items", {1, 5, 10});

// ===========================================================================
// Benchmark 5: Many repeated fields — measures per-field sync overhead at scale
// ===========================================================================
static void BM_protobuf_many_repeated(nvbench::state& state)
{
  auto const num_rows       = static_cast<int>(state.get_int64("num_rows"));
  auto const num_rep_fields = static_cast<int>(state.get_int64("num_rep_fields"));

  int const num_rep_str = std::max(1, num_rep_fields / 5);
  int const num_rep_int = num_rep_fields - num_rep_str;

  ManyRepeatedFieldsCase many_case{num_rep_int, num_rep_str};
  auto ctx = many_case.build_context();

  std::mt19937 rng(42);
  auto messages   = many_case.generate_messages(num_rows, /*avg_elems=*/3, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_many_repeated)
  .set_name("Protobuf Many Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("num_rep_fields", {10, 30, 50});
