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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <nvbench/nvbench.cuh>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/row_conversion/row_conversion.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_utilities.hpp>

void fixed_width(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  bool const to_row = state.get_bool("to_row");
  auto const table = create_random_table(cycle_dtypes({cudf::type_id::INT8,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::INT16,
                                                       cudf::type_id::INT64,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::BOOL8,
                                                       cudf::type_id::UINT16,
                                                       cudf::type_id::UINT8,
                                                       cudf::type_id::UINT64},
                                                      212),
                                         row_count{n_rows});

  std::vector<cudf::data_type> schema;
  cudf::size_type bytes_per_row = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto t = table->get_column(i).type();
    schema.push_back(t);
    bytes_per_row += cudf::size_of(t);
  }

  auto rows = cudf::jni::convert_to_rows_fixed_width_optimized(table->view());

  state.exec(nvbench::exec_tag::sync,
  [&](nvbench::launch& launch) {
      if (to_row) {
        auto _rows = cudf::jni::convert_to_rows_fixed_width_optimized(table->view());
      } else {
        for (cudf::lists_column_view const l : rows) {
          auto out = cudf::jni::convert_from_rows_fixed_width_optimized(l->view(), schema);
        }
      }
  });

  state.add_buffer_size(n_rows, "trc", "Total Rows");
  state.add_global_memory_reads<int64_t>(bytes_per_row * table->num_rows());
}

static void variable_or_fixed_width(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  bool const to_row = state.get_bool("to_row");
  bool const include_strings = state.get_bool("include_strings");
  std::vector<cudf::type_id> const table_types = include_strings ? {cudf::type_id::INT8,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::INT16,
                                                       cudf::type_id::INT64,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::BOOL8,
                                                       cudf::type_id::STRING,
                                                       cudf::type_id::UINT16,
                                                       cudf::type_id::UINT8,
                                                       cudf::type_id::UINT64} :
                                             {cudf::type_id::INT8,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::INT16,
                                                       cudf::type_id::INT64,
                                                       cudf::type_id::INT32,
                                                       cudf::type_id::BOOL8,
                                                       cudf::type_id::UINT16,
                                                       cudf::type_id::UINT8,
                                                       cudf::type_id::UINT64};

  auto const table = create_random_table(cycle_dtypes(table_types), 155), row_count{n_rows});

  cudf::size_type bytes_per_row = 0;
  cudf::size_type string_bytes  = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto t = table->get_column(i).type();
    if (is_fixed_width(t)) {
      bytes_per_row += cudf::size_of(t);
    } else if (t.id() == cudf::type_id::STRING) {
      auto sc = cudf::strings_column_view(table->get_column(i));
      string_bytes += sc.chars_size();
    }
  }

  state.exec(nvbench::exec_tag::sync,
  [&](nvbench::launch& launch) {
    auto new_rows = cudf::jni::convert_to_rows(table->view());
          if (to_row) {
        auto _rows = cudf::jni::convert_to_rows(table->view());
      } else {
        for (cudf::lists_column_view const l : rows) {
          auto out = cudf::jni::convert_from_rows(l->view(), schema);
        }
      }
  });

  state.add_buffer_size(n_rows, "trc", "Total Rows");
  state.add_global_memory_reads<int64_t>(bytes_per_row * table->num_rows());
}

NVBENCH_BENCH(fixed_width_to_row)
    .set_name("Fixed Width Only")
    .add_int64_axis("num_rows", {1 * 1024 * 1024, 4 * 1024 * 1024});
    .add_bool_axis("to_row", {true, false});

NVBENCH_BENCH(variable_or_fixed_width_to_row)
    .set_name("Fixed or Variable Width")
    .add_int64_axis("num_rows", {1 * 1024 * 1024, 4 * 1024 * 1024}
    .add_bool("to_row", {true, false})
    .add_bool("include_strings", {true, false}));
