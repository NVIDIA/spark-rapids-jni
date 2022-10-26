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

#include <cast_string.hpp>

#include <benchmarks/common/generate_input.hpp>

#include <nvbench/nvbench.cuh>

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_utilities.hpp>

void string_to_float(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const float_tbl = create_random_table({cudf::type_id::FLOAT32}, row_count{n_rows});
  auto const float_col = float_tbl->get_column(0);
  auto const string_col = cudf::strings::from_floats(float_col.view());

  state.exec(nvbench::exec_tag::sync,
  [&](nvbench::launch& launch) {
      auto rows = spark_rapids_jni::string_to_float(cudf::data_type{cudf::type_id::FLOAT32}, string_col->view(), false, cudf::get_default_stream());
  });
}

NVBENCH_BENCH(string_to_float)
    .set_name("Strings to Float Cast")
    .add_int64_axis("num_rows", {1 * 1024 * 1024, 100 * 1024 * 1024});
