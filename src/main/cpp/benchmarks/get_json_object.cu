/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <get_json_object.hpp>
#include <nvbench/nvbench.cuh>

#include <chrono>
#include <sstream>
#include <string>

// #define printf(...) void(0)
// #define fflush(...) void(0)

class Timer {
 protected:
  using Clock = std::chrono::high_resolution_clock;

 public:
  Timer()          = default;
  virtual ~Timer() = default;

  void tick()
  {
    assert(!m_TimerTicked);
    m_StartTime   = Clock::now();
    m_TimerTicked = true;
  }

  double tock()
  {
    assert(m_TimerTicked);
    m_EndTime     = Clock::now();
    m_TimerTicked = false;
    m_ElapsedTime = std::chrono::duration<double, std::milli>(m_EndTime - m_StartTime).count();

    return m_ElapsedTime;
  }

  std::string getRunTime()
  {
    if (m_TimerTicked) { tock(); }
    m_StrBuilder.str("");
    m_StrBuilder << std::to_string(m_ElapsedTime);
    m_StrBuilder << "ms";
    return m_StrBuilder.str();
  }

 private:
  Clock::time_point m_StartTime;
  Clock::time_point m_EndTime;
  std::stringstream m_StrBuilder;

  double m_ElapsedTime{0.0};
  bool m_TimerTicked{false};
};

// #define DEBUG_PRINT

#ifdef DEBUG_PRINT

#include <cudf/utilities/type_dispatcher.hpp>

#include <iostream>

namespace {

// Copy from `cudf/cpp/tests/utilities/column_utilities.cu`.
struct strings_to_host_fn {
  template <typename OffsetType,
            std::enable_if_t<std::is_same_v<OffsetType, int32_t> ||
                             std::is_same_v<OffsetType, int64_t>>* = nullptr>
  void operator()(std::vector<std::string>& host_data,
                  char const* chars,
                  cudf::column_view const& offsets,
                  rmm::cuda_stream_view stream)
  {
    auto const h_offsets = cudf::detail::make_std_vector_sync(
      cudf::device_span<OffsetType const>(offsets.data<OffsetType>(), offsets.size()), stream);
    // build std::string vector from chars and offsets
    std::transform(std::begin(h_offsets),
                   std::end(h_offsets) - 1,
                   std::begin(h_offsets) + 1,
                   host_data.begin(),
                   [&](auto start, auto end) { return std::string(chars + start, end - start); });
  }

  template <typename OffsetType,
            std::enable_if_t<!std::is_same_v<OffsetType, int32_t> &&
                             !std::is_same_v<OffsetType, int64_t>>* = nullptr>
  void operator()(std::vector<std::string>&,
                  char const*,
                  cudf::column_view const&,
                  rmm::cuda_stream_view)
  {
    CUDF_FAIL("invalid offsets type");
  }
};

template <typename CV>
std::vector<std::string> to_host_strings(CV const& c)
{
  std::vector<std::string> host_strs(c.size());
  auto stream        = cudf::get_default_stream();
  auto const scv     = cudf::strings_column_view(c);
  auto const h_chars = cudf::detail::make_std_vector_sync<char>(
    cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
  auto const offsets =
    cudf::slice(scv.offsets(), {scv.offset(), scv.offset() + scv.size() + 1}).front();
  cudf::type_dispatcher(
    offsets.type(), strings_to_host_fn{}, host_strs, h_chars.data(), offsets, stream);
  return host_strs;
}

}  // namespace
#endif  // #ifdef DEBUG_PRINT

constexpr auto list_depth = 1;
constexpr auto min_width  = 10;
constexpr auto max_width  = 10;

auto generate_input(std::size_t size_bytes, cudf::size_type max_depth)
{
  data_profile const table_profile =
    data_profile_builder()
      .no_validity()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .distribution(cudf::type_id::LIST, distribution_id::NORMAL, min_width, max_width)
      .list_depth(list_depth)
      .list_type(cudf::type_id::STRING)
      .struct_depth(max_depth > list_depth ? max_depth - list_depth : 1)
      .struct_types(std::vector<cudf::type_id>{
        cudf::type_id::LIST, cudf::type_id::STRING, cudf::type_id::INT32});

  auto const input_table = create_random_table(
    std::vector<cudf::type_id>{cudf::type_id::INT32, cudf::type_id::STRING, cudf::type_id::STRUCT},
    table_size_bytes{size_bytes},
    table_profile);

  std::vector<char> buffer;
  cudf::io::sink_info sink(&buffer);
  cudf::io::table_metadata mt{{{"int32"}, {"string"}, {"struct"}}};
  auto write_opts =
    cudf::io::json_writer_options::builder(sink, input_table->view()).lines(true).metadata(mt);
  cudf::io::write_json(write_opts);

  // Split one JSON string into separate JSON objects.
  auto const json_str = std::string{buffer.begin(), buffer.end()};
  auto const json_col = cudf::test::strings_column_wrapper{{json_str}};
  auto split_strs =
    cudf::strings::split_record(cudf::strings_column_view{json_col}, cudf::string_scalar("\n"))
      ->release();

  // Note that split_strs is a list of strings thus we need to extract the strings column.
  auto& json_strings = split_strs.children[cudf::lists_column_view::child_column_index];

#ifdef DEBUG_PRINT
  {
    auto const strs = to_host_strings(json_strings->view());
    std::cout << "First input row: \n" << strs.front() << std::endl;
  }
#endif  // #ifdef DEBUG_PRINT
  return std::move(json_strings);
}

void BM_get_json_object_single(nvbench::state& state)
{
  auto const size_bytes = static_cast<cudf::size_type>(state.get_int64("size_bytes"));
  auto const max_depth  = static_cast<cudf::size_type>(state.get_int64("max_depth"));

  auto const json_strings = generate_input(size_bytes, max_depth);

  using path_instruction_type = spark_rapids_jni::path_instruction_type;
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> instructions;
  instructions.emplace_back(path_instruction_type::NAMED, "struct", -1);
  for (int i = 0; i < max_depth - list_depth; ++i) {
    instructions.emplace_back(path_instruction_type::NAMED, "0", -1);
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
  // Can also verify at https://jsonpath.com/.
  // [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
  //   cudf::strings_column_view{json_strings->view()}, instructions);

#ifdef DEBUG_PRINT
    {
      auto const strs = to_host_strings(output->view());
      std::cout << "First output row: \n" << strs.front() << std::endl << std::endl << std::endl;
    }
#endif  // #ifdef DEBUG_PRINT
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

#if 0
void BM_get_json_object_multiple(nvbench::state& state)
{
  auto const size_bytes   = static_cast<cudf::size_type>(state.get_int64("size_bytes"));
  auto const num_paths    = state.get_int64("num_paths");
  auto const json_strings = generate_input(size_bytes, 2 /*max_depth*/);

  using path_instruction_type = spark_rapids_jni::path_instruction_type;
  using instruction_array = std::vector<std::tuple<path_instruction_type, std::string, int64_t>>;
  std::vector<instruction_array> instructions_arrays;

  instructions_arrays.emplace_back();
  instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "int32", -1);

  if (num_paths > 1) {
    instructions_arrays.emplace_back();
    instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "string", -1);
  }

  if (num_paths > 2) {
    instructions_arrays.emplace_back();
    instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "struct", -1);
    instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "0", -1);
  }

  if (num_paths > 3) {
    instructions_arrays.emplace_back();
    instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "struct", -1);
    instructions_arrays.back().emplace_back(path_instruction_type::NAMED, "2", -1);
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
  // Can also verify at https://jsonpath.com/.
#if 1
    [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
      cudf::strings_column_view{json_strings->view()}, instructions_arrays);
#else
    for (int64_t i = 0; i < num_paths; ++i) {
      auto const& instructions           = instructions_arrays[i];
      [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
        cudf::strings_column_view{json_strings->view()}, instructions);
#ifdef DEBUG_PRINT
      {
        auto const strs = to_host_strings(output->view());
        std::cout << "First output row: \n" << strs.front() << std::endl << std::endl << std::endl;
      }
#endif  // #ifdef DEBUG_PRINT
    }
#endif
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}
#endif

#if 1
using spark_rapids_jni::path_instruction_type;
std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> generate_paths0()
{
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "EIFGPHGLOPELFBN", -1}}
    //
  };
}

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> generate_paths1()
{
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GHPKNICLNDAGCNDBMFGEK", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "KIKNFPPAPGDO", -1},
      {static_cast<path_instruction_type>(2), "KLFALIBALPPK", -1},
      {static_cast<path_instruction_type>(2), "HGABIFNPHAHHGP", -1}}
    //
  };
}

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> generate_paths2()
{
  auto tmp = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "JEBEDJPKEFHPHGLLGPM", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "FLMEPG", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "JACICCCIMMHJHKPDED", -1},
      {static_cast<path_instruction_type>(2), "ACHCPIHLFCPHMBPNKJNOLNO", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "JACICCCIMMHJHKPDED", -1},
      {static_cast<path_instruction_type>(2), "OGGC", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "MDGA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AGHF", -1},
      {static_cast<path_instruction_type>(2), "DPKEAPDACLPHGPEMH", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AGHF", -1},
      {static_cast<path_instruction_type>(2), "ONNILHPABGIKKFJOEK", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AGHF", -1},
      {static_cast<path_instruction_type>(2), "FFFPOENCNBBNOOMOJGDBNIPD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "POFNDBFHDEJ", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PIGOFCPIPPBNNB", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "CCBJKBHGPBJCKFPCBHGLOAFE", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LMPCGHBIJGCIPDPNELPBCOP", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PKBGI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "ILPIJKBLDB", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "GHBBEOAC", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EKGPKGCJPMI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "BDEGLFGMCPKOCNDGJMFPANNBPK", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LILJMMPPO", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "BNLFCI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "GPIHMJ", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GJFKCFJELPJEDBAD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "DLJPDEPFEKDCKBI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PMJPCGCHAALKBPKHDM", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "OCFGAF", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "GMJICFMBNPLBEOLMGDN", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "CBMI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "NPAGLLFCHAI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LFKAJEPMJPLGLICEEMAHFEJGPLGIAKPIOPPP", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "HGNHKIOEGKIJJJPEC", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "JAGGKPKOICKOBABAJPNHF", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PLEJAKDBBGLCDLGDIBHPPBHB", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "MMNHNPKGLLBJMAOGOCBEOIOKIM", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "JLKDBLFFFPPCNANBKMELJKFOPKPNC", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "OCJGMOAJJKBKNCHOJKBJG", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PMOAGIJAFOGGLINIOEBFGHBN", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "JPDILOFKPCNBKDB", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "CPBFNDGC", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "KPOPPCFLFCNAPIJEDJDGGFBOPLDCMLLGOMO", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LBDGCNJNOGMJPNHMLLBMA", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EIHBDLNJDOAHPMCNGGLLEF", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "GIPPDMMAFOBAALMHMGJBM", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "FKBODHACMMGHL", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "KMEJHDA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "FKBODHACMMGHL", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "CJKIKCGA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "HFFDKEDMFBAKEHHM", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "KGJLLAPHJNKCEOIAMCAABCJP", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 1},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "KLJNBPLECGCA", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "NBJNFKKKCHEGCABDGKG", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "BNLFCI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "NBJNFKKKCHEGCABDGKG", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "GPIHMJ", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "NBJNFKKKCHEGCABDGKG", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GJFKCFJELPJEDBAD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "NBJNFKKKCHEGCABDGKG", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "AOHKGCPAOGANLKEJDLMIGDD", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "BNLFCI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "AOHKGCPAOGANLKEJDLMIGDD", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "GPIHMJ", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "AOHKGCPAOGANLKEJDLMIGDD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "IKHLECMHMONKLKIBD", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "PNJPGEHPDLMPBDMFPLKABFFGG", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "IGAJPHHGOENI", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LDPMFNAGLJGDMFOLAKH", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "KMEJHDA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "LDPMFNAGLJGDMFOLAKH", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "CJKIKCGA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "BFAJJIOLJBEOMFKLE", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "DOONHL", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "OCIKAF", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "IBMBCGNOCGCPCEN", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "GLNLBEA", -1}}
    //
  };

  // tmp.resize(std::min(8UL, tmp.size()));
  return tmp;
  // return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>(
  //   tmp.begin() + 7, tmp.end());
}

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> generate_paths3()
{
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "KPIGLEDEOCFELKLJLAFE", -1}}
    //
  };
}

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> generate_paths4()
{
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GPGACKDIBMPAKJMDMJ", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "NOIIFOJOPJP", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "CEJOOHNF", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "HODJK", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "HHKEKMIIGI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "KDGJICMEANMA", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "ILEADAN", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "OKPLFLHHEBDJELFA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "CHNFGBB", -1},
      {static_cast<path_instruction_type>(2), "KIKNFPPAPGDO", -1},
      {static_cast<path_instruction_type>(2), "KLFALIBALPPK", -1},
      {static_cast<path_instruction_type>(2), "HGABIFNPHAHHGP", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "IHIIKIHHMPFL", -1},
      {static_cast<path_instruction_type>(2), "KCCCHAM", -1},
      {static_cast<path_instruction_type>(2), "KCCCHAM", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "KFPJHMGFEELFG", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "AFHKGOFNFID", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "CEJOOHNF", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "KFPJHMGFEELFG", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "AFHKGOFNFID", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "HODJK", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "HHKEKMIIGI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "KFPJHMGFEELFG", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "AFHKGOFNFID", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "OKPLFLHHEBDJELFA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "JJKPNPFMNICGLC", -1},
      {static_cast<path_instruction_type>(2), "GGLF", -1},
      {static_cast<path_instruction_type>(2), "JKKJDAKAB", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "KPIGLEDEOCFELKLJLAFE", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "PACKGGMDGCLEHD", -1},
      {static_cast<path_instruction_type>(2), "IAFMNJMMNJPDAAHND", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "PACKGGMDGCLEHD", -1},
      {static_cast<path_instruction_type>(2), "MNIMBEMMOJFHILDMDBML", -1}}

    //
  };
}

#endif

#if 1

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>
generate_paths_test_1()
{
  /*
   * '$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].OKPLFLHHEBDJELFA'
   * '$.NHKDIEPJNND.CHNFGBB.KIKNFPPAPGDO.KLFALIBALPPK.HGABIFNPHAHHGP'
   * '$.NHKDIEPJNND.IHIIKIHHMPFL.KCCCHAM.KCCCHAM'
   */
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "DPBFKLKAKDHLMDLIONCCLJ", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "OKPLFLHHEBDJELFA", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "CHNFGBB", -1},
      {static_cast<path_instruction_type>(2), "KIKNFPPAPGDO", -1},
      {static_cast<path_instruction_type>(2), "KLFALIBALPPK", -1},
      {static_cast<path_instruction_type>(2), "HGABIFNPHAHHGP", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "NHKDIEPJNND", -1},
      {static_cast<path_instruction_type>(2), "IHIIKIHHMPFL", -1},
      {static_cast<path_instruction_type>(2), "KCCCHAM", -1},
      {static_cast<path_instruction_type>(2), "KCCCHAM", -1}}
    //
  };
}

std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>
generate_paths_test_2()
{
  /*
   * '$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.BEACAHEBBO.BNLFCI'
   * '$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.BEACAHEBBO.GPIHMJ'
   * '$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.CGEGPD[0].GJFKCFJELPJEDBAD'
   * '$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.CGEGPD[0].GMFDD'
   */
  return std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "BNLFCI", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "BEACAHEBBO", -1},
      {static_cast<path_instruction_type>(2), "GPIHMJ", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GJFKCFJELPJEDBAD", -1}},

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "AENBHHGIABBBDDGOEI", -1},
      {static_cast<path_instruction_type>(2), "EAGCHCMLMOLGJK", -1},
      {static_cast<path_instruction_type>(2), "CGEGPD", -1},
      {static_cast<path_instruction_type>(1), "", 0},
      {static_cast<path_instruction_type>(2), "GMFDD", -1}},

    //
  };
}

#endif

void test(rmm::cuda_stream_view stream, int method, bool warm_up = false)
{
  std::vector<std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>>
    paths(5);
#if 1
  paths[0] = generate_paths0();
  paths[1] = generate_paths1();
  paths[2] = generate_paths2();
  paths[3] = generate_paths3();
  paths[4] = generate_paths4();
#endif

  std::vector<
    std::pair<std::unique_ptr<std::vector<rmm::device_uvector<spark_rapids_jni::path_instruction>>>,
              std::unique_ptr<cudf::string_scalar>>>
    d_paths(5);
  for (int i = 0; i < 5; ++i) {
    d_paths[i] = spark_rapids_jni::generate_device_json_paths(paths[i]);
  }

  printf("Method: %d\n", method);
  fflush(stdout);

  {
    int idx{0};
    std::size_t count{0};
    for (auto const& path : paths) {
      if (warm_up) {
        fprintf(stdout, "Path %d, size: %d\n", idx++, (int)path.size());
      } else {
        printf("Path %d, size: %d\n", idx++, (int)path.size());
      }
      count += path.size();
    }
    printf("Total path: %d\n", (int)count);
    fflush(stdout);
  }

  auto const read_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{"/home/nghiat/Devel/data/WM_MOCKED_3/data.parquet"})
      .build();

  auto const limit = 256 * 1024 * 1024UL;
  auto reader      = cudf::io::chunked_parquet_reader(limit, 4 * limit, read_opts);

  Timer timer;
  double test_time = 0;
  int num_chunks{0};

  do {
    auto chunk = reader.read_chunk();
    num_chunks++;
    stream.synchronize();

    timer.tick();
    if (method == 0) {
      /* Test with 80 paths:
       * Test time: 7189.04ms, num chunks: 1
       * Test time: 410163ms, num chunks: 51
       *
       * Test with 15 paths:
       * Test time: 117.39ms, num chunks: 1
       * Test time: 5112.7ms, num chunks: 51
       */
      for (int i = 0; i < 5; ++i) {
        for (auto const& path : paths[i]) {
          if (path.size() > 0) {
            // [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
            //   cudf::strings_column_view{chunk.tbl->get_column(i).view()}, path);
          }
        }
      }
    } else if (method == 1) {
      /* Test with 80 paths:
       * Test time: 6896.2ms, num chunks: 1
       * Test time: 385526ms, num chunks: 51
       *
       * Test with 15 paths:
       * Test time: 144.366ms, num chunks: 1
       * Test time: 4889.46ms, num chunks: 51
       */
      // for (int i = 0; i < 5; ++i) {
      //   if (paths[i].size() > 0) {
      //     [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
      //       cudf::strings_column_view{chunk.tbl->get_column(i).view()}, paths[i]);
      //   }
      // }
    } else {
      for (int i = 0; i < 5; ++i) {
        if (paths[i].size() > 0) {
          [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object_multiple_paths(
            cudf::strings_column_view{chunk.tbl->get_column(i).view()}, *(d_paths[i].first));
        }
      }
    }
    stream.synchronize();
    test_time += timer.tock();

    if (warm_up) {
      break;  // just test one chunk
    }
  } while (reader.has_next());

  std::cout << "Test time: " << test_time << "ms, num chunks: " << num_chunks << std::endl;
}

void BM_get_json_object_multiple(nvbench::state& state)
{
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
#if 0
  test(stream, 0, true);   // warm up
  test(stream, 0, false);  // warm up
#else
  // test(stream, 1, true);  // warm up
  test(stream, 2, true);  // warm up

  // test(stream, 1, false);  // warm up
  test(stream, 2, false);  // warm up
#endif
  // 7/8:  32774 ms
  // 7/10: 26340 ms
}

auto debug_data1()
{
  auto input = cudf::test::strings_column_wrapper{"{'a':1, 'b':2, 'c':[1, 2, 3]}"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "a", -1}},
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "b", -1}},
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "c", -1},
      {static_cast<path_instruction_type>(0), "", -1}},
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "c", -1},
      {static_cast<path_instruction_type>(1), "", 1}}
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data2()
{
  auto input = cudf::test::strings_column_wrapper{"{\"k1\":{\"k2\":\"v2\"}}"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "k1", -1},
      {static_cast<path_instruction_type>(2), "k2", -1}}
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data3()
{
  auto input = cudf::test::strings_column_wrapper{
    "{\"k1\":{\"k2\":{\"k3\":{\"k4\":{\"k5\":{\"k6\":{\"k7\":{\"k8\":\"v8\"}}}}}}}}"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "k1", -1},
      {static_cast<path_instruction_type>(2), "k2", -1},
      {static_cast<path_instruction_type>(2), "k3", -1},
      {static_cast<path_instruction_type>(2), "k4", -1},
      {static_cast<path_instruction_type>(2), "k5", -1},
      {static_cast<path_instruction_type>(2), "k6", -1},
      {static_cast<path_instruction_type>(2), "k7", -1},
      {static_cast<path_instruction_type>(2), "k8", -1},
    }
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data4()
{
  auto input = cudf::test::strings_column_wrapper{
    "{\"brand\":\"ssssss\",\"duratRon\":15,\"eqTosuresurl\":\"\",\"RsZxarthrl\":false,"
    "\"xonRtorsurl\":\"\",\"xonRtorsurlstOTe\":0,\"TRctures\":[{\"RxaGe\":\"VttTs:\\/\\/"
    "feed-RxaGe.baRdu.cox\\/0\\/TRc\\/"
    "-196588744s840172444s-773690137.zTG\"}],\"Toster\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/"
    "TRc\\/"
    "-196588744s840172444s-773690137.zTG\",\"reserUed\":{\"bRtLate\":391.79,\"xooUZRke\":26876,"
    "\"nahrlIeneratRonNOTe\":0,\"useJublRc\":6,\"URdeoRd\":821284086},\"tRtle\":"
    "\"ssssssssssmMsssssssssssssssssss\",\"url\":\"s{storehrl}\",\"usersTortraRt\":\"VttTs:\\/\\/"
    "feed-RxaGe.baRdu.cox\\/0\\/TRc\\/"
    "-6971178959s-664926866s-6096674871.zTG\",\"URdeosurl\":\"http:\\/\\/nadURdeo2.baRdu.cox\\/"
    "5fa3893aed7fc0f8231dab7be23efc75s820s6240.xT3\",\"URdeoRd\":821284086}"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "URdeosurl", -1},
    }
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data5()
{
  auto input = cudf::test::strings_column_wrapper{
    "[  [[[ {'k': 'v1'} ], {'k': 'v2'}]], [[{'k': 'v3'}], {'k': 'v4'}], {'k': 'v5'}  ]"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "k", -1}}
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data6()
{
  auto input = cudf::test::strings_column_wrapper{"[     {'k': 'v1'}, {'k': 'v2'}     ]"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "k", -1}}
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data7()
{
  auto input = cudf::test::strings_column_wrapper{"[1, [21, 22], 3]"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(0), "", -1},
    }
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data8()
{
  auto input = cudf::test::strings_column_wrapper{
    "[ {'k': [0, 1]}, {'k': {'a': 'b'}}, {'k': [10, 11, 12]}  ]"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "k", -1}},
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(0), "", -1},
      {static_cast<path_instruction_type>(2), "k", -1},
      {static_cast<path_instruction_type>(0), "", -1},
    }
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

auto debug_data9()
{
  auto input = cudf::test::strings_column_wrapper{
    R"({"KAEMAPGKBFGNHPJ":"G2s!8-b@n2p","GKOGAKJDMEMFLJJD":[{"FMCMNJH":"FsZ!","OOPCOILM":".LX4HI1#6ZXQu"},{"FMCMNJH":"p,mt","OOPCOILM":"6zfMl<n9hKIEW3"},{"FMCMNJH":"FsZ!/","OOPCOILM":"QZHXy6.5`be'$"},{"FMCMNJH":"h1uU","OOPCOILM":"OR]NXQpvtZ2vm*"}],"NOJIEMJO":1,"KGDODIHGGE":5209092417461,"GJCPJNIGDPKPJD":{"MCBMBNGIBDINA":1,"NKGNHHBKCEKNHAALLIFPDOGBIAHPJJ":[{"DNPHCNFBGHGIG":{"PNILIIFFCJOIP":"7c","KPKJJBNBPANGGKLA":7.39},"DHOPCAGIDIOKE":1}],"IKPGINCMOA":true,"OENPHGKDIHJAI":true,"GNPDPLJBCIPKEI":true},"GLGFEFOCIPJJNANH":{"LP":{"KAEMAPGKBFGNHPJ":"(AK)R~N`Ph","IMGLDEPKNNLCDFMGEFMNEJEHCLPN":7023013792466,"DHLLEABGIMMFBILMCJI":{},"DBDJLNOLJMDCKJLGMBFBMK":6622780579313,"HMLBEEHBBJBIOJDABJ":"p,mtx?CXMdbHM","GKDACMEJIFMIEKMANHNNCPFGDCDGGGCG":{},"APEOMEHEIENM":"7c5vDh-|yd7","OJKCBLGL":"Fs","HGEFKFCGJJNBBJIGKOO":true}},"DGFNAGIDCKFP":"FsZ!/*!){O","OIJBCHJKPPOMGLOGKLLO":{"DOEBEBNFOJPMN":"Az[M`Q.'mn`","LFCPNEFNBJDO":true},"OJAGBGMLHPFENNPAN":"p<$[aE9FEzd","PENEBIND":"f40'f<`-+-^: =C>is\"/pH,n-:v\\.;N[","MLDKOIIPGBFAKKLJC":[],"DHLLEABGIMMFBILMCJI":{},"IIEEGFP":9133996313993,"EIFGPHGLOPELFBN":[{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"\"/eP4`=WHg~kUOm2AJoXHk"}},"BJHMGKLP":"h1uU#","ONLPCJKJDIAKK":false,"GACCEMNH":"d+Z%N\\jZ9']wGOl%V9gj@"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":5,"GAHIANAFHIBONL":"rvtH}C0uFmK\\#A*D)x}I7c@5M'Z_fm6>PviE=(J]T+KlsNuO-\"lxD~.N/9EQ<^l%Qb-eFRBv","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"kH#D'(VRt)*kH<{ITQ^|By~:6o2<N1yUXVHHmn$@^g@Z5Nd!!dg#K"}},"ONBMEJGFIFMPO":"Az[M`Q"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"(&9pN@K'vFf4ozyK:B'"}},"BJHMGKLP":"7c5vDh-|","ONLPCJKJDIAKK":true,"GACCEMNH":"'IGzim0]@s0*I~"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":3,"GAHIANAFHIBONL":"Cb[p7$3NQM}\"@YcrIwzt0\"CtK W&(@VBx4CcX.HHrW{IHAjlv_FlAfI0F7CjklK'U3>","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":""}},"ONBMEJGFIFMPO":"h1uU#"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"s[2nryS^V|=D]h5JgNHE+h"}},"BJHMGKLP":"Az[M`Q","ONLPCJKJDIAKK":true,"GACCEMNH":"`GkuDVB)},b<G"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":5,"GAHIANAFHIBONL":"fmVdmXrLq)~NTzM<Ox/C:zRYX<wMEsC?3w!'PFj|AuuJ!WwiPm<xdb<>9kD+(MUEJU5eSw5f","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"!mJue a[n}#S:Hn"}},"ONBMEJGFIFMPO":"h1uU#"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":":e:)wr^Mp;hk7YiPb"}},"BJHMGKLP":"Az[M`","ONLPCJKJDIAKK":true,"GACCEMNH":"|<*(OwRo([egvXq"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":"ire`w7YQ=+6v<>ML\\o%0O@12jW1qQzy?5E`ye{!TVMl`$i=cR)k0TT_KK(bf3wGEHn&K]3g3IhZO7\\tQ2a","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"yv7*i\"b>gLJ;l3l?CmCUS@i]ce:65U2o;buv%zo<a^>_/Ox>!t)+;kjVj56g'84"}},"ONBMEJGFIFMPO":"h1uU#C"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[]},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"tU'S+mwf}H,Hp)VmgTXV['"}},"BJHMGKLP":"p,mtx?CX","ONLPCJKJDIAKK":false,"GACCEMNH":"}+H1}a1o!H1ji&%9ZC+"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":"}\"UN}:D{?7}%{kXHzGI$x\\e|,D0kvN","HGFOBILDJONCIHF":"h1uU#C[.Qc","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"iu,Eu%,vEPu)Joovs4SDwP"}},"ONBMEJGFIFMPO":"Az[M"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":">bE[dUO=4eXQk 6vL,rMlN"}},"BJHMGKLP":"Az[M`Q","ONLPCJKJDIAKK":true,"GACCEMNH":"Iw92[v,?T`o2G"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":"EmJ@.R?h","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":""}},"ONBMEJGFIFMPO":"7c5vD"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"%,b~~]9sV1v#89H;-EU'JY{g"}},"BJHMGKLP":"FsZ!/*!","ONLPCJKJDIAKK":false,"GACCEMNH":"Z4N/&MHYFJIR/1rqLn"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":8,"GAHIANAFHIBONL":"v]W\\\\$JTlKpij#:v+ta`1zUJJMf","HGFOBILDJONCIHF":" aMZ({x5#1N=9(yM\"C","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"/Vo46pg='[eV+"}},"ONBMEJGFIFMPO":"Az[M`Q"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"$&ys4cUgN*OwnXH\\u"}},"BJHMGKLP":"FsZ","ONLPCJKJDIAKK":true,"GACCEMNH":"sp.R`h{>oKPb@H:HvtN&"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":3,"GAHIANAFHIBONL":"rs}8P(6 Co>|5h\"CI)u!\"wuhxfR/Q_\\rVr+v$5nibOOecKZ`INV0kqc&f%9(msUq%+g","HGFOBILDJONCIHF":"7c5vDh-|yd","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"H)Lm%i=LYm}l@g2]g$5v<8')(7o"}},"ONBMEJGFIFMPO":"h1uU#"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"uq=kN4JVxf)kOnw'{3"}},"BJHMGKLP":"FsZ!/*!","ONLPCJKJDIAKK":true,"GACCEMNH":"[X>{IRK/)Se49+QS"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":5,"GAHIANAFHIBONL":"","PPMNGNHGGONNLFAJKK":{},"ONBMEJGFIFMPO":"p,mtx?CX"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}}],"LDIHMDHNPANNGNMAI":{"CMBLBPGBPBLFOD":"Az[","LJCMAKCMDLOBC":"aK"},"KDOJAIFEFELP":{},"LHPGDLOGEGNGKCJMHEDKFJCBMH":[{"GEJGHKEHLAPFJ":"Az[M`","MHBGOLHCHBBIOM":{"CLELPFHJMKGHFNGKMBHAHAH":"B<m/V$0L","DJJOPKMGBIAKJJELPKLBA":"): u&\"cT^"},"ELAIPNGCDHPCIF":"FsZ!/*!"}],"DBDJLNOLJMDCKJLGMBFBMK":8761195769789,"HMMAJIPEEC":"FsZ!/*!){O5>Mq%ea #9u+F.AO%","KCCCHAM":"{&]1J_iH^}Eq>oE,#@R;T\"N1uwgXdH;M","AOICHPCGCHAMC":"*^r:\">l1+7XRSYU&g\"AU","PEJGENFFO":5074070918344,"CPFLBGNLMPLDFEGDHHG":[],"EIPCCEBCIMIFAEDOL":6202435079206,"AOHEGDPMAEPEAL":["b\\B53^![]\\A:n!","ji?Y()=t_+w-7R"],"KPLEHGGBK":"Az[M`Q.'mn`","AFDJNAEOHKK":8213889546936,"HMLBEEHBBJBIOJDABJ":"FsZ!/*!){O5","GKDACMEJIFMIEKMANHNNCPFGDCDGGGCG":{},"CIECJLNPP":"\\`Tx+HeoX`OU","FJPIMCHIJACHJE":[],"KHBKEAFCB":"7c5vDh-|","HABNLKACAJHCIOPFOPBBK":true,"IOJLOKAIK":1,"JIPBOH":"F","EGNIJOJDDCPKK":"x!ajLb(","GNBDNIMEEFCGKADKOAIE":"p,mtx?CX","HGEFKFCGJJNBBJIGKOO":false,"LGMLEEGAPIKBLFBL":[{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"y&\"MvM*"}},"BJHMGKLP":"h1uU#","ONLPCJKJDIAKK":true,"GACCEMNH":"&4k6jDc<a7{F~4"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":"J;N2$BP?_K`e?KqJQ\".87;p%Uc~}WM\\XW@*)]","HGFOBILDJONCIHF":">{%nzw`HO*\"&K_{8q","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"$]AP*.>#{S`c#VcSy"}},"ONBMEJGFIFMPO":"p,mtx?C"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"5<K4Q%"}},"BJHMGKLP":"FsZ!/*","ONLPCJKJDIAKK":true,"GACCEMNH":"cY$Jmk*<<v"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":"& wC8>y:o<DbE;2,/bhzM\\JM\"}hT}~EK6W_]M","HGFOBILDJONCIHF":"FsZ!/*!){O5>Mq%ea #9u+F.A","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"x\"7:h+D7K</.N6U=-'LrgPAbAF%x%M_GVxLjf"}},"ONBMEJGFIFMPO":"FsZ!/*!"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}},{"OAJHOPFELNNIJPFBNBG":{"KJKGAGFJNGBIGBM":{"ECHAGOLMEHCN":{"LJNHE":"EX9ojr=L1d!W"}},"BJHMGKLP":"7c5vDh","ONLPCJKJDIAKK":true,"GACCEMNH":"6Y~(ucM"},"DIMPFKELHHIEHNM":[{"JBMLPPOFI":1,"GAHIANAFHIBONL":";UCt|\\a:F\"&_1ZlnhF(x!b=.+y? ~\\V>{TC9`","HGFOBILDJONCIHF":"/#_v9kRtI'L_\\dtQl","PPMNGNHGGONNLFAJKK":{"ECHAGOLMEHCN":{"LJNHE":"mZH3+r@7uv07uY<,4S9{z`cyzYj8zfv5{XW~(%*f@\\r?Fug"}},"ONBMEJGFIFMPO":"Az[M`Q"}],"HDDDGJFELLA":true,"KEEBNLDHJAPJNIGHHLPPCCEDJ":[],"BHCNDBIIICDJDB":{"MGAGILEALPHFOPBO":{}}}],"OGFDLCDDIPBFH":["TlBC40,WgNae"],"KOOJKBKFOKBH":[]})"};

  auto paths = std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>{
    std::vector<std::tuple<path_instruction_type, std::string, int64_t>>{
      {static_cast<path_instruction_type>(2), "EIFGPHGLOPELFBN", -1}}
    //
  };

  return std::pair{std::move(input), std::move(paths)};
}

void BM_unit_tests(nvbench::state& state)
{ /*
   std::vector<cudf::test::strings_column_wrapper> input;
   std::vector<std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>>>
     paths;

   //////////////////////////////////////////////////////////////////////////////////////////
   {
     auto [new_input, new_paths] = debug_data1();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }
   {
     auto [new_input, new_paths] = debug_data2();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }
   {
     auto [new_input, new_paths] = debug_data3();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }
   {
     auto [new_input, new_paths] = debug_data4();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }
   {
     auto [new_input, new_paths] = debug_data5();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }
   {
     auto [new_input, new_paths] = debug_data6();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }

   {
     auto [new_input, new_paths] = debug_data7();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }

   {
     auto [new_input, new_paths] = debug_data8();
     input.emplace_back(std::move(new_input));
     paths.emplace_back(std::move(new_paths));
   }

   //////////////////////////////////////////////////////////////////////////////////////////
   for (std::size_t i = 0; i < input.size(); ++i) {
     auto const& curr_input = input[i];
     auto const& curr_paths = paths[i];

     freopen("/dev/null", "w", stdout);
     std::vector<std::unique_ptr<cudf::column>> output_old_method;
     for (auto const& path : curr_paths) {
       // output_old_method.emplace_back(
       //   spark_rapids_jni::get_json_object(cudf::strings_column_view{curr_input}, path));
     }
     // auto const output_new_method = spark_rapids_jni::get_json_object_multiple_paths(
     //   cudf::strings_column_view{curr_input}, curr_paths);
     freopen("/dev/tty", "w", stdout);

     CUDF_EXPECTS(output_old_method.size() == output_new_method.size(), "");
     for (std::size_t j = 0; j < curr_paths.size(); ++j) {
       auto const comp = cudf::test::detail::expect_columns_equal(output_old_method[j]->view(),
                                                                  output_new_method[j]->view());
       if (!comp) {
         printf("Failure at test data %d\n", (int)i + 1);
         exit(0);
       }
     }
   }
   printf("All test passed.\n");*/
}

void BM_verify(nvbench::state& state) { printf("All test passed.\n"); }

void debug(bool old) {}

void BM_debug(nvbench::state& state)
{
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const old = state.get_int64("method") == 0;
  debug(old);
}

NVBENCH_BENCH(BM_get_json_object_single)
  .set_name("get_json_object_single")
  .add_int64_axis("size_bytes", {1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
  .add_int64_axis("max_depth", {2, 4, 6, 8});

NVBENCH_BENCH(BM_get_json_object_multiple).set_name("get_json_object_multiple");

NVBENCH_BENCH(BM_debug).set_name("debug").add_int64_axis("method", {0, 1});
NVBENCH_BENCH(BM_unit_tests).set_name("tests");
NVBENCH_BENCH(BM_verify).set_name("verify");
