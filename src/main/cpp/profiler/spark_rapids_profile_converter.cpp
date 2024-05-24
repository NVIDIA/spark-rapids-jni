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

/* A tool that converts a spark-rapids profile binary into other forms. */

#if 0
#include <stdexcept>
#define FLATBUFFERS_ASSERT(x)                                     \
  do {                                                            \
    if (!(x)) { throw std::runtime_error("flatbuffers assert"); } \
  } while (0)
#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE
#endif

#include "profiler_generated.h"
#include "spark_rapids_jni_version.h"

#include <cupti.h>
#include <cxxabi.h>
#include <flatbuffers/idl.h>

#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spark_rapids_jni::profiler {
extern char const* Profiler_Schema;
}

struct program_options {
  std::optional<std::filesystem::path> output_path;
  bool help       = false;
  bool json       = false;
  bool nvtxt      = false;
  int json_indent = 2;
  bool version    = false;
};

struct event {
  enum struct type_id { API, DEVICE, KERNEL, MARKER, MARKER_DATA, MEMCPY, MEMSET, OVERHEAD };
  type_id id;
  void const* fb_data;
};

struct thread_id {
  uint32_t pid;
  uint32_t tid;

  bool operator==(thread_id const& o) const { return pid == o.pid && tid == o.tid; }
};

template <>
struct std::hash<thread_id> {
  std::size_t operator()(thread_id const& t) const
  {
    return std::hash<uint32_t>{}(t.pid) ^ (std::hash<uint32_t>{}(t.tid) << 1);
  }
};

struct stream_id {
  uint32_t device;
  uint32_t context;
  uint32_t stream;

  bool operator==(stream_id const& s) const
  {
    return device == s.device && context == s.context && stream == s.stream;
  }
};

template <>
struct std::hash<stream_id> {
  std::size_t operator()(stream_id const& s) const
  {
    return std::hash<uint32_t>{}(s.device) ^ (std::hash<uint32_t>{}(s.context) << 1) ^
           (std::hash<uint32_t>{}(s.stream) << 2);
  }
};

struct event_streams {
  std::unordered_map<thread_id, std::vector<event>> cpu;
  std::unordered_map<stream_id, std::vector<event>> gpu;
};

void print_usage()
{
  std::cout << "spark_rapids_profile_converter [OPTION]... profilebin" << std::endl;
  std::cout << R"(
Converts the spark-rapids profile in profile.bin into other forms.

  -h, --help                show this usage message
  -j, --json                convert to JSON, default output is stdout
  -i, --json-indent=INDENT  indentation to use for JSON. 0 is no indent, less than 0 also removes newlines
  -o, --output=PATH         use PATH as the output filename
  -t. --nvtxt               convert to NVTXT, default output is stdout
  -V, --version             print the version number
  )" << std::endl;
}

void print_version()
{
  std::cout << "spark_rapids_profile_converter " << spark_rapids_jni::Version << std::endl;
}

std::pair<program_options, std::vector<std::string_view>> parse_options(
  std::vector<std::string_view> args)
{
  program_options opts{};
  std::string_view long_output("--output=");
  std::string_view long_json_indent("--json-indent=");
  bool seen_output      = false;
  bool seen_json_indent = false;
  auto argp             = args.begin();
  while (argp != args.end()) {
    if (*argp == "-o" || *argp == "--output") {
      if (seen_output) { throw std::runtime_error("output path cannot be specified twice"); }
      seen_output = true;
      if (++argp != args.end()) {
        opts.output_path = std::make_optional(*argp++);
      } else {
        throw std::runtime_error("missing argument for output path");
      }
    } else if (argp->substr(0, long_output.size()) == long_output) {
      if (seen_output) { throw std::runtime_error("output path cannot be specified twice"); }
      seen_output = true;
      argp->remove_prefix(long_output.size());
      if (argp->empty()) {
        throw std::runtime_error("missing argument for output path");
      } else {
        opts.output_path = std::make_optional(*argp++);
      }
    } else if (*argp == "-h" || *argp == "--help") {
      opts.help = true;
      ++argp;
    } else if (*argp == "-i" || *argp == "--json-indent") {
      if (seen_json_indent) { throw std::runtime_error("JSON indent cannot be specified twice"); }
      seen_json_indent = true;
      if (++argp != args.end()) {
        auto [ptr, err] = std::from_chars(argp->data(), argp->end(), opts.json_indent);
        if (err != std::errc() || ptr != argp->end()) {
          throw std::runtime_error("invalid JSON indent value");
        }
        ++argp;
      } else {
        throw std::runtime_error("missing argument for JSON indent");
      }
    } else if (argp->substr(0, long_json_indent.size()) == long_json_indent) {
      if (seen_json_indent) { throw std::runtime_error("JSON indent cannot be specified twice"); }
      seen_json_indent = true;
      argp->remove_prefix(long_json_indent.size());
      if (argp->empty()) {
        throw std::runtime_error("missing argument for JSON indent");
      } else {
        auto [ptr, err] = std::from_chars(argp->data(), argp->end(), opts.json_indent);
        if (err != std::errc() || ptr != argp->end()) {
          throw std::runtime_error("invalid JSON indent value");
        }
        ++argp;
      }
    } else if (*argp == "-j" || *argp == "--json") {
      if (opts.nvtxt) { throw std::runtime_error("JSON and NVTXT output are mutually exclusive"); }
      opts.json = true;
      ++argp;
    } else if (*argp == "-t" || *argp == "--nvtxt") {
      if (opts.json) { throw std::runtime_error("JSON and NVTXT output are mutually exclusive"); }
      opts.nvtxt = true;
      ++argp;
    } else if (*argp == "-V" || *argp == "--version") {
      opts.version = true;
      ++argp;
    } else if (argp->empty()) {
      throw std::runtime_error("empty argument");
    } else if (argp->at(0) == '-') {
      throw std::runtime_error(std::string("unrecognized option: ") + std::string(*argp));
    } else {
      break;
    }
  }
  return std::make_pair(opts, std::vector<std::string_view>(argp, args.end()));
}

void checked_read(std::ifstream& in, char* buffer, size_t size)
{
  in.read(buffer, size);
  if (in.fail()) {
    if (in.eof()) {
      throw std::runtime_error("Unexpected EOF");
    } else {
      throw std::runtime_error(std::strerror(errno));
    }
  }
}

flatbuffers::uoffset_t read_flatbuffer_size(std::ifstream& in)
{
  flatbuffers::uoffset_t fb_size;
  checked_read(in, reinterpret_cast<char*>(&fb_size), sizeof(fb_size));
  return flatbuffers::EndianScalar(fb_size);
}

std::unique_ptr<std::vector<char>> read_flatbuffer(std::ifstream& in)
{
  auto size = read_flatbuffer_size(in);
  // Allocate a buffer that can hold the flatbuffer along with the prefixed size.
  // SizePrefixed APIs require size to be at the front of the buffer and alignment
  // of fields is planned out with that size.
  auto buffer   = std::make_unique<std::vector<char>>(size + sizeof(flatbuffers::uoffset_t));
  auto size_ptr = reinterpret_cast<flatbuffers::uoffset_t*>(buffer->data());
  *size_ptr     = size;
  checked_read(in, buffer->data() + sizeof(flatbuffers::uoffset_t), size);
  return buffer;
}

std::ofstream open_output(std::filesystem::path const& path,
                          std::ios::openmode mode = std::ios::out)
{
  if (std::filesystem::exists(path)) {
    throw std::runtime_error(path.string() + " already exists");
  }
  std::ofstream out(path, mode);
  out.exceptions(std::ios::badbit);
  return out;
}

template <typename T>
T const* validate_fb(std::vector<char> const& fb, std::string_view const& name)
{
  flatbuffers::Verifier::Options verifier_opts;
  verifier_opts.assert = true;
  flatbuffers::Verifier verifier(
    reinterpret_cast<uint8_t const*>(fb.data()), fb.size(), verifier_opts);
  if (not verifier.VerifySizePrefixedBuffer<T>(nullptr)) {
    throw std::runtime_error(std::string("malformed ") + std::string(name) + " record");
  }
  return flatbuffers::GetSizePrefixedRoot<T>(fb.data());
}

void verify_profile_header(std::ifstream& in)
{
  auto fb_ptr = read_flatbuffer(in);
  auto header = validate_fb<spark_rapids_jni::profiler::ProfileHeader>(*fb_ptr, "profile header");
  auto magic  = header->magic();
  if (magic == nullptr) {
    throw std::runtime_error("does not appear to be a spark-rapids profile");
  }
  if (magic->str() != "spark-rapids profile") {
    std::ostringstream oss;
    oss << "bad profile magic, expected 'spark-rapids profile' found '" << magic->str() << "'";
    throw std::runtime_error(oss.str());
  }
  auto version = header->version();
  if (version != 1) {
    std::ostringstream oss;
    oss << "unsupported profile version: " << version;
    throw std::runtime_error(oss.str());
  }
}

void convert_to_nsys_rep(std::ifstream& in,
                         std::string_view const& in_filename,
                         program_options const& opts)
{
  event_streams events;
  size_t num_dropped_records = 0;
  while (!in.eof()) {
    auto fb_ptr = read_flatbuffer(in);
    auto records =
      validate_fb<spark_rapids_jni::profiler::ActivityRecords>(*fb_ptr, "ActivityRecords");
    auto api = records->api();
    if (api != nullptr) {
      for (int i = 0; i < api->size(); ++i) {
        auto a = api->Get(i);
        thread_id tid{a->process_id(), a->thread_id()};
        event e{event::type_id::API, a};
        auto it = events.cpu.find(tid);
        if (it == events.cpu.end()) {
          events.cpu.emplace(tid, std::initializer_list<event>{e});
        } else {
          it->second.push_back(e);
        }
      }
    }
    auto device = records->device();
    if (device != nullptr) { std::cerr << "NUM DEVICES=" << device->size() << std::endl; }
    auto dropped = records->dropped();
    if (dropped != nullptr) {
      for (int i = 0; i < dropped->size(); ++i) {
        auto d = dropped->Get(i);
        num_dropped_records += d->num_dropped();
      }
    }
    auto kernel = records->kernel();
    if (kernel != nullptr) { std::cerr << "NUM KERNEL=" << kernel->size() << std::endl; }
    auto marker = records->marker();
    if (marker != nullptr) { std::cerr << "NUM MARKERS=" << marker->size() << std::endl; }
    auto marker_data = records->marker_data();
    if (marker_data != nullptr) {
      std::cerr << "NUM MARKER DATA=" << marker_data->size() << std::endl;
      for (int i = 0; i < marker_data->size(); ++i) {
        std::cerr << "MARKER DATA " << i << std::endl;
        auto md = marker_data->Get(i);
        std::cerr << " FLAGS: " << md->flags();
        std::cerr << " ID: " << md->id();
        std::cerr << " COLOR: " << md->color();
        std::cerr << " CATEGORY: " << md->category() << std::endl;
      }
    }
    auto memcpy = records->memcpy();
    if (memcpy != nullptr) { std::cerr << "NUM MEMCPY=" << memcpy->size() << std::endl; }
    auto memset = records->memset();
    if (device != nullptr) { std::cerr << "NUM MEMSET=" << memset->size() << std::endl; }
    auto overhead = records->overhead();
    if (overhead != nullptr) { std::cerr << "NUM OVERHEADS=" << overhead->size() << std::endl; }

    in.peek();
  }
  if (not in.eof()) { throw std::runtime_error(std::strerror(errno)); }
  if (num_dropped_records) {
    std::cerr << "Warning: " << num_dropped_records
              << " records were noted as dropped in the profile" << std::endl;
  }
}

void convert_to_json(std::ifstream& in, std::ostream& out, program_options const& opts)
{
  flatbuffers::Parser parser;
  if (parser.Parse(spark_rapids_jni::profiler::Profiler_Schema) != 0) {
    std::runtime_error("Internal error: Unable to parse profiler schema");
  }
  parser.opts.strict_json = true;
  parser.opts.indent_step = opts.json_indent;
  while (!in.eof()) {
    auto fb_ptr = read_flatbuffer(in);
    auto records =
      validate_fb<spark_rapids_jni::profiler::ActivityRecords>(*fb_ptr, "ActivityRecords");
    std::string json;
    char const* err =
      flatbuffers::GenText(parser, fb_ptr->data() + sizeof(flatbuffers::uoffset_t), &json);
    if (err != nullptr) { throw std::runtime_error(std::string("Error generating JSON: ") + err); }
    out << json;

    in.peek();
  }
  if (not in.eof()) { throw std::runtime_error(std::strerror(errno)); }
}

char const* get_api_name(spark_rapids_jni::profiler::ApiActivity const* a)
{
  char const* name = nullptr;
  switch (a->kind()) {
    case spark_rapids_jni::profiler::ApiKind_Driver:
      cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, a->cbid(), &name);
      break;
    case spark_rapids_jni::profiler::ApiKind_Runtime:
      cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid(), &name);
      break;
    default: {
      std::ostringstream oss;
      oss << "unsupported API kind: " << a->kind();
      throw std::runtime_error(oss.str());
    }
  }
  return name;
}

std::string demangle(char const* s)
{
  int status      = 0;
  char* demangled = abi::__cxa_demangle(s, nullptr, nullptr, &status);
  if (status == 0) {
    std::string result(demangled);
    free(demangled);
    return result;
  } else {
    return s;
  }
}

std::string memcpy_to_string(spark_rapids_jni::profiler::MemcpyActivity const* m)
{
  char const* kind_str;
  char const* pinned = "";
  switch (m->copy_kind()) {
    case spark_rapids_jni::profiler::MemcpyKind_HtoD:
      kind_str = "HtoD";
      if (m->src_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { pinned = " Pinned"; }
      break;
    case spark_rapids_jni::profiler::MemcpyKind_DtoH:
      kind_str = "DtoH";
      if (m->dst_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { pinned = " Pinned"; }
      break;
    case spark_rapids_jni::profiler::MemcpyKind_HtoA:
      kind_str = "HtoA";
      if (m->dst_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { pinned = " Pinned"; }
      break;
    case spark_rapids_jni::profiler::MemcpyKind_AtoH:
      kind_str = "AtoH";
      if (m->dst_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { pinned = " Pinned"; }
      break;
    case spark_rapids_jni::profiler::MemcpyKind_AtoA: kind_str = "AtoA"; break;
    case spark_rapids_jni::profiler::MemcpyKind_AtoD: kind_str = "AtoD"; break;
    case spark_rapids_jni::profiler::MemcpyKind_DtoA: kind_str = "DtoA"; break;
    case spark_rapids_jni::profiler::MemcpyKind_DtoD: kind_str = "DtoD"; break;
    case spark_rapids_jni::profiler::MemcpyKind_HtoH:
      kind_str = "HtoH";
      if (m->src_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned &&
          m->dst_kind() == m->src_kind()) {
        pinned = " Pinned";
      }
      break;
    case spark_rapids_jni::profiler::MemcpyKind_PtoP: kind_str = "PtoP"; break;
    case spark_rapids_jni::profiler::MemcpyKind_Unknown: kind_str = "Unknown"; break;
    default: kind_str = "Unknown"; break;
  }
  std::ostringstream oss;
  oss << kind_str << pinned;
  oss << " " << m->bytes() << " bytes";
  if (m->flags() == spark_rapids_jni::profiler::MemcpyFlags_Async) { oss << " async"; }
  return oss.str();
}

const char* memcpy_to_color(spark_rapids_jni::profiler::MemcpyActivity const* m)
{
  switch (m->copy_kind()) {
    case spark_rapids_jni::profiler::MemcpyKind_HtoD:
      if (m->src_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { return "MediumPurple"; }
      return "Gold";
    case spark_rapids_jni::profiler::MemcpyKind_DtoH:
      if (m->dst_kind() == spark_rapids_jni::profiler::MemoryKind_Pinned) { return "MediumPurple"; }
      return "Gold";
    case spark_rapids_jni::profiler::MemcpyKind_HtoA:
    case spark_rapids_jni::profiler::MemcpyKind_AtoH:
    case spark_rapids_jni::profiler::MemcpyKind_AtoA:
    case spark_rapids_jni::profiler::MemcpyKind_AtoD:
    case spark_rapids_jni::profiler::MemcpyKind_DtoA: return "Gold";
    case spark_rapids_jni::profiler::MemcpyKind_DtoD: return "Gold";
    case spark_rapids_jni::profiler::MemcpyKind_HtoH: return "Ivory";
    case spark_rapids_jni::profiler::MemcpyKind_PtoP: return "LightSalmon";
    case spark_rapids_jni::profiler::MemcpyKind_Unknown:
    default: return "DarkRed";
  }
}

std::string memset_to_string(spark_rapids_jni::profiler::MemsetActivity const* m)
{
  std::ostringstream oss;
  oss << "Memset " << m->bytes() << " bytes to " << m->value();
  if (m->flags() == spark_rapids_jni::profiler::MemsetFlags_Async) { oss << " async"; }
  return oss.str();
}

char const* overhead_kind_to_string(spark_rapids_jni::profiler::OverheadKind k)
{
  switch (k) {
    case spark_rapids_jni::profiler::OverheadKind_Unknown: return "Unknown";
    case spark_rapids_jni::profiler::OverheadKind_DriverCompiler: return "Driver compiler";
    case spark_rapids_jni::profiler::OverheadKind_CUptiBufferFlush: return "Buffer flush";
    case spark_rapids_jni::profiler::OverheadKind_CUptiInstrumentation: return "Instrumentation";
    case spark_rapids_jni::profiler::OverheadKind_CUptiResource: return "Resource";
    default: return "Unknown";
  }
}

// Convert a CUPTI thread ID to an NVTXT thread ID.
uint32_t to_nvtxt_tid(uint32_t tid)
{
  // NVTXT thread IDs are limited to 24-bit.
  // Take the upper 24 bits which empirically are the most unique bits returned by CUPTI.
  return tid >> 8;
}

void convert_to_nvtxt(std::ifstream& in, std::ostream& out, program_options const& opts)
{
  struct marker_start {
    uint64_t timestamp;
    uint32_t process_id;
    uint32_t thread_id;
    uint32_t color;
    uint32_t category;
    std::string name;
  };
  std::unordered_set<stream_id> streams_seen;
  std::unordered_map<int, spark_rapids_jni::profiler::MarkerData const*> marker_data_map;
  std::unordered_map<int, marker_start> marker_start_map;
  size_t num_dropped_records = 0;
  out << "@NameProcess,ProcessId,Name" << std::endl;
  out << "NameProcess,0,\"GPU\"" << std::endl;
  out << "@NameOsThread,ProcessId,ThreadId,Name" << std::endl;
  out << "@RangePush,Time,ProcessId,ThreadId,CategoryId,Color,Message" << std::endl;
  out << "@RangePop,Time,ProcessId,ThreadId" << std::endl;
  out << "TimeBase=Relative" << std::endl;
  out << "Payload=0" << std::endl;
  while (!in.eof()) {
    auto fb_ptr = read_flatbuffer(in);
    auto records =
      validate_fb<spark_rapids_jni::profiler::ActivityRecords>(*fb_ptr, "ActivityRecords");
    auto dropped = records->dropped();
    if (dropped != nullptr) {
      for (int i = 0; i < dropped->size(); ++i) {
        auto d = dropped->Get(i);
        num_dropped_records += d->num_dropped();
      }
    }
    auto api = records->api();
    if (api != nullptr) {
      for (int i = 0; i < api->size(); ++i) {
        auto a = api->Get(i);
        out << "RangePush," << a->start() << "," << a->process_id() << ","
            << to_nvtxt_tid(a->thread_id()) << ",0,PaleGreen"
            << ","
            << "\"" << get_api_name(a) << "\"" << std::endl;
        out << "RangePop," << a->end() << "," << a->process_id() << ","
            << to_nvtxt_tid(a->thread_id()) << std::endl;
      }
    }
    auto marker_data = records->marker_data();
    if (marker_data != nullptr) {
      for (int i = 0; i < marker_data->size(); ++i) {
        auto m              = marker_data->Get(i);
        auto [it, inserted] = marker_data_map.insert({m->id(), m});
        if (not inserted) {
          std::ostringstream oss;
          oss << "duplicate marker data for " << m->id();
          throw std::runtime_error(oss.str());
        }
      }
    }
    auto marker = records->marker();
    if (marker != nullptr) {
      for (int i = 0; i < marker->size(); ++i) {
        auto m         = marker->Get(i);
        auto object_id = m->object_id();
        if (object_id != nullptr) {
          uint32_t process_id = object_id->process_id();
          uint32_t thread_id  = to_nvtxt_tid(object_id->thread_id());
          if (process_id == 0) {
            // abusing thread ID as stream ID since NVTXT does not support GPU activity directly
            thread_id = object_id->stream_id();
            // TODO: Ignoring device ID and context here
            auto [it, inserted] = streams_seen.insert(stream_id{0, 0, thread_id});
            if (inserted) { out << "NameOsThread,0,\"Stream " << thread_id << "\"" << std::endl; }
          }
          if (m->flags() & spark_rapids_jni::profiler::MarkerFlags_Start) {
            auto it           = marker_data_map.find(m->id());
            uint32_t color    = 0x444444;
            uint32_t category = 0;
            if (it != marker_data_map.end()) {
              color    = it->second->color();
              category = it->second->category();
            }
            marker_start ms{
              m->timestamp(), process_id, thread_id, color, category, m->name()->str()};
            auto [ignored, inserted] = marker_start_map.insert({m->id(), ms});
            if (not inserted) {
              std::ostringstream oss;
              oss << "duplicate marker start for ID " << m->id();
              throw std::runtime_error(oss.str());
            }
          } else if (m->flags() & spark_rapids_jni::profiler::MarkerFlags_End) {
            auto it = marker_start_map.find(m->id());
            if (it != marker_start_map.end()) {
              auto const& ms = it->second;
              out << "RangePush," << ms.timestamp << "," << ms.process_id << "," << ms.thread_id
                  << "," << ms.category << "," << ms.color << ","
                  << "\"" << ms.name << "\"" << std::endl;
              out << "RangePop," << m->timestamp() << "," << ms.process_id << "," << ms.thread_id
                  << std::endl;
              marker_start_map.erase(it);
            } else {
              std::cerr << "Ignoring marker end without start for ID " << m->id() << std::endl;
            }
          } else {
            std::cerr << "Ignoring marker with unsupported flags: " << m->flags() << std::endl;
          }
        } else {
          std::cerr << "Marker " << m->id() << " has no object ID" << std::endl;
        }
      }
    }
    marker_data_map.clear();
    auto kernel = records->kernel();
    if (kernel != nullptr) {
      for (int i = 0; i < kernel->size(); ++i) {
        auto k              = kernel->Get(i);
        uint32_t process_id = 0;
        // abusing thread ID as stream ID since NVTXT does not support GPU activity directly
        uint32_t thread_id = k->stream_id();
        // TODO: Ignoring device ID and context here
        auto [it, inserted] = streams_seen.insert(stream_id{0, 0, thread_id});
        if (inserted) {
          out << "NameOsThread,0," << thread_id << ",\"Stream " << thread_id << "\"" << std::endl;
        }
        out << "RangePush," << k->start() << "," << process_id << "," << thread_id << ",0,Blue"
            << ","
            << "\"" << demangle(k->name()->c_str()) << "\"" << std::endl;
        out << "RangePop," << k->end() << "," << process_id << "," << thread_id << std::endl;
      }
    }
    auto memcpy = records->memcpy();
    if (memcpy != nullptr) {
      for (int i = 0; i < memcpy->size(); ++i) {
        auto m              = memcpy->Get(i);
        uint32_t process_id = 0;
        // abusing thread ID as stream ID since NVTXT does not support GPU activity directly
        uint32_t thread_id = m->stream_id();
        // TODO: Ignoring device ID and context here
        auto [it, inserted] = streams_seen.insert(stream_id{0, 0, thread_id});
        if (inserted) {
          out << "NameOsThread,0," << thread_id << ",\"Stream " << thread_id << "\"" << std::endl;
        }
        out << "RangePush," << m->start() << "," << process_id << "," << thread_id << ",0,"
            << memcpy_to_color(m) << ","
            << "\"" << memcpy_to_string(m) << "\"" << std::endl;
        out << "RangePop," << m->end() << "," << process_id << "," << thread_id << std::endl;
      }
    }
    auto memset = records->memset();
    if (memset != nullptr) {
      for (int i = 0; i < memset->size(); ++i) {
        auto m              = memset->Get(i);
        uint32_t process_id = 0;
        // abusing thread ID as stream ID since NVTXT does not support GPU activity directly
        uint32_t thread_id = m->stream_id();
        // TODO: Ignoring device ID and context here
        auto [it, inserted] = streams_seen.insert(stream_id{0, 0, thread_id});
        if (inserted) {
          out << "NameOsThread,0," << thread_id << ",\"Stream " << thread_id << "\"" << std::endl;
        }
        out << "RangePush," << m->start() << "," << process_id << "," << thread_id << ",0,Olive"
            << ","
            << "\"" << memset_to_string(m) << "\"" << std::endl;
        out << "RangePop," << m->end() << "," << process_id << "," << thread_id << std::endl;
      }
    }
    auto overhead = records->overhead();
    if (overhead != nullptr) {
      for (int i = 0; i < overhead->size(); ++i) {
        auto o         = overhead->Get(i);
        auto object_id = o->object_id();
        if (object_id != nullptr) {
          uint32_t process_id = object_id->process_id();
          uint32_t thread_id  = to_nvtxt_tid(object_id->thread_id());
          if (process_id == 0) {
            // abusing thread ID as stream ID since NVTXT does not support GPU activity directly
            thread_id = object_id->stream_id();
            // TODO: Ignoring device ID and context here
            auto [it, inserted] = streams_seen.insert(stream_id{0, 0, thread_id});
            if (inserted) { out << "NameOsThread,0,\"Stream " << thread_id << "\"" << std::endl; }
          }
          out << "RangePush," << o->start() << "," << process_id << "," << thread_id
              << ",0,OrangeRed"
              << ","
              << "\"" << overhead_kind_to_string(o->overhead_kind()) << "\"" << std::endl;
          out << "RangePop," << o->end() << "," << process_id << "," << thread_id << std::endl;
        } else {
          std::cerr << "Overhead activity has no object ID" << std::endl;
        }
      }
    }

    in.peek();
  }
  if (num_dropped_records) {
    std::cerr << "Warning: " << num_dropped_records
              << " records were noted as dropped in the profile" << std::endl;
  }
}

int main(int argc, char* argv[])
{
  constexpr int RESULT_SUCCESS = 0;
  constexpr int RESULT_FAILURE = 1;
  constexpr int RESULT_USAGE   = 2;
  program_options opts;
  std::vector<std::string_view> files;
  if (argc < 2) {
    print_usage();
    return RESULT_USAGE;
  }
  std::vector<std::string_view> args(argv + 1, argv + argc);
  try {
    auto [options, inputs] = parse_options(args);
    opts                   = options;
    files                  = inputs;
  } catch (std::exception const& e) {
    std::cerr << "spark_rapids_profile_converter: " << e.what() << std::endl;
    print_usage();
    return RESULT_USAGE;
  }
  if (opts.help) {
    print_usage();
    return RESULT_USAGE;
  }
  if (opts.version) {
    print_version();
    return RESULT_SUCCESS;
  }
  if (files.size() != 1) {
    std::cerr << "Missing input file." << std::endl;
    print_usage();
    return RESULT_USAGE;
  }
  auto input_file = files.front();
  try {
    std::ifstream in(std::string(input_file), std::ios::binary | std::ios::in);
    in.exceptions(std::istream::badbit);
    verify_profile_header(in);
    if (opts.json) {
      if (opts.output_path) {
        std::ofstream out = open_output(opts.output_path.value());
        convert_to_json(in, out, opts);
      } else {
        convert_to_json(in, std::cout, opts);
      }
    } else if (opts.nvtxt) {
      if (opts.output_path) {
        std::ofstream out = open_output(opts.output_path.value());
        convert_to_nvtxt(in, out, opts);
      } else {
        convert_to_nvtxt(in, std::cout, opts);
      }
    } else {
      convert_to_nsys_rep(in, input_file, opts);
    }
  } catch (std::system_error const& e) {
    std::cerr << "Error converting " << input_file << ": " << e.code().message() << std::endl;
    return RESULT_FAILURE;
  } catch (std::exception const& e) {
    std::cerr << "Error converting " << input_file << ": " << e.what() << std::endl;
    return RESULT_FAILURE;
  }
  return RESULT_SUCCESS;
}
