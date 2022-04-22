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

#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <cwctype>

// TCompactProtocol requires some #defines to work right.
// This came from the parquet code itself...
#define SIGNED_RIGHT_SHIFT_IS 1
#define ARITHMETIC_RIGHT_SHIFT 1
#include <thrift/TApplicationException.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <cudf/detail/nvtx/ranges.hpp>
#include <generated/parquet_types.h>

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

namespace rapids {
namespace jni {

std::string unicode_to_lower(std::string & input) {
  // This is not great, but it works, I think, not sure because it is also dependent on the local, and
  // we might need to pass that down from the JVM???
  int wide_size = std::mbstowcs(nullptr, input.data(), 0);
  if (wide_size < 0) {
    throw std::invalid_argument("invalid character sequence...");
  }
  std::vector<wchar_t> wide(wide_size + 1);
  std::mbstowcs(wide.data(), input.data(), wide_size);
  for (auto wit = wide.begin(); wit != wide.end(); wit++) {
    *wit = std::towlower(*wit);
  }
  int mb_size = std::wcstombs(nullptr, wide.data(), 0);
  std::string ret(mb_size, '\0');
  std::wcstombs(ret.data(), wide.data(), mb_size);
  return ret;
}

struct parquet_filter {
  std::vector<int> schema_map;
  std::vector<int> schema_num_children;
  std::vector<int> chunk_map;
};

class name_tree {
public:
    name_tree(): children(), s_id(0), c_id(-1) {
    }
    name_tree(int s_id, int c_id): children(), s_id(s_id), c_id(c_id) {
    }

    void add_depth_first(std::vector<std::string> names, std::vector<int> num_children, int parent_num_children) {
      CUDF_FUNC_RANGE();
      int local_s_id = 0; // There is always a root on the schema
      int local_c_id = -1; // for columns it is just the leaf nodes
      auto num = names.size();
      std::vector<name_tree*> tree_stack;
      std::vector<int> num_children_stack;
      tree_stack.push_back(this);
      num_children_stack.push_back(parent_num_children);
      for(uint64_t i = 0; i < num; i++) {
        auto name = names[i];
        auto num_c = num_children[i];
        local_s_id++;
        int tmp_c_id = -1;
        if (num_c == 0) {
          // leaf node...
          local_c_id++;
          tmp_c_id = local_c_id;
        }
        tree_stack.back()->children.try_emplace(name, local_s_id, tmp_c_id);
        if (num_c > 0) {
          tree_stack.push_back(&tree_stack.back()->children[name]);
          num_children_stack.push_back(num_c);
        } else {
          // go back up the stack/tree removing children until we hit one with more children
          bool done = false;
          while (!done) {
              int parent_children_left = num_children_stack.back() - 1;
              if (parent_children_left > 0) {
                num_children_stack.back() = parent_children_left;
                done = true;
              } else {
                tree_stack.pop_back();
                num_children_stack.pop_back();
              }

              if (tree_stack.size() <= 0) {
                done = true;
              }
          }
        }
      }
      if (tree_stack.size() != 0 || num_children_stack.size() != 0) {
        throw std::invalid_argument("DIDN'T COSUME EVERYTHING...");
      }
    }

    parquet_filter filter_schema(std::vector<parquet::format::SchemaElement> & schema, bool ignore_case) {
      CUDF_FUNC_RANGE();
      // The maps are sorted so we can compress the tree...
      // These are the outputs of the computation
      std::map<int, int, std::less<int>> chunk_map;
      std::map<int, int, std::less<int>> schema_map;
      std::map<int, int, std::less<int>> num_children_map;
      // Start off with 0 children in the root, will add more as we go
      schema_map[0] = 0;
      num_children_map[0] = 0;

      // num_children_stack and tree_stack hold the current state as we walk though schema
      std::vector<int> num_children_stack;
      std::vector<name_tree*> tree_stack;
      tree_stack.push_back(this);
      num_children_stack.push_back(schema[0].num_children);

      uint64_t chunk_index = 0;
      // We are skipping over the first entry in the schema because it is always the root entry, and
      //  we already processed it
      for (uint64_t schema_index = 1; schema_index < schema.size(); schema_index++) {
        int num_children = 0;
        // num_children should always be set if the type is not set, but just to be safe we keep them separate
        if (schema[schema_index].__isset.num_children) {
          num_children = schema[schema_index].num_children;
        }
        std::string name;
        if (ignore_case) {
          name = unicode_to_lower(schema[schema_index].name);
        } else {
          name = schema[schema_index].name;
        }
        name_tree * found = nullptr;
        if (tree_stack.back() != nullptr) {
          auto found_it = tree_stack.back()->children.find(name);
          if (found_it != tree_stack.back()->children.end()) {
            found = &(found_it->second);
            int parent_mapped_schema_index = tree_stack.back()->s_id;
            num_children_map[parent_mapped_schema_index]++;
          }
        }

        if (schema[schema_index].__isset.type) {
          if (found != nullptr) {
            int mapped_chunk_index = found->c_id;
            int mapped_schema_index = found->s_id;
            chunk_map[mapped_chunk_index] = chunk_index;
            schema_map[mapped_schema_index] = schema_index;
            num_children_map[mapped_schema_index] = 0;
          }
          // this is a leaf node
          chunk_index++;
        } else {
          // a non-leaf node
          if (found != nullptr) {
            int mapped_schema_index = found->s_id;
            schema_map[mapped_schema_index] = schema_index;
            num_children_map[mapped_schema_index] = 0;
          }
        }

        if (num_children > 0) {
          tree_stack.push_back(found);
          num_children_stack.push_back(num_children);
        } else {
          // go back up the stack/tree removing children until we hit one with more children
          bool done = false;
          while (!done) {
            int parent_children_left = num_children_stack.back() - 1;
            if (parent_children_left > 0) {
              num_children_stack.back() = parent_children_left;
              done = true;
            } else {
              tree_stack.pop_back();
              num_children_stack.pop_back();
            }
 
            if (tree_stack.size() <= 0) {
              done = true;
            }
          }
        }
      }
      std::vector<int> final_schema_map;
      for (auto it = schema_map.begin(); it != schema_map.end(); it++) {
        final_schema_map.push_back(it->second);
      }

      std::vector<int> final_num_children_map;
      for (auto it = num_children_map.begin(); it != num_children_map.end(); it++) {
        final_num_children_map.push_back(it->second);
      }

      std::vector<int> final_chunk_map;
      for (auto it = chunk_map.begin(); it != chunk_map.end(); it++) {
        final_chunk_map.push_back(it->second);
      }

      return parquet_filter{final_schema_map, final_num_children_map, final_chunk_map};
    }
private:
    std::map<std::string, name_tree> children;
    // The following IDs are the position that they should be in when output in a filteres footer, except
    // that if there are any missing columns in the actual data the gaps need to be removed.
    // schema ID
    int s_id;
    // Column chunk and Column order ID
    int c_id;
};

static bool invalid_file_offset(long start_index, long pre_start_index, long pre_compressed_size) {
  bool invalid = false;
  //assert preStartIndex <= startIndex;
  // checking the first rowGroup
  if (pre_start_index == 0 && start_index != 4) {
    invalid = true;
    return invalid;
  }

  //calculate start index for other blocks
  int64_t min_start_index = pre_start_index + pre_compressed_size;
  if (start_index < min_start_index) {
    // a bad offset detected, try first column's offset
    // can not use minStartIndex in case of padding
    invalid = true;
  }

  return invalid;
}

static int64_t get_offset(parquet::format::ColumnChunk & column_chunk) {
  auto md = column_chunk.meta_data;
  int64_t offset = md.data_page_offset;
  if (md.__isset.dictionary_page_offset && offset > md.dictionary_page_offset) {
    offset = md.dictionary_page_offset;
  }
  return offset;
}

static std::vector<parquet::format::RowGroup> filter_groups(parquet::format::FileMetaData & meta, 
        int64_t part_offset, int64_t part_length) {
    // TODO should combine filter columns with filter_groups so we don't copy as much data...
    CUDF_FUNC_RANGE();
    // This is based off of the java parquet_mr code to find the groups in a range... 
    auto num_row_groups = meta.row_groups.size();
    int64_t pre_start_index = 0;
    int64_t pre_compressed_size = 0;
    bool first_column_with_metadata = true;
    if (num_row_groups > 0) {
        first_column_with_metadata = meta.row_groups[0].columns[0].__isset.meta_data;
    }

    std::vector<parquet::format::RowGroup> filtered_groups;
    for (uint64_t rg_i = 0; rg_i < num_row_groups; rg_i++) {
        parquet::format::RowGroup & row_group = meta.row_groups[rg_i];
        int64_t total_size = 0;
        int64_t start_index;
        auto column_chunk = row_group.columns[0];
        if (first_column_with_metadata) {
            start_index = get_offset(column_chunk);
        } else {
          //assert rowGroup.isSetFile_offset();
          //assert rowGroup.isSetTotal_compressed_size();

          //the file_offset of first block always holds the truth, while other blocks don't :
          //see PARQUET-2078 for details
          start_index = row_group.file_offset;
          if (invalid_file_offset(start_index, pre_start_index, pre_compressed_size)) {
            //first row group's offset is always 4
            if (pre_start_index == 0) {
              start_index = 4;
            } else {
              // use minStartIndex(imprecise in case of padding, but good enough for filtering)
              start_index = pre_start_index + pre_compressed_size;
            }
          }
          pre_start_index = start_index;
          pre_compressed_size = row_group.total_compressed_size;
      }
      if (row_group.__isset.total_compressed_size) {
        total_size = row_group.total_compressed_size;
      } else {
        auto num_columns = row_group.columns.size();
        for (uint64_t cc_i = 0; cc_i < num_columns; cc_i++) {
            parquet::format::ColumnChunk & col = row_group.columns[cc_i];
            total_size += col.meta_data.total_compressed_size;
        }
      }

      int64_t mid_point = start_index + total_size / 2;
      if (mid_point >= part_offset && mid_point < (part_offset + part_length)) {
        filtered_groups.push_back(row_group);
      }
    }
    return filtered_groups;
}

void deserialize_parquet_footer(uint8_t* buffer, uint32_t len, parquet::format::FileMetaData * meta) {
  using ThriftBuffer = apache::thrift::transport::TMemoryBuffer;

  CUDF_FUNC_RANGE();
  // A lot of this came from the parquet source code...
  // Deserialize msg bytes into c++ thrift msg using memory transport.
  #if PARQUET_THRIFT_VERSION_MAJOR > 0 || PARQUET_THRIFT_VERSION_MINOR >= 14
  auto conf = std::make_shared<apache::thrift::TConfiguration>();
  conf->setMaxMessageSize(std::numeric_limits<int>::max());
  auto tmem_transport = std::make_shared<ThriftBuffer>(buffer, len, ThriftBuffer::OBSERVE, conf);
  #else
  auto tmem_transport = std::make_shared<ThriftBuffer>(buffer, len);
  #endif

  apache::thrift::protocol::TCompactProtocolFactoryT<ThriftBuffer> tproto_factory;
  // Protect against CPU and memory bombs
  tproto_factory.setStringSizeLimit(100 * 1000 * 1000);
  // Structs in the thrift definition are relatively large (at least 300 bytes).
  // This limits total memory to the same order of magnitude as stringSize.
  tproto_factory.setContainerSizeLimit(1000 * 1000);
  std::shared_ptr<apache::thrift::protocol::TProtocol> tproto =
      tproto_factory.getProtocol(tmem_transport);
  try {
    meta->read(tproto.get());
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << "Couldn't deserialize thrift: " << e.what() << "\n";
    throw std::runtime_error(ss.str());
  }
}

void filter_columns(std::vector<parquet::format::RowGroup> & groups, std::vector<int> & chunk_filter) {
  CUDF_FUNC_RANGE();
  for (auto group_it = groups.begin(); group_it != groups.end(); group_it++) {
    std::vector<parquet::format::ColumnChunk> new_chunks;
    for (auto it = chunk_filter.begin(); it != chunk_filter.end(); it++) {
      new_chunks.push_back(group_it->columns[*it]);
    }
    group_it->columns = std::move(new_chunks);
    // TODO for sorting_columns we need a reverse map...
    // TODO sorting_columns
    // TODO ordinal???
  }
}

}
}

extern "C" {

JNIEXPORT long JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_readAndFilter(JNIEnv *env, jclass,
                                                                                    jlong buffer,
                                                                                    jlong buffer_length,
                                                                                    jlong part_offset,
                                                                                    jlong part_length,
                                                                                    jobjectArray filter_col_names,
                                                                                    jintArray num_children,
                                                                                    jint parent_num_children,
                                                                                    jboolean ignore_case) {
  CUDF_FUNC_RANGE();
  try {
      auto meta = std::make_unique<parquet::format::FileMetaData>();
      uint32_t len = static_cast<uint32_t>(buffer_length);
      // We don't support encrypted parquet...
      rapids::jni::deserialize_parquet_footer(reinterpret_cast<uint8_t*>(buffer), len, meta.get());

      // Get the filter for the columns first...
      cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
      cudf::jni::native_jintArray n_num_children(env, num_children);

      rapids::jni::name_tree tree;
      tree.add_depth_first(n_filter_col_names.as_cpp_vector(),
              std::vector(n_num_children.begin(), n_num_children.end()),
              parent_num_children);
      auto filter = tree.filter_schema(meta->schema, ignore_case);

      // start by filtering the schema and the chunks
      std::size_t new_schema_size = filter.schema_map.size();
      std::vector<parquet::format::SchemaElement> new_schema(new_schema_size);
      for (std::size_t i = 0; i < new_schema_size; i++) {
        int orig_index = filter.schema_map[i];
        int new_num_children = filter.schema_num_children[i];
        new_schema[i] = meta->schema[orig_index];
        new_schema[i].num_children = new_num_children;
      }
      meta->schema = std::move(new_schema);
      if (meta->__isset.column_orders) {
        // TODO we might be able to drop some of this ordering copy, if they are all the same,
        //  but for now we will not worry about it.
        std::vector<parquet::format::ColumnOrder> new_order;
        for (auto it = filter.chunk_map.begin(); it != filter.chunk_map.end(); it++) {
          new_order.push_back(meta->column_orders[*it]);
        }
        meta->column_orders = std::move(new_order);
      }
      // Now we want to filter the columns out of each row group that we care about as we go.
      if (part_length >= 0) {
        meta->row_groups = std::move(rapids::jni::filter_groups(*meta, part_offset, part_length));
      }
      rapids::jni::filter_columns(meta->row_groups, filter.chunk_map);

      // TODO filter out some of the key/value meta that is large and we know we don't need

      return cudf::jni::release_as_jlong(meta);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_close(JNIEnv *env, jclass,
                                                                            jlong handle) {
  try {
    parquet::format::FileMetaData * ptr = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    delete ptr;
  }
  CATCH_STD(env, );
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_serializeThriftFile(JNIEnv *env, jclass,
                                                                                             jlong handle) {
  CUDF_FUNC_RANGE();
  try {
    parquet::format::FileMetaData * meta = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    // TODO at some point add in the PAR1 and the length when we get to that point...
    std::shared_ptr<apache::thrift::transport::TMemoryBuffer> transportOut(
            new apache::thrift::transport::TMemoryBuffer());
    apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer> factory;
    auto protocolOut = factory.getProtocol(transportOut);
    meta->write(protocolOut.get());
    uint8_t* buf_ptr;
    uint32_t buf_size;
    transportOut->getBuffer(&buf_ptr, &buf_size);

    // 12 extra is for the MAGIC thrift_footer length MAGIC
    jobject ret = cudf::jni::allocate_host_buffer(env, buf_size + 12, false);
    uint8_t* ret_addr = reinterpret_cast<uint8_t*>(cudf::jni::get_host_buffer_address(env, ret));
    ret_addr[0] = 'P';
    ret_addr[1] = 'A';
    ret_addr[2] = 'R';
    ret_addr[3] = '1';
    std::memcpy(ret_addr + 4, buf_ptr, buf_size);
    uint8_t * after = ret_addr + buf_size + 4;
    after[0] = static_cast<uint8_t>(0xFF & buf_size);
    after[1] = static_cast<uint8_t>(0xFF & (buf_size >> 8));
    after[2] = static_cast<uint8_t>(0xFF & (buf_size >> 16));
    after[3] = static_cast<uint8_t>(0xFF & (buf_size >> 24));
    after[4] = 'P';
    after[5] = 'A';
    after[6] = 'R';
    after[7] = '1';
    return ret;
  }
  CATCH_STD(env, nullptr);
}

}
