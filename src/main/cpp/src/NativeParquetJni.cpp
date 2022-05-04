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

/**
 * Convert a string to lower case. It uses std::tolower per character which has limitations
 * and may not produce the exact same result as the JVM does. This is probably good enough
 * for now.
 */
std::string unicode_to_lower(std::string const& input) {
  // get the size of the wide character result
    std::size_t wide_size = std::mbstowcs(nullptr, input.data(), 0);
  if (wide_size < 0) {
    throw std::invalid_argument("invalid character sequence");
  }

  std::vector<wchar_t> wide(wide_size + 1);
  // Set a null so we can get a proper output size from wcstombs. This is because 
  // we pass in a max length of 0, so it will only stop when it see the null character.
  wide.back() = 0;
  if (std::mbstowcs(wide.data(), input.data(), wide_size) != wide_size) {
    throw std::runtime_error("error during wide char converstion");
  }
  for (auto wit = wide.begin(); wit != wide.end(); ++wit) {
    *wit = std::towlower(*wit);
  }
  // Get the multi-byte result size
  std::size_t mb_size = std::wcstombs(nullptr, wide.data(), 0);
  if (mb_size < 0) {
    throw std::invalid_argument("unsupported wide character sequence");
  }
  // We are allocating a fixed size string so we can put the data directly into it
  // instead of going through a NUL terminated char* first. The NUL fill char is
  // just because we need to pass in a fill char. The value does not matter
  // because it will be overwritten. std::string itself will insert a NUL
  // terminator on the buffer it allocates internally. We don't need to worry about it.
  std::string ret(mb_size, '\0');
  if (std::wcstombs(ret.data(), wide.data(), mb_size) != mb_size) {
    throw std::runtime_error("error during multibyte char converstion");
  }
  return ret;
}

/**
 * Holds a set of "maps" that are used to rewrite various parts of the parquet metadata.
 * Generally each "map" is a gather map that pulls data from an input vector to be placed in
 * an output vector.
 */
struct column_pruning_maps {
  // gather map for pulling out items from the schema
  std::vector<int> schema_map;
  // Each SchemaElement also includes the number of children in it. This allows the vector
  // to be interpreted as a tree flattened depth first. These are the new values for num
  // children after the schema is gathered.
  std::vector<int> schema_num_children;
  // There are several places where a struct is stored only for a leaf column (like a column chunk)
  // This holds the gather map for those cases.
  std::vector<int> chunk_map;
};

/**
 * This class will handle processing column pruning for a schema. It is written as a class because
 * of JNI we are sending the names of the columns as a depth first list, like parquet does internally.
 */
class column_pruner {
public:
    /**
     * Create pruning filter from a depth first flattened tree of names and num_children.
     * The root entry is not included in names or in num_children, but parent_num_children
     * should hold how many entries there are in it.
     */
    column_pruner(const std::vector<std::string> & names, 
            const std::vector<int> & num_children, 
            int parent_num_children): children(), s_id(0), c_id(-1) {
      add_depth_first(names, num_children, parent_num_children);
    }

    column_pruner(int s_id, int c_id): children(), s_id(s_id), c_id(c_id) {
    }

    column_pruner(): children(), s_id(0), c_id(-1) {
    }

    /**
     * Given a schema from a parquet file create a set of pruning maps to prune columns from the rest of the footer
     */
    column_pruning_maps filter_schema(std::vector<parquet::format::SchemaElement> & schema, bool ignore_case) {
      // The following are all covered by follow on work in https://github.com/NVIDIA/spark-rapids-jni/issues/210
      // TODO the java code will fail if there is ambiguity in the names and ignore_case is true
      // so we need to figure that out too.
      // TODO there are a number of different way to represent a list or a map. We want to support all of them
      //  so we need a way to detect that schema is a list and group the parts we don't care about together.
      // TODO the java code verifies that the schema matches when it is looking at the columns or it throws
      // an exception. Sort of, It really just checks that it is a GroupType where it expects to find them
      //
      // With all of this in mind I think what we want to do is to pass down a full-ish schema, not just the names,
      // and the number of children. We need to know if it is a Map, an Array, a Struct or primitive.
      //
      // Then when we are walking the tree we need to keep track of if we are looking for a Map, an array or
      // a struct and match up the SchemaElement entries accordingly as we go.
      // If we see something that is off we need to throw an exception.
      //
      // To be able to handle the duplicates, I think we need to have some state in the column_pruner class
      // to say if we have matched a leaf node or not.
      // 
      // From the Parquet spec
      // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
      //
      // A repeated field that is neither contained by a LIST- or MAP-annotated group nor annotated by LIST
      // or MAP should be interpreted as a required list of required elements where the element type is the
      // type of the field.
      //
      // LIST must always annotate a 3-level structure:
      // <list-repetition> group <name> (LIST) {
      //   repeated group list {
      //     <element-repetition> <element-type> element;
      //   }
      // }
      // ...
      // However, these names may not be used in existing data and should not be enforced as errors when reading.
      // ...
      // Some existing data does not include the inner element layer. For backward-compatibility, the type of
      // elements in LIST-annotated structures should always be determined by the following rules:
      //
      //  1. If the repeated field is not a group, then its type is the element type and elements are required.
      //  2. If the repeated field is a group with multiple fields, then its type is the element type and
      //     elements are required.
      //  3. If the repeated field is a group with one field and is named either array or uses the
      //     LIST-annotated group's name with _tuple appended then the repeated type is the element
      //     type and elements are required.
      //  4. Otherwise, the repeated field's type is the element type with the repeated field's repetition.

      // MAP is used to annotate types that should be interpreted as a map from keys to values. MAP must
      // annotate a 3-level structure:
      //
      //  * The outer-most level must be a group annotated with MAP that contains a single field named
      //    key_value. The repetition of this level must be either optional or required and determines
      //    whether the list is nullable.
      //  * The middle level, named key_value, must be a repeated group with a key field for map keys
      //    and, optionally, a value field for map values.
      //  * The key field encodes the map's key type. This field must have repetition required and must
      //    always be present.
      //  * The value field encodes the map's value type and repetition. This field can be required,
      //    optional, or omitted.
      //
      // It is required that the repeated group of key-value pairs is named key_value and that its
      // fields are named key and value. However, these names may not be used in existing data and
      // should not be enforced as errors when reading.
      //
      // Some existing data incorrectly used MAP_KEY_VALUE in place of MAP. For backward-compatibility,
      // a group annotated with MAP_KEY_VALUE that is not contained by a MAP-annotated group should be
      // handled as a MAP-annotated group.

      // Parquet says that the map's value is optional, but Spark looks like it would throw an exception
      // if it ever actually saw that in practice, so we should too.
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
      std::vector<column_pruner*> tree_stack;
      tree_stack.push_back(this);
      if (schema.size() == 0) {
        throw std::invalid_argument("a root schema element must exist");
      }
      num_children_stack.push_back(schema[0].num_children);

      uint64_t chunk_index = 0;
      // We are skipping over the first entry in the schema because it is always the root entry, and
      //  we already processed it
      for (uint64_t schema_index = 1; schema_index < schema.size(); ++schema_index) {
        auto schema_item = schema[schema_index];
        // num_children is optional, but is supposed to be set for non-leaf nodes. That said leaf nodes
        // will have 0 children so we can just default to that.
        int num_children = 0;
        if (schema_item.__isset.num_children) {
          num_children = schema_item.num_children;
        }
        std::string name;
        if (ignore_case) {
          name = unicode_to_lower(schema_item.name);
        } else {
          name = schema_item.name;
        }
        column_pruner * found = nullptr;
        if (tree_stack.back() != nullptr) {
          // tree_stack can have a nullptr in it if the schema we are looking through
          // has an entry that does not match the tree
          auto found_it = tree_stack.back()->children.find(name);
          if (found_it != tree_stack.back()->children.end()) {
            found = &(found_it->second);
            int parent_mapped_schema_index = tree_stack.back()->s_id;
            ++num_children_map[parent_mapped_schema_index];

            int mapped_schema_index = found->s_id;
            schema_map[mapped_schema_index] = schema_index;
            num_children_map[mapped_schema_index] = 0;
          }
        }

        if (schema_item.__isset.type) {
          // this is a leaf node, it has a primitive type.
          if (found != nullptr) {
            int mapped_chunk_index = found->c_id;
            chunk_map[mapped_chunk_index] = chunk_index;
          }
          ++chunk_index;
        } 
        // else it is a non-leaf node it is group typed
        // chunks are only for leaf nodes

        // num_children and if the type is set or not should correspond to each other.
        //  By convention in parquet they should, but to be on the safe side I keep them
        //  separate.
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
 
            if (tree_stack.size() == 0) {
              done = true;
            }
          }
        }
      }

      // If there is a column that is missing from this file we need to compress the gather maps
      //  so there are no gaps
      std::vector<int> final_schema_map;
      final_schema_map.reserve(schema_map.size());
      for (auto it = schema_map.begin(); it != schema_map.end(); ++it) {
        final_schema_map.push_back(it->second);
      }

      std::vector<int> final_num_children_map;
      final_num_children_map.reserve(num_children_map.size());
      for (auto it = num_children_map.begin(); it != num_children_map.end(); ++it) {
        final_num_children_map.push_back(it->second);
      }

      std::vector<int> final_chunk_map;
      final_chunk_map.reserve(chunk_map.size());
      for (auto it = chunk_map.begin(); it != chunk_map.end(); ++it) {
        final_chunk_map.push_back(it->second);
      }

      return column_pruning_maps{std::move(final_schema_map),
          std::move(final_num_children_map),
          std::move(final_chunk_map)};
    }

private:

    void add_depth_first(std::vector<std::string> const& names,
            std::vector<int> const& num_children,
            int parent_num_children) {
      CUDF_FUNC_RANGE();
      if (parent_num_children == 0) {
        // There is no point in doing more the tree is empty, and it lets us avoid some corner cases
        // in the code below
        return;
      }
      int local_s_id = 0; // There is always a root on the schema
      int local_c_id = -1; // for columns it is just the leaf nodes
      auto num = names.size();
      std::vector<column_pruner*> tree_stack;
      std::vector<int> num_children_stack;
      tree_stack.push_back(this);
      num_children_stack.push_back(parent_num_children);
      for(uint64_t i = 0; i < num; ++i) {
        auto name = names[i];
        auto num_c = num_children[i];
        ++local_s_id;
        int tmp_c_id = -1;
        if (num_c == 0) {
          // leaf node...
          ++local_c_id;
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
        throw std::invalid_argument("DIDN'T CONSUME EVERYTHING...");
      }
    }

    std::map<std::string, column_pruner> children;
    // The following IDs are the position that they should be in when output in a filtered footer, except
    // that if there are any missing columns in the actual data the gaps need to be removed.
    // schema ID
    int s_id;
    // Column chunk and Column order ID
    int c_id;
};

static bool invalid_file_offset(long start_index, long pre_start_index, long pre_compressed_size) {
  bool invalid = false;
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

static int64_t get_offset(parquet::format::ColumnChunk const& column_chunk) {
  auto md = column_chunk.meta_data;
  int64_t offset = md.data_page_offset;
  if (md.__isset.dictionary_page_offset && offset > md.dictionary_page_offset) {
    offset = md.dictionary_page_offset;
  }
  return offset;
}

static std::vector<parquet::format::RowGroup> filter_groups(parquet::format::FileMetaData const& meta, 
        int64_t part_offset, int64_t part_length) {
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
    for (uint64_t rg_i = 0; rg_i < num_row_groups; ++rg_i) {
        parquet::format::RowGroup const& row_group = meta.row_groups[rg_i];
        int64_t total_size = 0;
        int64_t start_index;
        auto column_chunk = row_group.columns[0];
        if (first_column_with_metadata) {
            start_index = get_offset(column_chunk);
        } else {
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
        for (uint64_t cc_i = 0; cc_i < num_columns; ++cc_i) {
            parquet::format::ColumnChunk const& col = row_group.columns[cc_i];
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

void deserialize_parquet_footer(uint8_t * buffer, uint32_t len, parquet::format::FileMetaData * meta) {
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
  for (auto group_it = groups.begin(); group_it != groups.end(); ++group_it) {
    std::vector<parquet::format::ColumnChunk> new_chunks;
    for (auto it = chunk_filter.begin(); it != chunk_filter.end(); ++it) {
      new_chunks.push_back(group_it->columns[*it]);
    }
    group_it->columns = std::move(new_chunks);
  }
}

}
}

extern "C" {

JNIEXPORT long JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_readAndFilter(JNIEnv * env, jclass,
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

    rapids::jni::column_pruner pruner(n_filter_col_names.as_cpp_vector(),
            std::vector(n_num_children.begin(), n_num_children.end()),
            parent_num_children);
    auto filter = pruner.filter_schema(meta->schema, ignore_case);

    // start by filtering the schema and the chunks
    std::size_t new_schema_size = filter.schema_map.size();
    std::vector<parquet::format::SchemaElement> new_schema(new_schema_size);
    for (std::size_t i = 0; i < new_schema_size; ++i) {
      int orig_index = filter.schema_map[i];
      int new_num_children = filter.schema_num_children[i];
      new_schema[i] = meta->schema[orig_index];
      new_schema[i].num_children = new_num_children;
    }
    meta->schema = std::move(new_schema);
    if (meta->__isset.column_orders) {
      std::vector<parquet::format::ColumnOrder> new_order;
      for (auto it = filter.chunk_map.begin(); it != filter.chunk_map.end(); ++it) {
        new_order.push_back(meta->column_orders[*it]);
      }
      meta->column_orders = std::move(new_order);
    }
    // Now we want to filter the columns out of each row group that we care about as we go.
    if (part_length >= 0) {
      meta->row_groups = std::move(rapids::jni::filter_groups(*meta, part_offset, part_length));
    }
    rapids::jni::filter_columns(meta->row_groups, filter.chunk_map);

    return cudf::jni::release_as_jlong(meta);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_close(JNIEnv * env, jclass,
                                                                            jlong handle) {
  try {
    parquet::format::FileMetaData * ptr = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    delete ptr;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumRows(JNIEnv * env, jclass,
                                                                                 jlong handle) {
  try {
    parquet::format::FileMetaData * ptr = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    long ret = 0;
    for(auto it = ptr->row_groups.begin(); it != ptr->row_groups.end(); ++it) {
      ret = ret + it->num_rows;
    }
    return ret;
  }
  CATCH_STD(env, -1);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumColumns(JNIEnv * env, jclass,
                                                                                     jlong handle) {
  try {
    parquet::format::FileMetaData * ptr = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    int ret = 0;
    if (ptr->schema.size() > 0) {
      if (ptr->schema[0].__isset.num_children) {
        ret = ptr->schema[0].num_children;
      }
    }
    return ret;
  }
  CATCH_STD(env, -1);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_serializeThriftFile(JNIEnv * env, jclass,
                                                                                             jlong handle) {
  CUDF_FUNC_RANGE();
  try {
    parquet::format::FileMetaData * meta = reinterpret_cast<parquet::format::FileMetaData *>(handle);
    std::shared_ptr<apache::thrift::transport::TMemoryBuffer> transportOut(
            new apache::thrift::transport::TMemoryBuffer());
    apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer> factory;
    auto protocolOut = factory.getProtocol(transportOut);
    meta->write(protocolOut.get());
    uint8_t * buf_ptr;
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
