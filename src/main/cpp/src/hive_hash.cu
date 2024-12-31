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

#include "hash.cuh"
#include "hash.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

using hive_hash_value_t = int32_t;

constexpr hive_hash_value_t HIVE_HASH_FACTOR = 31;
constexpr hive_hash_value_t HIVE_INIT_HASH   = 0;

hive_hash_value_t __device__ inline compute_int(int32_t key) { return key; }

hive_hash_value_t __device__ inline compute_long(int64_t key)
{
  return (static_cast<uint64_t>(key) >> 32) ^ key;
}

hive_hash_value_t __device__ inline compute_bytes(int8_t const* data, cudf::size_type const len)
{
  hive_hash_value_t ret = HIVE_INIT_HASH;
  for (auto i = 0; i < len; i++) {
    ret = ret * HIVE_HASH_FACTOR + static_cast<int32_t>(data[i]);
  }
  return ret;
}

template <typename Key>
struct hive_hash_function {
  // 'seed' is not used in 'hive_hash_function', but required by 'element_hasher'.
  constexpr hive_hash_function(uint32_t) {}

  [[nodiscard]] hive_hash_value_t __device__ inline operator()(Key const& key) const
  {
    CUDF_UNREACHABLE("Unsupported type for hive hash");
  }
};  // struct hive_hash_function

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<int8_t const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<bool>::operator()(bool const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int8_t>::operator()(int8_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int16_t>::operator()(
  int16_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int32_t>::operator()(
  int32_t const& key) const
{
  return compute_int(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int64_t>::operator()(
  int64_t const& key) const
{
  return compute_long(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<float>::operator()(float const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_int     = reinterpret_cast<int32_t const*>(&normalized);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<double>::operator()(double const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_long    = reinterpret_cast<int64_t const*>(&normalized);
  return compute_long(*p_long);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_D>::operator()(
  cudf::timestamp_D const& key) const
{
  auto* p_int = reinterpret_cast<int32_t const*>(&key);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_us>::operator()(
  cudf::timestamp_us const& key) const
{
  auto time_as_long            = *reinterpret_cast<int64_t const*>(&key);
  constexpr int MICRO_PER_SEC  = 1000000;
  constexpr int NANO_PER_MICRO = 1000;

  int64_t ts  = time_as_long / MICRO_PER_SEC;
  int64_t tns = (time_as_long % MICRO_PER_SEC) * NANO_PER_MICRO;

  int64_t result = ts;
  result <<= 30;
  result |= tns;

  result = (static_cast<uint64_t>(result) >> 32) ^ result;
  return static_cast<hive_hash_value_t>(result);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * This functor produces the same result as "HiveHash" in Spark for supported types.
 *
 * @tparam hash_function Hash functor to use for hashing elements. Must be hive_hash_function.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class hive_device_row_hasher {
 public:
  CUDF_HOST_DEVICE hive_device_row_hasher(Nullate check_nulls,
                                          cudf::table_device_view t,
                                          cudf::column_device_view* flattened_column_views,
                                          cudf::size_type* first_child_index,
                                          cudf::size_type* nested_column_map) noexcept
    : _check_nulls{check_nulls},
      _table{t},
      _flattened_column_views{flattened_column_views},
      _first_child_index{first_child_index},
      _nested_column_map{nested_column_map}
  {
    // Error out if passed an unsupported hash_function
    static_assert(std::is_base_of_v<hive_hash_function<int>, hash_function<int>>,
                  "hive_device_row_hasher only supports the 'hive_hash_function' hash function");
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      HIVE_INIT_HASH,
      cuda::proclaim_return_type<hive_hash_value_t>(
        [=] __device__(auto const hash, auto const col_idx) {
          auto cur_hash = cudf::type_dispatcher(
            column.type(),
            element_hasher_adapter{
              nulls, col_idx, flattened_column_views, first_child_index, nested_column_map},
            column,
            row_index);
          return HIVE_HASH_FACTOR * hash + cur_hash;
        }));
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    using hash_functor_t = cudf::experimental::row::hash::element_hasher<hash_function, Nullate>;

    __device__ element_hasher_adapter(Nullate check_nulls,
                                      cudf::size_type col_idx,
                                      cudf::column_device_view* flattened_column_views,
                                      cudf::size_type* first_child_index,
                                      cudf::size_type* nested_column_map) noexcept
      : hash_functor{check_nulls, HIVE_INIT_HASH, HIVE_INIT_HASH},
        _col_idx{col_idx},
        _flattened_column_views{flattened_column_views},
        _first_child_index{first_child_index},
        _nested_column_map{nested_column_map}
    {
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ hive_hash_value_t operator()(cudf::column_device_view const& col,
                                            cudf::size_type row_index) const noexcept
    {
      return this->hash_functor.template operator()<T>(col, row_index);
    }

    /**
     * @brief A structure to keep track of the computation for nested types.
     */
    struct col_stack_frame {
     private:
      cudf::size_type _col_idx;     // the column index in the flattened array
      cudf::size_type _row_idx;     // the index of the row in the column
      int _idx_to_process;          // the index of child or element to process next
      hive_hash_value_t _cur_hash;  // current hash value of the column

     public:
      __device__ col_stack_frame() = default;

      __device__ void init(cudf::size_type col_idx, cudf::size_type row_idx)
      {
        _col_idx        = col_idx;
        _row_idx        = row_idx;
        _idx_to_process = 0;
        _cur_hash       = HIVE_INIT_HASH;
      }

      __device__ void update_cur_hash(hive_hash_value_t hash)
      {
        _cur_hash = _cur_hash * HIVE_HASH_FACTOR + hash;
      }

      __device__ hive_hash_value_t get_hash() { return _cur_hash; }

      __device__ int get_and_inc_idx_to_process() { return _idx_to_process++; }

      __device__ int get_idx_to_process() { return _idx_to_process; }

      __device__ cudf::size_type get_col_idx() { return _col_idx; }

      __device__ cudf::size_type get_row_idx() { return _row_idx; }
    };

    /**
     * @brief Functor to compute the hive hash value for a nested column.
     *
     * This functor produces the same result as "HiveHash" in Spark for structs and lists.
     * The pseudocode of Spark's HiveHash function for structs is as follows:
     *
     * hive_hash_value_t hive_hash(NestedType element) {
     *    hive_hash_value_t hash = HIVE_INIT_HASH;
     *    for (int i = 0; i < element.num_child(); i++) {
     *        hash = hash * HIVE_HASH_FACTOR + hive_hash(element.get_child(i));
     *    }
     *    return hash;
     * }
     *
     * In the cases of lists, the hash value is computed by a similar way but we iterate through the
     * list elements instead of through the child columns' elements.
     *
     * This functor uses a stack to simulate the recursive process of the above pseudocode.
     * When an element is popped from the stack, it means that the hash value of it has been
     * computed. Therefore, we should update the parent's `cur_hash` upon popping the element.
     *
     * The algorithm is as follows:
     *
     * 1. Initialize the stack and push the root column into the stack.
     * 2. While the stack is not empty:
     *    a. Get the top element of the stack. Don't pop it until it is processed.
     *    b. If the column is a structs column:
     *        i.  If all child columns are processed, pop the element and update `cur_hash` of its
     *            parent column.
     *        ii. Otherwise, process the next child column.
     *    c. If the column is a lists column, process it by a similar way as structs column but
     *       iterating through the list elements instead of child columns' elements.
     * 3. Return the hash value of the root column.
     *
     * For example, consider the following nested column: `Struct<Struct<int, float>, decimal>`
     *
     *      S1
     *     / \
     *    S2  d
     *   / \
     *  i   f
     *
     * - First, S1 is pushed into the stack. Then, S2 is pushed into the stack.
     * - S2's hash value can be computed directly because its children are of primitive types.
     *   When S2 is popped, S1's `cur_hash` is updated to S2's hash value.
     * - Now the top of the stack is S1. The next child to process is d. S1's `cur_hash` is updated
     *   to `hash(S2) * HIVE_HASH_FACTOR + hash(d)`, which is the hash value of S1.
     * - When S1 is popped, the hash value of the root column is returned.
     *
     * As lists columns have a different interface from structs columns, we need to handle them
     * separately.
     *
     * For example, consider that the input column is of type `List<List<int>>`.
     * Assume that the element at `row_index` is: [[1, 0], null, [2, null]]
     * Since the stack_frame should contain a column that consists of only one row, the input column
     * should be sliced. The sliced column is noted as L1 here.
     *
     *     L1
     *     |
     *     L2
     *     |
     *     i
     *
     * List level L1:
     * |Index|      List<list<int>>    |
     * |-----|-------------------------|
     * |0    |[[1, 0], null, [2, null]]|
     * length: 1
     * Offsets: 0, 3
     *
     * List level L2:
     * |Index|List<int>|
     * |-----|---------|
     * |0    |[1, 0]   |
     * |1    |null     |
     * |2    |[2, null]|
     * length: 3
     * Offsets: 0, 2, 2, 4
     * null_mask: 101
     *
     * Int level i:
     * |Index|int |
     * |-----|----|
     * |0    |1   |
     * |1    |0   |
     * |2    |2   |
     * |3    |null|
     * length: 4
     * null_mask: 0111
     *
     * Since the underlying data loses the null information of the top-level list column, computing
     * hash values using the underlying data merely can yield different results compared to Spark.
     * For example, [[1, 0], [2, null]] has the same underlying data as [[1, 0], null, [2, null]].
     * However, they have different hive hash values, as null values affect the hash value.
     *
     * And sublist structure factors in the hash value. For example, [[1, 0], [2]] and [[1], [0, 2]]
     * have different hive hash values.
     *
     * The computation process for lists columns in this solution is as follows:
     *            L1              List<list<int>>
     *            |
     *            L2              List<int>
     *      /     |     \
     *    L2[0] L2[1] L2[2]       int
     *     |            |
     *    i1           i2         int
     *
     * Note: L2、i1、i2 are all temporary columns, which would not be pushed into the stack.
     * If the child column is of primitive type, the hash value of the list column can be directly
     * computed.
     *
     * @tparam T The type of the column.
     * @param col The column device view.
     * @param row_index The index of the row to compute the hash for.
     * @return The computed hive hash value.
     *
     * @note This function is only enabled for nested column types.
     */
    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ hive_hash_value_t operator()(cudf::column_device_view const&,
                                            cudf::size_type row_index) const noexcept
    {
      auto curr_col_idx = _nested_column_map[_col_idx];
      auto curr_row_idx = row_index;

      col_stack_frame col_stack[MAX_STACK_DEPTH];
      int stack_size = 0;
      col_stack[stack_size++].init(curr_col_idx, curr_row_idx);

      while (stack_size > 0) {
        col_stack_frame& top = col_stack[stack_size - 1];
        curr_col_idx         = top.get_col_idx();
        curr_row_idx         = top.get_row_idx();
        auto const& curr_col = _flattened_column_views[curr_col_idx];
        // Do not pop it until it is processed. The definition of `processed` is:
        // - For structs, it is when all child columns are processed.
        // - For lists, it is when all elements in the list are processed.
        if (curr_col.type().id() == cudf::type_id::STRUCT) {
          if (top.get_idx_to_process() == curr_col.num_child_columns()) {
            if (--stack_size > 0) { col_stack[stack_size - 1].update_cur_hash(top.get_hash()); }
          } else {
            while (top.get_idx_to_process() < curr_col.num_child_columns()) {
              auto idx              = top.get_and_inc_idx_to_process();
              auto child_col_idx    = _first_child_index[curr_col_idx] + idx;
              auto const& child_col = _flattened_column_views[child_col_idx];
              // If the child is of primitive type, accumulate child hash into struct hash
              if (child_col.type().id() != cudf::type_id::LIST &&
                  child_col.type().id() != cudf::type_id::STRUCT) {
                auto child_hash =
                  cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
                    child_col.type(), this->hash_functor, child_col, curr_row_idx);
                top.update_cur_hash(child_hash);
              } else {
                col_stack[stack_size++].init(child_col_idx, curr_row_idx);
                break;
              }
            }
          }
        } else if (curr_col.type().id() == cudf::type_id::LIST) {
          // Get the child column of the list column
          auto offsets_col_idx     = _first_child_index[curr_col_idx];
          auto child_col_idx       = offsets_col_idx + 1;
          auto const& offsets_col  = _flattened_column_views[offsets_col_idx];
          auto const& child_col    = _flattened_column_views[child_col_idx];
          auto child_row_idx_begin = offsets_col.element<cudf::size_type>(curr_row_idx);
          auto child_row_idx_end   = offsets_col.element<cudf::size_type>(curr_row_idx + 1);

          // If the child column is of primitive type, directly compute the hash value of the list
          if (child_col.type().id() != cudf::type_id::LIST &&
              child_col.type().id() != cudf::type_id::STRUCT) {
            auto single_level_list_hash = cudf::detail::accumulate(
              thrust::counting_iterator(child_row_idx_begin),
              thrust::counting_iterator(child_row_idx_end),
              HIVE_INIT_HASH,
              [child_col, hasher = this->hash_functor] __device__(auto hash, auto element_index) {
                auto cur_hash = cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
                  child_col.type(), hasher, child_col, element_index);
                return HIVE_HASH_FACTOR * hash + cur_hash;
              });
            top.update_cur_hash(single_level_list_hash);
            if (--stack_size > 0) { col_stack[stack_size - 1].update_cur_hash(top.get_hash()); }
          } else {
            if (top.get_idx_to_process() == child_row_idx_end - child_row_idx_begin) {
              if (--stack_size > 0) { col_stack[stack_size - 1].update_cur_hash(top.get_hash()); }
            } else {
              // Push the next element into the stack
              col_stack[stack_size++].init(child_col_idx,
                                           child_row_idx_begin + top.get_and_inc_idx_to_process());
            }
          }
        }
      }
      return col_stack[0].get_hash();
    }

   private:
    hash_functor_t const hash_functor;
    cudf::size_type const _col_idx;
    cudf::column_device_view* _flattened_column_views;
    cudf::size_type* _first_child_index;
    cudf::size_type* _nested_column_map;
  };

  Nullate const _check_nulls;
  cudf::table_device_view const _table;
  cudf::column_device_view* _flattened_column_views;
  cudf::size_type* _first_child_index;
  cudf::size_type* _nested_column_map;
};

void check_nested_depth(cudf::table_view const& input)
{
  using column_checker_fn_t = std::function<int(cudf::column_view const&)>;

  column_checker_fn_t get_nested_depth = [&](cudf::column_view const& col) {
    if (col.type().id() == cudf::type_id::LIST) {
      auto const child_col = cudf::lists_column_view(col).child();
      return 1 + get_nested_depth(child_col);
    } else if (col.type().id() == cudf::type_id::STRUCT) {
      int max_child_depth = 0;
      for (auto child = col.child_begin(); child != col.child_end(); ++child) {
        max_child_depth = std::max(max_child_depth, get_nested_depth(*child));
      }
      return 1 + max_child_depth;
    } else {  // Primitive type
      return 0;
    }
  };

  for (auto i = 0; i < input.num_columns(); i++) {
    cudf::column_view const& col = input.column(i);
    CUDF_EXPECTS(get_nested_depth(col) <= MAX_STACK_DEPTH,
                 "The " + std::to_string(i) +
                   "-th column exceeds the maximum allowed nested depth. " +
                   "Current depth: " + std::to_string(get_nested_depth(col)) + ", " +
                   "Maximum allowed depth: " + std::to_string(MAX_STACK_DEPTH));
  }
}

}  // namespace

std::unique_ptr<cudf::column> hive_hash(cudf::table_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hive_hash_value_t>()),
                                          input.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  check_nested_depth(input);

  // `flattened_column_views` only contains nested columns and columns that result from flattening
  // nested columns
  std::vector<cudf::column_view> flattened_column_views;
  // `first_child_index` has the same size as `flattened_column_views`
  std::vector<cudf::size_type> first_child_index;
  // `nested_column_map` has the same size as `input.num_columns()`, and it maps the column index in
  // `input` to the index in `flattened_column_views`
  std::vector<cudf::size_type> nested_column_map;

  for (auto const& root_col : input) {
    if (root_col.type().id() == cudf::type_id::LIST ||
        root_col.type().id() == cudf::type_id::STRUCT) {
      // Construct the `flattened_column_views` by level order traversal
      nested_column_map.push_back(static_cast<cudf::size_type>(flattened_column_views.size()));
      flattened_column_views.push_back(root_col);
      // flattened_column_views[idx] is the next column to process
      for (auto idx = nested_column_map.back();
           idx < static_cast<cudf::size_type>(flattened_column_views.size());
           idx++) {
        auto const col = flattened_column_views[idx];
        if (col.type().id() == cudf::type_id::LIST) {
          first_child_index.push_back(static_cast<cudf::size_type>(flattened_column_views.size()));
          flattened_column_views.push_back(cudf::lists_column_view(col).offsets());
          flattened_column_views.push_back(cudf::lists_column_view(col).get_sliced_child(stream));
        } else if (col.type().id() == cudf::type_id::STRUCT) {
          first_child_index.push_back(static_cast<cudf::size_type>(flattened_column_views.size()));
          for (auto child_idx = 0; child_idx < col.num_children(); child_idx++) {
            flattened_column_views.push_back(
              cudf::structs_column_view(col).get_sliced_child(child_idx, stream));
          }
        } else {
          first_child_index.push_back(-1);
        }
      }
    } else {
      nested_column_map.push_back(-1);
    }
  }

  [[maybe_unused]] auto [device_view_owners, flattened_column_device_views] =
    cudf::contiguous_copy_column_device_views<cudf::column_device_view>(flattened_column_views,
                                                                        stream);
  auto first_child_index_view = cudf::detail::make_device_uvector_async(
    first_child_index, stream, cudf::get_current_device_resource_ref());
  auto nested_column_map_view = cudf::detail::make_device_uvector_async(
    nested_column_map, stream, cudf::get_current_device_resource_ref());

  bool const nullable   = has_nested_nulls(input);
  auto const input_view = cudf::table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   output_view.begin<hive_hash_value_t>(),
                   output_view.end<hive_hash_value_t>(),
                   hive_device_row_hasher<hive_hash_function, bool>(nullable,
                                                                    *input_view,
                                                                    flattened_column_device_views,
                                                                    first_child_index_view.data(),
                                                                    nested_column_map_view.data()));

  // Push data from host vectors `first_child_index` and `nested_column_map` to device
  // before they are destroyed.
  stream.synchronize();
  return output;
}

}  // namespace spark_rapids_jni
