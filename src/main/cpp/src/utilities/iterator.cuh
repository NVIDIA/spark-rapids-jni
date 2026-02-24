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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>


#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

namespace spark_rapids_jni::util {

/**
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `cuda::counting_iterator` within the range [0, INT_MAX].
 * Note: This is practically identical to `cudf::detail::make_counting_transform_iterator`,
 * except that it uses `cuda::counting_iterator` instead of `thrust::counting_iterator`.
 *
 * Example:
 * @code{.cpp}
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);});
 * iter[0] == 0
 * iter[1] == 1
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * @endcode
 *
 * @param start The starting value of the counting iterator (must be size_type or smaller type).
 * @param f The unary function to apply to the counting iterator.
 * @return A transform iterator that applies `f` to a counting iterator
 */
template <typename CountingIterType, typename UnaryFunction>
CUDF_HOST_DEVICE inline auto make_counting_transform_iterator(CountingIterType start,
                                                              UnaryFunction f)
{
  // Check if the `start` for counting_iterator is of size_type or a smaller integral type
  static_assert(
    cuda::std::is_integral_v<CountingIterType> and
      cuda::std::numeric_limits<CountingIterType>::digits <=
        cuda::std::numeric_limits<cudf::size_type>::digits,
    "The `start` for the counting_transform_iterator must be size_type or smaller type");

  return thrust::make_transform_iterator(cuda::make_counting_iterator(start), f);
}

/**
 * @brief Constructs a pair iterator over a column's values and its validity.
 * Note: This is identical to `cudf::detail::make_pair_iterator`.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<Element, bool>`.
 *
 * If an element at position `i` is valid (or `has_nulls == false`), then for `p = *(iter + i)`,
 * `p.first` contains the value of the element at `i` and `p.second == true`.
 *
 * Else, if the element at `i` is null, then the value of `p.first` is undefined and `p.second ==
 * false`. `pair(column[i], validity)`. `validity` is `true` if `has_nulls=false`. `validity` is
 * validity of the element at `i` if `has_nulls=true` and the column is nullable.
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 * @param column The column to iterate
 * @return auto Iterator that returns valid column elements, and validity of the
 * element in a pair
 */
template <typename Element, bool has_nulls = false>
auto make_pair_iterator(cudf::column_device_view const& column)
{
  return column.pair_begin<Element, has_nulls>();
}

/**
 * @brief value accessor for scalar with valid data.
 * Note: This is identical to `cudf::detail::scalar_value_accessor`.
 * The unary functor returns data of Element type of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_value_accessor {
  using ScalarType       = cudf::scalar_type_t<Element>;
  using ScalarDeviceType = cudf::scalar_device_type_t<Element>;
  ScalarDeviceType const dscalar;  ///< scalar device view

  scalar_value_accessor(cudf::scalar const& scalar_value)
    : dscalar(cudf::get_scalar_device_view(
        static_cast<ScalarType&>(const_cast<cudf::scalar&>(scalar_value))))
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
                 "the data type mismatch");
  }

  __device__ inline Element const operator()(cudf::size_type) const { return dscalar.value(); }
};

/**
 * @brief Optional accessor for a scalar
 * Note: This is identical to `cudf::detail::scalar_optional_accessor`.
 *
 * The `scalar_optional_accessor` always returns a `cuda::std::optional` of the scalar.
 * The validity of the optional is determined by the `Nullate` parameter which may
 * be one of the following:
 *
 * - `nullate::YES` means that the scalar may be valid or invalid and the optional returned
 *    will contain a value only if the scalar is valid.
 *
 * - `nullate::NO` means the caller attests that the scalar will always be valid,
 *    no checks will occur and `cuda::std::optional{column[i]}` will return a value
 *    for each `i`.
 *
 * - `nullate::DYNAMIC` defers the assumption of nullability to runtime and the caller
 *    specifies if the scalar may be valid or invalid.
 *    For `DYNAMIC{true}` the return value will be a `cuda::std::optional{scalar}` when the
 *      scalar is valid and a `cuda::std::optional{}` when the scalar is invalid.
 *    For `DYNAMIC{false}` the return value will always be a `cuda::std::optional{scalar}`.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Element, typename Nullate>
struct scalar_optional_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = cuda::std::optional<Element>;

  scalar_optional_accessor(cudf::scalar const& scalar_value, Nullate with_nulls)
    : scalar_value_accessor<Element>(scalar_value), has_nulls{with_nulls}
  {
  }

  __device__ inline value_type const operator()(cudf::size_type) const
  {
    if (has_nulls && !super_t::dscalar.is_valid()) { return value_type{cuda::std::nullopt}; }

    if constexpr (cudf::is_fixed_point<Element>()) {
      using namespace numeric;
      using rep        = typename Element::rep;
      auto const value = super_t::dscalar.rep();
      auto const scale = scale_type{super_t::dscalar.type().scale()};
      return Element{scaled_integer<rep>{value, scale}};
    } else {
      return Element{super_t::dscalar.value()};
    }
  }

  Nullate has_nulls{};
};

/**
 * @brief pair accessor for scalar.
 * Note: This is identical to `cudf::detail::scalar_pair_accessor`.
 *
 * The unary functor returns a pair of data of Element type and bool validity of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_pair_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = cuda::std::pair<Element, bool>;
  scalar_pair_accessor(cudf::scalar const& scalar_value) : scalar_value_accessor<Element>(scalar_value) {}

  __device__ inline value_type const operator()(cudf::size_type) const
  {
    return {Element(super_t::dscalar.value()), super_t::dscalar.is_valid()};
  }
};

/**
 * @brief Constructs a constant device pair iterator over a scalar's value and its validity.
 * Note: This is identical to `cudf::detail::make_pair_iterator`.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<Element, bool>`.
 *
 * If scalar is valid, then for `p = *(iter + i)`, `p.first` contains
 * the value of the scalar and `p.second == true`.
 *
 * Else, if the scalar is null, then the value of `p.first` is undefined and `p.second == false`.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of elements in the scalar
 * @tparam bool unused. This template parameter exists to enforce same
 * template interface as @ref make_pair_iterator(column_device_view const&).
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar, and validity of the scalar in a pair
 */
template <typename Element, bool = false>
auto inline make_pair_iterator(cudf::scalar const& scalar_value)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(cuda::make_constant_iterator<cudf::size_type>(0),
                                         scalar_pair_accessor<Element>{scalar_value});
}

}  // namespace spark_rapids_jni::util
