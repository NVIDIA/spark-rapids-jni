/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "round_float.hpp"

#include <cuda/std/type_traits>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <cmath>
#include <type_traits>

namespace spark_rapids_jni {


inline float __device__ generic_round(float f) { return roundf(f); }
inline double __device__ generic_round(double d) { return ::round(d); }

inline float __device__ generic_round_half_even(float f) { return rintf(f); }
inline double __device__ generic_round_half_even(double d) { return rint(d); }

inline __device__ float generic_modf(float a, float* b) { return modff(a, b); }
inline __device__ double generic_modf(double a, double* b) { return modf(a, b); }


template <typename T>
struct half_up_zero {
  T n;  // unused in the decimal_places = 0 case
  __device__ T operator()(T e)
  {
    return generic_round(e);
  }
};

template <typename T>
struct half_up_positive {
  T n;
  __device__ T operator()(T e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round(fractional_part * n) / n;
  }
};

template <typename T>
struct half_up_negative {
  T n;
  __device__ T operator()(T e)
  {
    return generic_round(e / n) * n;
  }
};

template <typename T>
struct half_even_zero {
  T n;  // unused in the decimal_places = 0 case
  __device__ T operator()(T e)
  {
    return generic_round_half_even(e);
  }
};

template <typename T>
struct half_even_positive {
  T n;
  __device__ T operator()(T e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round_half_even(fractional_part * n) / n;
  }
};

template <typename T>
struct half_even_negative {
  T n;
  __device__ T operator()(T e)
  {
    return generic_round_half_even(e / n) * n;
  }
};

template <typename T, template <typename> typename RoundFunctor>
std::unique_ptr<cudf::column> round_with(cudf::column_view const& input,
                                         int32_t decimal_places,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
{
  using Functor = RoundFunctor<T>;

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              cudf::detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();
  T const n     = std::pow(10, std::abs(decimal_places));

  thrust::transform(
    rmm::exec_policy(stream), input.begin<T>(), input.end<T>(), out_view.begin<T>(), Functor{n});

  result->set_null_count(input.null_count());

  return result;
}

struct round_type_dispatcher {
  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Type not supported for spark_rapids_jni::round");
  }
  
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           int32_t decimal_places,
                                           cudf::rounding_method method,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    // clang-format off
    switch (method) {
      case cudf::rounding_method::HALF_UP:
        if      (decimal_places == 0) return round_with<T, half_up_zero      >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_up_positive  >(input, decimal_places, stream, mr);
        else                          return round_with<T, half_up_negative  >(input, decimal_places, stream, mr);
      case cudf::rounding_method::HALF_EVEN:
        if      (decimal_places == 0) return round_with<T, half_even_zero    >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_even_positive>(input, decimal_places, stream, mr);
        else                          return round_with<T, half_even_negative>(input, decimal_places, stream, mr);
      default: CUDF_FAIL("Undefined rounding method");
    }
    // clang-format on
  }
};

std::unique_ptr<cudf::column> round(cudf::column_view const& input,
                                    int32_t decimal_places,
                                    cudf::rounding_method method,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(cudf::is_numeric(input.type()) || cudf::is_fixed_point(input.type()),
               "Only integral/floating point/fixed point currently supported.");

  if(!cudf::is_floating_point(input.type())) {
    return cudf::round_decimal(input, decimal_places, method, stream, mr);
  }
  if (input.is_empty()) {
    return cudf::empty_like(input);
  }

  return cudf::type_dispatcher(
    input.type(), round_type_dispatcher{}, input, decimal_places, method, stream, mr);
}

} // namespace spark_rapids_jni