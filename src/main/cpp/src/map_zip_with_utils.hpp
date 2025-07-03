#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

using namespace cudf;

namespace spark_rapids_jni {

/**
 * @brief Zip two lists columns element-wise to create key-value pairs
 *
 * This function combines two lists columns by zipping their elements together to form
 * key-value pairs. Each element from `col1` is paired with the corresponding element
 * from `col2` with the same key. If a value cannot be matched to a key, it is replaced with null.
 *
 * @param col1 The first lists column (keys)
 * @param col2 The second lists column (values)
 * @param stream CUDA stream for asynchronous execution (default: default stream)
 * @param mr Memory resource for device memory allocation (default: current device resource)
 *
 * @return A unique pointer to the column of zipped maps
 *
 * @note Both input columns must have the same number of rows.
 *
 * @note The function preserves the null mask and validity of the input columns.
 *
 * ```
 */
[[maybe_unused]] std::unique_ptr<cudf::column> map_zip(
  cudf::lists_column_view const& col1,
  cudf::lists_column_view const& col2,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni