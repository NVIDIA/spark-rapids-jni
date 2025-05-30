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

#include "version.hpp"

#include <stdexcept>

namespace spark_rapids_jni {

platform_type get_platform_type(int platform_ordinal)
{
  switch (platform_ordinal) {
    case PLATFORM_SPARK: return platform_type::SPARK;
    case PLATFORM_DATABRICKS: return platform_type::DATABRICKS;
    case PLATFORM_CLOUDERA: return platform_type::CLOUDERA;
    default:
      throw std::invalid_argument("Invalid platform ordinal: " + std::to_string(platform_ordinal));
  }
}

}  // namespace spark_rapids_jni
