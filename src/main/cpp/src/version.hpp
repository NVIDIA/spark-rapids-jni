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

#pragma once

namespace spark_rapids_jni {

// NOTE: MUST keep sync with Platform.java
constexpr int PLATFORM_SPARK      = 0;
constexpr int PLATFORM_DATABRICKS = 1;
constexpr int PLATFORM_CLOUDERA   = 2;

enum class platform_type { SPARK, DATABRICKS, CLOUDERA };

/**
 * @brief Get the platform type based on the integer value from JNI.
 */
platform_type get_platform_type(int platform_ordinal);

}  // namespace spark_rapids_jni
