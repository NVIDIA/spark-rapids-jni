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

/**
 * @brief Enum class representing different Spark platform types.
 * The values must match the ordinal values defined in SparkPlatformType.java.
 * - VANILLA_SPARK: Represents the standard Apache Spark platform.
 * - DATABRICKS: Represents the Databricks platform.
 * - CLOUDERA: Represents the Cloudera platform.
 * - NUM_PLATFORMS: Represents the total number of platforms defined.
 */
enum class spark_platform_type { VANILLA_SPARK = 0, DATABRICKS, CLOUDERA, NUM_PLATFORMS };

class spark_system {
 public:
  /**
   * @brief Constructor to initialize the spark system with platform type and version.
   * NOTE: The `spark_platform_ordinal` MUST keep sync with SparkPlatformType.java
   * @param spark_platform The platform ordinal value.
   * @param major Major version number.
   * @param minor Minor version number.
   * @param patch Patch version number.
   */
  spark_system(int spark_platform_ordinal, int major_, int minor_, int patch_)
    : platform(spark_platform_ordinal), major(major_), minor(minor_), patch(patch_)
  {
  }

  bool isVanillaSpark() const { return spark_platform() == spark_platform_type::VANILLA_SPARK; }
  bool isDatabricks() const { return spark_platform() == spark_platform_type::DATABRICKS; }

  bool isVanilla_320() const { return isVanillaSpark() && major == 3 && minor == 2 && patch == 0; }

  bool isVanilla_400_or_later() const
  {
    return isVanillaSpark() &&
           (major > 4 || (major == 4 && minor >= 0) || (major == 4 && minor == 0 && patch >= 0));
  }

  bool isDatabricks_14_3_or_later() const
  {
    return (isDatabricks() && (major > 14 || (major == 14 && minor >= 3)));
  }

 private:
  spark_platform_type spark_platform() const { return static_cast<spark_platform_type>(platform); }

  int platform, major, minor, patch;
};

}  // namespace spark_rapids_jni
