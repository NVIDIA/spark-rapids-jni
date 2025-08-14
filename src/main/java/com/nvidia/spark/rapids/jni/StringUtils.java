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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

class StringUtils {

  // Stores the sequence ID of calling generate UUIDs.
  private static AtomicLong sequence = new AtomicLong(0);

  /**
   * Generate a seed for UUID generation.
   * The seed is generated based on the current time in nanoseconds, the process
   * name, the GPU UUID, and a sequence ID that increments with each call to
   * this method. This method ensures (do best effort) that the seed is unique
   * across different runs, including different Spark jobs, different executions
   * of the same job, set backward of the clock, etc.
   *
   * @return A seed for UUID generation.
   */
  private static long randomSeed() {
    long seed = System.nanoTime();
    String processName = ManagementFactory.getRuntimeMXBean().getName();
    byte[] gpuUUID = Cuda.getGpuUuid();
    seed = seed * 37 + processName.hashCode();
    seed = seed * 37 + Arrays.hashCode(gpuUUID);
    seed = seed * 37 + sequence.incrementAndGet();
    return seed;
  }

  /**
   * Generate a column of UUIDs (String type) with `rowCount` rows.
   * Spark uses `Truly Random or Pseudo-Random` UUID type which is described in
   * the section 4.4 of [RFC4122](https://datatracker.ietf.org/doc/html/rfc4122),
   * The variant in UUID is 2 and the version in UUID is 4. This implementation
   * generates UUIDs in the same format, but does not generate the same UUIDs as
   * Spark. This function is indeterministic, meaning that it will generate
   * different UUIDs each time it is called, even with the same row count.
   * The UUIDs are generated using a seed based on the current time, process name,
   * GPU UUID and running sequence index, ensuring uniqueness across different
   * runs.
   *
   * E.g.: "123e4567-e89b-12d3-a456-426614174000"
   *
   * @param rowCount Number of UUIDs to generate
   * @return ColumnVector containing UUIDs
   */
  public static ColumnVector randomUUIDs(int rowCount) {
    long seed = randomSeed();
    return new ColumnVector(randomUUIDs(rowCount, seed));
  }

  /**
   * Only for test purpose, please do not use it in production.
   */
  static ColumnVector randomUUIDsWithSeed(int rowCount, long seed) {
    return new ColumnVector(randomUUIDs(rowCount, seed));
  }

  private static native long randomUUIDs(int rowCount, long seed);
}
