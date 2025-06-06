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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;

import static java.util.Objects.requireNonNull;

/**
 * Right now this is just to provide access to the underlying C++ APIs. In the future it should hopefully
 * look more like the CPU KudoSerializer.
 */
public class KudoGpuSerializer {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static DeviceMemoryBuffer[] splitAndSerializeToDevice(Table table, int... splits) {
    DeviceMemoryBuffer[] ret = new DeviceMemoryBuffer[2];
    boolean success = false;
    try {
      long[] values = splitAndSerializeToDevice(table.getNativeView(), splits);
      ret[0] = DeviceMemoryBuffer.fromRmm(values[0], values[1], values[2]);
      ret[1] = DeviceMemoryBuffer.fromRmm(values[3], values[4], values[5]);
      success = true;
      return ret;
    } finally {
      if (!success) {
        if (ret[0] != null) {
          ret[0].close();
        }
        if (ret[1] != null) {
          ret[1].close();
        }
      }
    }
  }

  public static Table assembleFromDeviceRaw(Schema schema,
                                            DeviceMemoryBuffer partitions,
                                            DeviceMemoryBuffer offsets) {
    return new Table(assembleFromDeviceRawNative(
        partitions.getAddress(), partitions.getLength(),
        offsets.getAddress(), offsets.getLength(),
        schema.getFlattenedNumChildren(),
        schema.getFlattenedTypeIds(),
        schema.getFlattenedTypeScales()));
  }

  /**
   * Split the input table and serialize it to a device buffer.
   * @param tableNativeView the native view of the table to split
   * @param splits the row indices to split around
   * @return an array of 6 longs. These represent two device buffers each with 3 values.
   * (address, length, rmmAddress) The first tuple holds all the serialized data. The second tuple holds
   * size_t offsets into the first to show where the splits are.
   */
  private static native long[] splitAndSerializeToDevice(long tableNativeView, int[] splits);


  private static native long[] assembleFromDeviceRawNative(long partAddr, long partLen,
                                                           long offsetAddr, long offsetLen,
                                                           int[] flattenedNumChildren,
                                                           int[] flattenedTypeIds,
                                                           int[] flattenedTypeScales);

}