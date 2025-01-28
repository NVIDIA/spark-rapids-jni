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

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;

import static java.util.Objects.requireNonNull;

/**
 * Provides serializer and deserializer APIs similar to the java KudoSerializer
 * but are GPU accelerated. The APIs provided do not match KudoSerializer directly
 * because they operate slightly differently.
 */
public class KudoGpuSerializer {

  private final Schema schema;
  // TODO a lot of this might need to change so we can so we can get the APIs right.
  private final int flattenedColumnCount;

  public KudoGpuSerializer(Schema schema) {
    requireNonNull(schema, "schema is null");
    this.schema = schema;
    this.flattenedColumnCount = schema.getFlattenedColumnNames().length;
  }

  /**
   * This splits and serializes the table based on the indices passed in.
   * @param table the table to be split up. The table must match the schema passed in.
   * @param indices the indices into the table that will be used to split it up. These follow the same
   *                pattern as contiguous split.
   * @return a host memory buffers for each split.
   */
  public HostMemoryBuffer[] splitAndSerializeToHost(Table table, int... indices) {
    // TODO some how we need to assert that the schema matches what was returned from the processing...
    return null;
  }

  /**
   * Split the input table and serialize it to a device buffer.
   * @param tableNativeView the native view of the table to split
   * @param indices the row indices to split around
   * @param flattenedTypeIds the type ids that are expected to be returned (validation?? do we care??)
   * @param flattenedNumChildren the num children that are expected to be returned (validation?? do we care??)
   * @param flattenedScale the decimal scales that are expected to be returned (validation?? do we care??)
   * @return an array of 2 DeviceMemoryBuffers. The first holds all the serialized data. The second holds
   * size_t offsets into the first to show where the splits are.
   */
  // TODO make the return values like gather_maps_to_java in CUDF.
  private static native DeviceMemoryBuffer[] splitAndSerializeToDevice(long tableNativeView,
                                                       int[] indices,
                                                       int[] flattenedTypeIds,
                                                       int[] flattenedNumChildren,
                                                       int[] flattenedScale);

  public Table mergeToTable(DeviceMemoryBuffer ... buffers) {
    // TODO but how do we know what to read???
    return null;
  }

  // TODO we need a way to read from an input stream into a HostMemoryBuffer. This partly exist in the
  //  KudoSerializer except the header is not written out serialized too, like it would need to be here.
}