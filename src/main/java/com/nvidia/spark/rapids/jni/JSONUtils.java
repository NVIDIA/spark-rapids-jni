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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

public class JSONUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static ColumnVector getJsonObject(ColumnVector input, int[] path_ins_types, String[] path_ins_names,
  long[] path_ins_indexes) {
    assert(input.getType().equals(DType.STRING)) : "column must be a String";
    assert(path_ins_types.length == path_ins_names.length) : "path_ins_types and path_ins_names must have the same size";
    assert(path_ins_types.length == path_ins_indexes.length) : "path_ins_types and path_ins_indexes must have the same size";
    return new ColumnVector(getJsonObject(input.getNativeView(), path_ins_types, path_ins_names, path_ins_indexes));
  }

  private static native long getJsonObject(long input, int[] path_ins_types, String[] path_ins_names, long[] path_ins_indexes);
}
