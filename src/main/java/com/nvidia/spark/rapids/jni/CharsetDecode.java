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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

public class CharsetDecode {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /** Charset constant for GBK encoding */
  public static final int GBK = 0;

  /**
   * Decode a binary column from the specified charset encoding to a UTF-8 strings column.
   *
   * @param cv Binary column (LIST of UINT8) to decode
   * @param charset Charset identifier (use constants like {@link #GBK})
   * @return UTF-8 string column
   */
  public static ColumnVector decode(ColumnView cv, int charset) {
    if (charset != GBK) {
      throw new IllegalArgumentException("Unsupported charset: " + charset);
    }
    return new ColumnVector(decodeNative(cv.getNativeView(), charset));
  }

  private static native long decodeNative(long nativeColumnView, int charset) throws CudfException;
}
