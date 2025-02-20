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

public class GpuListSliceUtils {
    static{
        NativeDepsLoader.loadNativeDeps();
    }

    /**
     * Slices a lists column, beginning at the specified integer start offset and
     * including up to the given integer length.
     *
     * @param cv    the column of lists to slice
     * @param start the integer offset at which to begin slicing
     * @param length the integer length of elements to include in the slice
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, int start, int length) {
        return new ColumnVector(listSliceIntInt(cv.getNativeView(), start, length));
    }

    /**
     * Slices a lists column, beginning at the specified integer start offset and
     * including up to the lengths specified by another column view.
     *
     * @param cv    the column of lists to slice
     * @param start the integer offset at which to begin slicing
     * @param length the column view specifying the lengths for each list slice
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, int start, ColumnView length) {
        return new ColumnVector(listSliceIntCol(cv.getNativeView(), start, length.getNativeView()));
    }

    /**
     * Slices a lists column, beginning at the offsets specified by a column view and
     * including up to a fixed integer length.
     *
     * @param cv    the column of lists to slice
     * @param start the column view specifying the start offsets for each list slice
     * @param length the integer length of elements to include in the slice
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, ColumnView start, int length) {
        return new ColumnVector(listSliceColInt(cv.getNativeView(), start.getNativeView(), length));
    }

    /**
     * Slices a lists column, beginning at the offsets specified by one column view and
     * including up to the lengths specified by another column view.
     *
     * @param cv    the column of lists to slice
     * @param start the column view specifying the start offsets for each list slice
     * @param length the column view specifying the lengths for each list slice
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, ColumnView start, ColumnView length) {
        return new ColumnVector(listSliceColCol(cv.getNativeView(), start.getNativeView(), length.getNativeView()));
    }

    private static native long listSliceIntInt(long listColumnView, int start, int length);

    private static native long listSliceIntCol(long listColumnView, int start, long lengthColumnView);

    private static native long listSliceColInt(long listColumnView, long startColumnView, int length);

    private static native long listSliceColCol(long listColumnView, long startColumnView, long lengthColumnView);
}
