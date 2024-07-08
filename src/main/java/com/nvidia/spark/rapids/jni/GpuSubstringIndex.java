/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

public class GpuSubstringIndex {
    static{
        NativeDepsLoader.loadNativeDeps();
    }

    public static ColumnVector gpuSubstringIndex(ColumnView cv, String delimiter, int count){
        return new ColumnVector(gpuSubstringIndex(cv.getNativeView(), delimiter, count));
    }


//    public static ColumnVector substringIndex(ColumnView cv, ColumnView delimiterView, int count){
//        assert delimiterView.getType().equals(DType.STRING) : "column type must be a String";
//        return new ColumnVector(substringIndexColumn(cv.getNativeView(), delimiterView.getNativeView(), count));
//    }

    private static native long gpuSubstringIndex(long columnView, String delimiter, int count) throws CudfException;

    //private static native long substringIndexColumn(long columnView, long delimiterView, int count) throws CudfException;


}
