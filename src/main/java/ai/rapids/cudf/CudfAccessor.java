/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

package ai.rapids.cudf;

// TODO: properly expose these functions in the actual Scalar API and remove this layer.
// https://github.com/NVIDIA/spark-rapids-jni/issues/1307
public class CudfAccessor {
  public static long getScalarHandle(Scalar s) {
    return s.getScalarHandle();
  }

  public static Scalar scalarFromHandle(DType type, long scalarHandle) {
    return new Scalar(type, scalarHandle);
  }
}