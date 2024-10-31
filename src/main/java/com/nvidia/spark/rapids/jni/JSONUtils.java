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

import java.util.List;

public class JSONUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int MAX_PATH_DEPTH = getMaxJSONPathDepth();

  public enum PathInstructionType {
    WILDCARD,
    INDEX,
    NAMED
  }

  public static class PathInstructionJni {
    // type: byte, name: String, index: int
    private final byte type;
    private final String name;
    private final int index;

    public PathInstructionJni(PathInstructionType type, String name, long index) {
      this.type = (byte) type.ordinal();
      this.name = name;
      if (index > Integer.MAX_VALUE) {
        throw new IllegalArgumentException("index is too large " + index);
      }
      this.index = (int) index;
    }

    public PathInstructionJni(PathInstructionType type, String name, int index) {
      this.type = (byte) type.ordinal();
      this.name = name;
      this.index = index;
    }
  }

  /**
   * Extract a JSON path from a JSON column. The path is processed in a Spark compatible way.
   * @param input the string column containing JSON
   * @param pathInstructions the instructions for the path processing
   * @return the result of processing the path
   */
  public static ColumnVector getJsonObject(ColumnVector input, PathInstructionJni[] pathInstructions) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int numTotalInstructions = pathInstructions.length;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < pathInstructions.length; i++) {
      PathInstructionJni current = pathInstructions[i];
      typeNums[i] = current.type;
      names[i] = current.name;
      indexes[i] = current.index;
    }
    return new ColumnVector(getJsonObject(input.getNativeView(), typeNums, names, indexes));
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths) {
    return getJsonObjectMultiplePaths(input, paths, -1, -1);
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @param memoryBudgetBytes a budget that is used to limit the amount of memory
   *                          that is used when processing the paths. This is a soft limit.
   *                          A value <= 0 disables this and all paths will be processed in parallel.
   * @param parallelOverride Set a maximum number of paths to be processed in parallel. The memory
   *                         budget can limit how many paths can be processed in parallel. This overrides
   *                         that automatically calculated value with a set value for benchmarking purposes.
   *                         A value <= 0 disables this.
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int[] pathOffsets = new int[paths.size() + 1];
    int offset = 0;
    for (int i = 0; i < paths.size(); i++) {
      pathOffsets[i] = offset;
      offset += paths.get(i).size();
    }
    pathOffsets[paths.size()] = offset;
    int numTotalInstructions = offset;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths.get(i).size(); j++) {
        PathInstructionJni current = paths.get(i).get(j);
        typeNums[pathOffsets[i] + j] = current.type;
        names[pathOffsets[i] + j] = current.name;
        indexes[pathOffsets[i] + j] = current.index;
      }
    }
    long[] ptrs = getJsonObjectMultiplePaths(input.getNativeView(), typeNums,
        names, indexes, pathOffsets, memoryBudgetBytes, parallelOverride);
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }


  /**
   * Extract key-value pairs for each output map from the given json strings. These key-value are
   * copied directly as substrings of the input without any type conversion.
   * <p>
   * Since there is not any validity check, the output of this function may be different from
   * what generated by Spark's `from_json` function. Situations that can lead to
   * different/incorrect outputs may include:<br>
   * - The value in the input json string is invalid, such as 'abc' value for an integer key.<br>
   * - The value string can be non-clean format for floating-point type, such as '1.00000'.
   * <p>
   * The output of these situations should all be NULL or a value '1.0', respectively. However, this
   * function will just simply copy the input value strings to the output.
   *
   * @param input The input strings column in which each row specifies a json object
   * @param opts The options for parsing JSON strings
   * @return A map column (i.e., a column of type {@code List<Struct<String,String>>}) in
   * which the key-value pairs are extracted directly from the input json strings
   */
  public static ColumnVector extractRawMapFromJsonString(ColumnView input, JSONOptions opts) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(extractRawMapFromJsonString(input.getNativeView(),
        opts.isNormalizeSingleQuotes(),
        opts.leadingZerosAllowed(),
        opts.nonNumericNumbersAllowed(),
        opts.unquotedControlChars()));
  }

  /**
   * A class to hold the result when concatenating JSON strings.
   * <p>
   * A long with the concatenated data, the result also contains a vector that indicates
   * whether each row in the input is null or empty, and the delimiter used for concatenation.
   */
  public static class ConcatenatedJson implements AutoCloseable {
    public final ColumnVector isNullOrEmpty;
    public final DeviceMemoryBuffer data;
    public final char delimiter;

    public ConcatenatedJson(ColumnVector isNullOrEmpty, DeviceMemoryBuffer data, char delimiter) {
      this.isNullOrEmpty = isNullOrEmpty;
      this.data = data;
      this.delimiter = delimiter;
    }

    @Override
    public void close() {
      isNullOrEmpty.close();
      data.close();
    }
  }

  /**
   * Concatenate JSON strings in the input column into a single JSON string.
   * <p>
   * During concatenation, the function also generates a boolean vector that indicates whether
   * each row in the input is null or empty. The delimiter used for concatenation is also returned.
   *
   * @param input The input strings column to concatenate
   * @return A {@link ConcatenatedJson} object that contains the concatenated output
   */
  public static ConcatenatedJson concatenateJsonStrings(ColumnView input) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    long[] concatenated = concatenateJsonStrings(input.getNativeView());
    return new ConcatenatedJson(new ColumnVector(concatenated[0]),
        DeviceMemoryBuffer.fromRmm(concatenated[1], concatenated[2], concatenated[3]),
        (char) concatenated[4]);
  }

  /**
   * Create a structs column from the given children columns and a boolean column specifying
   * the rows at which the output column.should be null.
   * <p>
   * Note that the children columns are expected to have null rows at the same positions indicated
   * by the input isNull column.
   *
   * @param children The children columns of the output structs column
   * @param isNull A boolean column specifying the rows at which the output column should be null
   * @return A structs column created from the given children and the isNull column
   */
  public static ColumnVector makeStructs(ColumnView[] children, ColumnView isNull) {
    long[] handles = new long[children.length];
    for (int i = 0; i < children.length; i++) {
      handles[i] = children[i].getNativeView();
    }
    return new ColumnVector(makeStructs(handles, isNull.getNativeView()));
  }

  private static native int getMaxJSONPathDepth();

  private static native long getJsonObject(long input,
                                           byte[] typeNums,
                                           String[] names,
                                           int[] indexes);

  private static native long[] getJsonObjectMultiplePaths(long input,
                                                          byte[] typeNums,
                                                          String[] names,
                                                          int[] indexes,
                                                          int[] pathOffsets,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride);


  private static native long extractRawMapFromJsonString(long input,
                                                         boolean normalizeSingleQuotes,
                                                         boolean leadingZerosAllowed,
                                                         boolean nonNumericNumbersAllowed,
                                                         boolean unquotedControlChars);

  private static native long[] concatenateJsonStrings(long input);

  private static native long makeStructs(long[] children, long isNull);
}
