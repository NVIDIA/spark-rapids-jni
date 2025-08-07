/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;

public class ParseURI {
  static {
    NativeDepsLoader.loadNativeDeps();
  }


  /**
   * Parse protocol for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with protocol data extracted.
   */
  public static ColumnVector parseURIProtocol(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseProtocol(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseProtocol(uriColumn.getNativeView(), false));
    }
  }

  /**
   * Parse host for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with host data extracted.
   */
  public static ColumnVector parseURIHost(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseHost(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseHost(uriColumn.getNativeView(), false));
    }
  }

  /**
   * Parse query for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, throw exception on invalid URLs.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQuery(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseQuery(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseQuery(uriColumn.getNativeView(), false));
    }
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param query The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithLiteral(ColumnView uriColumn, String query, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseQueryWithLiteral(uriColumn.getNativeView(), query, true));
    } else {
      return new ColumnVector(parseQueryWithLiteral(uriColumn.getNativeView(), query, false));
    }
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param queryColumn The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithColumn(ColumnView uriColumn, ColumnView queryColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    assert queryColumn.getType().equals(DType.STRING) : "Query type must be String";
    if (failOnError) {
      return new ColumnVector(parseQueryWithColumn(uriColumn.getNativeView(), queryColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseQueryWithColumn(uriColumn.getNativeView(), queryColumn.getNativeView(), false));
    }
  }

  /**
   * Parse path for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with the URI path extracted.
   */
  public static ColumnVector parseURIPath(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parsePath(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parsePath(uriColumn.getNativeView(), false));
    }
  }

  private static native long parseProtocol(long inputColumnHandle, boolean ansiMode);
  private static native long parseHost(long inputColumnHandle, boolean ansiMode);
  private static native long parseQuery(long inputColumnHandle, boolean ansiMode);
  private static native long parseQueryWithLiteral(long inputColumnHandle, String query, boolean ansiMode);
  private static native long parseQueryWithColumn(long inputColumnHandle, long queryColumnHandle, boolean ansiMode);
  private static native long parsePath(long inputColumnHandle, boolean ansiMode);
}
