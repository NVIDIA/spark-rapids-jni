/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package com.nvidia.spark.rapids.jni.schema;

import ai.rapids.cudf.HostColumnVectorCore;
import java.util.List;

/**
 * A post order visitor for visiting a list of host columns in a schema.
 *
 * <p>
 * <p>
 * For example, if we have three columns A, B, and C with following types:
 *
 * <ul>
 *    <li> A: <code>struct { int a1; long a2} </code> </li>
 *    <li> B: <code>list { int b1} </code> </li>
 *    <li> C: <code>string c1 </code> </li>
 * </ul>
 * <p>
 * The order of visiting will be:
 * <ol>
 *     <li> Visit primitive column a1 </li>
 *     <li> Visit primitive column a2</li>
 *     <li> Visit struct column A, with results from a1 and a2</li>
 *     <li> Previsit list column B</li>
 *     <li> Visit primitive column b1</li>
 *     <li> Visit list column B</li>
 *     <li> Visit primitive column c1</li>
 * </ol>
 *
 * </p>
 *
 * @param <T> Return type when visiting intermediate nodes.
 */
public interface HostColumnsVisitor<T> {
  /**
   * Visit a struct column.
   *
   * @param col      the struct column to visit
   * @param children the results of visiting the children
   * @return the result of visiting the struct column
   */
  T visitStruct(HostColumnVectorCore col, List<T> children);

  /**
   * Visit a list column before actually visiting its child.
   *
   * @param col the list column to visit
   * @return the result of visiting the list column
   */
  T preVisitList(HostColumnVectorCore col);

  /**
   * Visit a list column after visiting its child.
   *
   * @param col            the list column to visit
   * @param preVisitResult the result of visiting the list column before visiting its child
   * @param childResult    the result of visiting the child
   * @return the result of visiting the list column
   */
  T visitList(HostColumnVectorCore col, T preVisitResult, T childResult);

  /**
   * Visit a column that is a primitive type.
   *
   * @param col the column to visit
   * @return the result of visiting the column
   */
  T visit(HostColumnVectorCore col);
}
