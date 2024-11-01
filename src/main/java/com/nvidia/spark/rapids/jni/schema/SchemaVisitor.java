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

import ai.rapids.cudf.Schema;

import java.util.List;

/**
 * A post order visitor for schemas.
 *
 * <p>
 *
 * For example, if our schema consists of three fields A, B, and C with following types:
 *
 * <ul>
 *    <li> A: <code>struct { int a1; long a2} </code> </li>
 *    <li> B: <code>list { int b1} </code> </li>
 *    <li> C: <code>string c1 </code> </li>
 * </ul>
 *
 * The order of visiting will be:
 * <ol>
 *     <li> Visit primitive field a1 </li>
 *     <li> Visit primitive field a2</li>
 *     <li> Visit struct field A, with results from a1 and a2</li>
 *     <li> Previsit list field B</li>
 *     <li> Visit primitive field b1</li>
 *     <li> Visit list field B with results from b1 and previsit result. </li>
 *     <li> Visit primitive field c1</li>
 *     <li> Visit top schema with results from fields A, B, and C</li>
 * </ol>
 *
 * </p>
 *
 * @param <T> Return type when visiting intermediate nodes.
 * @param <R> Return type after processing all children values.
 */
public interface SchemaVisitor<T, R> {
    /**
     * Visit the top level schema.
     * @param schema the top level schema to visit
     * @param children the results of visiting the children
     * @return the result of visiting the top level schema
     */
    R visitTopSchema(Schema schema, List<T> children);

    /**
     * Visit a struct schema.
     * @param structType the struct schema to visit
     * @param children the results of visiting the children
     * @return the result of visiting the struct schema
     */
    T visitStruct(Schema structType, List<T> children);

    /**
     * Visit a list schema before actually visiting its child.
     * @param listType the list schema to visit
     * @return the result of visiting the list schema
     */
    T preVisitList(Schema listType);

    /**
     * Visit a list schema after visiting its child.
     * @param listType the list schema to visit
     * @param preVisitResult the result of visiting the list schema before visiting its child
     * @param childResult the result of visiting the child
     * @return the result of visiting the list schema
     */
    T visitList(Schema listType, T preVisitResult, T childResult);

    /**
     * Visit a primitive type.
     * @param primitiveType the primitive type to visit
     * @return the result of visiting the primitive type
     */
    T visit(Schema primitiveType);
}
