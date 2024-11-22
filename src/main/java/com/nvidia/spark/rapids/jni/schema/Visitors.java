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

import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.Schema;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A utility class for visiting a schema or a list of host columns.
 */
public class Visitors {
    /**
     * Visiting a schema in post order. For more details, see {@link SchemaVisitor}.
     *
     * @param schema the schema to visit
     * @param visitor the visitor to use
     * @param <T> Return type when visiting intermediate nodes. See {@link SchemaVisitor}
     * @param <P> Return type when previsiting a list. See {@link SchemaVisitor}
     * @param <R> Return type after processing all children values. See {@link SchemaVisitor}
     * @return the result of visiting the schema
     */
    public static <T, P, R> R visitSchema(Schema schema, SchemaVisitor<T, P, R> visitor) {
        Objects.requireNonNull(schema, "schema cannot be null");
        Objects.requireNonNull(visitor, "visitor cannot be null");

        List<T> childrenResult = IntStream.range(0, schema.getNumChildren())
                .mapToObj(i -> visitSchemaInner(schema.getChild(i), visitor))
                .collect(Collectors.toList());

        return visitor.visitTopSchema(schema, childrenResult);
    }

    private static <T, P, R> T visitSchemaInner(Schema schema, SchemaVisitor<T, P, R> visitor) {
        switch (schema.getType().getTypeId()) {
            case STRUCT:
                List<T> children = IntStream.range(0, schema.getNumChildren())
                        .mapToObj(childIdx -> visitSchemaInner(schema.getChild(childIdx), visitor))
                        .collect(Collectors.toList());
                return visitor.visitStruct(schema, children);
            case LIST:
                P preVisitResult = visitor.preVisitList(schema);
                T childResult = visitSchemaInner(schema.getChild(0), visitor);
                return visitor.visitList(schema, preVisitResult, childResult);
            default:
                return visitor.visit(schema);
        }
    }


    /**
     * Visiting a list of host columns in post order. For more details, see {@link HostColumnsVisitor}.
     *
     * @param cols the list of host columns to visit
     * @param visitor the visitor to use
     * @param <T> Return type when visiting intermediate nodes. See {@link HostColumnsVisitor}
     */
    public static <T> void visitColumns(HostColumnVector[] cols,
                                        HostColumnsVisitor<T> visitor) {
        Objects.requireNonNull(cols, "cols cannot be null");
        Objects.requireNonNull(visitor, "visitor cannot be null");

        for (HostColumnVector col : cols) {
            visitColumn(col, visitor);
        }

    }

    private static <T> T visitColumn(HostColumnVectorCore col, HostColumnsVisitor<T> visitor) {
        switch (col.getType().getTypeId()) {
            case STRUCT:
                List<T> children = IntStream.range(0, col.getNumChildren())
                        .mapToObj(childIdx -> visitColumn(col.getChildColumnView(childIdx), visitor))
                        .collect(Collectors.toList());
                return visitor.visitStruct(col, children);
            case LIST:
                T preVisitResult = visitor.preVisitList(col);
                T childResult = visitColumn(col.getChildColumnView(0), visitor);
                return visitor.visitList(col, preVisitResult, childResult);
            default:
                return visitor.visit(col);
        }
    }
}
