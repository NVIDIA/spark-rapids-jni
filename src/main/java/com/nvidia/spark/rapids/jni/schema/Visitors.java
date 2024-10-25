package com.nvidia.spark.rapids.jni.schema;

import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.Schema;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Visitors {
    public static <T, R> R visitSchema(Schema schema, SchemaVisitor<T, R> visitor) {
        Objects.requireNonNull(schema, "schema cannot be null");
        Objects.requireNonNull(visitor, "visitor cannot be null");

        List<T> childrenResult = IntStream.range(0, schema.getNumChildren())
                .mapToObj(i -> visitSchemaInner(schema.getChild(i), visitor))
                .collect(Collectors.toList());

        return visitor.visitTopSchema(schema, childrenResult);
    }

    private static <T, R> T visitSchemaInner(Schema schema, SchemaVisitor<T, R> visitor) {
        switch (schema.getType().getTypeId()) {
            case STRUCT:
                List<T> children = IntStream.range(0, schema.getNumChildren())
                        .mapToObj(childIdx -> visitSchemaInner(schema.getChild(childIdx), visitor))
                        .collect(Collectors.toList());
                return visitor.visitStruct(schema, children);
            case LIST:
                T preVisitResult = visitor.preVisitList(schema);
                T childResult = visitSchemaInner(schema.getChild(0), visitor);
                return visitor.visitList(schema, preVisitResult, childResult);
            default:
                return visitor.visit(schema);
        }
    }


    /**
     * Entry point for visiting a schema with columns.
     */
    public static <T, R> R visitColumns(List<HostColumnVector> cols,
                                                  HostColumnsVisitor<T, R> visitor) {
        Objects.requireNonNull(cols, "cols cannot be null");
        Objects.requireNonNull(visitor, "visitor cannot be null");

        List<T> childrenResult = new ArrayList<>(cols.size());

        for (HostColumnVector col : cols) {
            childrenResult.add(visitSchema(col, visitor));
        }

        return visitor.visitTopSchema(childrenResult);
    }

    private static <T, R> T visitSchema(HostColumnVectorCore col, HostColumnsVisitor<T, R> visitor) {
        switch (col.getType().getTypeId()) {
            case STRUCT:
                List<T> children = IntStream.range(0, col.getNumChildren())
                        .mapToObj(childIdx -> visitSchema(col.getChildColumnView(childIdx), visitor))
                        .collect(Collectors.toList());
                return visitor.visitStruct(col, children);
            case LIST:
                T preVisitResult = visitor.preVisitList(col);
                T childResult = visitSchema(col.getChildColumnView(0), visitor);
                return visitor.visitList(col, preVisitResult, childResult);
            default:
                return visitor.visit(col);
        }
    }
}
