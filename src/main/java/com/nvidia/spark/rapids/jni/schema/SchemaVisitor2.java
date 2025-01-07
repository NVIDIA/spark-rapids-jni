package com.nvidia.spark.rapids.jni.schema;

import ai.rapids.cudf.Schema;

import java.util.List;

public interface SchemaVisitor2 {
    /**
     * Visit the top level schema.
     * @param schema the top level schema to visit
     * @return the result of visiting the top level schema
     */
    void visitTopSchema(Schema schema);

    /**
     * Visit a struct schema.
     * @param structType the struct schema to visit
     * @return the result of visiting the struct schema
     */
    void visitStruct(Schema structType);

    /**
     * Visit a list schema before actually visiting its child.
     * @param listType the list schema to visit
     * @return the result of visiting the list schema
     */
    void preVisitList(Schema listType);

    /**
     * Visit a list schema after visiting its child.
     * @param listType the list schema to visit
     * @return the result of visiting the list schema
     */
    void visitList(Schema listType);

    /**
     * Visit a primitive type.
     * @param primitiveType the primitive type to visit
     * @return the result of visiting the primitive type
     */
    void visit(Schema primitiveType);
}
