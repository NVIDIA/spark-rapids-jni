package com.nvidia.spark.rapids.jni.schema;

import ai.rapids.cudf.HostColumnVectorCore;

import java.util.List;

public interface HostColumnsVisitor<T, R> {
    R visitTopSchema(List<T> children);

    T visitStruct(HostColumnVectorCore col, List<T> children);

    T preVisitList(HostColumnVectorCore col);
    T visitList(HostColumnVectorCore col, T preVisitResult, T childResult);

    T visit(HostColumnVectorCore col);
}
