package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class HostMergeResultCalc implements SchemaVisitor2 {
    private final List<ColumnOffsetInfo> columnOffsetInfos;
    private final int[] nullCounts;
    private final int[] rowCounts;
    private final HostMemoryBuffer hostMemoryBuffer;

    private final ColumnViewInfo[] columnViewInfos;
    private int curColIdx = 0;
    private KudoHostMergeResult result;

    HostMergeResultCalc(List<ColumnOffsetInfo> columnOffsetInfos, int[] nullCounts, int[] rowCounts,
                        HostMemoryBuffer hostMemoryBuffer) {
        this.columnOffsetInfos = columnOffsetInfos;
        this.nullCounts = nullCounts;
        this.rowCounts = rowCounts;
        this.columnViewInfos = new ColumnViewInfo[columnOffsetInfos.size()];
        this.hostMemoryBuffer = hostMemoryBuffer;
        this.result = null;
    }

    public KudoHostMergeResult getResult() {
        return result;
    }

    @Override
    public void visitTopSchema(Schema schema) {
        result = new KudoHostMergeResult(schema, hostMemoryBuffer, columnViewInfos);
    }

    @Override
    public void visitStruct(Schema structType) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(structType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
    }

    @Override
    public void preVisitList(Schema listType) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(listType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
    }

    @Override
    public void visitList(Schema listType) {
    }

    @Override
    public void visit(Schema primitiveType) {
        columnViewInfos[curColIdx] = new ColumnViewInfo(primitiveType.getType(),
                columnOffsetInfos.get(curColIdx), nullCounts[curColIdx], rowCounts[curColIdx]);
        curColIdx++;
    }
}
