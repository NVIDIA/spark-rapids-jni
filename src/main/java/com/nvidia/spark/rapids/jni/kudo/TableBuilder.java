package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;

import java.util.List;

import static com.nvidia.spark.rapids.jni.TableUtils.ensure;
import static java.util.Objects.requireNonNull;

public class TableBuilder implements SchemaVisitor<Object, Table> {
    // Current column index
    private int curIdx;
    private final DeviceMemoryBuffer buffer;
    private final List<ColumnViewInfo> colViewInfoList;

    public TableBuilder(List<ColumnViewInfo> colViewInfoList, DeviceMemoryBuffer buffer) {
        requireNonNull(colViewInfoList, "colViewInfoList cannot be null");
        ensure(!colViewInfoList.isEmpty(), "colViewInfoList cannot be empty");
        requireNonNull(buffer, "Device buffer can't be null!");

        this.curIdx = 0;
        this.buffer = buffer;
        this.colViewInfoList = colViewInfoList;
    }

    @Override
    public Table visitTopSchema(Schema schema, List<Object> children) {
        try (CloseableArray<ColumnVector> arr = CloseableArray.wrap(new ColumnVector[children.size()])) {
            for (int i = 0; i < children.size(); i++) {
                long colView = (long) children.get(i);
                arr.set(i, RefUtils.fromViewWithContiguousAllocation(colView, buffer));
            }

            return new Table(arr.getArray());
        }
    }

    @Override
    public Long visitStruct(Schema structType, List<Object> children) {
        ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

        long[] childrenView = children.stream().mapToLong(o -> (long) o).toArray();
        long columnView = colViewInfo.buildColumnView(buffer, childrenView);
        curIdx += 1;
        return columnView;
    }

    @Override
    public ColumnViewInfo preVisitList(Schema listType) {
        ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

        curIdx += 1;
        return colViewInfo;
    }

    @Override
    public Long visitList(Schema listType, Object preVisitResult, Object childResult) {
        ColumnViewInfo colViewInfo = (ColumnViewInfo) preVisitResult;

        long[] children = new long[] { (long) childResult };

        return colViewInfo.buildColumnView(buffer, children);
    }

    @Override
    public Long visit(Schema primitiveType) {
        ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

        long columnView = colViewInfo.buildColumnView(buffer, null);
        curIdx += 1;
        return columnView;
    }

    private ColumnViewInfo getCurrentColumnViewInfo() {
        return colViewInfoList.get(curIdx);
    }
}
