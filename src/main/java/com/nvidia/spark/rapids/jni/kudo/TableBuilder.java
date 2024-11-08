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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.Arms;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;

import java.util.ArrayList;
import java.util.List;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static java.util.Objects.requireNonNull;

/**
 * This class is used to build a cudf table from a list of column view info, and a device buffer.
 */
class TableBuilder implements SchemaVisitor<Object, Table>, AutoCloseable {
  // Current column index
  private int curColumnIdx;
  private final DeviceMemoryBuffer buffer;
  private final List<ColumnViewInfo> colViewInfoList;
  private final List<ColumnView> columnViewList;

  public TableBuilder(List<ColumnViewInfo> colViewInfoList, DeviceMemoryBuffer buffer) {
    requireNonNull(colViewInfoList, "colViewInfoList cannot be null");
    ensure(!colViewInfoList.isEmpty(), "colViewInfoList cannot be empty");
    requireNonNull(buffer, "Device buffer can't be null!");

    this.curColumnIdx = 0;
    this.buffer = buffer;
    this.colViewInfoList = colViewInfoList;
    this.columnViewList = new ArrayList<>(colViewInfoList.size());
  }

  @Override
  public Table visitTopSchema(Schema schema, List<Object> children) {
    try (CloseableArray<ColumnVector> arr = CloseableArray.wrap(new ColumnVector[children.size()])) {
      for (int i = 0; i < children.size(); i++) {
        ColumnView colView = (ColumnView) children.get(i);
        arr.set(i, ColumnVector.fromViewWithContiguousAllocation(colView.getNativeView(), buffer));
      }

      return new Table(arr.getArray());
    }
  }

  @Override
  public ColumnView visitStruct(Schema structType, List<Object> children) {
    ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

    ColumnView[] childrenView = children.stream().map(o -> (ColumnView) o).toArray(ColumnView[]::new);
    ColumnView columnView = colViewInfo.buildColumnView(buffer, childrenView);
    curColumnIdx += 1;
    columnViewList.add(columnView);
    return columnView;
  }

  @Override
  public ColumnViewInfo preVisitList(Schema listType) {
    ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

    curColumnIdx += 1;
    return colViewInfo;
  }

  @Override
  public ColumnView visitList(Schema listType, Object preVisitResult, Object childResult) {
    ColumnViewInfo colViewInfo = (ColumnViewInfo) preVisitResult;

    ColumnView[] children = new ColumnView[]{(ColumnView) childResult};

    ColumnView view = colViewInfo.buildColumnView(buffer, children);
    columnViewList.add(view);
    return view;
  }

  @Override
  public ColumnView visit(Schema primitiveType) {
    ColumnViewInfo colViewInfo = getCurrentColumnViewInfo();

    ColumnView columnView = colViewInfo.buildColumnView(buffer, null);
    curColumnIdx += 1;
    columnViewList.add(columnView);
    return columnView;
  }

  private ColumnViewInfo getCurrentColumnViewInfo() {
    return colViewInfoList.get(curColumnIdx);
  }

  @Override
  public void close() throws Exception {
    Arms.closeAll(columnViewList);
  }
}
