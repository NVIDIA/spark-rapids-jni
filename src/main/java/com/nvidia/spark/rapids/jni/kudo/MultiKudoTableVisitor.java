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

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import com.nvidia.spark.rapids.jni.schema.SchemaVisitor;

import java.io.OutputStream;
import java.util.*;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static java.lang.Math.toIntExact;

/**
 * This class provides a base class for visiting multiple kudo tables, e.g. it helps to maintain internal states during
 * visiting multi kudo tables, which makes it easier to do some calculations based on them.
 * <br/>
 * The schema used when visiting these kudo tables must be same as the schema used when creating these kudo tables.
 */
abstract class MultiKudoTableVisitor<T, R> implements SchemaVisitor<T, R> {
    private final List<KudoTable> tables;
    private final long[] currentValidityOffsets;
    private final long[] currentOffsetOffsets;
    private final long[] currentDataOffset;
    private final Deque<SliceInfo>[] sliceInfoStack;
    private final Deque<Long> totalRowCountStack;
    // A temporary variable to keep if current column has null
    private boolean hasNull;
    private int currentIdx;
    // Temporary buffer to store data length of string column to avoid repeated allocation
    private final int[] strDataLen;
    // Temporary variable to calcluate total data length of string column
    private long totalStrDataLen;

    protected MultiKudoTableVisitor(List<KudoTable> inputTables) {
        Objects.requireNonNull(inputTables, "tables cannot be null");
        ensure(!inputTables.isEmpty(), "tables cannot be empty");
        this.tables = inputTables instanceof ArrayList ? inputTables : new ArrayList<>(inputTables);
        this.currentValidityOffsets = new long[tables.size()];
        this.currentOffsetOffsets = new long[tables.size()];
        this.currentDataOffset = new long[tables.size()];
        this.sliceInfoStack = new Deque[tables.size()];
        for (int i = 0; i < tables.size(); i++) {
            this.currentValidityOffsets[i] = 0;
            KudoTableHeader header = tables.get(i).getHeader();
            this.currentOffsetOffsets[i] = header.getValidityBufferLen();
            this.currentDataOffset[i] = header.getValidityBufferLen() + header.getOffsetBufferLen();
            this.sliceInfoStack[i] = new ArrayDeque<>(16);
            this.sliceInfoStack[i].add(new SliceInfo(header.getOffset(), header.getNumRows()));
        }
        long totalRowCount = tables.stream().mapToLong(t -> t.getHeader().getNumRows()).sum();
        this.totalRowCountStack = new ArrayDeque<>(16);
        totalRowCountStack.addLast(totalRowCount);
        this.hasNull = true;
        this.currentIdx = 0;
        this.strDataLen = new int[tables.size()];
        this.totalStrDataLen = 0;
    }

    List<KudoTable> getTables() {
        return tables;
    }

    @Override
    public R visitTopSchema(Schema schema, List<T> children) {
        return doVisitTopSchema(schema, children);
    }

    protected abstract R doVisitTopSchema(Schema schema, List<T> children);

    @Override
    public T visitStruct(Schema structType, List<T> children) {
        updateHasNull();
        T t = doVisitStruct(structType, children);
        updateOffsets(false, false, false, -1);
        currentIdx += 1;
        return t;
    }

    protected abstract T doVisitStruct(Schema structType, List<T> children);

    @Override
    public T preVisitList(Schema listType) {
        updateHasNull();
        T t = doPreVisitList(listType);
        updateOffsets(true, false, true, Integer.BYTES);
        currentIdx += 1;
        return t;
    }

    protected abstract T doPreVisitList(Schema listType);

    @Override
    public T visitList(Schema listType, T preVisitResult, T childResult) {
        T t = doVisitList(listType, preVisitResult, childResult);
        for (int tableIdx = 0; tableIdx < tables.size(); tableIdx++) {
            sliceInfoStack[tableIdx].removeLast();
        }
        totalRowCountStack.removeLast();
        return t;
    }

    protected abstract T doVisitList(Schema listType, T preVisitResult, T childResult);

    @Override
    public T visit(Schema primitiveType) {
        updateHasNull();
        if (primitiveType.getType().hasOffsets()) {
            // string type
            updateDataLen();
        }

        T t = doVisit(primitiveType);
        if (primitiveType.getType().hasOffsets()) {
            updateOffsets(true, true, false, -1);
        } else {
            updateOffsets(false, true, false, primitiveType.getType().getSizeInBytes());
        }
        currentIdx += 1;
        return t;
    }

    protected abstract T doVisit(Schema primitiveType);

    private void updateHasNull() {
        hasNull = false;
        for (KudoTable table : tables) {
            if (table.getHeader().hasValidityBuffer(currentIdx)) {
                hasNull = true;
                return;
            }
        }
    }

    // For string column only
    private void updateDataLen() {
        totalStrDataLen = 0;
        // String's data len needs to be calculated from offset buffer
        for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
            SliceInfo sliceInfo = sliceInfoOf(tableIdx);
            if (sliceInfo.getRowCount() > 0) {
                int offset = offsetOf(tableIdx, 0);
                int endOffset = offsetOf(tableIdx, toIntExact(sliceInfo.getRowCount()));

                strDataLen[tableIdx] = endOffset - offset;
                totalStrDataLen += strDataLen[tableIdx];
            } else {
                strDataLen[tableIdx] = 0;
            }
        }
    }

    private void updateOffsets(boolean updateOffset, boolean updateData, boolean updateSliceInfo, int sizeInBytes) {
        long totalRowCount = 0;
        for (int tableIdx = 0; tableIdx < tables.size(); tableIdx++) {
            SliceInfo sliceInfo = sliceInfoOf(tableIdx);
            if (sliceInfo.getRowCount() > 0) {
                if (updateSliceInfo) {
                    int startOffset = offsetOf(tableIdx, 0);
                    int endOffset = offsetOf(tableIdx, toIntExact(sliceInfo.getRowCount()));
                    int rowCount = endOffset - startOffset;
                    totalRowCount += rowCount;

                    sliceInfoStack[tableIdx].addLast(new SliceInfo(startOffset, rowCount));
                }

                if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
                    currentValidityOffsets[tableIdx] += padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
                }

                if (updateOffset) {
                    currentOffsetOffsets[tableIdx] += padForHostAlignment((sliceInfo.getRowCount() + 1) * Integer.BYTES);
                    if (updateData) {
                        // string type
                        currentDataOffset[tableIdx] += padForHostAlignment(strDataLen[tableIdx]);
                    }
                    // otherwise list type
                } else {
                    if (updateData) {
                        // primitive type
                        currentDataOffset[tableIdx] += padForHostAlignment(sliceInfo.getRowCount() * sizeInBytes);
                    }
                }

            } else {
                if (updateSliceInfo) {
                    sliceInfoStack[tableIdx].addLast(new SliceInfo(0, 0));
                }
            }
        }

        if (updateSliceInfo) {
            totalRowCountStack.addLast(totalRowCount);
        }
    }

    // Below parts are information about current column

    protected long getTotalRowCount() {
        return totalRowCountStack.getLast();
    }


    protected boolean hasNull() {
        return hasNull;
    }

    protected SliceInfo sliceInfoOf(int tableIdx) {
        return sliceInfoStack[tableIdx].getLast();
    }

    protected HostMemoryBuffer memoryBufferOf(int tableIdx) {
        return tables.get(tableIdx).getBuffer();
    }

    protected int offsetOf(int tableIdx, long rowIdx) {
        long startOffset = currentOffsetOffsets[tableIdx];
        return tables.get(tableIdx).getBuffer().getInt(startOffset + rowIdx * Integer.BYTES);
    }

    protected long validifyBufferOffset(int tableIdx) {
        if (tables.get(tableIdx).getHeader().hasValidityBuffer(currentIdx)) {
            return currentValidityOffsets[tableIdx];
        } else {
            return -1;
        }
    }

    protected void copyDataBuffer(HostMemoryBuffer dst, long dstOffset, int tableIdx, int dataLen) {
        long startOffset = currentDataOffset[tableIdx];
        dst.copyFromHostBuffer(dstOffset, tables.get(tableIdx).getBuffer(), startOffset, dataLen);
    }

    protected long getTotalStrDataLen() {
        return totalStrDataLen;
    }

    protected int getStrDataLenOf(int tableIdx) {
        return strDataLen[tableIdx];
    }

    protected int getCurrentIdx() {
        return currentIdx;
    }

    public int getTableSize() {
        return this.tables.size();
    }
}
