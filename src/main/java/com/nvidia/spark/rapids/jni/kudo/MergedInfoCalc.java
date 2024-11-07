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

import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.getValidityLengthInBytes;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padFor64byteAlignment;


/**
 * This class is used to calculate column offsets of merged buffer.
 */
class MergedInfoCalc extends MultiKudoTableVisitor<Void, Void> {
    // Total data len in gpu, which accounts for 64 byte alignment
    private long totalDataLen;
    // Column offset in gpu device buffer, it has one field for each flattened column
    private final List<ColumnOffsetInfo> columnOffsets;

    public MergedInfoCalc(List<KudoTable> tables) {
        super(tables);
        this.totalDataLen = 0;
        this.columnOffsets = new ArrayList<>(tables.get(0).getHeader().getNumColumns()) ;
    }

    @Override
    protected Void doVisitTopSchema(Schema schema, List<Void> children) {
        return null;
    }

    @Override
    protected Void doVisitStruct(Schema structType, List<Void> children) {
        long validityBufferLen = 0;
        long validityOffset = -1;
        if (hasNull()) {
            validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(getTotalRowCount()));
            validityOffset = totalDataLen;
            totalDataLen += validityBufferLen;
        }

        columnOffsets.add(new ColumnOffsetInfo(validityOffset, -1, -1, 0));
        return null;
    }

    @Override
    protected Void doPreVisitList(Schema listType) {
        long validityBufferLen = 0;
        long validityOffset = -1;
        if (hasNull()) {
            validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(getTotalRowCount()));
            validityOffset = totalDataLen;
            totalDataLen += validityBufferLen;
        }

        long offsetBufferLen = 0;
        long offsetBufferOffset = -1;
        if (getTotalRowCount() > 0) {
            offsetBufferLen = padFor64byteAlignment((getTotalRowCount() + 1) * Integer.BYTES);
            offsetBufferOffset = totalDataLen;
            totalDataLen += offsetBufferLen;
        }


        columnOffsets.add(new ColumnOffsetInfo(validityOffset, offsetBufferOffset, -1, 0));
        return null;
    }

    @Override
    protected Void doVisitList(Schema listType, Void preVisitResult, Void childResult) {
        return null;
    }

    @Override
    protected Void doVisit(Schema primitiveType) {
        // String type
        if (primitiveType.getType().hasOffsets()) {
            long validityBufferLen = 0;
            long validityOffset = -1;
            if (hasNull()) {
                validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(getTotalRowCount()));
                validityOffset = totalDataLen;
                totalDataLen += validityBufferLen;
            }

            long offsetBufferLen = 0;
            long offsetBufferOffset = -1;
            if (getTotalRowCount() > 0) {
                offsetBufferLen = padFor64byteAlignment((getTotalRowCount() + 1) * Integer.BYTES);
                offsetBufferOffset = totalDataLen;
                totalDataLen += offsetBufferLen;
            }

            long dataBufferLen = 0;
            long dataBufferOffset = -1;
            if (getTotalStrDataLen() > 0) {
                dataBufferLen = padFor64byteAlignment(getTotalStrDataLen());
                dataBufferOffset = totalDataLen;
                totalDataLen += dataBufferLen;
            }

            columnOffsets.add(new ColumnOffsetInfo(validityOffset, offsetBufferOffset, dataBufferOffset, dataBufferLen));
        } else {
            long totalRowCount = getTotalRowCount();
            long validityBufferLen = 0;
            long validityOffset = -1;
            if (hasNull()) {
                validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(totalRowCount));
                validityOffset = totalDataLen;
                totalDataLen += validityBufferLen;
            }

            long offsetBufferOffset = -1;

            long dataBufferLen = 0;
            long dataBufferOffset = -1;
            if (totalRowCount > 0) {
                dataBufferLen = padFor64byteAlignment(totalRowCount * primitiveType.getType().getSizeInBytes());
                dataBufferOffset = totalDataLen;
                totalDataLen += dataBufferLen;
            }

            columnOffsets.add(new ColumnOffsetInfo(validityOffset, offsetBufferOffset, dataBufferOffset, dataBufferLen));
        }

        return null;
    }


    public long getTotalDataLen() {
        return totalDataLen;
    }

    List<ColumnOffsetInfo> getColumnOffsets() {
        return Collections.unmodifiableList(columnOffsets);
    }

    @Override
    public String toString() {
        return "MergedInfoCalc{" +
                "totalDataLen=" + totalDataLen +
                ", columnOffsets=" + columnOffsets +
                '}';
    }

    static MergedInfoCalc calc(Schema schema, List<KudoTable> table) {
        MergedInfoCalc calc = new MergedInfoCalc(table);
        Visitors.visitSchema(schema, calc);
        return calc;
    }
}
