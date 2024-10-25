package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;


class SerializedTableHeaderCalc implements HostColumnsVisitor<Void, SerializedTableHeader> {
    private final SliceInfo root;
    private final List<Boolean> hasValidityBuffer = new ArrayList<>(1024);
    private long validityBufferLen;
    private long offsetBufferLen;
    private long totalDataLen;

    private Deque<SliceInfo> sliceInfos = new ArrayDeque<>();

    SerializedTableHeaderCalc(long rowOffset, long numRows) {
        this.root = new SliceInfo(rowOffset, numRows);
        this.totalDataLen = 0;
        sliceInfos.addLast(this.root);
    }

    @Override
    public SerializedTableHeader visitTopSchema(List<Void> children) {
        byte[] hasValidityBuffer = new byte[this.hasValidityBuffer.size()];
        for (int i = 0; i < this.hasValidityBuffer.size(); i++) {
            hasValidityBuffer[i] = (byte) (this.hasValidityBuffer.get(i) ? 1 : 0);
        }
        return new SerializedTableHeader(root.offset, root.rowCount,
                validityBufferLen, offsetBufferLen,
                totalDataLen, hasValidityBuffer);
    }

    @Override
    public Void visitStruct(HostColumnVectorCore col, List<Void> children) {
        SliceInfo parent = sliceInfos.getLast();

        long validityBufferLength = 0;
        if (col.hasValidityVector()) {
            validityBufferLength = padForHostAlignment(parent.getValidityBufferInfo().getBufferLength());
        }

        this.validityBufferLen += validityBufferLength;

        totalDataLen += validityBufferLength;
        hasValidityBuffer.add(col.getValidity() != null);
        return null;
    }

    @Override
    public Void preVisitList(HostColumnVectorCore col) {
        SliceInfo parent = sliceInfos.getLast();


        long validityBufferLength = 0;
        if (col.hasValidityVector() && parent.rowCount > 0) {
            validityBufferLength = padForHostAlignment(parent.getValidityBufferInfo().getBufferLength());
        }

        long offsetBufferLength = 0;
        if (col.getOffsets() != null && parent.rowCount > 0) {
            offsetBufferLength = padForHostAlignment((parent.rowCount + 1) * Integer.BYTES);
        }

        this.validityBufferLen += validityBufferLength;
        this.offsetBufferLen += offsetBufferLength;
        this.totalDataLen += validityBufferLength + offsetBufferLength;

        hasValidityBuffer.add(col.getValidity() != null);

        SliceInfo current;

        if (col.getOffsets() != null) {
            long start = col.getOffsets().getInt(parent.offset * Integer.BYTES);
            long end = col.getOffsets().getInt((parent.offset + parent.rowCount) * Integer.BYTES);
            long rowCount = end - start;
            current = new SliceInfo(start, rowCount);
        } else {
            current = new SliceInfo(0, 0);
        }

        sliceInfos.addLast(current);
        return null;
    }

    @Override
    public Void visitList(HostColumnVectorCore col, Void preVisitResult, Void childResult) {
        sliceInfos.removeLast();

        return null;
    }


    @Override
    public Void visit(HostColumnVectorCore col) {
        SliceInfo parent = sliceInfos.peekLast();
        long validityBufferLen = calcPrimitiveDataLen(col, BufferType.VALIDITY, parent);
        long offsetBufferLen = calcPrimitiveDataLen(col, BufferType.OFFSET, parent);
        long dataBufferLen = calcPrimitiveDataLen(col, BufferType.DATA, parent);

        this.validityBufferLen += validityBufferLen;
        this.offsetBufferLen += offsetBufferLen;
        this.totalDataLen += validityBufferLen + offsetBufferLen + dataBufferLen;

        hasValidityBuffer.add(col.getValidity() != null);

        return null;
    }

    private long calcPrimitiveDataLen(HostColumnVectorCore col,
                                      BufferType bufferType,
                                      SliceInfo info) {
        switch (bufferType) {
            case VALIDITY:
                if (col.hasValidityVector() && info.getRowCount() > 0) {
                    return  padForHostAlignment(info.getValidityBufferInfo().getBufferLength());
                } else {
                    return 0;
                }
            case OFFSET:
                if (DType.STRING.equals(col.getType()) && info.getRowCount() > 0) {
                    return padForHostAlignment((info.rowCount + 1) * Integer.BYTES);
                } else {
                    return 0;
                }
            case DATA:
                if (DType.STRING.equals(col.getType())) {
                    if (col.getOffsets() != null) {
                        long startByteOffset = col.getOffsets().getInt(info.offset * Integer.BYTES);
                        long endByteOffset = col.getOffsets().getInt((info.offset + info.rowCount) * Integer.BYTES);
                        return padForHostAlignment(endByteOffset - startByteOffset);
                    } else {
                        return 0;
                    }
                } else {
                    if (col.getType().getSizeInBytes() > 0) {
                        return padForHostAlignment(col.getType().getSizeInBytes() * info.rowCount);
                    } else {
                        return 0;
                    }
                }
            default:
                throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);

        }
    }
}
