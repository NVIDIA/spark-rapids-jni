package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;


class SlicedBufferSerializer implements HostColumnsVisitor<Long, Long> {
    private final SliceInfo root;
    private final BufferType bufferType;
    private final DataWriter writer;

    private final Deque<SliceInfo> sliceInfos = new ArrayDeque<>();

    SlicedBufferSerializer(long rowOffset, long numRows, BufferType bufferType, DataWriter writer) {
        this.root = new SliceInfo(rowOffset, numRows);
        this.bufferType = bufferType;
        this.writer = writer;
        this.sliceInfos.addLast(root);
    }

    @Override
    public Long visitTopSchema(List<Long> children) {
        return children.stream().mapToLong(Long::longValue).sum();
    }

    @Override
    public Long visitStruct(HostColumnVectorCore col, List<Long> children) {
        SliceInfo parent = sliceInfos.peekLast();

        long bytesCopied = children.stream().mapToLong(Long::longValue).sum();
        try {
            switch (bufferType) {
                case VALIDITY:
                    bytesCopied += this.copySlicedValidity(col, parent);
                    return bytesCopied;
                case OFFSET:
                case DATA:
                    return bytesCopied;
                default:
                    throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Long preVisitList(HostColumnVectorCore col) {
        SliceInfo parent = sliceInfos.getLast();


        long bytesCopied = 0;
        try {
            switch (bufferType) {
                case VALIDITY:
                    bytesCopied = this.copySlicedValidity(col, parent);
                    break;
                case OFFSET:
                    bytesCopied = this.copySlicedOffset(col, parent);
                    break;
                case DATA:
                    break;
                default:
                    throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        SliceInfo current;
        if (col.getOffsets() != null) {
            long start = col.getOffsets()
                    .getInt(parent.offset * Integer.BYTES);
            long end = col.getOffsets().getInt((parent.offset + parent.rowCount) * Integer.BYTES);
            long rowCount = end - start;

            current = new SliceInfo(start, rowCount);
        } else {
            current = new SliceInfo(0, 0);
        }

        sliceInfos.addLast(current);
        return bytesCopied;
    }

    @Override
    public Long visitList(HostColumnVectorCore col, Long preVisitResult, Long childResult) {
        sliceInfos.removeLast();
        return preVisitResult + childResult;
    }

    @Override
    public Long visit(HostColumnVectorCore col) {
        SliceInfo parent = sliceInfos.getLast();
        try {
            switch (bufferType) {
                case VALIDITY:
                    return this.copySlicedValidity(col, parent);
                case OFFSET:
                    return this.copySlicedOffset(col, parent);
                case DATA:
                    return this.copySlicedData(col, parent);
                default:
                    throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private long copySlicedValidity(HostColumnVectorCore column, SliceInfo sliceInfo) throws IOException {
        if (column.getValidity() != null && sliceInfo.getRowCount() > 0) {
            HostMemoryBuffer buff = column.getValidity();
            long len = sliceInfo.getValidityBufferInfo().getBufferLength();
            writer.copyDataFrom(buff, sliceInfo.getValidityBufferInfo().getBufferOffset(),
                    len);
            return padForHostAlignment(writer, len);
        } else {
            return 0;
        }
    }

    private long copySlicedOffset(HostColumnVectorCore column, SliceInfo sliceInfo) throws IOException {
        if (sliceInfo.rowCount <= 0 || column.getOffsets() == null) {
            // Don't copy anything, there are no rows
            return 0;
        }
        long bytesToCopy = (sliceInfo.rowCount + 1) * Integer.BYTES;
        long srcOffset = sliceInfo.offset * Integer.BYTES;
        HostMemoryBuffer buff = column.getOffsets();
        writer.copyDataFrom(buff, srcOffset, bytesToCopy);
        return padForHostAlignment(writer, bytesToCopy);
    }

    private long copySlicedData(HostColumnVectorCore column, SliceInfo sliceInfo) throws IOException {
        if (sliceInfo.rowCount > 0) {
            DType type = column.getType();
            if (type.equals(DType.STRING)) {
                long startByteOffset = column.getOffsets().getInt(sliceInfo.offset * Integer.BYTES);
                long endByteOffset = column.getOffsets().getInt((sliceInfo.offset + sliceInfo.rowCount) * Integer.BYTES);
                long bytesToCopy = endByteOffset - startByteOffset;
                if (column.getData() == null) {
                    if (bytesToCopy != 0) {
                        throw new IllegalStateException("String column has no data buffer, " +
                                "but bytes to copy is not zero: " + bytesToCopy);
                    }

                    return 0;
                } else {
                    writer.copyDataFrom(column.getData(), startByteOffset, bytesToCopy);
                    return padForHostAlignment(writer, bytesToCopy);
                }
            } else if (type.getSizeInBytes() > 0) {
                long bytesToCopy = sliceInfo.rowCount * type.getSizeInBytes();
                long srcOffset = sliceInfo.offset * type.getSizeInBytes();
                writer.copyDataFrom(column.getData(), srcOffset, bytesToCopy);
                return padForHostAlignment(writer, bytesToCopy);
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }
}
