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

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.ColumnOffsetInfo.INVALID_OFFSET;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.getValidityLengthInBytes;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padFor64byteAlignment;
import static java.lang.Math.min;
import static java.lang.Math.toIntExact;
import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.Arms;
import com.nvidia.spark.rapids.jni.schema.Visitors;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * This class is used to merge multiple KudoTables into a single contiguous buffer, e.g. {@link KudoHostMergeResult},
 * which could be easily converted to a {@link ai.rapids.cudf.ContiguousTable}.
 */
class KudoTableMerger extends MultiKudoTableVisitor<Void, Void, KudoHostMergeResult> {
  private final List<ColumnOffsetInfo> columnOffsets;
  private final HostMemoryBuffer buffer;
  private final List<ColumnViewInfo> colViewInfoList;

  public KudoTableMerger(List<KudoTable> tables, HostMemoryBuffer buffer,
                         List<ColumnOffsetInfo> columnOffsets) {
    super(tables);
    requireNonNull(buffer, "buffer can't be null!");
    ensure(columnOffsets != null, "column offsets cannot be null");
    ensure(!columnOffsets.isEmpty(), "column offsets cannot be empty");
    this.columnOffsets = columnOffsets;
    this.buffer = buffer;
    this.colViewInfoList = new ArrayList<>(columnOffsets.size());
  }

  @Override
  protected KudoHostMergeResult doVisitTopSchema(Schema schema, List<Void> children) {
    return new KudoHostMergeResult(schema, buffer, colViewInfoList);
  }

  @Override
  protected Void doVisitStruct(Schema structType, List<Void> children) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = getTotalRowCount();
    colViewInfoList.add(new ColumnViewInfo(structType.getType(),
        offsetInfo, nullCount, totalRowCount));
    return null;
  }

  @Override
  protected Void doPreVisitList(Schema listType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = getTotalRowCount();
    deserializeOffsetBuffer(offsetInfo);

    colViewInfoList.add(new ColumnViewInfo(listType.getType(),
        offsetInfo, nullCount, totalRowCount));
    return null;
  }

  @Override
  protected Void doVisitList(Schema listType, Void preVisitResult, Void childResult) {
    return null;
  }

  @Override
  protected Void doVisit(Schema primitiveType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    int nullCount = deserializeValidityBuffer(offsetInfo);
    int totalRowCount = getTotalRowCount();
    if (primitiveType.getType().hasOffsets()) {
      deserializeOffsetBuffer(offsetInfo);
      deserializeDataBuffer(offsetInfo, OptionalInt.empty());
    } else {
      deserializeDataBuffer(offsetInfo, OptionalInt.of(primitiveType.getType().getSizeInBytes()));
    }

    colViewInfoList.add(new ColumnViewInfo(primitiveType.getType(),
        offsetInfo, nullCount, totalRowCount));

    return null;
  }

  private int deserializeValidityBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getValidity() != INVALID_OFFSET) {
      long offset = curColOffset.getValidity();
      long validityBufferSize = padFor64byteAlignment(getValidityLengthInBytes(getTotalRowCount()));
      try (HostMemoryBuffer validityBuffer = buffer.slice(offset, validityBufferSize)) {
        int nullCountTotal = 0;
        int startRow = 0;
        for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
          SliceInfo sliceInfo = sliceInfoOf(tableIdx);
          long validityOffset = validifyBufferOffset(tableIdx);
          if (validityOffset != INVALID_OFFSET) {
            nullCountTotal += copyValidityBuffer(validityBuffer, startRow,
                memoryBufferOf(tableIdx), toIntExact(validityOffset),
                sliceInfo);
          } else {
            appendAllValid(validityBuffer, startRow, sliceInfo.getRowCount());
          }

          startRow += sliceInfo.getRowCount();
        }
        return nullCountTotal;
      }
    } else {
      return 0;
    }
  }

  /**
   * Copy a sliced validity buffer to the destination buffer, starting at the given bit offset.
   *
   * @return Number of nulls in the validity buffer.
   */
  private static int copyValidityBuffer(HostMemoryBuffer dest, int startBit,
                                        HostMemoryBuffer src, int srcOffset,
                                        SliceInfo sliceInfo) {
    int nullCount = 0;
    int totalRowCount = toIntExact(sliceInfo.getRowCount() + sliceInfo.getValidityBufferInfo().getBeginBit());
    int curSrcIdx = sliceInfo.getValidityBufferInfo().getBeginBit();
    int curDestIdx = startBit;


    while (curSrcIdx < totalRowCount) {
      int leftRowCount = totalRowCount - curSrcIdx;

      int curDestOffset = (curDestIdx / 32) * Integer.BYTES;
      int curDestBitIdx = curDestIdx % 32;

      int curSrcOffset = srcOffset + (curSrcIdx / 32) * Integer.BYTES;
      int curSrcBitIdx = curSrcIdx % 32;

      // This is safe since we always have validity buffer 4 bytes padded
      int srcInt = src.getInt(curSrcOffset);
      srcInt = srcInt >>> curSrcBitIdx;

      if (dest.getLength() >= (curDestOffset + Integer.BYTES)) {
        // We have enough room to get an int
        int destInt = dest.getInt(curDestOffset);
        destInt &= (1 << curDestBitIdx) - 1;
        destInt |= srcInt << curDestBitIdx;
        dest.setInt(curDestOffset, destInt);

        int appendCount = min(leftRowCount, 32 - Math.max(curSrcBitIdx, curDestBitIdx));

        curDestIdx += appendCount;
        curSrcIdx += appendCount;
        if (appendCount == 32) {
          nullCount += 32 - Integer.bitCount(srcInt);
        } else {
          int mask = (1 << appendCount) - 1;
          nullCount += (appendCount  - Integer.bitCount(srcInt & mask));
        }
      } else {
        int destBufRemBytes = toIntExact(dest.getLength() - curDestOffset);
        byte[] destBytes = new byte[4];
        dest.getBytes(destBytes, 0, curDestOffset, destBufRemBytes);
        int destInt = ByteBuffer.wrap(destBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
        destInt &= (1 << curDestBitIdx) - 1;
        destInt |= srcInt << curDestBitIdx;

        ByteBuffer.wrap(destBytes).order(ByteOrder.LITTLE_ENDIAN).putInt(destInt);
        dest.setBytes(curDestOffset, destBytes, 0, destBufRemBytes);

        int appendCount = min(leftRowCount, destBufRemBytes * 8 - Math.max(curSrcBitIdx, curDestBitIdx));

        curDestIdx += appendCount;
        curSrcIdx += appendCount;
        int mask = (1 << appendCount) - 1;
        nullCount += (appendCount  - Integer.bitCount(srcInt & mask));
      }
    }

    int srcIdx = curSrcIdx;
    ensure(curSrcIdx == totalRowCount, () -> "Did not copy all of the validity buffer, total row count: " + totalRowCount +
        " current src idx: " + srcIdx);
    return nullCount;
  }

  private static void appendAllValid(HostMemoryBuffer dest, int startBit, int numRowsLong) {
    int numRows = toIntExact(numRowsLong);
    int curDestByteIdx = startBit / 8;
    int curDestBitIdx = startBit % 8;

    if (curDestBitIdx > 0) {
      int numBits = 8 - curDestBitIdx;
      int mask = ((1 << numBits) - 1) << curDestBitIdx;
      dest.setByte(curDestByteIdx, (byte) (dest.getByte(curDestByteIdx) | mask));
      curDestByteIdx += 1;
      numRows -= numBits;
    }

    if (numRows > 0) {
      int numBytes = (numRows + 7) / 8;
      dest.setMemory(curDestByteIdx, numBytes, (byte) 0xFF);
    }
  }

  private void deserializeOffsetBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getOffset() != INVALID_OFFSET) {
      long offset = curColOffset.getOffset();
      long bufferSize = Integer.BYTES * (getTotalRowCount() + 1);

      IntBuffer buf = buffer
          .asByteBuffer(offset, toIntExact(bufferSize))
          .order(ByteOrder.LITTLE_ENDIAN)
          .asIntBuffer();

      int accumulatedDataLen = 0;

      for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
        SliceInfo sliceInfo = sliceInfoOf(tableIdx);

        if (sliceInfo.getRowCount() > 0) {
          int rowCnt = sliceInfo.getRowCount();

          int firstOffset = offsetOf(tableIdx, 0);
          int lastOffset = offsetOf(tableIdx, rowCnt);

          for (int i = 0; i < rowCnt; i += 1) {
            buf.put(offsetOf(tableIdx, i) - firstOffset + accumulatedDataLen);
          }

          accumulatedDataLen += (lastOffset - firstOffset);
        }
      }

      buf.put(accumulatedDataLen);
    }
  }

  private void deserializeDataBuffer(ColumnOffsetInfo curColOffset, OptionalInt sizeInBytes) {
    if (curColOffset.getData() != INVALID_OFFSET && curColOffset.getDataBufferLen() > 0) {
      long offset = curColOffset.getData();
      long dataLen = curColOffset.getDataBufferLen();

      try (HostMemoryBuffer buf = buffer.slice(offset, dataLen)) {
        if (sizeInBytes.isPresent()) {
          // Fixed size type
          int elementSize = sizeInBytes.getAsInt();

          long start = 0;
          for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
            SliceInfo sliceInfo = sliceInfoOf(tableIdx);
            if (sliceInfo.getRowCount() > 0) {
              int thisDataLen = toIntExact(elementSize * sliceInfo.getRowCount());
              copyDataBuffer(buf, start, tableIdx, thisDataLen);
              start += thisDataLen;
            }
          }
        } else {
          // String type
          long start = 0;
          for (int tableIdx = 0; tableIdx < getTableSize(); tableIdx += 1) {
            int thisDataLen = getStrDataLenOf(tableIdx);
            copyDataBuffer(buf, start, tableIdx, thisDataLen);
            start += thisDataLen;
          }
        }
      }
    }
  }


  private ColumnOffsetInfo getCurColumnOffsets() {
    return columnOffsets.get(getCurrentIdx());
  }

  static KudoHostMergeResult merge(Schema schema, MergedInfoCalc mergedInfo) {
    List<KudoTable> serializedTables = mergedInfo.getTables();
    return Arms.closeIfException(HostMemoryBuffer.allocate(mergedInfo.getTotalDataLen()),
        buffer -> {
          KudoTableMerger merger =
              new KudoTableMerger(serializedTables, buffer, mergedInfo.getColumnOffsets());
          return Visitors.visitSchema(schema, merger);
        });
  }
}
