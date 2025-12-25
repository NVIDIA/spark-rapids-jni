/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector.*;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class ProtobufSimpleTest {

  private static byte[] encodeVarint(long value) {
    long v = value;
    byte[] tmp = new byte[10];
    int idx = 0;
    while ((v & ~0x7FL) != 0) {
      tmp[idx++] = (byte) ((v & 0x7F) | 0x80);
      v >>>= 7;
    }
    tmp[idx++] = (byte) (v & 0x7F);
    byte[] out = new byte[idx];
    System.arraycopy(tmp, 0, out, 0, idx);
    return out;
  }

  private static long zigzagEncode(long n) {
    return (n << 1) ^ (n >> 63);
  }

  private static byte[] encodeFixed32(int v) {
    return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
  }

  private static byte[] encodeFixed64(long v) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
  }

  private static Byte[] box(byte[] bytes) {
    if (bytes == null) return null;
    Byte[] out = new Byte[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      out[i] = bytes[i];
    }
    return out;
  }

  private static Byte[] concat(Byte[]... parts) {
    int len = 0;
    for (Byte[] p : parts) if (p != null) len += p.length;
    Byte[] out = new Byte[len];
    int pos = 0;
    for (Byte[] p : parts) {
      if (p != null) {
        System.arraycopy(p, 0, out, pos, p.length);
        pos += p.length;
      }
    }
    return out;
  }

  @Test
  void decodeVarintAndStringToStruct() {
    // message Msg { int64 id = 1; string name = 2; }
    // Row0: id=100, name="alice"
    Byte[] row0 = concat(
        new Byte[]{(byte) 0x08}, // field 1, varint
        box(encodeVarint(100)),
        new Byte[]{(byte) 0x12}, // field 2, len-delimited
        box(encodeVarint(5)),
        box("alice".getBytes()));

    // Row1: id=200, name missing
    Byte[] row1 = concat(
        new Byte[]{(byte) 0x08},
        box(encodeVarint(200)));

    // Row2: null input message
    Byte[] row2 = null;

    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector expectedId = ColumnVector.fromBoxedLongs(100L, 200L, null);
         ColumnVector expectedName = ColumnVector.fromStrings("alice", null, null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedId, expectedName);
         ColumnVector actualStruct = ProtobufSimple.decodeToStruct(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{0, 0})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void decodeMoreTypes() {
    // message Msg {
    //   uint32 u32 = 1;
    //   sint64 s64 = 2;
    //   fixed32 f32 = 3;
    //   bytes b = 4;
    // }
    Byte[] row0 = concat(
        new Byte[]{(byte) 0x08}, // field 1, varint
        box(encodeVarint(4000000000L)),
        new Byte[]{(byte) 0x10}, // field 2, varint
        box(encodeVarint(zigzagEncode(-1234567890123L))),
        new Byte[]{(byte) 0x1d}, // field 3, fixed32
        box(encodeFixed32(12345)),
        new Byte[]{(byte) 0x22}, // field 4, len-delimited
        box(encodeVarint(3)),
        box(new byte[]{1, 2, 3}));

    try (Table input = new Table.TestBuilder().column(row0).build();
         ColumnVector expectedU32 = ColumnVector.fromBoxedLongs(4000000000L); // cuDF doesn't have boxed UInt32 easily, use Longs for test if needed, but we want native id
         // Wait, I'll use direct values to avoid Boxing issues with UInt32
         ColumnVector expectedS64 = ColumnVector.fromBoxedLongs(-1234567890123L);
         ColumnVector expectedF32 = ColumnVector.fromBoxedInts(12345);
         ColumnVector expectedB = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT8)),
             Arrays.asList((byte) 1, (byte) 2, (byte) 3));
         ColumnVector actualStruct = ProtobufSimple.decodeToStruct(
             input.getColumn(0),
             new int[]{1, 2, 3, 4},
             new int[]{
                 DType.UINT32.getTypeId().getNativeId(),
                 DType.INT64.getTypeId().getNativeId(),
                 DType.INT32.getTypeId().getNativeId(),
                 DType.LIST.getTypeId().getNativeId()},
             new int[]{
                 ProtobufSimple.ENC_DEFAULT,
                 ProtobufSimple.ENC_ZIGZAG,
                 ProtobufSimple.ENC_FIXED,
                 ProtobufSimple.ENC_DEFAULT})) {
      // For UINT32, expectedU32 from fromBoxedLongs will be INT64.
      // I should use makeColumn to get exactly the right types for comparison.
      try (ColumnVector expectedU32Correct = expectedU32.castTo(DType.UINT32);
           ColumnVector expectedStructCorrect = ColumnVector.makeStruct(expectedU32Correct, expectedS64, expectedF32, expectedB)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStructCorrect, actualStruct);
      }
    }
  }
}


