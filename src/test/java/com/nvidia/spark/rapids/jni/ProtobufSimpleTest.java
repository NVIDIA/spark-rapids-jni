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
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

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

  private static Byte[] box(byte[] bytes) {
    Byte[] out = new Byte[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      out[i] = bytes[i];
    }
    return out;
  }

  private static Byte[] concat(Byte[]... parts) {
    int len = 0;
    for (Byte[] p : parts) len += p.length;
    Byte[] out = new Byte[len];
    int pos = 0;
    for (Byte[] p : parts) {
      System.arraycopy(p, 0, out, pos, p.length);
      pos += p.length;
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
}


