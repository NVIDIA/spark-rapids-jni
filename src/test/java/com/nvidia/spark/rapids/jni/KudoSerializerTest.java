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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import com.nvidia.spark.rapids.jni.kudo.KudoSerializer;
import com.nvidia.spark.rapids.jni.kudo.KudoTableHeader;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;

import static org.junit.jupiter.api.Assertions.*;

public class KudoSerializerTest {

  @Test
  public void testRowCountOnly() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    long bytesWritten = KudoSerializer.writeRowCountToStream(out, 5);
    assertEquals(28, bytesWritten);

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();

    assertEquals(0, header.getNumColumns());
    assertEquals(0, header.getOffset());
    assertEquals(5, header.getNumRows());
    assertEquals(0, header.getValidityBufferLen());
    assertEquals(0, header.getOffsetBufferLen());
    assertEquals(0, header.getTotalDataLen());
  }

  @Test
  public void testWriteSimple() throws Exception {
    KudoSerializer serializer = new KudoSerializer(buildSimpleTestSchema());

    try(Table t = buildSimpleTable()) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      long bytesWritten = serializer.writeToStream(t, out, 0, 4);
      assertEquals(189, bytesWritten);

      ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

      KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();
      assertEquals(7, header.getNumColumns());
      assertEquals(0, header.getOffset());
      assertEquals(4, header.getNumRows());
      assertEquals(24, header.getValidityBufferLen());
      assertEquals(40, header.getOffsetBufferLen());
      assertEquals(160, header.getTotalDataLen());

      // First integer column has no validity buffer
      assertFalse(header.hasValidityBuffer(0));
      for (int i=1; i<7; i++) {
        assertTrue(header.hasValidityBuffer(i));
      }
    }
  }

  private static Schema buildSimpleTestSchema() {
    Schema.Builder builder = Schema.builder();

    builder.addColumn(DType.INT32, "a");
    builder.addColumn(DType.STRING, "b");
    Schema.Builder listBuilder = builder.addColumn(DType.LIST, "c");
    listBuilder.addColumn(DType.INT32, "c1");

    Schema.Builder structBuilder =  builder.addColumn(DType.STRUCT, "d");
    structBuilder.addColumn(DType.INT8, "d1");
    structBuilder.addColumn(DType.INT64, "d2");

    return builder.build();
  }

  private static Table buildSimpleTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
        .column(1, 2, 3, 4)
        .column("1", "12", null, "45")
        .column(new Integer[]{1, null, 3}, new Integer[]{4, 5, 6}, null, new Integer[]{7, 8, 9})
        .column(st,  new HostColumnVector.StructData((byte)1, 11L),
            new HostColumnVector.StructData ((byte)2, null), null,
            new HostColumnVector.StructData((byte)3, 33L))
        .build();
  }
}
