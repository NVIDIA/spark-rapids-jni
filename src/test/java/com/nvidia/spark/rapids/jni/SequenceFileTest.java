/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the SequenceFile JNI parsing API.
 */
public class SequenceFileTest {

    private static final byte[] TEST_SYNC_MARKER = new byte[] {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10
    };

    /**
     * Build SequenceFile record data (without header).
     *
     * Record format:
     *   recordLength (int32 BE) = keyLen + valueLen
     *   keyLength (int32 BE)
     *   keyBytes
     *   valueBytes
     */
    private byte[] buildTestData(List<byte[][]> records, byte[] syncMarker, int insertSyncEvery) {
        ByteBuffer buffer = ByteBuffer.allocate(1024 * 1024);
        buffer.order(ByteOrder.BIG_ENDIAN);

        int recordCount = 0;
        for (byte[][] record : records) {
            byte[] key = record[0];
            byte[] value = record[1];

            // Insert sync marker if needed
            if (insertSyncEvery > 0 && recordCount > 0 && (recordCount % insertSyncEvery) == 0) {
                buffer.putInt(-1);  // Sync marker indicator
                buffer.put(syncMarker);
            }

            int recordLen = key.length + value.length;
            buffer.putInt(recordLen);
            buffer.putInt(key.length);
            buffer.put(key);
            buffer.put(value);

            recordCount++;
        }

        buffer.flip();
        byte[] result = new byte[buffer.remaining()];
        buffer.get(result);
        return result;
    }

    /**
     * Extract data from a LIST<UINT8> column as list of byte arrays.
     */
    private List<byte[]> extractListData(ColumnVector col) {
        List<byte[]> result = new ArrayList<>();
        try (HostMemoryBuffer offsetsBuffer = HostMemoryBuffer.allocate(
                (col.getRowCount() + 1) * Integer.BYTES)) {
            
            // Get offsets and child data
            // For this test, we'll use a simpler approach - just validate row count
            for (int i = 0; i < col.getRowCount(); i++) {
                // In a real implementation, we'd extract the actual bytes
                // For now, just verify the structure is correct
                result.add(new byte[0]);  // Placeholder
            }
        }
        return result;
    }

    @Test
    void testNullSyncMarker() {
        try (DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(100)) {
            assertThrows(IllegalArgumentException.class, () ->
                SequenceFile.parseSequenceFile(data, 100, null, true, true));
        }
    }

    @Test
    void testInvalidSyncMarkerSize() {
        byte[] invalidSync = new byte[] {0x01, 0x02, 0x03};  // Only 3 bytes
        try (DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(100)) {
            assertThrows(IllegalArgumentException.class, () ->
                SequenceFile.parseSequenceFile(data, 100, invalidSync, true, true));
        }
    }

    @Test
    void testNullDataBuffer() {
        assertThrows(IllegalArgumentException.class, () ->
            SequenceFile.parseSequenceFile(null, 100, TEST_SYNC_MARKER, true, true));
    }

    @Test
    void testNegativeDataSize() {
        try (DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(100)) {
            assertThrows(IllegalArgumentException.class, () ->
                SequenceFile.parseSequenceFile(data, -1, TEST_SYNC_MARKER, true, true));
        }
    }

    @Test
    void testNoColumnsRequested() {
        try (DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(100)) {
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                data, 0, TEST_SYNC_MARKER, false, false);
            assertEquals(0, result.length);
        }
    }

    @Test
    void testEmptyData() {
        try (DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(1)) {
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                data, 0, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(0, keyCol.getRowCount());
                assertEquals(0, valCol.getRowCount());
            }
        }
    }

    @Test
    void testSingleRecord() {
        byte[] key = "key1".getBytes();
        byte[] value = "value1".getBytes();
        
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {key, value});
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(1, keyCol.getRowCount());
                assertEquals(1, valCol.getRowCount());
            }
        }
    }

    @Test
    void testMultipleRecords() {
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {"k1".getBytes(), "v1".getBytes()});
        records.add(new byte[][] {"k22".getBytes(), "v222".getBytes()});
        records.add(new byte[][] {"k333".getBytes(), "v3".getBytes()});
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(3, keyCol.getRowCount());
                assertEquals(3, valCol.getRowCount());
            }
        }
    }

    @Test
    void testRecordsWithSyncMarkers() {
        List<byte[][]> records = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            records.add(new byte[][] {
                ("k" + i).getBytes(),
                ("value" + i).getBytes()
            });
        }
        
        // Insert sync markers every 3 records
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 3);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(10, keyCol.getRowCount());
                assertEquals(10, valCol.getRowCount());
            }
        }
    }

    @Test
    void testKeyOnly() {
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {"k1".getBytes(), "v1".getBytes()});
        records.add(new byte[][] {"k2".getBytes(), "v2".getBytes()});
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, false);
            
            assertEquals(1, result.length);
            try (ColumnVector keyCol = result[0]) {
                assertEquals(2, keyCol.getRowCount());
            }
        }
    }

    @Test
    void testValueOnly() {
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {"k1".getBytes(), "v1".getBytes()});
        records.add(new byte[][] {"k2".getBytes(), "v2".getBytes()});
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, false, true);
            
            assertEquals(1, result.length);
            try (ColumnVector valCol = result[0]) {
                assertEquals(2, valCol.getRowCount());
            }
        }
    }

    @Test
    void testLargeRecords() {
        Random rng = new Random(42);
        byte[] largeKey = new byte[10000];
        byte[] largeValue = new byte[50000];
        rng.nextBytes(largeKey);
        rng.nextBytes(largeValue);
        
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {largeKey, largeValue});
        records.add(new byte[][] {"small".getBytes(), "val".getBytes()});
        records.add(new byte[][] {largeKey, "tiny".getBytes()});
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(3, keyCol.getRowCount());
                assertEquals(3, valCol.getRowCount());
            }
        }
    }

    @Test
    void testCountRecords() {
        List<byte[][]> records = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            records.add(new byte[][] {
                ("k" + i).getBytes(),
                ("v" + i).getBytes()
            });
        }
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 10);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            long count = SequenceFile.countRecords(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER);
            
            assertEquals(100, count);
        }
    }

    @Test
    void testEmptyKeyAndValue() {
        List<byte[][]> records = new ArrayList<>();
        records.add(new byte[][] {new byte[0], new byte[0]});  // Both empty
        records.add(new byte[][] {"k".getBytes(), new byte[0]});  // Value empty
        records.add(new byte[][] {new byte[0], "v".getBytes()});  // Key empty
        records.add(new byte[][] {"k2".getBytes(), "v2".getBytes()});  // Normal
        
        byte[] hostData = buildTestData(records, TEST_SYNC_MARKER, 0);
        
        try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(hostData.length);
             DeviceMemoryBuffer deviceBuffer = DeviceMemoryBuffer.allocate(hostData.length)) {
            
            hostBuffer.setBytes(0, hostData, 0, hostData.length);
            deviceBuffer.copyFromHostBuffer(hostBuffer);
            
            ColumnVector[] result = SequenceFile.parseSequenceFile(
                deviceBuffer, hostData.length, TEST_SYNC_MARKER, true, true);
            
            assertEquals(2, result.length);
            try (ColumnVector keyCol = result[0];
                 ColumnVector valCol = result[1]) {
                assertEquals(4, keyCol.getRowCount());
                assertEquals(4, valCol.getRowCount());
            }
        }
    }
}
