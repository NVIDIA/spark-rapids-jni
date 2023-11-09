/*
 * Copyright (c)  2023, NVIDIA CORPORATION.
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

import java.time.ZoneId;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.rapids.cudf.ColumnVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TimeZoneTest {
    @BeforeAll
    static void loadTimezoneDatabase() {
        Executor executor = Executors.newFixedThreadPool(1);
        GpuTimeZoneDB.load(executor);
    }

    @Test
    void databaseLoadTest() {
        // Check for a few timezones
        GpuTimeZoneDB instance = GpuTimeZoneDB.getInstance();
        List transitions = instance.getHostFixedTransitions("UTC+8");
        assertNotNull(transitions);
        assertEquals(1, transitions.size());
        transitions = instance.getHostFixedTransitions("Asia/Shanghai");
        assertNotNull(transitions);
        ZoneId shanghai = ZoneId.of("Asia/Shanghai").normalized();
        assertEquals(shanghai.getRules().getTransitions().size() + 1, transitions.size());
    }

    @Test
    void convertToUTCTest() {
        try (ColumnVector input = ColumnVector.timestampSecondsFromBoxedLongs(0L);
             ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(
                 -28800L);
             ColumnVector actual = GpuTimeZoneDB.convertToUTC(input,
                 ZoneId.of("Asia/Shanghai"))) {
            assertColumnsAreEqual(expected, actual);
        }
    }
}
