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

public class SerializedTable implements AutoCloseable {
    private final SerializedTableHeader header;
    private final HostMemoryBuffer buffer;

    SerializedTable(SerializedTableHeader header, HostMemoryBuffer buffer) {
        this.header = header;
        this.buffer = buffer;
    }

    public SerializedTableHeader getHeader() {
        return header;
    }

    public HostMemoryBuffer getBuffer() {
        return buffer;
    }

    @Override
    public String toString() {
        return "SerializedTable{" +
                "header=" + header +
                ", buffer=" + buffer +
                '}';
    }

    @Override
    public void close() throws Exception {
        if (buffer != null) {
            buffer.close();
        }
    }
}
