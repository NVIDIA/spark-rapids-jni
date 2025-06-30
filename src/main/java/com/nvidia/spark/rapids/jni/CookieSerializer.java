/*
 *
 *  Copyright (c) 2025, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package com.nvidia.spark.rapids.jni;

public class CookieSerializer {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    // Class storing unique_ptr<std::vector<uint8_t>>
    // This is just a prototype and can be changed later
    public class NativeBuffer implements AutoCloseable {
        private long std_vector_handle;

        public NativeBuffer(long std_vector_handle) {
            this.std_vector_handle = std_vector_handle;
        }

        @Override
        public void close() {
            if (std_vector_handle != 0) {
                closeStdVector(std_vector_handle);
                std_vector_handle = 0;
            }
        }

        private native void closeStdVector(long std_vector_handle);
    }

    
    public static NativeBuffer serialize(long[] addrsSizes) {
        return new NativeBuffer(serialize(addrsSizes));
    }

    public static NativeBuffer serialize(HostMemoryBuffer... buffers) {
        long[] addrsSizes = new long[buffers.length * 2];
        for (int i = 0; i < buffers.length; i++) {
            addrsSizes[i * 2] = buffers[i].getAddress();
            addrsSizes[(i * 2) + 1] = buffers[i].getLength();
        }
        return new NativeBuffer(serialize(addrsSizes));
    }

    public static NativeBuffer[] deserialize(long addr, long size) {
        long[] bufferHandles = deserialize(addr, size);
        NativeBuffer[] buffers = new NativeBuffer[bufferHandles.length];
        for (int i = 0; i < bufferHandles.length; i++) {
            buffers[i] = new NativeBuffer(bufferHandles[i]);
        }
        return buffers;
    }

    private static native long serialize(long[] addrsSizes);

    private static native long[] deserialize(long addr, long size);
}
