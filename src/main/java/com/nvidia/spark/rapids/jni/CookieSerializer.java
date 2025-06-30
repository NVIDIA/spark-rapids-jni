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

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

public class CookieSerializer {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    // Class storing unique_ptr<std::vector<uint8_t>>
    // This is just a prototype and can be changed later
    public static class NativeBuffer implements AutoCloseable {
        private long address;
        private long length;
        private long handle;

        public NativeBuffer(long[] stdVectorInfo) {
            this.address = stdVectorInfo[0];
            this.length = stdVectorInfo[1];
            this.handle = stdVectorInfo[2];
        }

        public long getAddress() {
            return address;
        }

        public long getLength() {
            return length;
        }

        public long getLong(int index) {
            return getLongNative(address, index);
        }

        @Override
        public void close() {
            if (handle != 0) {
                closeStdVector(handle);
                handle = 0;
            }
        }

        private static native void closeStdVector(long std_vector_handle);

        
        private static native long getLongNative(long address, int index);
    }

    
    public static NativeBuffer serializeFromAddrsAndSizes(long[] addrsSizes) {
        return new NativeBuffer(serializeNative(addrsSizes));
    }

    public static NativeBuffer serialize(HostMemoryBuffer... buffers) {
        long[] addrsSizes = new long[buffers.length * 2];
        for (int i = 0; i < buffers.length; i++) {
            addrsSizes[i * 2] = buffers[i].getAddress();
            addrsSizes[(i * 2) + 1] = buffers[i].getLength();
        }
        return new NativeBuffer(serializeNative(addrsSizes));
    }

    public static NativeBuffer[] deserializeFromAddrsAndSizes(long addr, long size) {
        long[] bufferInfos = deserializeNative(addr, size);
        NativeBuffer[] buffers = new NativeBuffer[bufferInfos.length / 3];
        for (int i = 0; i < buffers.length; i++) {
            long[] bufferInfo = {bufferInfos[i * 3], bufferInfos[i * 3 + 1], bufferInfos[i * 3 + 2]};
            buffers[i] = new NativeBuffer(bufferInfo);
        }
        return buffers;
    }

    public static void serializeToFile(String outputFile, HostMemoryBuffer... buffers) {
        long[] addrsSizes = new long[buffers.length * 2];
        for (int i = 0; i < buffers.length; i++) {
            addrsSizes[i * 2] = buffers[i].getAddress();
            addrsSizes[(i * 2) + 1] = buffers[i].getLength();
        }
        serializeToFileNative(addrsSizes, outputFile);
    }

    public static NativeBuffer[] deserializeFromFile(String inputFile) {
        long[] bufferInfos = deserializeNative(inputFile);
        NativeBuffer[] buffers = new NativeBuffer[bufferInfos.length / 3];
        for (int i = 0; i < buffers.length; i++) {
            long[] bufferInfo = {bufferInfos[i * 3], bufferInfos[i * 3 + 1], bufferInfos[i * 3 + 2]};
            buffers[i] = new NativeBuffer(bufferInfo);
        }
        return buffers;
    }

    // TODO: fix names
    private static native long[] serializeNative(long[] addrsSizes);

    private static native void serializeToFileNative(long[] addrsSizes, String outputFile);

    private static native long[] deserializeNative(long addr, long size);

    private static native long[] deserializeNative(String inputFile);

    
}
