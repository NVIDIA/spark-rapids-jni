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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class CookieSerializerTest {

  private static class NoopCleaner extends MemoryBuffer.MemoryBufferCleaner {
    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      return true;
    }

    @Override
    public boolean isClean() {
      return true;
    }
  }

  private static final NoopCleaner cleaner = new NoopCleaner();

  @Test
  void simpleRoundTripTest() {
    try (HostMemoryBuffer input = new HostMemoryBuffer(1000L * Long.BYTES);) {
      for (int i = 0; i < 1000; i++) {
        input.setLong(i * Long.BYTES, i);
      }
      try (CookieSerializer.NativeBuffer serialized = CookieSerializer.serialize(input);
           CookieSerializer.NativeBuffer[] deserialized = CookieSerializer.deserialize(
            serialized.getAddress(), serialized.getLength());
           HostMemoryBuffer output = new HostMemoryBuffer(deserialized[0].getAddress(), 
           deserialized[0].getLength(), cleaner);
      ) {
         assertColumnsAreEqual(input, output);
      }
    }
  }
  
}
