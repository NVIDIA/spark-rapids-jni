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

/**
 * This class contains metrics for serializing table using kudo format.
 */
public class WriteMetrics {
  private long calcHeaderTime;
  private long copyHeaderTime;
  private long copyBufferTime;
  private long writtenBytes;


  public WriteMetrics() {
    this.calcHeaderTime = 0;
    this.copyHeaderTime = 0;
    this.copyBufferTime = 0;
    this.writtenBytes = 0;
  }

  /**
   * Get the time spent on calculating the header.
   */
  public long getCalcHeaderTime() {
    return calcHeaderTime;
  }

  /**
   * Get the time spent on copying the buffer.
   */
  public long getCopyBufferTime() {
    return copyBufferTime;
  }

  public void addCopyBufferTime(long time) {
    copyBufferTime += time;
  }

  /**
   * Get the time spent on copying the header.
   */
  public long getCopyHeaderTime() {
    return copyHeaderTime;
  }

  public void addCalcHeaderTime(long time) {
    calcHeaderTime += time;
  }

  public void addCopyHeaderTime(long time) {
    copyHeaderTime += time;
  }

  /**
   * Get the number of bytes written.
   */
  public long getWrittenBytes() {
    return writtenBytes;
  }

  public void addWrittenBytes(long bytes) {
    writtenBytes += bytes;
  }
}
