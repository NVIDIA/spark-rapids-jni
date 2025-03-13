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

package com.nvidia.spark.rapids.jni.kudo;

import java.io.OutputStream;

public class MergeOptions {
  private final DumpOption dumpOption;
  private final OutputStream outputStream;
  private final String filePath;

  public MergeOptions(DumpOption dumpOption, OutputStream outputStream, String filePath) {
    this.dumpOption = dumpOption;
    this.outputStream = outputStream;
    this.filePath = filePath;
  }

  public DumpOption getDumpOption() {
    return dumpOption;
  }

  public OutputStream getOutputStream() {
    return outputStream;
  }

  public String getFilePath() {
    return filePath;
  }
}
