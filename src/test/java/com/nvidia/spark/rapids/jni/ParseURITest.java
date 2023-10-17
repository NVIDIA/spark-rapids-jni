/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

import java.net.URI;
import java.net.URISyntaxException;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;

public class ParseURITest {
  @Test
  void parseURIToProtocolTest() {
    String[] testData = {"https://nvidia.com/https&#://nvidia.com",
      "https://http://www.nvidia.com",
      "filesystemmagicthing://bob.yaml",
      "nvidia.com:8080",
      "http://thisisinvalid.data/due/to-the_character%s/inside*the#url`~",
      "file:/absolute/path",
      "//www.nvidia.com",
      "#bob",
      "#this%doesnt#make//sense://to/me",
      "HTTP:&bob",
      "/absolute/path",
      "http://%77%77%77.%4EV%49%44%49%41.com",
      "https:://broken.url",
      "https://www.nvidia.com/q/This%20is%20a%20query",
      "http:/www.nvidia.com",
      "http://:www.nvidia.com/",
      "http:///nvidia.com/q",
      "https://www.nvidia.com:8080/q",
      "https://www.nvidia.com#8080",
      "file://path/to/cool/file",
      "http//www.nvidia.com/q"};
    
    String[] expectedStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String scheme = null;
      try {
        URI uri = new URI(testData[i]);
        scheme = uri.getScheme();
      } catch (URISyntaxException ex) {
        // leave the scheme null if URI is invalid
      }
      expectedStrings[i] = scheme;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expected = ColumnVector.fromStrings(expectedStrings);
      ColumnVector result = ParseURI.parseURIProtocol(v0)) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }
}
