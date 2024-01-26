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
  void testProtocol(String[] testData) {
    String[] expectedProtocolStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String scheme = null;
      try {
        URI uri = new URI(testData[i]);
        scheme = uri.getScheme();
      } catch (URISyntaxException ex) {
        // leave the scheme null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the scheme null if URI is null
      }
      expectedProtocolStrings[i] = scheme;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedProtocol = ColumnVector.fromStrings(expectedProtocolStrings);
      ColumnVector protocolResult = ParseURI.parseURIProtocol(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedProtocol, protocolResult);
    }
  }

  void testHost(String[] testData) {
    String[] expectedHostStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String host = null;
      try {
        URI uri = new URI(testData[i]);
        host = uri.getHost();
      } catch (URISyntaxException ex) {
        // leave the host null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the host null if URI is null
      }

      expectedHostStrings[i] = host;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedHost = ColumnVector.fromStrings(expectedHostStrings);
      ColumnVector hostResult = ParseURI.parseURIHost(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedHost, hostResult);
    }
  }

  void testQuery(String[] testData) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      expectedQueryStrings[i] = query;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQuery(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  void testQuery(String[] testData, String param) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      String subquery = null;

      if (query != null) {
        String[] pairs = query.split("&");
        for (String pair : pairs) {
          int idx = pair.indexOf("=");
          if (idx > 0 && pair.substring(0, idx).equals(param)) {
            subquery = pair.substring(idx + 1);
            break;
          }
        }
      }
      expectedQueryStrings[i] = subquery;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQueryWithLiteral(v0, param)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  void testQuery(String[] testData, String[] params) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      String subquery = null;

      if (query != null) {
        String[] pairs = query.split("&");
        for (String pair : pairs) {
          int idx = pair.indexOf("=");
          if (idx > 0 && pair.substring(0, idx).equals(params[i])) {
            subquery = pair.substring(idx + 1);
            break;
          }
        }
      }
      expectedQueryStrings[i] = subquery;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector p0 = ColumnVector.fromStrings(params);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQueryWithColumn(v0, p0)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  @Test
  void parseURISparkTest() {
    String[] testData = {
      "https://nvidia.com/https&#://nvidia.com",
      "https://http://www.nvidia.com",
      // commented out until https://github.com/NVIDIA/spark-rapids/issues/10036 is fixed
      //"http://www.nvidia.com/object.php?object=ะก-Ð%9Fะฑ-ะฟ-ะกÑ%82Ñ%80ะตะปÑ%8Cะฝะฐ-Ñ%83ะป-Ð%97ะฐะฒะพะดÑ%81ะบะฐÑ%8F.htm",
      "http://www.nvidia.com/object.php?object=ะก-Ðะฑ-ะฟ-ะกÑÑะตะปÑ%20ะฝะฐ-Ñะป-ÐะฐะฒะพะดÑะบะฐÑ.htm",
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
      "http//www.nvidia.com/q",
      "http://?",
      "http://#",
      "http://??",
      "http://??/",
      "http://user:pass@host/file;param?query;p2",
      "http://foo.bar/abc/\\\\\\http://foo.bar/abc.gif\\\\\\",
      "nvidia.com:8100/servlet/impc.DisplayCredits?primekey_in=2000041100:05:14115240636",
      "https://nvidia.com/2Ru15Ss ",
      "http://www.nvidia.com/xmlrpc//##",
      "www.nvidia.com:8080/expert/sciPublication.jsp?ExpertId=1746&lenList=all",
      "www.nvidia.com:8080/hrcxtf/view?docId=ead/00073.xml&query=T.%20E.%20Lawrence&query-join=and",
      "www.nvidia.com:81/Free.fr/L7D9qw9X4S-aC0&amp;D4X0/Panels&amp;solutionId=0X54a/cCdyncharset=UTF-8&amp;t=01wx58Tab&amp;ps=solution/ccmd=_help&amp;locale0X1&amp;countrycode=MA/",
      "http://www.nvidia.com/tags.php?%2F88\323\351\300ึณวน\331\315\370%2F",
      "http://www.nvidia.com//wp-admin/includes/index.html#9389#123",
      "http://[1:2:3:4:5:6:7::]",
      "http://[::2:3:4:5:6:7:8]",
      "http://[fe80::7:8%eth0]",
      "http://[fe80::7:8%1]",
      "http://www.nvidia.com/picshow.asp?id=106&mnid=5080&classname=\271\253ืฐฦช",
      "http://-.~_!$&'()*+,;=:%40:80%2f::::::@nvidia.com:443",
      "http://userid:password@nvidia.com:8080/",
      "https://www.nvidia.com/path?param0=1&param2=3&param4=5%206",
      "https:// /?params=5&cloth=0&metal=1",
      "https://[2001:db8::2:1]:443/parms/in/the/uri?a=b",
      "https://[::1]/?invalid=param&f„⁈.=7",
      "https://[::1]/?invalid=param&~.=!@&^",
      "userinfo@www.nvidia.com/path?query=1#Ref",
      "",
      null,
      "https://www.nvidia.com/?cat=12",
      "www.nvidia.com/vote.php?pid=50",
      "https://www.nvidia.com/vote.php?=50",
    };

      String[] queries = {
        "a",
        "h",
        // commented out until https://github.com/NVIDIA/spark-rapids/issues/10036 is fixed
        //"object",
        "object",
        "a",
        "h",
        "a",
        "f",
        "g",
        "a",
        "a",
        "f",
        "g",
        "a",
        "a",
        "b",
        "a",
        "",
        "a",
        "a",
        "a",
        "a",
        "b",
        "a",
        "q",
        "b",
        "a",
        "query",
        "a",
        "primekey_in",
        "a",
        "q",
        "ExpertId",
        "query",
        "solutionId",
        "f",
        "param",
        "",
        "q",
        "a",
        "f",
        "mnid=5080",
        "f",
        "a",
        "param4",
        "cloth",
        "a",
        "invalid",
        "invalid",
        "query",
        "a",
        "f",
        "query",
        "query",
        ""
      };

    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
    testQuery(testData, queries);
  }

  @Test
  void parseURIUTF8Test() {
    String[] testData = {
      "https:// /path/to/file",
      "https://nvidia.com/%4EV%49%44%49%41",
      "http://%77%77%77.%4EV%49%44%49%41.com",
      "http://✪↩d⁚f„⁈.ws/123"};

    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
  }

  @Test
  void parseURIIP4Test() {
    String[] testData = {
      "https://192.168.1.100/",
      "https://192.168.1.100:8443/",
      "https://192.168.1.100.5/",
      "https://192.168.1/",
      "https://280.100.1.1/",
      "https://182.168..100/path/to/file"};

    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
  }

  @Test
  void parseURIIP6Test() {
    String[] testData = {
      "https://[fe80::]",
      "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
      "https://[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
      "https://[2001:db8::1:0]",
      "http://[2001:db8::2:1]",
      "https://[::1]",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
      "https://[2001:db8:3333:4444:5555:6666:1.2.3.4]/path/to/file",
      "https://[2001:db8:3333:4444:5555:6666:7777:8888:1.2.3.4]/path/to/file",
      "https://[::db8:3333:4444:5555:6666:1.2.3.4]/path/to/file]",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
      "https://[2001:]db8:85a3:8d3:1319:8a2e:370:7348/",
      "https://[][][][]nvidia.com/",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348:2001:db8:85a3]/path",
      "http://[1:2:3:4:5:6:7::]",
      "http://[::2:3:4:5:6:7:8]",
      "http://[fe80::7:8%eth0]",
      "http://[fe80::7:8%1]",
    };
    
    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
  }
}
