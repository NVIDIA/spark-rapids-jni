#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/bin/bash

set -ex

# Assuming you are in the root of your git repository
MODIFIED_FILES=$(git diff --name-only "origin/${GITHUB_BASE_REF}")

SRC_DIR="src/main/java/"
TEST_SRC_DIR="src/test/java/"
# Filter out the Java files that have been modified
JAVA_FILES=()
for FILE in $MODIFIED_FILES; do
  if [[ $FILE == *.java ]]; then
    if [[ $FILE == $SRC_DIR* ]]; then
      JAVA_FILES+=("${FILE#"$SRC_DIR"}") # Remove the src/main/java/ prefix
    elif [[ $FILE == $TEST_SRC_DIR* ]]; then
      JAVA_FILES+=("${FILE#"$TEST_SRC_DIR"}") # Remove the src/test/java/ prefix
    fi
  fi
done

# If there are Java files to check, run Checkstyle on them
if [ ${#JAVA_FILES[@]} -ne 0 ]; then
  mvn checkstyle:check -Dcheckstyle.includes="$(echo "${JAVA_FILES[@]}" | tr ' ' ',')"
else
  echo "No Java files modified, skipping Checkstyle."
fi
