#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

###
# Script to deploy spark-rapids-jni jar files along with other classifiers,
# such as cudaXXX, sources, javadoc.
#
# Argument(s):
#   SIGN_FILE: true/false, whether to sign the jar/pom file to de deployed
#
# Used environment(s):
#   CLASSIFIERS:    The classifier list of the jars to be deployed
#   SERVER_ID:      The repository id for this deployment.
#   SERVER_URL:     The url where to deploy artifacts.
#   GPG_PASSPHRASE: The passphrase used to sign files, only required when <SIGN_FILE> is true.
###

set -ex

SIGN_FILE=$1
OUT_PATH=$WORKSPACE/$OUT

cd $WORKSPACE/
REL_VERSION=$(mvn exec:exec -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.version}')

echo "REL_VERSION: $REL_VERSION, OUT_PATH: $OUT_PATH \
SERVER_URL: $SERVER_URL, SERVER_ID: $SERVER_ID"

###### Build the deploy command ######
if [ "$SIGN_FILE" == true ]; then
    DEPLOY_CMD="mvn -B gpg:sign-and-deploy-file -s ci/settings.xml -Dgpg.passphrase=$GPG_PASSPHRASE"
else
    DEPLOY_CMD="mvn -B deploy:deploy-file -s ci/settings.xml"
fi
echo "Deploy CMD: $DEPLOY_CMD"

###### Build types/files from classifiers ######
FPATH="$OUT_PATH/rapids-4-spark-jni-$REL_VERSION"
CLASS_TYPES=''
CLASS_FILES=''
ORI_IFS="$IFS"
IFS=','
for CLASS in $CLASSIFIERS; do
    CLASS_TYPES="${CLASS_TYPES},jar"
    CLASS_FILES="${CLASS_FILES},${FPATH}-${CLASS}.jar"
done
# Remove the first char ','
CLASS_TYPES=${CLASS_TYPES#*,}
CLASS_FILES=${CLASS_FILES#*,}
IFS="$ORI_IFS"

###### Copy jar so we strip off classifier  #######
# Use the first classifier(aka jar file) as the default jar
FIRST_FILE=${CLASS_FILES%%,*}
cp -f "$FIRST_FILE" "$FPATH.jar"

###### Deploy spark-rapids-jni jar with all its additions ######
$DEPLOY_CMD -Durl=$SERVER_URL -DrepositoryId=$SERVER_ID \
            -Dfile=$FPATH.jar -DpomFile=spark-rapids-jni/pom.xml \
            -Dfiles=$CLASS_FILES \
            -Dtypes=$CLASS_TYPES \
            -Dclassifiers=$CLASSIFIERS
