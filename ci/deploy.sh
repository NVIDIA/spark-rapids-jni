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
#   OUT:            The path where jar files are, relative to $WORKSPACE
#   CLASSIFIERS:    The classifier list of the jars to be deployed
#   SERVER_ID:      The repository id for this deployment.
#   SERVER_URL:     The url where to deploy artifacts.
#   NVSEC_CFG_FILE: The nvsec credentials used to sign via 3S service, only required when <SIGN_FILE> is true.
###

set -ex

SIGN_FILE=$1
WORKSPACE=${WORKSPACE:-`pwd`}
OUT=${OUT:-'out'}
#Set the absolute path for 'out'
OUT_PATH=$WORKSPACE/$OUT

cd $WORKSPACE/
REL_VERSION=$(mvn exec:exec -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.version}')

echo "REL_VERSION: $REL_VERSION, OUT_PATH: $OUT_PATH \
        SERVER_URL: $SERVER_URL, SERVER_ID: $SERVER_ID"

###### Build types/files from classifiers ######
FPATH="$OUT_PATH/spark-rapids-jni-$REL_VERSION"
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

###### Build the deploy command ######
MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -DretryFailedDeploymentCount=3 -B"
DEPLOY_CMD="$MVN -B deploy:deploy-file -Durl=$SERVER_URL -DrepositoryId=$SERVER_ID \
    -DgroupId=com.nvidia -DartifactId=spark-rapids-jni -Dversion=$REL_VERSION -s ci/settings.xml"
SIGN_CMD='eval nvsec sign --job-name "Spark Jar Signing" --description "Sign artifact with 3s"'
echo "Deploy CMD: $DEPLOY_CMD, sign CMD: $SIGN_CMD"

###### sign with nvsec 3s #######
if [ "$SIGN_FILE" == true ]; then
    # Apply nvsec configs
    cp $NVSEC_CFG_FILE ~/.nvsec.cfg
    # nvsec add the '-signature' suffix to signed file, upload with packaging '.asc' to meet Sonatype requirement
    $SIGN_CMD --file $FPATH.jar --out-dir $OUT_PATH
    $DEPLOY_CMD -Dfile=$FPATH.jar-signature -Dpackaging=jar.asc
    $SIGN_CMD --file pom.xml --out-dir ./
    $DEPLOY_CMD -Dfile=pom.xml-signature -Dpackaging=pom.asc
    SIGN_CLASS="sources"
    $SIGN_CMD --file $FPATH-$SIGN_CLASS.jar --out-dir $OUT_PATH
    $DEPLOY_CMD -Dfile=$FPATH-$SIGN_CLASS.jar-signature -Dclassifier=$SIGN_CLASS -Dpackaging=jar.asc
    SIGN_CLASS="javadoc"
    $SIGN_CMD --file $FPATH-$SIGN_CLASS.jar --out-dir $OUT_PATH
    $DEPLOY_CMD -Dfile=$FPATH-$SIGN_CLASS.jar-signature -Dclassifier=$SIGN_CLASS -Dpackaging=jar.asc
    for SIGN_CLASS in ${CLASSIFIERS//,/ }; do
        $SIGN_CMD --file $FPATH-$SIGN_CLASS.jar --out-dir $OUT_PATH
        $DEPLOY_CMD -Dfile=$FPATH-$SIGN_CLASS.jar-signature -Dclassifier=$SIGN_CLASS -Dpackaging=jar.asc
    done
fi

###### Deploy spark-rapids-jni jar with all its additions ######
$DEPLOY_CMD -Dfile=$FPATH.jar \
            -DpomFile=pom.xml \
            -Dfiles=$CLASS_FILES \
            -Dsources=$FPATH-sources.jar \
            -Djavadoc=$FPATH-javadoc.jar \
            -Dtypes=$CLASS_TYPES \
            -Dclassifiers=$CLASSIFIERS
