#!/bin/bash
#
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

# NOTE:
#     this script is for jenkins only, and should not be used for local development
#     run with ci/Dockerfile in jenkins:
#         scl enable devtoolset-11 rh-python38 "ci/submodule-sync.sh"

set -ex

OWNER=${OWNER:-"NVIDIA"}
REPO=${REPO:-"spark-rapids-jni"}
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
REPO_LOC="github.com/${OWNER}/${REPO}.git"

GIT_AUTHOR_NAME="spark-rapids automation"
GIT_COMMITTER_NAME="spark-rapids automation"
GIT_AUTHOR_EMAIL="70000568+nvauto@users.noreply.github.com"
GIT_COMMITTER_EMAIL="70000568+nvauto@users.noreply.github.com"
git submodule update --init --recursive

cudf_prev_sha=$(git -C thirdparty/cudf rev-parse HEAD)

INTERMEDIATE_HEAD=bot-submodule-sync-${REF}
# try cleanup remote first if no open PR for HEAD targeting BASE
$WORKSPACE/.github/workflows/action-helper/python/cleanup-bot-branch \
  --owner=${OWNER} --repo=${REPO} --head=${INTERMEDIATE_HEAD} --base=${REF} --token=${GIT_TOKEN} || true

remote_head=$(git ls-remote --heads origin ${INTERMEDIATE_HEAD})
if [[ -z $remote_head ]]; then
  git checkout -b ${INTERMEDIATE_HEAD} origin/${REF}
else
  git fetch origin ${INTERMEDIATE_HEAD} ${REF}
  git checkout -b ${INTERMEDIATE_HEAD} origin/${INTERMEDIATE_HEAD}
  git merge origin/${REF}
fi

# sync up cudf from remote
git submodule update --remote --merge
cudf_sha=$(git -C thirdparty/cudf rev-parse HEAD)
if [[ "${cudf_sha}" == "${cudf_prev_sha}" ]]; then
  echo "Submodule is up to date."
  exit 0
fi

echo "Try update cudf submodule to ${cudf_sha}..."
git add .
git diff-index --quiet HEAD || git commit -s -m "Update submodule cudf to ${cudf_sha}"
sha=$(git rev-parse HEAD)

echo "Test against ${cudf_sha}..."

MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -B"
set +e
${MVN} verify ${MVN_MIRROR} \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=ON -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest \
  -DBUILD_TESTS=ON
verify_status=$?
set -e

test_pass="False"
if [[ "${verify_status}" == "0" ]]; then
  echo "Test passed, will try merge the change"
  test_pass="True"
else
  echo "Test failed, will update the result"
fi

# push the intermediate branch and create PR against REF
# if test passed, it will try auto-merge the PR
# if test failed, it will only comment the test result in the PR
git push https://${GIT_USER}:${GIT_TOKEN}@${REPO_LOC} ${INTERMEDIATE_HEAD}
sleep 30 # sleep for a while to avoid inconsistent sha between HEAD branch and GitHub REST API
$WORKSPACE/.github/workflows/action-helper/python/submodule-sync \
  --owner=${OWNER} \
  --repo=${REPO} \
  --head=${INTERMEDIATE_HEAD} \
  --base=${REF} \
  --sha=${sha} \
  --cudf_sha=${cudf_sha} \
  --token=${GIT_TOKEN} \
  --passed=${test_pass} \
  --delete_head=True

exit $verify_status # always exit return code of mvn verify at the end
