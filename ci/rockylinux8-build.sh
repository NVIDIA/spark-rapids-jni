set -ex

nvidia-smi

git submodule update --init --recursive

PARALLEL_LEVEL=${PARALLEL_LEVEL:-16}
scl enable gcc-toolset-9 "mvn clean package ${MVN_MIRROR}  \
  -Psource-javadoc \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=OFF -Dtest=*,!CuFileTest"
