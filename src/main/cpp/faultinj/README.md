# Fault Injection Tool for CUDA Runtime and CUDA Driver API

The goal of this tool is to increase testability of the failure handling logic
in CUDA applications.

It's especially important when CUDA is embedded in a higher-level fault-tolerant framework
to ensure that CUDA failures are handled correctly in that
- fatal errors leaving a GPU in unusable state are detected and such GPUs are prevented from executing retries for
a failed computation
- non-fatal errors are retried without losing valuable compute resources
- error handling logic does not cause deadlocks and other sort of unresponsiveness.

This tool allows creating test scenarios and triggering CUDA errors at will
with some degree of configurability to determine whether an individual CUDA process
or the higher-level framework such as Apache Spark remain usable when it should
or gracefully exits with an actionable error message when the errors are irrecoverable.

## Deployment

The tool is designed with automated testing and interactive testing use cases in mind. The tool is a dynamically linked library `libcufaultinj.so` that is loaded by the CUDA process via CUDA Driver API `cuInit` if it's provided via the `CUDA_INJECTION64_PATH` environment variable.

As an example it can be used to test RAPIDS Accelerator for Apache Spark. Consult documentation to find
how to set these variables correctly in the context of the framework under test.

### Local Mode
Spark local mode is a single CUDA process. We can test is as any standalone single-process application.

```bash
CUDA_INJECTION64_PATH=$PWD/target/cmake-build/faultinj/libcufaultinj.so \
FAULT_INJECTOR_CONFIG_PATH=src/test/cpp/faultinj/test_faultinj.json \
$SPARK_HOME/bin/pyspark \
  --jars $SPARK_RAPIDS_REPO/dist/target/rapids-4-spark_2.12-22.08.0-SNAPSHOT-cuda11.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin
```
### Distributed Mode
```bash
$SPARK_HOME/bin/spark-shell \
  --jars $SPARK_RAPIDS_REPO/dist/target/rapids-4-spark_2.12-22.08.0-SNAPSHOT-cuda11.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --files ./target/cmake-build/faultinj/libcufaultinj.so,./src/test/cpp/faultinj/test_faultinj.json \
  --conf spark.executorEnv.CUDA_INJECTION64_PATH=./libcufaultinj.so \
  --conf spark.executorEnv.FAULT_INJECTOR_CONFIG_PATH=test_faultinj.json \
  --conf spark.rapids.memory.gpu.minAllocFraction=0 \
  --conf spark.rapids.memory.gpu.allocFraction=0.2 \
  --master spark://hostname:7077
```
When we configure the executor environment spark.executorEnv.CUDA_INJECTION64_PATH
we have to use a path separator in the value ./libcufaultinj.so with the leading dot
to make sure that dlopen loads the library file submitted. Otherwise it will assume a
locally installed library accessible to the dynamic linker via LD_LIBRARY_PATH and similar mechanisms.
See [dlopen man page](https://man7.org/linux/man-pages/man3/dlopen.3.html)

## Fault injection configuration

Fault injection configuration is provided via the `FAULT_INJECTOR_CONFIG_PATH` environment variable.
It's a set of rules to apply fault injection when CUDA Drvier or Runtime is matched by either
- function name such as [`cudaLaunchKernel_ptsz`](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-default-cli)
- or callback id such as [`214`](https://gitlab.com/nvidia/headers/cuda-individual/cupti/-/blob/main/cupti_runtime_cbid.h#L224)
- or wildcard * to match all API
with a given probability.

Example config:
```json
{
    "logLevel": 1,
    "dynamic": true,
    "cudaRuntimeFaults": {
        "cudaLaunchKernel_ptsz": {
            "percent": 0,
            "injectionType": 0,
            "interceptionCount": 1
        }
    },
    "cudaDriverFaults": {
        "cuMemFreeAsync_ptsz": {
            "percent": 0,
            "injectionType": 2,
            "substituteReturnCode": 999,
            "interceptionCount": 1
        }
    }
}
```

| Top level configuration | Type | Description |
|-------------------------|------|-------------|
| logLevel | Number | Set numeric [log level](https://github.com/gabime/spdlog/blob/d546201f127c306ec8a0082d57562a05a049af77/include/spdlog/common.h#L198-L204) |
| dynamic | Boolean | Whether to re-apply config on config-file modification for interactive use |
| cudaDriverFaults | Object | Maps a string Driver function name, [CUPTI Driver callback id](https://gitlab.com/nvidia/headers/cuda-individual/cupti/-/blob/cuda-11.5.1/cupti_driver_cbid.h#L9), "*" to a fault injection config |
| cudaRuntimeFaults | Object | Maps a string Runtime function name, [CUPTI Runtime callback id](https://gitlab.com/nvidia/headers/cuda-individual/cupti/-/blob/cuda-11.5.1/cupti_driver_cbid.h#L9), "*" to a fault injection config |


| Fault injection configuration | Type | Description |
|-------------------------------|------|-------------|
| injectionType | Number | Numeric value of FaultInjectionType: 0 (PTX trap), 1 (device assert), 2 (replace CUDA return code by configured `substituteReturnCode`) |
| interceptionCount | Number | How many consecutive matched callbacks should be sampled for fault injection |
| percent | Number | Probability in percent whether a failure is injected |

