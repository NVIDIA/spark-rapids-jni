/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"

#include <rmm/mr/pool_memory_resource.hpp>

#include <cerrno>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {
/**
 * @brief Host memory resource that allocates pageable memory via posix_memalign and
 * pre-touches each page in parallel to amortize first-touch page-fault cost at
 * allocation time rather than at first DtoH copy.
 *
 * Intended to be wrapped in `rmm::mr::pool_memory_resource` so the upstream alloc
 * (and its pre-touch work) happens at pool growth time, with cheap sub-allocation
 * from there on. Designed as a graceful fallback when the pinned pool is exhausted:
 * a pre-touched pageable destination is fault-free and reaches ~85% of pinned DtoH
 * bandwidth on this hardware, vs ~8% for fresh malloc'd pageable.
 */
class pretouched_pageable_host_memory_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit pretouched_pageable_host_memory_resource(int pretouch_threads)
    : pretouch_threads_{pretouch_threads}
  {
  }

  pretouched_pageable_host_memory_resource(pretouched_pageable_host_memory_resource const&) =
    delete;
  pretouched_pageable_host_memory_resource& operator=(
    pretouched_pageable_host_memory_resource const&) = delete;

 private:
  static constexpr std::size_t page_size_ = 4096;
  static constexpr std::size_t alignment_ = rmm::CUDA_ALLOCATION_ALIGNMENT;
  int pretouch_threads_;

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override
  {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    int rc    = ::posix_memalign(&ptr, alignment_, bytes);
    if (rc != 0 || ptr == nullptr) { RMM_FAIL("posix_memalign failed", rmm::out_of_memory); }
    pretouch_parallel(ptr, bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) noexcept override
  {
    ::free(ptr);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<pretouched_pageable_host_memory_resource const*>(&other) != nullptr;
  }

  void pretouch_parallel(void* base, std::size_t bytes)
  {
    int const threads = std::max(1, pretouch_threads_);
    std::vector<std::thread> ts;
    ts.reserve(threads);
    std::size_t per = (bytes + threads - 1) / static_cast<std::size_t>(threads);
    per             = (per + page_size_ - 1) & ~(page_size_ - 1);
    for (int i = 0; i < threads; ++i) {
      std::size_t off = static_cast<std::size_t>(i) * per;
      if (off >= bytes) break;
      std::size_t end = std::min(off + per, bytes);
      ts.emplace_back([base, off, end]() {
        auto* c = static_cast<volatile char*>(base);
        for (std::size_t k = off; k < end; k += page_size_)
          c[k] = 0;
      });
    }
    for (auto& t : ts)
      t.join();
  }

  friend void get_property(pretouched_pageable_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }
};
}  // namespace

using rmm_pageable_pool_t = rmm::mr::pool_memory_resource<pretouched_pageable_host_memory_resource>;

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_newPageablePoolMemoryResource(
  JNIEnv* env, jclass clazz, jlong init, jlong max, jint pretouch_threads)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = new rmm_pageable_pool_t(
      new pretouched_pageable_host_memory_resource(static_cast<int>(pretouch_threads)), init, max);
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_releasePageablePoolMemoryResource(
  JNIEnv* env, jclass clazz, jlong pool_ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<rmm_pageable_pool_t*>(pool_ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_allocFromPageablePool(JNIEnv* env,
                                                                                     jclass clazz,
                                                                                     jlong pool_ptr,
                                                                                     jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = reinterpret_cast<rmm_pageable_pool_t*>(pool_ptr);
    void* ret = pool->allocate(rmm::cuda_stream_view{}, size);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH_BEGIN(env, 0)
  catch (...) { return -1; }
  // -1 indicates failure; 0 indicates success but null ptr (e.g. 0-byte alloc).
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_freeFromPageablePool(
  JNIEnv* env, jclass clazz, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool  = reinterpret_cast<rmm_pageable_pool_t*>(pool_ptr);
    void* cptr = reinterpret_cast<void*>(ptr);
    pool->deallocate(rmm::cuda_stream_view{}, cptr, size);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
