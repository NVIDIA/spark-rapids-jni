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

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>

#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {

static constexpr std::size_t page_size_ = 4096;

void pretouch_parallel(void* base, std::size_t bytes, int threads)
{
  int const n = std::max(1, threads);
  std::vector<std::thread> ts;
  ts.reserve(n);
  std::size_t per = (bytes + n - 1) / static_cast<std::size_t>(n);
  per             = (per + page_size_ - 1) & ~(page_size_ - 1);
  for (int i = 0; i < n; ++i) {
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

/**
 * @brief Simple coalescing best-fit pool over a single malloc'd backing buffer.
 *
 * RMM's pool_memory_resource now requires device-accessible upstream resources, so
 * we can't use it for pageable host memory. This pool pre-touches all backing pages
 * at construction time (amortizing first-touch page-fault cost) and then provides
 * lock-protected sub-allocation with adjacent-block coalescing on free.
 */
struct pageable_pool {
  void* base;
  std::size_t pool_size;
  std::mutex mtx;
  std::map<void*, std::size_t> free_blocks;   // addr -> size, sorted for coalescing
  std::map<void*, std::size_t> alloc_blocks;  // addr -> size, for tracking on free

  pageable_pool(std::size_t size, int pretouch_threads)
  {
    void* ptr = nullptr;
    int rc    = ::posix_memalign(&ptr, rmm::CUDA_ALLOCATION_ALIGNMENT, size);
    if (rc != 0 || ptr == nullptr) {
      RMM_FAIL("posix_memalign failed for pageable pool", rmm::out_of_memory);
    }
    base      = ptr;
    pool_size = size;
    pretouch_parallel(base, pool_size, pretouch_threads);
    free_blocks[base] = pool_size;
  }

  ~pageable_pool() { ::free(base); }

  void* try_allocate(std::size_t bytes)
  {
    bytes = (bytes + rmm::CUDA_ALLOCATION_ALIGNMENT - 1) & ~(rmm::CUDA_ALLOCATION_ALIGNMENT - 1);
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
      if (it->second < bytes) continue;
      void* ptr             = it->first;
      std::size_t remaining = it->second - bytes;
      free_blocks.erase(it);
      if (remaining > 0) { free_blocks[static_cast<char*>(ptr) + bytes] = remaining; }
      alloc_blocks[ptr] = bytes;
      return ptr;
    }
    return nullptr;
  }

  void free(void* ptr, std::size_t size)
  {
    std::size_t aligned =
      (size + rmm::CUDA_ALLOCATION_ALIGNMENT - 1) & ~(rmm::CUDA_ALLOCATION_ALIGNMENT - 1);
    std::lock_guard<std::mutex> lock(mtx);
    alloc_blocks.erase(ptr);
    auto [it, _] = free_blocks.emplace(ptr, aligned);
    // coalesce with next
    auto next = std::next(it);
    if (next != free_blocks.end() && static_cast<char*>(it->first) + it->second == next->first) {
      it->second += next->second;
      free_blocks.erase(next);
    }
    // coalesce with prev
    if (it != free_blocks.begin()) {
      auto prev = std::prev(it);
      if (static_cast<char*>(prev->first) + prev->second == it->first) {
        prev->second += it->second;
        free_blocks.erase(it);
      }
    }
  }
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_newPageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_size, jlong /* max_size unused */, jint pretouch_threads)
{
  JNI_TRY
  {
    auto* pool =
      new pageable_pool(static_cast<std::size_t>(pool_size), static_cast<int>(pretouch_threads));
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_releasePageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_ptr)
{
  JNI_TRY { delete reinterpret_cast<pageable_pool*>(pool_ptr); }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_allocFromPageablePool(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong pool_ptr,
                                                                                     jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<pageable_pool*>(pool_ptr);
    void* ret  = pool->try_allocate(static_cast<std::size_t>(size));
    return ret == nullptr ? -1L : reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH(env, -1);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_freeFromPageablePool(
  JNIEnv* env, jclass, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<pageable_pool*>(pool_ptr);
    void* cptr = reinterpret_cast<void*>(ptr);
    pool->free(cptr, static_cast<std::size_t>(size));
  }
  JNI_CATCH(env, );
}

}  // extern "C"
