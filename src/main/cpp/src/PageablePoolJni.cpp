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
#include <rmm/mr/detail/coalescing_free_list.hpp>

#include <cstdlib>
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
  // Divide work ceiling-fashion so every byte is covered, then round up to a page boundary
  // so each thread's range starts/ends on a page — preventing two threads from touching the same
  // page.
  std::size_t per = (bytes + n - 1) / static_cast<std::size_t>(n);
  per             = (per + page_size_ - 1) & ~(page_size_ - 1);
  for (int i = 0; i < n; ++i) {
    std::size_t off = static_cast<std::size_t>(i) * per;
    if (off >= bytes) break;
    std::size_t end = std::min(off + per, bytes);
    ts.emplace_back([base, off, end]() {
      // volatile prevents the compiler from eliding these as dead stores — the write's
      // only purpose is to fault in the page, not to store a meaningful value.
      auto* c = static_cast<volatile char*>(base);
      for (std::size_t k = off; k < end; k += page_size_)
        c[k] = 0;
    });
  }
  for (auto& t : ts)
    t.join();
}

using free_list_t = rmm::mr::detail::coalescing_free_list;
using block_t     = rmm::mr::detail::block;

inline std::size_t align_up(std::size_t bytes)
{
  return (bytes + rmm::CUDA_ALLOCATION_ALIGNMENT - 1) & ~(rmm::CUDA_ALLOCATION_ALIGNMENT - 1);
}

/**
 * @brief Coalescing best-fit pool over a single posix_memalign'd backing buffer.
 *
 * RMM's pool_memory_resource requires a device-accessible upstream resource (i.e. one whose
 * pointers GPU kernels can dereference directly). Pageable malloc memory does not satisfy that
 * property on PCIe systems — it is only usable as a cudaMemcpyAsync destination via CUDA's
 * internal staging mechanism, not for in-kernel pointer access. We therefore cannot use
 * pool_memory_resource here and instead build our own pool using RMM's coalescing_free_list
 * (rmm/mr/detail/coalescing_free_list.hpp), the same free-list implementation that
 * pool_memory_resource uses internally.
 *
 * All blocks within the single backing buffer are marked is_head=false so that coalescing_free_list
 * treats them as contiguous sub-allocations that may be merged freely.
 *
 * Pre-touching all backing pages at construction time amortizes first-touch page-fault cost,
 * allowing DtoH copies into pool-allocated buffers to reach near-pinned bandwidth rather than
 * incurring per-page faults at copy time.
 */
struct pageable_pool {
  void* base;
  std::size_t pool_size;
  std::mutex mtx;
  free_list_t free_list;

  /**
   * @brief Construct the pool: allocate the backing buffer, pre-touch all pages in parallel,
   *        then register the entire region as a single free block.
   *
   * @throws rmm::out_of_memory if the backing allocation fails.
   *
   * @param size             total pool size in bytes
   * @param pretouch_threads number of threads used to fault in backing pages at construction time
   */
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
    // is_head=false: all sub-blocks are within one contiguous upstream allocation and may coalesce.
    free_list.insert(block_t{static_cast<char*>(base), pool_size, false});
  }

  /**
   * @brief Destroy the pool, releasing the backing allocation.
   *
   * Any outstanding allocations become dangling after this call.
   */
  ~pageable_pool() { ::free(base); }

  /**
   * @brief Try to sub-allocate @p bytes from the pool using a best-fit search.
   *
   * The request is rounded up to rmm::CUDA_ALLOCATION_ALIGNMENT. If the chosen block is larger
   * than needed, the remainder is returned to the free list.
   *
   * @param bytes requested allocation size in bytes
   * @return pointer to the allocated region, or nullptr if no free block is large enough
   */
  void* try_allocate(std::size_t bytes)
  {
    bytes = align_up(bytes);
    std::lock_guard<std::mutex> lock(mtx);
    block_t blk = free_list.get_block(bytes);
    if (!blk.is_valid()) { return nullptr; }
    if (blk.size() > bytes) {
      free_list.insert(block_t{blk.pointer() + bytes, blk.size() - bytes, false});
    }
    return blk.pointer();
  }

  /**
   * @brief Return a previously allocated block to the pool, coalescing with adjacent free blocks.
   *
   * @param ptr  pointer returned by a prior call to try_allocate
   * @param size size passed to that try_allocate call (will be rounded up to alignment)
   */
  void free(void* ptr, std::size_t size)
  {
    std::lock_guard<std::mutex> lock(mtx);
    free_list.insert(block_t{static_cast<char*>(ptr), align_up(size), false});
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
  JNI_CATCH_BEGIN(env, 0)
  catch (...) { return -1; }  // Catch and suppress all exceptions.
  // The return value of -1 indicates that the allocation failed.
  // This is different from the return value of 0, which indicates that the allocation succeeded
  // but the returned pointer is null (such cases can be due to allocating 0 bytes).
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
