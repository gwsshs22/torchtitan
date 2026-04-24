#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace leto {

struct LetoFreeMemCallback final : public c10::FreeMemoryCallback {
  bool Execute() override {
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
      std::fprintf(stderr,
                   "[leto] cudaMemGetInfo failed: %s\n",
                   cudaGetErrorString(err));
    } else {
      std::fprintf(stderr,
                   "[leto] free=%.2f MiB / total=%.2f MiB\n",
                   static_cast<double>(free_b) / (1024.0 * 1024.0),
                   static_cast<double>(total_b) / (1024.0 * 1024.0));
    }
    return false;
  }
};

}  // namespace leto

namespace c10 {
namespace {
REGISTER_FREE_MEMORY_CALLBACK(leto_free_mem_callback,
                              ::leto::LetoFreeMemCallback)
}
}  // namespace c10
