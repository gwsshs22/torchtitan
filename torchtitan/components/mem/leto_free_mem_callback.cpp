#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace leto {

static std::mutex g_mu;
static int g_threshold_mb = 0;
static py::object g_kill_cb;  // default-constructed = null PyObject*

struct LetoFreeMemCallback final : public c10::FreeMemoryCallback {
  bool Execute() override {
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
      std::fprintf(stderr,
                   "[leto] cudaMemGetInfo failed: %s\n",
                   cudaGetErrorString(err));
      return false;
    }
    int free_mb = static_cast<int>(free_b / (1024 * 1024));
    int total_mb = static_cast<int>(total_b / (1024 * 1024));
    std::fprintf(stderr,
                 "[leto] free=%d MiB / total=%d MiB\n",
                 free_mb, total_mb);

    int threshold;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      threshold = g_threshold_mb;
    }
    if (threshold <= 0 || free_mb >= threshold) {
      return false;
    }

    // Below threshold — invoke the registered Python kill callback under
    // the GIL. Synchronous: when the call returns, the callback has
    // finished (e.g. issued a blocking gRPC and waited for the standby's
    // GPU memory to be reclaimed).
    py::gil_scoped_acquire gil;
    py::object cb;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      if (!g_kill_cb || g_kill_cb.is_none()) {
        return false;
      }
      cb = g_kill_cb;  // refcount bump (safe: GIL held)
    }
    try {
      py::object ret = cb();
      return py::cast<bool>(ret);
    } catch (py::error_already_set& e) {
      e.discard_as_unraisable("leto_free_mem_callback");
      return false;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[leto] kill callback threw: %s\n", e.what());
      return false;
    }
  }
};

}  // namespace leto

namespace c10 {
namespace {
REGISTER_FREE_MEMORY_CALLBACK(leto_free_mem_callback,
                              ::leto::LetoFreeMemCallback)
}
}  // namespace c10

PYBIND11_MODULE(leto_free_mem_callback, m) {
  m.doc() = "Leto free-memory callback. Invokes a registered Python callable "
            "synchronously when free GPU MiB falls below a configured "
            "threshold during CUDACachingAllocator cache-miss expansion.";

  m.def(
      "set_threshold_mb",
      [](int mb) {
        std::lock_guard<std::mutex> lk(leto::g_mu);
        leto::g_threshold_mb = mb;
      },
      py::arg("mb"));

  m.def(
      "set_kill_callback",
      [](py::object cb) {
        std::lock_guard<std::mutex> lk(leto::g_mu);
        leto::g_kill_cb = std::move(cb);
      },
      py::arg("cb"));

  m.def("clear_kill_callback", []() {
    std::lock_guard<std::mutex> lk(leto::g_mu);
    leto::g_kill_cb = py::object();  // null; safer at static teardown
  });

  m.def("get_threshold_mb", []() {
    std::lock_guard<std::mutex> lk(leto::g_mu);
    return leto::g_threshold_mb;
  });
}
