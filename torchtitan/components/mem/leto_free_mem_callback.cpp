#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace leto {

static std::mutex g_mu;
static int g_threshold_mb = 0;
static py::object g_kill_cb;  // default-constructed = null PyObject*
// Number of times we've passed the threshold check and invoked the
// kill callback. Lets Python observe whether a fire happened (e.g. the
// OOM-test loop disables the safeguard once it sees the first fire).
static std::atomic<unsigned int> g_num_kill_standby_called{0};

// NVML state. nvmlInit_v2 is reference-counted in libnvidia-ml, so it
// safely coexists with any other NVML initialization in the process
// (e.g. pynvml from leto's progressive init).
static std::once_flag g_nvml_init_once;
static nvmlReturn_t g_nvml_init_status = NVML_ERROR_UNINITIALIZED;
static std::mutex g_nvml_handle_mu;
static nvmlDevice_t g_nvml_handle{};
static bool g_nvml_handle_ready = false;

static void init_nvml_once() {
  std::call_once(g_nvml_init_once, []() {
    g_nvml_init_status = nvmlInit_v2();
    if (g_nvml_init_status != NVML_SUCCESS) {
      std::fprintf(stderr, "[leto] nvmlInit_v2 failed: %s\n",
                   nvmlErrorString(g_nvml_init_status));
    }
  });
}

static bool resolve_nvml_handle(nvmlDevice_t* out) {
  {
    std::lock_guard<std::mutex> lk(g_nvml_handle_mu);
    if (g_nvml_handle_ready) {
      *out = g_nvml_handle;
      return true;
    }
  }
  // Map the current CUDA device to its NVML handle via PCI bus ID.
  // PCI is the stable identity even when CUDA_VISIBLE_DEVICES has
  // remapped the CUDA-side enumeration.
  int cuda_dev = 0;
  cudaError_t cerr = cudaGetDevice(&cuda_dev);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "[leto] cudaGetDevice failed: %s\n",
                 cudaGetErrorString(cerr));
    return false;
  }
  cudaDeviceProp prop;
  cerr = cudaGetDeviceProperties(&prop, cuda_dev);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "[leto] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(cerr));
    return false;
  }
  char pci[64];
  std::snprintf(pci, sizeof(pci), "%08X:%02X:%02X.0",
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
  nvmlDevice_t handle;
  nvmlReturn_t nerr = nvmlDeviceGetHandleByPciBusId_v2(pci, &handle);
  if (nerr != NVML_SUCCESS) {
    std::fprintf(stderr,
                 "[leto] nvmlDeviceGetHandleByPciBusId_v2(%s) failed: %s\n",
                 pci, nvmlErrorString(nerr));
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(g_nvml_handle_mu);
    g_nvml_handle = handle;
    g_nvml_handle_ready = true;
  }
  *out = handle;
  return true;
}

static bool get_free_mb_nvml(int* free_mb_out, int* total_mb_out) {
  init_nvml_once();
  if (g_nvml_init_status != NVML_SUCCESS) return false;
  nvmlDevice_t handle;
  if (!resolve_nvml_handle(&handle)) return false;
  // Use nvmlDeviceGetMemoryInfo_v2 when available (NVML API >= 12, i.e.
  // CUDA 11.6+). v2 splits used / free / reserved into three buckets;
  // computing total - used gives free + reserved, which catches a
  // teardown that releases memory into `reserved` rather than `free`.
  // Fall back to the v1 API on older NVML (e.g. mew1's system-package
  // /usr/include/nvml.h ships NVML_API_VERSION 11, where only
  // nvmlMemory_t / nvmlDeviceGetMemoryInfo exist). In v1 the relation is
  // total = used + free, so total - used == free; the teardown-shift
  // case isn't observable, but the OOM-threshold trigger is correct.
#if defined(NVML_API_VERSION) && NVML_API_VERSION >= 12
  nvmlMemory_v2_t mem;
  mem.version = nvmlMemory_v2;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo_v2(handle, &mem);
  const char* nvml_api_name = "nvmlDeviceGetMemoryInfo_v2";
#else
  nvmlMemory_t mem;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo(handle, &mem);
  const char* nvml_api_name = "nvmlDeviceGetMemoryInfo";
#endif
  if (nerr != NVML_SUCCESS) {
    std::fprintf(stderr, "[leto] %s failed: %s\n",
                 nvml_api_name, nvmlErrorString(nerr));
    return false;
  }
  unsigned long long free_b = (mem.total > mem.used)
      ? (mem.total - mem.used)
      : 0ULL;
  *free_mb_out = static_cast<int>(free_b / (1024ULL * 1024ULL));
  if (total_mb_out != nullptr) {
    *total_mb_out = static_cast<int>(mem.total / (1024ULL * 1024ULL));
  }
  return true;
}

struct LetoFreeMemCallback final : public c10::FreeMemoryCallback {
  bool Execute() override {
    // Hot path: cheap per-context query via the CUDA runtime. NVML is
    // a driver IPC and ~5-50x more expensive, so we keep it out of
    // the every-cache-miss path and only consult it once we've already
    // decided to fire.
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
      std::fprintf(stderr,
                   "[leto] cudaMemGetInfo failed: %s\n",
                   cudaGetErrorString(err));
      return false;
    }
    int free_mb_cuda = static_cast<int>(free_b / (1024ULL * 1024ULL));
    int total_mb_cuda = static_cast<int>(total_b / (1024ULL * 1024ULL));

    int threshold;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      threshold = g_threshold_mb;
    }
    if (threshold <= 0 || free_mb_cuda >= threshold) {
      return false;
    }
    g_num_kill_standby_called.fetch_add(1, std::memory_order_relaxed);

    std::fprintf(stderr,
             "[leto] Kill standby. free=%d MiB / total=%d MiB (cuda), threshold=%d\n",
             free_mb_cuda, total_mb_cuda, threshold);
    // Below threshold — invoke the registered Python kill callback under
    // the GIL. Synchronous: when the call returns, the standby's
    // worker-controller-side kill has been awaited; the kernel may still
    // be tearing down the dead context, which is what the NVML poll
    // below verifies.
    //
    // The callback returns (memory_freed, killed_pid):
    //   memory_freed: did any standby kill happen in this OOM episode
    //   killed_pid:   pid of the standby training proc on this rank's
    //                 GPU (0 if no specific pid is known — e.g. the
    //                 standby was mid-startup and that rank had not yet
    //                 registered when the kill landed).
    bool memory_freed = false;
    unsigned int killed_pid = 0;
    {
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
        py::tuple t = py::cast<py::tuple>(ret);
        memory_freed = py::cast<bool>(t[0]);
        killed_pid = py::cast<unsigned int>(t[1]);
      } catch (py::error_already_set& e) {
        e.discard_as_unraisable("leto_free_mem_callback");
        return false;
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[leto] kill callback threw: %s\n", e.what());
        return false;
      }
    }

    if (!memory_freed) {
      return false;
    }
    if (killed_pid == 0) {
      // Kill happened but no specific pid to wait for (standby's matching
      // rank was not registered when the kill landed). Trust the
      // callback and let PyTorch retry the allocation.
      std::fprintf(stderr,
                   "[leto] kill confirmed; no pid to track\n");
      return true;
    }

    // Poll NVML compute-running processes every 100ms for up to 10s,
    // waiting for `killed_pid` to disappear from this GPU. Mirrors the
    // enumeration used by PyTorch's
    // CUDACachingAllocator::reportProcessMemoryInfo.
    nvmlDevice_t handle;
    if (!resolve_nvml_handle(&handle)) {
      return false;
    }
    std::fprintf(stderr,
                 "[leto] waiting for killed pid=%u to leave GPU\n",
                 killed_pid);
    auto poll_start = std::chrono::steady_clock::now();
    auto deadline = poll_start + std::chrono::seconds(10);
    // Header maps the unversioned call to nvmlDeviceGetComputeRunningProcesses_v3,
    // which takes nvmlProcessInfo_t (== _v2_t). pid lives at the same offset in
    // every version, so the v3 struct is fine for our purposes.
    std::vector<nvmlProcessInfo_t> procs(8);
    while (std::chrono::steady_clock::now() < deadline) {
      unsigned int count = static_cast<unsigned int>(procs.size());
      nvmlReturn_t r;
      while ((r = nvmlDeviceGetComputeRunningProcesses(
                  handle, &count, procs.data())) ==
             NVML_ERROR_INSUFFICIENT_SIZE) {
        procs.resize(count);
      }
      if (r != NVML_SUCCESS) {
        std::fprintf(stderr,
                     "[leto] nvmlDeviceGetComputeRunningProcesses failed: %s\n",
                     nvmlErrorString(r));
        return false;
      }
      bool still_present = false;
      for (unsigned int i = 0; i < count; ++i) {
        if (procs[i].pid == killed_pid) {
          still_present = true;
          break;
        }
      }
      if (!still_present) {
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - poll_start)
                .count();
        std::fprintf(stderr,
                     "[leto] killed pid=%u gone from GPU; kill confirmed "
                     "after %lldms\n",
                     killed_pid, static_cast<long long>(elapsed_ms));
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - poll_start)
            .count();
    std::fprintf(stderr,
                 "[leto] poll timeout after %lldms; pid=%u still present "
                 "on GPU\n",
                 static_cast<long long>(elapsed_ms), killed_pid);
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

  m.def("get_num_kill_standby_called", []() {
    return leto::g_num_kill_standby_called.load(std::memory_order_relaxed);
  });

  m.def("reset_num_kill_standby_called", []() {
    leto::g_num_kill_standby_called.store(0, std::memory_order_relaxed);
  });
}
