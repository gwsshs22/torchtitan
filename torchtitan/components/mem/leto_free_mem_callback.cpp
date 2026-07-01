#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/Allocator.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
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

// Request-size tracking, fed by LetoRecordingAllocator (defined below).
// The thread-local holds the size of the allocation currently in flight
// on *this* thread; because the FreeMemoryCallback fires synchronously
// deeper in the same malloc, Execute() reads the exact pending request
// off it. The global atomic mirrors the most recent request across all
// threads for observability (logging / tests) only -- it is never used
// for the kill decision, which must be per-thread to stay race-free.
thread_local size_t g_request_bytes = 0;
static std::atomic<size_t> g_last_request_bytes{0};

static inline void record_request(size_t n) {
  g_request_bytes = n;
  g_last_request_bytes.store(n, std::memory_order_relaxed);
}

// Request-aware reclaim criterion (opt-in; off preserves legacy behavior).
// When enabled, the callback fires iff free GPU MiB is below the pending
// request MiB plus g_request_margin_mb of headroom, instead of the legacy
// absolute floor `free < g_threshold_mb`. Both guarded by g_mu.
static bool g_request_aware = false;
static int g_request_margin_mb = 0;

// ---------------------------------------------------------------------------
// Reservation broker — standby memory reservation against the active allocator
// ---------------------------------------------------------------------------
//
// The active process is the single broker for any GPU memory the co-located
// standby may use. A standby posts a Reserve(bytes) request into a per-rank
// shared-memory ledger; the broker thread (here, in the active process)
// grants iff the device can spare it, accounting the standby at its
// *reservation* rather than its current usage. The FreeMemCallback then
// reclaims (kills) the standby before the active grows into reserved memory.
// See the plan: the no-OOM guarantee factors into
//   (A) active enforces:  others_used + granted <= capacity - margin
//   (B) protocol asserts:  standby_actual <= granted
// All memory is read from NVML (driver ground truth), never cudaMemGetInfo.

// Shared-memory ledger layout. Fixed, 8-byte-aligned; the Python standby side
// mmaps the same file with a matching struct format (see progressive.py).
// Single-producer-per-field + a seq handshake make a process-shared mutex
// unnecessary on x86-64 (naturally-aligned loads/stores are atomic; we add
// acquire/release fences via __atomic builtins for ordering + portability).
constexpr uint32_t RESV_LEDGER_MAGIC = 0x4C54524Eu;  // "LTRN"
constexpr int RESV_VERDICT_PENDING = -1;
constexpr int RESV_VERDICT_DENY = 0;
constexpr int RESV_VERDICT_GRANT = 1;

struct ReservationLedger {
  uint32_t magic;          // 0  : RESV_LEDGER_MAGIC once initialized
  int32_t standby_pid;     // 4  : standby-written; NVML proc to exclude (0=none)
  uint32_t standby_epoch;  // 8  : standby-written; bumped per new standby
  uint32_t req_seq;        // 12 : standby-written; bumped *last* to publish a req
  int64_t req_bytes;       // 16 : standby-written; requested reservation
  uint32_t resp_seq;       // 24 : broker-written; bumped *last*; == req_seq when done
  int32_t resp_verdict;    // 28 : broker-written; GRANT/DENY
  int64_t granted;         // 32 : broker-published cumulative grant (observability)
  int64_t effective_free;  // 40 : broker-published effective_free bytes (observability)
  int64_t standby_actual;  // 48 : broker-published NVML per-pid usage (observability)
  int64_t pad;             // 56
};
static_assert(sizeof(ReservationLedger) == 64, "ledger layout/size mismatch");

template <typename T>
static inline T resv_load_acq(const T* p) {
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
template <typename T>
static inline void resv_store_rel(T* p, T v) {
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

// Broker state. `g_broker_mutex` serializes a grant decision against the
// active process's own allocations (the lock-serialized rigor that closes the
// grant/alloc race). `g_granted` is the cumulative grant, kept here (not in
// shm); atomic so the FreeMemCallback can read it lock-free post-init.
static std::mutex g_broker_mutex;
static std::atomic<bool> g_reservation_active{false};  // gates per-alloc lock
static std::atomic<int64_t> g_granted{0};
static std::atomic<int> g_reservation_margin_mb{0};
static uint32_t g_standby_epoch_seen = 0;  // broker-thread-local; under g_broker_mutex

// NVML snapshot refreshed by the broker thread (~every poll), read cheaply by
// the FreeMemCallback (NVML itself is too slow for the per-cache-miss path).
static std::atomic<int64_t> g_cached_free_bytes{0};
static std::atomic<int64_t> g_cached_standby_actual_bytes{0};
static std::atomic<bool> g_cached_valid{false};

// Invariant (B): standby_actual <= granted. A breach is a protocol violation
// (a task over-allocated past its profiled delta, or skipped reserving); the
// accounting clamps to stay safe, and we count breaches for tests/observability.
constexpr int64_t RESV_ASSERT_TOL_BYTES = int64_t{256} * 1024 * 1024;
static std::atomic<uint64_t> g_assert_b_violations{0};

static ReservationLedger* g_ledger = nullptr;  // mmap'd; never unmapped
// Heap-allocated and leaked rather than a static std::thread object, so a
// process that exits without calling stop_broker() doesn't std::terminate on
// a still-joinable thread during static destruction.
static std::thread* g_broker_thread = nullptr;
static std::atomic<bool> g_broker_running{false};
static int g_broker_device = 0;

// Acquire g_broker_mutex around the delegated allocation iff a reservation
// phase is in progress, so no active allocation is in flight during a grant.
// `locked_` is captured at construction so the dtor unlock matches the ctor
// lock even if the flag flips mid-allocation.
struct MaybeBrokerLock {
  bool locked_;
  MaybeBrokerLock()
      : locked_(g_reservation_active.load(std::memory_order_acquire)) {
    if (locked_) g_broker_mutex.lock();
  }
  ~MaybeBrokerLock() {
    if (locked_) g_broker_mutex.unlock();
  }
  MaybeBrokerLock(const MaybeBrokerLock&) = delete;
  MaybeBrokerLock& operator=(const MaybeBrokerLock&) = delete;
};

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

// Read device free bytes (NVML, allocatable `free` bucket) and the GPU memory
// used by `target_pid` (0 if absent or NVML can't attribute per-process
// memory). NVML, not cudaMemGetInfo — the driver view is ground truth.
static bool read_nvml_mem(nvmlDevice_t handle, int target_pid,
                          int64_t* free_b_out, int64_t* pid_used_b_out) {
#if defined(NVML_API_VERSION) && NVML_API_VERSION >= 12
  nvmlMemory_v2_t mem;
  mem.version = nvmlMemory_v2;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo_v2(handle, &mem);
#else
  nvmlMemory_t mem;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo(handle, &mem);
#endif
  if (nerr != NVML_SUCCESS) {
    std::fprintf(stderr, "[leto] broker nvmlDeviceGetMemoryInfo failed: %s\n",
                 nvmlErrorString(nerr));
    return false;
  }
  const int64_t total_b = static_cast<int64_t>(mem.total);
  *free_b_out = static_cast<int64_t>(mem.free);

  int64_t pid_used = 0;
  if (target_pid > 0) {
    std::vector<nvmlProcessInfo_t> procs(16);
    unsigned int count = static_cast<unsigned int>(procs.size());
    nvmlReturn_t r;
    while ((r = nvmlDeviceGetComputeRunningProcesses(
                handle, &count, procs.data())) ==
           NVML_ERROR_INSUFFICIENT_SIZE) {
      procs.resize(count);
    }
    if (r != NVML_SUCCESS) {
      std::fprintf(stderr,
                   "[leto] broker nvmlDeviceGetComputeRunningProcesses "
                   "failed: %s\n",
                   nvmlErrorString(r));
      return false;
    }
    for (unsigned int i = 0; i < count; ++i) {
      if (static_cast<int>(procs[i].pid) == target_pid) {
        unsigned long long u = procs[i].usedGpuMemory;
        // NVML reports (ull)-1 when per-process memory is unavailable; treat
        // as 0 (conservative — don't credit it back into effective_free).
        if (u != static_cast<unsigned long long>(-1) &&
            static_cast<int64_t>(u) <= total_b) {
          pid_used = static_cast<int64_t>(u);
        }
        break;
      }
    }
  }
  *pid_used_b_out = pid_used;
  return true;
}

// Broker thread (active process). Refreshes the NVML snapshot read by the
// FreeMemCallback hot path, resets the grant ledger on a new standby epoch,
// and services reservation requests under g_broker_mutex (so a grant is
// serialized against the active's own allocations).
static void broker_loop() {
  // The broker thread needs the active rank's CUDA device so NVML handle
  // resolution (via PCI bus id) targets the right GPU.
  cudaSetDevice(g_broker_device);
  nvmlDevice_t handle{};
  while (g_broker_running.load(std::memory_order_acquire)) {
    if (resolve_nvml_handle(&handle)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  uint32_t last_req_seq = 0;
  while (g_broker_running.load(std::memory_order_acquire)) {
    ReservationLedger* L = g_ledger;
    if (L == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    const int target_pid = resv_load_acq(&L->standby_pid);
    const uint32_t epoch = resv_load_acq(&L->standby_epoch);

    // 1) Refresh the NVML snapshot for the cheap reclaim path.
    int64_t free_b = 0, pid_used_b = 0;
    const bool ok = read_nvml_mem(handle, target_pid, &free_b, &pid_used_b);
    if (ok) {
      g_cached_free_bytes.store(free_b, std::memory_order_relaxed);
      g_cached_standby_actual_bytes.store(pid_used_b, std::memory_order_relaxed);
      g_cached_valid.store(true, std::memory_order_release);
    }

    // 2) A fresh standby instance (new epoch) voids any prior grant.
    if (epoch != g_standby_epoch_seen) {
      std::lock_guard<std::mutex> lk(g_broker_mutex);
      g_granted.store(0, std::memory_order_relaxed);
      g_standby_epoch_seen = epoch;
    }

    // 3) Service a pending reservation request. `req_bytes` is the standby's
    // *cumulative* target footprint (not an increment), so a re-request after
    // a peer-denied task is idempotent — we only raise `granted` toward the
    // target and never double-count.
    const uint32_t req_seq = resv_load_acq(&L->req_seq);
    if (ok && req_seq != last_req_seq) {
      const int64_t req_bytes = resv_load_acq(&L->req_bytes);
      int verdict = RESV_VERDICT_DENY;
      int64_t granted_after = 0, eff_free = 0;
      {
        std::lock_guard<std::mutex> lk(g_broker_mutex);
        // Read NVML fresh under the lock so the grant is consistent with the
        // active's allocations (which also take g_broker_mutex).
        int64_t free_now = 0, pid_now = 0;
        if (read_nvml_mem(handle, target_pid, &free_now, &pid_now)) {
          int64_t granted_cur = g_granted.load(std::memory_order_relaxed);
          const int64_t protected_b = std::max<int64_t>(0, granted_cur - pid_now);
          eff_free = free_now - protected_b;
          const int64_t margin_b =
              static_cast<int64_t>(
                  g_reservation_margin_mb.load(std::memory_order_relaxed)) *
              1024 * 1024;
          const int64_t needed = req_bytes - granted_cur;
          if (needed <= 0) {
            verdict = RESV_VERDICT_GRANT;  // already covered; no change
          } else if (eff_free - margin_b >= needed) {
            granted_cur = req_bytes;  // raise grant to the cumulative target
            g_granted.store(granted_cur, std::memory_order_relaxed);
            verdict = RESV_VERDICT_GRANT;
          }
          granted_after = granted_cur;
          resv_store_rel(&L->effective_free, eff_free);
          resv_store_rel(&L->standby_actual, pid_now);
        }
        resv_store_rel(&L->granted, granted_after);
      }
      resv_store_rel(&L->resp_verdict, verdict);
      resv_store_rel(&L->resp_seq, req_seq);  // bump last → signals done
      last_req_seq = req_seq;
      std::fprintf(
          stderr,
          "[leto] broker req_seq=%u bytes=%lld -> %s (granted=%lld, "
          "eff_free=%lld)\n",
          req_seq, static_cast<long long>(req_bytes),
          verdict == RESV_VERDICT_GRANT ? "GRANT" : "DENY",
          static_cast<long long>(granted_after),
          static_cast<long long>(eff_free));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
  }
}

// mmap the per-rank shm ledger. Captures the active rank's CUDA device and
// primes the NVML handle on this (correctly-deviced) thread.
static bool attach_reservation_ledger(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0) {
    std::fprintf(stderr,
                 "[leto] attach_reservation_ledger open(%s) failed: %s\n",
                 path.c_str(), std::strerror(errno));
    return false;
  }
  if (::ftruncate(fd, sizeof(ReservationLedger)) != 0) {
    std::fprintf(stderr, "[leto] attach_reservation_ledger ftruncate failed: %s\n",
                 std::strerror(errno));
    ::close(fd);
    return false;
  }
  void* p = ::mmap(nullptr, sizeof(ReservationLedger), PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {
    std::fprintf(stderr, "[leto] attach_reservation_ledger mmap failed: %s\n",
                 std::strerror(errno));
    return false;
  }
  g_ledger = static_cast<ReservationLedger*>(p);
  resv_store_rel(&g_ledger->magic, RESV_LEDGER_MAGIC);
  cudaGetDevice(&g_broker_device);
  init_nvml_once();
  nvmlDevice_t tmp{};
  resolve_nvml_handle(&tmp);  // prime the handle cache on the right device
  return true;
}

static bool start_broker(int margin_mb) {
  if (g_ledger == nullptr) {
    std::fprintf(stderr, "[leto] start_broker: no ledger attached\n");
    return false;
  }
  if (g_broker_running.load(std::memory_order_acquire)) {
    g_reservation_margin_mb.store(margin_mb, std::memory_order_relaxed);
    return true;  // already running
  }
  g_reservation_margin_mb.store(margin_mb, std::memory_order_relaxed);
  g_broker_running.store(true, std::memory_order_release);
  // Publish the lock gate before the thread can grant. Safe: the caller (main
  // thread) is not concurrently allocating, so no unlocked allocation is in
  // flight to race the first grant.
  g_reservation_active.store(true, std::memory_order_release);
  g_broker_thread = new std::thread(broker_loop);
  return true;
}

static void stop_broker() {
  if (!g_broker_running.load(std::memory_order_acquire)) return;
  g_broker_running.store(false, std::memory_order_release);
  if (g_broker_thread != nullptr && g_broker_thread->joinable()) {
    g_broker_thread->join();
    delete g_broker_thread;
    g_broker_thread = nullptr;
  }
  g_reservation_active.store(false, std::memory_order_release);
}

// Void the cumulative grant (e.g. after the standby is killed/activated).
// Lock-free on purpose: it may be invoked from the kill callback, which runs
// inside the FreeMemCallback while g_broker_mutex is already held by the
// allocate wrapper -- re-locking would deadlock. g_granted is atomic; a blunt
// store is safe (a concurrent grant can only happen for a live standby, and a
// reset only follows the standby's death, so they don't overlap).
static void reset_granted() {
  g_granted.store(0, std::memory_order_relaxed);
  if (g_ledger != nullptr) resv_store_rel(&g_ledger->granted, int64_t{0});
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

    // Decide whether this allocation warrants reclaiming the standby.
    const int request_mb =
        static_cast<int>(g_request_bytes / (1024ULL * 1024ULL));
    bool should_fire = false;
    int margin_mb_used = 0;

    // Read NVML *fresh* for the reclaim decision when in reservation mode.
    // The callback only runs on a cache-miss, so this is not the per-alloc
    // hot path; a fresh read avoids acting on a stale snapshot during a fast
    // active-growth burst (the broker's ~15ms cache can lag multiple GB). Fall
    // back to the cached snapshot only if the fresh NVML read fails.
    bool used_reservation = false;
    if (g_reservation_active.load(std::memory_order_acquire)) {
      const int64_t req_a = static_cast<int64_t>(g_request_bytes);
      const int64_t granted = g_granted.load(std::memory_order_relaxed);
      const int target_pid =
          g_ledger != nullptr ? resv_load_acq(&g_ledger->standby_pid) : 0;
      int64_t nvml_free = 0, standby_actual = 0;
      bool have = false;
      nvmlDevice_t handle{};
      if (resolve_nvml_handle(&handle) &&
          read_nvml_mem(handle, target_pid, &nvml_free, &standby_actual)) {
        have = true;
        g_cached_free_bytes.store(nvml_free, std::memory_order_relaxed);
        g_cached_standby_actual_bytes.store(standby_actual, std::memory_order_relaxed);
        g_cached_valid.store(true, std::memory_order_release);
      } else if (g_cached_valid.load(std::memory_order_acquire)) {
        nvml_free = g_cached_free_bytes.load(std::memory_order_relaxed);
        standby_actual =
            g_cached_standby_actual_bytes.load(std::memory_order_relaxed);
        have = true;
      }
      if (have) {
        used_reservation = true;
        // Account the standby at its reservation, not its current usage:
        // effective_free = nvml_free - max(0, granted - standby_actual).
        // Reclaim iff this allocation would push effective_free below margin.
        // We already hold g_broker_mutex here (taken in the allocate wrapper
        // for the reservation phase), so granted is consistent with grants.
        const int64_t protected_b = std::max<int64_t>(0, granted - standby_actual);
        const int64_t eff_free = nvml_free - protected_b;
        margin_mb_used = g_reservation_margin_mb.load(std::memory_order_relaxed);
        const int64_t margin_b = static_cast<int64_t>(margin_mb_used) * 1024 * 1024;

        // (B) standby_actual <= granted. A breach means the standby physically
        // holds more than it reserved; the max-clamp above accounts it at
        // `actual` so safety holds -- this is a loud, rate-limited diagnostic.
        if (standby_actual > granted + RESV_ASSERT_TOL_BYTES) {
          const uint64_t n =
              g_assert_b_violations.fetch_add(1, std::memory_order_relaxed);
          if ((n & (n + 1)) == 0) {  // print at n = 0,1,3,7,15,...
            std::fprintf(stderr,
                         "[leto] ASSERT(B) VIOLATED: standby_actual=%lld MiB > "
                         "granted=%lld MiB (+tol); accounting at actual\n",
                         static_cast<long long>(standby_actual / (1024 * 1024)),
                         static_cast<long long>(granted / (1024 * 1024)));
          }
        }
        should_fire = (eff_free - req_a < margin_b);
        if (should_fire) {
          std::fprintf(stderr,
                       "[leto] Kill standby (reservation). nvml_free=%lld MiB, "
                       "standby_actual=%lld MiB, granted=%lld MiB, "
                       "eff_free=%lld MiB, request=%d MiB, margin=%d MiB\n",
                       static_cast<long long>(nvml_free / (1024 * 1024)),
                       static_cast<long long>(standby_actual / (1024 * 1024)),
                       static_cast<long long>(granted / (1024 * 1024)),
                       static_cast<long long>(eff_free / (1024 * 1024)),
                       request_mb, margin_mb_used);
        }
      }
    }
    if (!used_reservation) {
      // Legacy / request-aware path (cudaMemGetInfo, no reservation ledger).
      int threshold;
      bool request_aware;
      int request_margin_mb;
      {
        std::lock_guard<std::mutex> lk(g_mu);
        threshold = g_threshold_mb;
        request_aware = g_request_aware;
        request_margin_mb = g_request_margin_mb;
      }
      if (request_aware) {
        should_fire = free_mb_cuda < request_mb + request_margin_mb;
      } else {
        should_fire = threshold > 0 && free_mb_cuda < threshold;
      }
      if (should_fire) {
        std::fprintf(stderr,
                     "[leto] Kill standby. free=%d MiB / total=%d MiB (cuda), "
                     "request=%d MiB, margin=%d MiB, threshold=%d, "
                     "request_aware=%d\n",
                     free_mb_cuda, total_mb_cuda, request_mb, request_margin_mb,
                     threshold, request_aware ? 1 : 0);
      }
    }

    if (!should_fire) {
      return false;
    }
    g_num_kill_standby_called.fetch_add(1, std::memory_order_relaxed);
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
namespace cuda {
namespace CUDACachingAllocator {

// Transparent pass-through allocator. Delegates every operation to the
// real (native) CUDA allocator captured at install time; the *only*
// behavioral change is that the three size-bearing entry points stamp the
// requested byte count via leto::record_request before delegating, so a
// FreeMemoryCallback firing synchronously deeper inside the same malloc
// can read the exact pending request. name() mirrors the wrapped allocator
// so backend-gated features (expandable segments, CUDA graph pools, NCCL
// registration) still observe "native". Every other method forwards
// verbatim -- including the non-pure virtuals the native allocator
// overrides, which would otherwise fall through to base defaults that
// throw "not supported" and silently break MemPool / graph / history APIs.
struct LetoRecordingAllocator final : public CUDAAllocator {
  CUDAAllocator* real_ = nullptr;
  void set_real(CUDAAllocator* r) { real_ = r; }

  using CUDAAllocator::recordStream;  // un-hide the base-class overload set

  // --- size-bearing entry points: record, then delegate ---
  // MaybeBrokerLock serializes the delegated allocation against reservation
  // grants during the reservation phase (no-op otherwise). The
  // FreeMemCallback fires *inside* the delegate, i.e. already under the lock.
  DataPtr allocate(size_t n) override {
    ::leto::record_request(n);
    ::leto::MaybeBrokerLock lk;
    return real_->allocate(n);
  }
  void* raw_alloc(size_t nbytes) override {
    ::leto::record_request(nbytes);
    ::leto::MaybeBrokerLock lk;
    return real_->raw_alloc(nbytes);
  }
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    ::leto::record_request(nbytes);
    ::leto::MaybeBrokerLock lk;
    return real_->raw_alloc_with_stream(nbytes, stream);
  }

  // --- everything else: verbatim forwarding to the real allocator ---
  void raw_delete(void* ptr) override { real_->raw_delete(ptr); }
  void init(int device_count) override { real_->init(device_count); }
  bool initialized() override { return real_->initialized(); }
  double getMemoryFraction(c10::DeviceIndex device) override {
    return real_->getMemoryFraction(device);
  }
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    real_->setMemoryFraction(fraction, device);
  }
  std::vector<StreamSegmentSize> getExpandableSegmentSizes(
      c10::DeviceIndex device) override {
    return real_->getExpandableSegmentSizes(device);
  }
  void enable(bool value) override { real_->enable(value); }
  bool isEnabled() const override { return real_->isEnabled(); }
  void emptyCache(MempoolId_t mempool_id = {0, 0}) override {
    real_->emptyCache(mempool_id);
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
    real_->cacheInfo(device, largestBlock);
  }
  void* getBaseAllocation(void* ptr, size_t* size) override {
    return real_->getBaseAllocation(ptr, size);
  }
  void recordStream(const DataPtr& ptr, CUDAStream stream) override {
    real_->recordStream(ptr, stream);
  }
  CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override {
    return real_->getDeviceStats(device);
  }
  void resetAccumulatedStats(c10::DeviceIndex device) override {
    real_->resetAccumulatedStats(device);
  }
  void resetPeakStats(c10::DeviceIndex device) override {
    real_->resetPeakStats(device);
  }
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
    return real_->getMemoryInfo(device);
  }
  SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) override {
    return real_->snapshot(mempool_id);
  }
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) override {
    real_->beginAllocateToPool(device, mempool_id, std::move(filter));
  }
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    real_->endAllocateToPool(device, mempool_id);
  }
  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    real_->releasePool(device, mempool_id);
  }
  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    return real_->getPoolUseCount(device, mempool_id);
  }
  void createOrIncrefPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::shared_ptr<CUDAAllocator> allocator = nullptr) override {
    real_->createOrIncrefPool(device, mempool_id, std::move(allocator));
  }
  void setUseOnOOM(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      bool use_on_oom) override {
    real_->setUseOnOOM(device, mempool_id, use_on_oom);
  }
  void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    real_->setNoSplit(device, mempool_id);
  }
  bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return real_->checkPoolLiveAllocations(
        device, mempool_id, expected_live_allocations);
  }
  ShareableHandle shareIpcHandle(void* ptr) override {
    return real_->shareIpcHandle(ptr);
  }
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    return real_->getIpcDevPtr(std::move(handle));
  }
  bool isHistoryEnabled() override { return real_->isHistoryEnabled(); }
  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) override {
    real_->recordHistory(
        enabled,
        std::move(context_recorder),
        alloc_trace_max_entries,
        when,
        clearHistory,
        skip_actions);
  }
  void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& md) override {
    real_->recordAnnotation(md);
  }
  void pushCompileContext(std::string& md) override {
    real_->pushCompileContext(md);
  }
  void popCompileContext() override { real_->popCompileContext(); }
  void setUserMetadata(const std::string& metadata) override {
    real_->setUserMetadata(metadata);
  }
  std::string getUserMetadata() override { return real_->getUserMetadata(); }
  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    real_->attachOutOfMemoryObserver(std::move(observer));
  }
  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
    real_->attachAllocatorTraceTracker(std::move(tracker));
  }
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override {
    real_->enablePeerAccess(dev, dev_to_access);
  }
  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override {
    return real_->memcpyAsync(
        dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
  }
  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    return real_->getCheckpointState(device, id);
  }
  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) override {
    return real_->setCheckpointPoolState(device, std::move(pps));
  }
  bool is_simple_data_ptr(const DataPtr& data_ptr) const override {
    return real_->is_simple_data_ptr(data_ptr);
  }
  DeleterFnPtr raw_deleter() const override { return real_->raw_deleter(); }
  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    real_->copy_data(dest, src, count);
  }
  std::string name() override { return real_->name(); }
};

}  // namespace CUDACachingAllocator
}  // namespace cuda
}  // namespace c10

namespace leto {

namespace cca = c10::cuda::CUDACachingAllocator;

// Process-lifetime singleton. Heap-allocated and never freed: it must
// outlive every allocation, including teardown-time frees that run after
// atexit handlers.
static cca::LetoRecordingAllocator* g_recording_allocator = nullptr;

// Install the recording allocator as the current CUDA allocator. Safe to
// call at any time -- even after allocations already exist -- because the
// wrapper delegates to the very same native allocator, so blocks allocated
// before and after the swap are freed by the identical backend (this is
// what changeCurrentAllocator's "not initialized" guard protects against,
// and why we can skip it). Idempotent; returns false if no backend
// allocator is present yet.
static bool install_recording_allocator() {
  cca::CUDAAllocator* current = cca::allocator.load();
  if (current == nullptr) {
    return false;
  }
  if (g_recording_allocator != nullptr && current == g_recording_allocator) {
    return true;  // already installed
  }
  if (g_recording_allocator == nullptr) {
    g_recording_allocator = new cca::LetoRecordingAllocator();
  }
  // Publish real_ before the pointer so any thread that observes the new
  // allocator via the seq_cst store below also observes a valid delegate.
  g_recording_allocator->set_real(current);
  cca::allocator.store(g_recording_allocator);
  // Also win the c10::GetAllocator(kCUDA) registry -- the secondary path
  // some generic code uses -- with higher priority than the backend's
  // priority-0 entry.
  c10::SetAllocator(
      c10::DeviceType::CUDA, g_recording_allocator, /*priority=*/255);
  return true;
}

static bool recording_allocator_installed() {
  return g_recording_allocator != nullptr &&
      cca::allocator.load() == g_recording_allocator;
}

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

  // --- request-aware reclaim (RecordingAllocator) ---

  m.def("install_recording_allocator", []() {
    return leto::install_recording_allocator();
  });

  m.def("is_recording_allocator_installed", []() {
    return leto::recording_allocator_installed();
  });

  m.def(
      "set_request_aware",
      [](bool enabled) {
        std::lock_guard<std::mutex> lk(leto::g_mu);
        leto::g_request_aware = enabled;
      },
      py::arg("enabled"));

  m.def("get_request_aware", []() {
    std::lock_guard<std::mutex> lk(leto::g_mu);
    return leto::g_request_aware;
  });

  m.def(
      "set_request_margin_mb",
      [](int mb) {
        std::lock_guard<std::mutex> lk(leto::g_mu);
        leto::g_request_margin_mb = mb;
      },
      py::arg("mb"));

  m.def("get_request_margin_mb", []() {
    std::lock_guard<std::mutex> lk(leto::g_mu);
    return leto::g_request_margin_mb;
  });

  // Most recent allocation request observed by the RecordingAllocator, in
  // MiB, across all threads. Observability only (logging / tests); the kill
  // decision uses the per-thread value, not this.
  m.def("get_last_request_mb", []() {
    return static_cast<int>(
        leto::g_last_request_bytes.load(std::memory_order_relaxed) /
        (1024ULL * 1024ULL));
  });

  // --- reservation broker (standby memory reservation) ---

  m.def(
      "attach_reservation_ledger",
      [](const std::string& path) {
        return leto::attach_reservation_ledger(path);
      },
      py::arg("path"));

  m.def(
      "start_broker",
      [](int margin_mb) {
        py::gil_scoped_release nogil;  // the broker thread takes no GIL
        return leto::start_broker(margin_mb);
      },
      py::arg("margin_mb"));

  m.def("stop_broker", []() {
    py::gil_scoped_release nogil;
    leto::stop_broker();
  });

  m.def("reset_granted", []() { leto::reset_granted(); });

  m.def(
      "set_reservation_margin_mb",
      [](int mb) {
        leto::g_reservation_margin_mb.store(mb, std::memory_order_relaxed);
      },
      py::arg("mb"));

  m.def("get_reservation_margin_mb", []() {
    return leto::g_reservation_margin_mb.load(std::memory_order_relaxed);
  });

  m.def("is_broker_running", []() {
    return leto::g_broker_running.load(std::memory_order_acquire);
  });

  // Observability: cumulative grant + last cached NVML snapshot (MiB).
  m.def("get_granted_mb", []() {
    return static_cast<int>(
        leto::g_granted.load(std::memory_order_relaxed) / (1024 * 1024));
  });

  m.def("get_cached_free_mb", []() {
    return static_cast<int>(
        leto::g_cached_free_bytes.load(std::memory_order_relaxed) /
        (1024 * 1024));
  });

  m.def("get_cached_standby_actual_mb", []() {
    return static_cast<int>(
        leto::g_cached_standby_actual_bytes.load(std::memory_order_relaxed) /
        (1024 * 1024));
  });

  m.def("get_assert_b_violations", []() {
    return leto::g_assert_b_violations.load(std::memory_order_relaxed);
  });

  m.def("reset_assert_b_violations", []() {
    leto::g_assert_b_violations.store(0, std::memory_order_relaxed);
  });

  m.attr("LEDGER_NBYTES") = static_cast<int>(sizeof(leto::ReservationLedger));
  m.attr("LEDGER_MAGIC") = static_cast<unsigned int>(leto::RESV_LEDGER_MAGIC);
  m.attr("VERDICT_PENDING") = leto::RESV_VERDICT_PENDING;
  m.attr("VERDICT_DENY") = leto::RESV_VERDICT_DENY;
  m.attr("VERDICT_GRANT") = leto::RESV_VERDICT_GRANT;
  // Field byte offsets, so the Python standby writes the identical layout.
  m.attr("OFF_MAGIC") = static_cast<int>(offsetof(leto::ReservationLedger, magic));
  m.attr("OFF_STANDBY_PID") =
      static_cast<int>(offsetof(leto::ReservationLedger, standby_pid));
  m.attr("OFF_STANDBY_EPOCH") =
      static_cast<int>(offsetof(leto::ReservationLedger, standby_epoch));
  m.attr("OFF_REQ_SEQ") =
      static_cast<int>(offsetof(leto::ReservationLedger, req_seq));
  m.attr("OFF_REQ_BYTES") =
      static_cast<int>(offsetof(leto::ReservationLedger, req_bytes));
  m.attr("OFF_RESP_SEQ") =
      static_cast<int>(offsetof(leto::ReservationLedger, resp_seq));
  m.attr("OFF_RESP_VERDICT") =
      static_cast<int>(offsetof(leto::ReservationLedger, resp_verdict));
  m.attr("OFF_GRANTED") =
      static_cast<int>(offsetof(leto::ReservationLedger, granted));
  m.attr("OFF_EFFECTIVE_FREE") =
      static_cast<int>(offsetof(leto::ReservationLedger, effective_free));
  m.attr("OFF_STANDBY_ACTUAL") =
      static_cast<int>(offsetof(leto::ReservationLedger, standby_actual));
}
