#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <ctime>
#include <atomic>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cstdarg>
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

// Compact wall-clock timestamp (HH:MM:SS.mmm) so the reservation-broker and
// free-memory-callback event series can be correlated with the [titan] logs.
static const char* leto_ts() {
  static thread_local char buf[24];
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  struct tm tmv;
  localtime_r(&tv.tv_sec, &tmv);
  std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", tmv.tm_hour,
                tmv.tm_min, tmv.tm_sec, static_cast<int>(tv.tv_usec / 1000));
  return buf;
}

// Rank tag from torchrun's env (RANK, else LOCAL_RANK, else "?"): multiple
// ranks' stderr interleave in one per-node log file.
static const char* leto_rank() {
  static const std::string tag = [] {
    const char* r = std::getenv("RANK");
    if (r == nullptr) r = std::getenv("LOCAL_RANK");
    return std::string(r != nullptr ? r : "?");
  }();
  return tag.c_str();
}

// Single-write log line "[leto r<rank> HH:MM:SS.mmm] <msg>" so concurrent
// ranks' lines don't shear inside the shared stderr pipe.
static void leto_log(const char* fmt, ...) {
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "[leto r%s %s] %s\n", leto_rank(), leto_ts(), msg);
}

// ---------------------------------------------------------------------------
// Releasable-memory estimation (page-exact, snapshot-based)
// ---------------------------------------------------------------------------

// Mirror of DeviceCachingAllocator::round_size, upper-bounded: 512 B
// granularity by default; when roundup_power2_divisions is configured the
// allocator rounds to the next power-of-2 division, which never exceeds
// the next power of two -- bound by that.
static int64_t round_request(int64_t size) {
  constexpr int64_t kMinBlockSize = 512;
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  }
  const auto divisions = c10::CachingAllocator::AcceleratorAllocatorConfig::
      roundup_power2_divisions(static_cast<size_t>(size));
  if (divisions > 1 &&
      size > kMinBlockSize * static_cast<int64_t>(divisions)) {
    const int64_t floor_p2 = int64_t{1}
        << (63 - __builtin_clzll(static_cast<uint64_t>(size)));
    return floor_p2 == size ? size : floor_p2 << 1;
  }
  return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

// Upper bound on the physical bytes a request will newly map if it cannot
// be served from cache. The expandable path maps round_size(req) rounded
// up to whole pages (2 MiB small pool, large_segment_size() -- 20 MiB
// default -- large pool) within page-aligned mapped ranges, so newly
// mapped bytes never exceed the page-rounded request. The non-expandable
// path cudaMallocs get_allocation_size(req), at least kLargeBuffer for
// large-pool requests -- the max() keeps the bound valid there too.
static int64_t request_physical_ub(int64_t req) {
  constexpr int64_t kSmallSize = int64_t{1} << 20;    // <= 1 MiB: small pool
  constexpr int64_t kSmallBuffer = int64_t{2} << 20;  // small-pool page
  constexpr int64_t kLargeBuffer = int64_t{20} << 20;
  if (req <= kSmallSize) {
    return kSmallBuffer;
  }
  const int64_t page = static_cast<int64_t>(
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size());
  const int64_t r = round_request(req);
  return std::max((r + page - 1) / page * page, kLargeBuffer);
}

// Sound lower bound on the bytes release_cached_blocks()/emptyCache() can
// return to the driver right now, replaying the allocator's release logic
// over snapshot(): a non-expandable segment frees iff nothing in it is
// active; an expandable segment unmaps only whole pages strictly inside
// each inactive block. Do NOT use `reserved - allocated - inactive_split`
// instead: expandable segments keep no inactive_split stat, so that formula
// credits unreleasable sub-page fragments (a 945 MiB phantom on the 2026-07
// AWS OOM). Event-pending blocks and private pools are counted as
// unreleasable, keeping this an under-estimate. snapshot() re-takes the
// recursive device allocator lock — safe from every caller here.
static int64_t releasable_lower_bound(c10::DeviceIndex device) {
  const auto snap = c10::cuda::CUDACachingAllocator::snapshot();
  const int64_t large_page = static_cast<int64_t>(
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size());
  constexpr int64_t small_page = int64_t{2} << 20;  // kSmallBuffer
  int64_t out = 0;
  for (const auto& seg : snap.segments) {
    if (seg.device != device) {
      continue;
    }
    if (seg.owner_private_pool_id.first != 0 ||
        seg.owner_private_pool_id.second != 0) {
      continue;
    }
    if (!seg.is_expandable) {
      if (seg.active_size == 0) {
        out += static_cast<int64_t>(seg.total_size);
      }
      continue;
    }
    const int64_t page = seg.is_large ? large_page : small_page;
    int64_t off = static_cast<int64_t>(seg.address);
    for (const auto& b : seg.blocks) {
      const int64_t sz = static_cast<int64_t>(b.size);
      if (!b.active) {
        const int64_t lo = (off + page - 1) / page * page;
        const int64_t hi = (off + sz) / page * page;
        if (hi > lo) {
          out += hi - lo;
        }
      }
      off += sz;
    }
  }
  return out;
}

// Mapped-but-free bytes an expandable-segments expansion will REUSE when
// serving Execute()'s own request (a guaranteed miss): the allocator fuses
// the mapped free tail before the lowest-address unmapped block into the
// new mapping and maps only the shortfall (see find_expandable_block /
// try_allocate_expandable_block / map_block in pytorch's
// CUDACachingAllocator.cpp). snapshot() reports one SegmentInfo per mapped
// range, so a range's trailing inactive bytes (page-floored) are exactly
// that reusable tail; we take the lowest-address candidate per
// (stream, pool), mirroring the allocator's scan order. Approximation: VA
// extents aren't in the snapshot, but expandable reservations dwarf any
// request, so the first chain is the served one. Cost: one snapshot walk
// (~1.8 ms at 18k blocks), paid only for requests >= kReuseWalkMinRequest.
static int64_t tail_reuse_bytes(
    c10::DeviceIndex device, cudaStream_t stream, bool small_pool) {
  const auto snap = c10::cuda::CUDACachingAllocator::snapshot();
  const int64_t large_page = static_cast<int64_t>(
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size());
  constexpr int64_t small_page = int64_t{2} << 20;  // kSmallBuffer
  int64_t best = -1;
  uintptr_t best_addr = 0;
  for (const auto& seg : snap.segments) {
    if (seg.device != device || !seg.is_expandable) {
      continue;
    }
    if (seg.owner_private_pool_id.first != 0 ||
        seg.owner_private_pool_id.second != 0) {
      continue;
    }
    if (seg.stream != stream || seg.is_large == small_pool) {
      continue;
    }
    // Trailing run of inactive blocks at the end of this mapped range,
    // page-floored: expansion maps whole pages, so only whole free pages
    // adjacent to the unmapped remainder are reusable without new mapping.
    int64_t tail = 0;
    for (auto it = seg.blocks.rbegin(); it != seg.blocks.rend(); ++it) {
      if (it->active) {
        break;
      }
      tail += static_cast<int64_t>(it->size);
    }
    const int64_t page = seg.is_large ? large_page : small_page;
    tail = tail / page * page;
    if (best < 0 || seg.address < best_addr) {
      best = tail;
      best_addr = seg.address;
    }
  }
  return best < 0 ? 0 : best;
}

// Below this request size the reuse walk isn't worth 1.8 ms: charging the
// request in full errs deny-side by less than the margin.
constexpr int64_t kReuseWalkMinRequest = int64_t{64} << 20;

// Last-publish component snapshot (diagnostics; relaxed atomics). Written by
// compute_used_estimate so the verbose fmcb line can print the estimate's
// breakdown and the broker can print the estimate's age (it only moves on
// this rank's cache misses).
static std::atomic<int64_t> g_pub_total{0};
static std::atomic<int64_t> g_pub_free{0};
static std::atomic<int64_t> g_pub_actual{0};
static std::atomic<int64_t> g_pub_req_charge{0};  // request's net phys charge
static std::atomic<int64_t> g_pub_reuse{0};       // expandable tail credited
static std::atomic<int64_t> g_pub_released{0};    // forced self-release credit
static std::atomic<int64_t> g_pub_at_ms{0};       // publish time (mono_ms)

static int64_t mono_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Guards g_kill_cb (a py::object) only.
static std::mutex g_mu;
static py::object g_kill_cb;  // default-constructed = null PyObject*
// Number of times the reclaim decision fired and invoked the kill callback.
// Lets Python/tests observe whether a fire happened.
static std::atomic<unsigned int> g_num_kill_standby_called{0};

// Size of the allocation currently in flight on THIS thread, stamped by
// LetoRecordingAllocator. The FreeMemoryCallback fires synchronously deeper
// in the same malloc, so Execute() reads its exact pending request here.
thread_local size_t g_request_bytes = 0;

static inline void record_request(size_t n) { g_request_bytes = n; }


// ---------------------------------------------------------------------------
// Reservation broker — standby memory reservation against the active allocator
// ---------------------------------------------------------------------------
//
// The active process brokers all GPU memory the co-located standby may use,
// via a per-rank shared-memory ledger and a two-phase protocol (RESERVE soft
// claim -> GRANT commit; see req_type below). Defense layers, in order:
//   admission     RESERVE iff est + margin + cum <= capacity
//                 (margin = growth allowance for the active; its ONLY role)
//   enforcement   allocator cap = total - granted: the active recycles its
//                 cache at the boundary instead of growing into the grant
//   cancellation  Execute() retracts a soft claim the device can no longer
//                 back (free, nothing was allocated)
//   kill          last resort, only when the in-flight allocation provably
//                 cannot proceed without reclaimed memory
// Memory truth flows one way: Execute() reads NVML per cache miss and
// publishes the estimate; the broker only consumes it.

// Shared-memory ledger layout. Fixed, 8-byte-aligned; the Python standby side
// mmaps the same file with a matching struct format (see progressive.py).
// Single-producer-per-field + a seq handshake make a process-shared mutex
// unnecessary on x86-64 (naturally-aligned loads/stores are atomic; we add
// acquire/release fences via __atomic builtins for ordering + portability).
constexpr uint32_t RESV_LEDGER_MAGIC = 0x4C54524Eu;  // "LTRN"
constexpr int RESV_VERDICT_PENDING = -1;
constexpr int RESV_VERDICT_DENY = 0;
constexpr int RESV_VERDICT_GRANT = 1;

// Two-phase request types (req_type). A task executes only after EVERY rank
// passed both phases:
//   RESERVE  — soft claim: admitted against est + granted + delta, held in
//              g_reserved_pending, cancellable by Execute() under pressure
//              at zero cost (nothing was allocated).
//   GRANT    — hard commit: allowed iff the reservation is still alive;
//              converts reserved -> granted (kill-protected, budget-capped).
//   ROLLBACK — abort the round: lower granted back to the pre-task
//              cumulative (req_bytes) and clear the reservation. Sent to
//              every rank when any rank's phase-2 GRANT was denied, so a
//              unilateral phase-2 commit never lingers as phantom pressure.
constexpr int RESV_REQ_RESERVE = 1;
constexpr int RESV_REQ_GRANT = 2;
constexpr int RESV_REQ_ROLLBACK = 3;

struct ReservationLedger {
  uint32_t magic;          // 0  : RESV_LEDGER_MAGIC once initialized
  int32_t standby_pid;     // 4  : standby-written; NVML proc to exclude (0=none)
  uint32_t standby_epoch;  // 8  : standby-written; bumped per new standby
  uint32_t req_seq;        // 12 : standby-written; bumped *last* to publish a req
  int64_t req_bytes;       // 16 : standby-written; cumulative target (RESERVE/
                           //      GRANT) or rollback-to cumulative (ROLLBACK)
  uint32_t resp_seq;       // 24 : broker-written; bumped *last*; == req_seq when done
  int32_t resp_verdict;    // 28 : broker-written; GRANT/DENY
  int64_t granted;         // 32 : broker-published cumulative grant (observability)
  int64_t effective_free;  // 40 : broker-published grantable bytes (observability)
  int64_t standby_actual;  // 48 : layout-reserved; no longer written
  int32_t req_type;        // 56 : standby-written; RESV_REQ_* (before req_seq)
  int32_t pad;             // 60
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

static std::atomic<bool> g_reservation_active{false};  // gates reservation path
// Committed cumulative grant. Writers (broker commit, resets) run under
// g_state_mutex and call apply_reservation_budget() after, which derives the
// cap from the CURRENT value, so interleavings converge.
static std::atomic<int64_t> g_granted{0};
// Pending phase-1 reservation (delta of the one outstanding task; 0 = none).
// Soft: counted by RESERVE admission and Execute's cancel, NOT by the
// allocator budget — the active may grow into it, which is what triggers
// cancellation. Mutated only under g_state_mutex.
static std::atomic<int64_t> g_reserved_pending{0};
// Standby pid mirrored out of the shm ledger by the broker each tick, so
// Execute() reads a plain atomic instead of touching the ledger.
static std::atomic<int> g_standby_pid{0};
static std::atomic<int> g_reservation_margin_mb{0};
static uint32_t g_standby_epoch_seen = 0;  // broker-thread-local

// Allocator-budget enforcement of the reservation (see
// apply_reservation_budget). Captured at attach/start_broker.
static int64_t g_device_total_bytes = 0;
static double g_orig_fraction = 1.0;
static bool g_orig_fraction_saved = false;

// ---------------------------------------------------------------------------
// used_estimate — the PHYSICAL device-usage prediction Execute() publishes
// per cache miss (see compute_used_estimate):
//   est = max(0, total − free − standby_actual) + req_charge      [bytes]
// Pure measurement: no margin, no cache credit (nothing converts the
// active's cache into free memory on the standby's behalf — crediting it
// caused the 2026-07 standby OOMs). Starts at INT64_MAX so the broker
// denies until the first publish (seeded once at start_broker).
//
// (est, g_granted, g_reserved_pending) are mutated only under
// g_state_mutex, keeping admission and the kill/cancel decisions mutually
// consistent. The broker's critical section is the admission arithmetic +
// one store (~ns; it must never stall an allocation). LOCK ORDER: Execute
// enters holding the device-allocator lock, so g_state_mutex is strictly
// INNERMOST — no allocator API, NVML, or GIL while holding it. The atomics
// stay atomic for lock-free readers (logging, budget, pybind getters).
static std::mutex g_state_mutex;
static std::atomic<int64_t> g_used_estimate{INT64_MAX};

// Kill latch — at most ONE reclaim per standby generation. Armed whenever
// the broker services a request (a live standby is asking); consumed by
// Execute() on fire, and ONLY on fire (short-circuit order), so a gated
// keep never burns the generation's one kill.
static std::atomic<bool> g_kill_armed{false};
// NVML device capacity (mem.total), captured at the start_broker seed read
// and constant thereafter. Kept NVML-consistent with used_estimate (the
// CUDA device-prop total differs by the driver-reserved region).
static std::atomic<int64_t> g_nvml_total_bytes{0};

// Invariant (B): standby_actual <= granted. A breach is a protocol violation
// (a task over-allocated past its profiled delta, or skipped reserving); the
// accounting clamps to stay safe, and we count breaches for tests/observability.
constexpr int64_t RESV_ASSERT_TOL_BYTES = int64_t{256} * 1024 * 1024;
static std::atomic<uint64_t> g_assert_b_violations{0};

// Verbose decision logging: every keep evaluation and every idempotent
// re-grant. Off by default — at MoE cache-miss burst rates the synchronous
// stderr writes (NFS-backed node files) are themselves a hot-path cost.
// RELEASE / KILL / DENY / grant-raising decisions are always logged.
static const bool g_verbose_decisions =
    std::getenv("LETO_FMCB_VERBOSE") != nullptr;

static ReservationLedger* g_ledger = nullptr;  // mmap'd; never unmapped
// Heap-allocated and leaked rather than a static std::thread object, so a
// process that exits without calling stop_broker() doesn't std::terminate on
// a still-joinable thread during static destruction.
static std::thread* g_broker_thread = nullptr;
static std::atomic<bool> g_broker_running{false};
static int g_broker_device = 0;

// NVML state. nvmlInit_v2 is reference-counted in libnvidia-ml, so it
// safely coexists with any other NVML initialization in the process
// (e.g. pynvml from leto's progressive init).
static std::once_flag g_nvml_init_once;
static nvmlReturn_t g_nvml_init_status = NVML_ERROR_UNINITIALIZED;
static std::mutex g_nvml_handle_mu;  // slow init path only; fast path is lock-free
static nvmlDevice_t g_nvml_handle{};
static std::atomic<bool> g_nvml_handle_ready{false};

static void init_nvml_once() {
  std::call_once(g_nvml_init_once, []() {
    g_nvml_init_status = nvmlInit_v2();
    if (g_nvml_init_status != NVML_SUCCESS) {
      leto_log("nvmlInit_v2 failed: %s", nvmlErrorString(g_nvml_init_status));
    }
  });
}

static bool resolve_nvml_handle(nvmlDevice_t* out) {
  // Lock-free fast path: the handle is written once before the release
  // store below, then immutable — Execute() never takes a lock here.
  if (g_nvml_handle_ready.load(std::memory_order_acquire)) {
    *out = g_nvml_handle;
    return true;
  }
  // Map the current CUDA device to its NVML handle via PCI bus ID.
  // PCI is the stable identity even when CUDA_VISIBLE_DEVICES has
  // remapped the CUDA-side enumeration.
  int cuda_dev = 0;
  cudaError_t cerr = cudaGetDevice(&cuda_dev);
  if (cerr != cudaSuccess) {
    leto_log("cudaGetDevice failed: %s", cudaGetErrorString(cerr));
    return false;
  }
  cudaDeviceProp prop;
  cerr = cudaGetDeviceProperties(&prop, cuda_dev);
  if (cerr != cudaSuccess) {
    leto_log("cudaGetDeviceProperties failed: %s", cudaGetErrorString(cerr));
    return false;
  }
  char pci[64];
  std::snprintf(pci, sizeof(pci), "%08X:%02X:%02X.0",
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
  nvmlDevice_t handle;
  nvmlReturn_t nerr = nvmlDeviceGetHandleByPciBusId_v2(pci, &handle);
  if (nerr != NVML_SUCCESS) {
    leto_log("nvmlDeviceGetHandleByPciBusId_v2(%s) failed: %s", pci,
             nvmlErrorString(nerr));
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(g_nvml_handle_mu);
    if (!g_nvml_handle_ready.load(std::memory_order_relaxed)) {
      g_nvml_handle = handle;
      g_nvml_handle_ready.store(true, std::memory_order_release);
    }
  }
  *out = handle;
  return true;
}

// Read device total/free bytes (NVML) and the GPU memory used by
// `target_pid` (0 if absent or NVML can't attribute per-process memory).
// NVML, not cudaMemGetInfo — the driver view is ground truth. Uses the v2
// memory API when the build's NVML has it, else v1.
static bool read_nvml_mem(nvmlDevice_t handle, int target_pid,
                          int64_t* free_b_out, int64_t* total_b_out,
                          int64_t* pid_used_b_out) {
#if defined(NVML_API_VERSION) && NVML_API_VERSION >= 12
  nvmlMemory_v2_t mem;
  mem.version = nvmlMemory_v2;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo_v2(handle, &mem);
#else
  nvmlMemory_t mem;
  nvmlReturn_t nerr = nvmlDeviceGetMemoryInfo(handle, &mem);
#endif
  if (nerr != NVML_SUCCESS) {
    leto_log("broker nvmlDeviceGetMemoryInfo failed: %s",
             nvmlErrorString(nerr));
    return false;
  }
  const int64_t total_b = static_cast<int64_t>(mem.total);
  if (total_b_out != nullptr) *total_b_out = total_b;
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
      leto_log("broker nvmlDeviceGetComputeRunningProcesses failed: %s",
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

// Enforce the reservation as an allocator budget: while anything is granted,
// cap the active's mapped bytes at (device_total - granted) via
// setMemoryFraction. The malloc slow path then RECYCLES cached blocks at the
// boundary instead of growing into the grant. No margin: the margin is
// admission-only policy, so the cap is exactly the committed reservation.
// granted==0 restores the pre-install fraction. Argument-free by design: it
// derives the fraction from the CURRENT g_granted, so concurrent writers
// converge. Callable from the broker thread and the kill path (the device
// allocator lock is recursive).
static void apply_reservation_budget() {
  if (g_device_total_bytes <= 0) return;
  const int64_t granted_b = g_granted.load(std::memory_order_relaxed);
  double frac = g_orig_fraction;
  int64_t cap_b = g_device_total_bytes;
  if (granted_b > 0) {
    cap_b = g_device_total_bytes - granted_b;
    frac = std::max(
        0.0,
        std::min(g_orig_fraction,
                 static_cast<double>(cap_b) /
                     static_cast<double>(g_device_total_bytes)));
  }
  try {
    c10::cuda::CUDACachingAllocator::setMemoryFraction(
        frac, static_cast<c10::DeviceIndex>(g_broker_device));
  } catch (const std::exception& e) {
    leto_log("setMemoryFraction(%.4f) failed: %s", frac, e.what());
  }
  // Always-on diagnostic (grant/reset/epoch events only): both sides of the
  // cap, so whether it can bind is observable per event.
  int64_t alloc_reserved_b = -1, alloc_alloc_b = -1;
  try {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        static_cast<c10::DeviceIndex>(g_broker_device));
    constexpr size_t kAgg = static_cast<size_t>(
        c10::CachingDeviceAllocator::StatType::AGGREGATE);
    alloc_reserved_b = stats.reserved_bytes[kAgg].current;
    alloc_alloc_b = stats.allocated_bytes[kAgg].current;
  } catch (const std::exception&) {
  }
  leto_log(
      "budget: granted=%lld cap=%lld frac=%.4f | alloc_reserved=%lld "
      "alloc_allocated=%lld device_total=%lld MiB binding=%s",
      static_cast<long long>(granted_b / (1024 * 1024)),
      static_cast<long long>(cap_b / (1024 * 1024)),
      frac,
      static_cast<long long>(alloc_reserved_b >= 0
                                 ? alloc_reserved_b / (1024 * 1024)
                                 : -1),
      static_cast<long long>(alloc_alloc_b >= 0
                                 ? alloc_alloc_b / (1024 * 1024)
                                 : -1),
      static_cast<long long>(g_device_total_bytes / (1024 * 1024)),
      (alloc_reserved_b >= 0 && cap_b < alloc_reserved_b) ? "YES" : "NO");
}

// One fresh NVML read -> the PHYSICAL used estimate:
//   est        = max(0, total - free - standby_actual) + req_charge
//   req_charge = max(0, phys_ub(request) - tail_reuse) - released
// `request` is Execute()'s own in-flight allocation — a guaranteed miss, so
// only the expandable tail can serve it in place; `released` is the cache
// the release ladder will recycle when the charge exceeds free (serve-path
// bound: net consumption never exceeds ~free while the cache covers the
// shortfall). Charging the request makes the estimate post-malloc-correct.
// request == 0 (seed / pybind refresh) charges nothing.
//
// No margin (admission-only policy), no cache credit (nothing converts the
// active's cache into free memory on the standby's behalf), and no charge
// for other threads' lock-queued allocations (each either hits the cache or
// fires its own Execute before mapping).
//
// Computes and returns; the CALLER stores into g_used_estimate under
// g_state_mutex. Returns false (out-params untouched) on NVML failure.
static bool compute_used_estimate(nvmlDevice_t handle, int target_pid,
                                  int64_t request_bytes,
                                  int64_t* est_out,
                                  int64_t* free_b_out, int64_t* total_b_out,
                                  int64_t* standby_actual_out) {
  int64_t free_b = 0, total_b = 0, actual_b = 0;
  if (!read_nvml_mem(handle, target_pid, &free_b, &total_b, &actual_b)) {
    return false;
  }
  int64_t req_charge = 0;
  int64_t reuse_b = 0;
  int64_t released_b = 0;
  if (request_bytes > 0) {
    const int64_t req_ub = request_physical_ub(request_bytes);
    if (request_bytes >= kReuseWalkMinRequest) {
      try {
        const bool small_pool = request_bytes <= (int64_t{1} << 20);
        reuse_b = tail_reuse_bytes(
            c10::cuda::current_device(),
            c10::cuda::getCurrentCUDAStream().stream(), small_pool);
      } catch (const std::exception& e) {
        // Snapshot unavailable -> no reuse credit (conservative).
        leto_log("tail_reuse_bytes failed: %s", e.what());
      }
    }
    req_charge = std::max<int64_t>(0, req_ub - reuse_b);
    // Serve-path bound: past physical free the expansion fails and the
    // allocator recycles its own cache, so the net consumption is
    // charge - released (an estimate above device capacity would be
    // physically impossible). The walk runs only in this boundary case;
    // the already-credited tail is excluded to avoid double counting.
    if (req_charge > free_b) {
      int64_t releasable_b = 0;
      try {
        releasable_b = releasable_lower_bound(c10::cuda::current_device());
      } catch (const std::exception& e) {
        // Walk unavailable -> assume nothing releasable (conservative).
        leto_log("releasable_lower_bound failed: %s", e.what());
      }
      const int64_t coverable = std::max<int64_t>(0, releasable_b - reuse_b);
      released_b = std::min<int64_t>(req_charge - free_b, coverable);
      req_charge -= released_b;
    }
  }
  const int64_t used_excl_b =
      std::max<int64_t>(0, total_b - free_b - actual_b);
  *est_out = used_excl_b + req_charge;
  // Component snapshot for diagnostics (verbose fmcb line + broker
  // staleness annotation).
  g_pub_total.store(total_b, std::memory_order_relaxed);
  g_pub_free.store(free_b, std::memory_order_relaxed);
  g_pub_actual.store(actual_b, std::memory_order_relaxed);
  g_pub_req_charge.store(req_charge, std::memory_order_relaxed);
  g_pub_reuse.store(reuse_b, std::memory_order_relaxed);
  g_pub_released.store(released_b, std::memory_order_relaxed);
  g_pub_at_ms.store(mono_ms(), std::memory_order_relaxed);
  if (free_b_out != nullptr) *free_b_out = free_b;
  if (total_b_out != nullptr) *total_b_out = total_b;
  if (standby_actual_out != nullptr) *standby_actual_out = actual_b;
  return true;
}

// Tick only paces ledger polling (an mmap read + a few atomic loads);
// nothing heavy is on this path, so latency is the only trade-off: a grant
// response may lag a request by up to one tick, dwarfed by the standby's
// 1s deny-retry pacing.
constexpr int kBrokerTickMs = 500;

// Broker thread (active process). Polls the shm ledger, resets grant state
// on a new standby epoch, and services RESERVE/GRANT/ROLLBACK requests. It
// performs NO NVML/driver calls (except the never-published fallback seed):
// decisions consume the Execute-published estimate. (A per-tick NVML timer
// here once serialized 8 ranks on the driver lock and slowed training ~6x.)
static void broker_loop() {
  // No CUDA/NVML setup: the broker never touches the driver. It reads the
  // Callback-published estimate and the mmap'd ledger only.
  uint32_t last_req_seq = 0;
  while (g_broker_running.load(std::memory_order_acquire)) {
    ReservationLedger* L = g_ledger;
    if (L == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    const int target_pid = resv_load_acq(&L->standby_pid);
    g_standby_pid.store(target_pid, std::memory_order_relaxed);
    const uint32_t epoch = resv_load_acq(&L->standby_epoch);

    // 1) A fresh standby instance (new epoch) voids any prior grant AND
    // disarms the kill latch: a new generation must not inherit an arm from
    // its predecessor's requests (stale arms fired at freshly-rotated
    // groups and cascaded through 22 generations in ~4 min, 2026-07 run).
    if (epoch != g_standby_epoch_seen) {
      {
        std::lock_guard<std::mutex> lk(g_state_mutex);
        g_granted.store(0, std::memory_order_relaxed);
        g_reserved_pending.store(0, std::memory_order_relaxed);
        g_kill_armed.store(false, std::memory_order_relaxed);
      }
      g_standby_epoch_seen = epoch;
      apply_reservation_budget();  // outside the mutex (allocator lock)
    }

    // 2) Service a pending reservation request. `req_bytes` is the standby's
    // *cumulative* target footprint (not an increment), so a re-request after
    // a peer-denied task is idempotent — we only raise `granted` toward the
    // target and never double-count. Admission is a few atomic loads and a
    // compare — no locks, no NVML, nothing that can stall an allocation:
    //   GRANT iff used_estimate + new_reservation <= capacity
    // (margin is folded into used_estimate at publish). used_estimate is
    // INT64_MAX until the first publish, so requests deny rather than grant
    // blind. The CAS ensures a concurrent reset_granted (standby died) is
    // never overwritten by a raise meant for that dead standby.
    const uint32_t req_seq = resv_load_acq(&L->req_seq);
    if (req_seq != last_req_seq) {
      // A live standby is asking — re-arm the (at most once per generation)
      // kill latch. Harmless while nothing is granted/held: the reclaim
      // gate still requires standby_actual > 0 || reserved > 0.
      g_kill_armed.store(true, std::memory_order_relaxed);
      const int64_t req_bytes = resv_load_acq(&L->req_bytes);
      // Fallback seed — ONLY when nothing has ever been published (the
      // start_broker seed failed, e.g. a transient NVML error): the broker
      // may compute the estimate itself rather than deny forever. Never
      // refreshes an already-published value — Execute() owns the estimate.
      if (g_used_estimate.load(std::memory_order_relaxed) == INT64_MAX) {
        nvmlDevice_t handle{};
        int64_t est = 0, total_b = 0;
        if (resolve_nvml_handle(&handle) &&
            compute_used_estimate(handle, target_pid, /*request_bytes=*/0,
                                  &est, nullptr, &total_b, nullptr)) {
          std::lock_guard<std::mutex> lk(g_state_mutex);
          g_used_estimate.store(est, std::memory_order_relaxed);
          if (g_nvml_total_bytes.load(std::memory_order_relaxed) == 0) {
            g_nvml_total_bytes.store(total_b, std::memory_order_relaxed);
          }
        }
      }
      const int64_t capacity =
          g_nvml_total_bytes.load(std::memory_order_relaxed);
      const int req_type = resv_load_acq(&L->req_type);
      int verdict = RESV_VERDICT_DENY;
      int64_t used_est = 0, granted_before = 0, granted_after = 0;
      int64_t reserved_after = 0;
      {
        // Admission critical section — arithmetic + the state transition
        // only (~ns; see g_state_mutex discipline). No ledger writes, no
        // allocator APIs, no logging inside. The broker only READS the
        // estimate; freshness policing is Execute()'s cancellation.
        std::lock_guard<std::mutex> lk(g_state_mutex);
        used_est = g_used_estimate.load(std::memory_order_relaxed);
        granted_before = g_granted.load(std::memory_order_relaxed);
        granted_after = granted_before;
        const int64_t delta = req_bytes - granted_before;
        if (req_type == RESV_REQ_ROLLBACK) {
          // Abort the round: req_bytes = the pre-task cumulative. Lower a
          // unilateral phase-2 commit back to it and clear the soft claim.
          if (granted_before > req_bytes) {
            g_granted.store(req_bytes, std::memory_order_relaxed);
            granted_after = req_bytes;
          }
          g_reserved_pending.store(0, std::memory_order_relaxed);
          verdict = RESV_VERDICT_GRANT;  // ack
        } else if (delta <= 0) {
          // Already covered by the committed cumulative (idempotent retry
          // of an earlier task): nothing to reserve or convert.
          if (req_type == RESV_REQ_GRANT) {
            g_reserved_pending.store(0, std::memory_order_relaxed);
          }
          verdict = RESV_VERDICT_GRANT;
        } else if (req_type == RESV_REQ_RESERVE) {
          // Phase 1 — soft claim, admitted against the physical estimate
          // plus the EXPLICIT margin (est is a pure measurement; margin is
          // policy and belongs to admission): est + margin + cum <= cap.
          const int64_t margin_b =
              static_cast<int64_t>(
                  g_reservation_margin_mb.load(std::memory_order_relaxed)) *
              1024 * 1024;
          if (capacity > 0 && used_est + margin_b <= capacity - req_bytes) {
            g_reserved_pending.store(delta, std::memory_order_relaxed);
            verdict = RESV_VERDICT_GRANT;
          } else {
            g_reserved_pending.store(0, std::memory_order_relaxed);
          }
        } else if (req_type == RESV_REQ_GRANT) {
          // Phase 2 — hard commit iff the reservation survived Execute's
          // cancellation since phase 1. No re-check of the estimate: the
          // cancel path is strictly fresher (per cache miss) than any
          // broker-side test could be.
          if (g_reserved_pending.load(std::memory_order_relaxed) >= delta) {
            g_granted.store(req_bytes, std::memory_order_relaxed);
            granted_after = req_bytes;
            g_reserved_pending.store(0, std::memory_order_relaxed);
            verdict = RESV_VERDICT_GRANT;
          }
          // else: DENY — reservation canceled (or never made); the standby
          // rolls back every rank and redoes both phases.
        }
        reserved_after = g_reserved_pending.load(std::memory_order_relaxed);
      }
      const int64_t needed = req_bytes - granted_before;
      if (granted_after != granted_before) {
        apply_reservation_budget();  // outside the mutex (allocator lock)
      }
      resv_store_rel(&L->effective_free,
                     capacity > 0 ? std::max<int64_t>(0, capacity - used_est)
                                  : int64_t{0});
      resv_store_rel(&L->granted, granted_after);
      resv_store_rel(&L->resp_verdict, verdict);
      resv_store_rel(&L->resp_seq, req_seq);  // bump last → signals done
      last_req_seq = req_seq;
      // Diagnostic: DENYs and state-changing decisions always; idempotent
      // re-acks (needed<=0, one per peer-denied unanimity round) only
      // with LETO_FMCB_VERBOSE=1.
      if (verdict == RESV_VERDICT_DENY || needed > 0 || g_verbose_decisions) {
        // est_age: how stale the consumed estimate is (published on this
        // rank's LAST cache miss). Components of that publish are on the
        // matching verbose fmcb line.
        const int64_t est_age_ms =
            mono_ms() - g_pub_at_ms.load(std::memory_order_relaxed);
        const char* type_name =
            req_type == RESV_REQ_RESERVE
                ? "RESERVE"
                : (req_type == RESV_REQ_GRANT
                       ? "GRANT"
                       : (req_type == RESV_REQ_ROLLBACK ? "ROLLBACK"
                                                        : "UNKNOWN"));
        leto_log(
            "broker req_seq=%u %s request=%lld MiB -> %s | "
            "used_est=%lld (age=%lldms) capacity=%lld "
            "granted_before=%lld needed=%lld granted_after=%lld "
            "reserved_after=%lld MiB",
            req_seq, type_name,
            static_cast<long long>(req_bytes / (1024 * 1024)),
            verdict == RESV_VERDICT_GRANT ? "OK" : "DENY",
            static_cast<long long>(used_est / (1024 * 1024)),
            static_cast<long long>(est_age_ms),
            static_cast<long long>(capacity / (1024 * 1024)),
            static_cast<long long>(granted_before / (1024 * 1024)),
            static_cast<long long>(needed / (1024 * 1024)),
            static_cast<long long>(granted_after / (1024 * 1024)),
            static_cast<long long>(reserved_after / (1024 * 1024)));
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kBrokerTickMs));
  }
}

// mmap the per-rank shm ledger. Captures the active rank's CUDA device and
// primes the NVML handle on this (correctly-deviced) thread.
static bool attach_reservation_ledger(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0) {
    leto_log("attach_reservation_ledger open(%s) failed: %s", path.c_str(),
             std::strerror(errno));
    return false;
  }
  if (::ftruncate(fd, sizeof(ReservationLedger)) != 0) {
    leto_log("attach_reservation_ledger ftruncate failed: %s",
             std::strerror(errno));
    ::close(fd);
    return false;
  }
  void* p = ::mmap(nullptr, sizeof(ReservationLedger), PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {
    leto_log("attach_reservation_ledger mmap failed: %s",
             std::strerror(errno));
    return false;
  }
  g_ledger = static_cast<ReservationLedger*>(p);
  resv_store_rel(&g_ledger->magic, RESV_LEDGER_MAGIC);
  cudaGetDevice(&g_broker_device);
  {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, g_broker_device) == cudaSuccess) {
      g_device_total_bytes = static_cast<int64_t>(prop.totalGlobalMem);
    }
  }
  init_nvml_once();
  nvmlDevice_t tmp{};
  resolve_nvml_handle(&tmp);  // prime the handle cache on the right device
  return true;
}

static bool start_broker(int margin_mb) {
  if (g_ledger == nullptr) {
    leto_log("start_broker: no ledger attached");
    return false;
  }
  if (g_broker_running.load(std::memory_order_acquire)) {
    g_reservation_margin_mb.store(margin_mb, std::memory_order_relaxed);
    return true;  // already running
  }
  g_reservation_margin_mb.store(margin_mb, std::memory_order_relaxed);
  // Baseline for the reservation budget: whatever fraction the trainer had
  // before install is what granted==0 restores.
  if (!g_orig_fraction_saved) {
    try {
      g_orig_fraction = c10::cuda::CUDACachingAllocator::getMemoryFraction(
          static_cast<c10::DeviceIndex>(g_broker_device));
      g_orig_fraction_saved = true;
    } catch (const std::exception& e) {
      leto_log("getMemoryFraction failed: %s", e.what());
    }
  }
  // Seed the estimate once (also captures NVML capacity, the constant the
  // broker admits against): the standby's first request may arrive before
  // the first post-install cache miss publishes anything, and the estimate
  // starts at INT64_MAX (deny-until-published). request=0: no allocation
  // is in flight on this (install) path.
  {
    nvmlDevice_t handle{};
    int64_t est = 0, total_b = 0;
    const int target_pid = resv_load_acq(&g_ledger->standby_pid);
    g_standby_pid.store(target_pid, std::memory_order_relaxed);
    if (resolve_nvml_handle(&handle) &&
        compute_used_estimate(handle, target_pid, /*request_bytes=*/0, &est,
                              nullptr, &total_b, nullptr)) {
      std::lock_guard<std::mutex> lk(g_state_mutex);
      g_used_estimate.store(est, std::memory_order_relaxed);
      g_nvml_total_bytes.store(total_b, std::memory_order_relaxed);
    }
  }
  g_broker_running.store(true, std::memory_order_release);
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
  g_granted.store(0, std::memory_order_relaxed);
  apply_reservation_budget();  // restore the pre-install fraction
}

// Void the cumulative grant (e.g. after the standby is killed/activated).
// Callable from the kill callback, which runs inside the FreeMemCallback
// deep in the allocator's malloc path — g_state_mutex is innermost there
// (Execute holds no leto lock at kill-callback time), so taking it here is
// safe; the budget re-derives from the current value at apply time.
static void reset_granted() {
  {
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_granted.store(0, std::memory_order_relaxed);
    g_reserved_pending.store(0, std::memory_order_relaxed);
  }
  if (g_ledger != nullptr) resv_store_rel(&g_ledger->granted, int64_t{0});
  // Restore the allocator budget (kill path runs inside the malloc chain —
  // the device allocator lock is recursive, so re-entering is safe, and the
  // retried allocation must not stay capped by the dead standby's grant).
  apply_reservation_budget();
}

// Wait (<=10s, 1ms poll) until the device shows at least `need_b` free
// bytes via cudaMemGetInfo. Used to confirm a standby kill's reclaim:
// the memory is usable the moment the driver tears the context down —
// no need to wait for the pid to leave NVML's compute-process list.
static bool wait_for_free_at_least(int64_t need_b) {
  auto poll_start = std::chrono::steady_clock::now();
  auto deadline = poll_start + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    size_t cu_free = 0, cu_total = 0;
    if (cudaMemGetInfo(&cu_free, &cu_total) != cudaSuccess) {
      leto_log("cudaMemGetInfo failed during reclaim wait");
      return false;
    }
    if (static_cast<int64_t>(cu_free) >= need_b) {
      auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - poll_start)
              .count();
      leto_log("reclaim confirmed: free=%lld MiB >= %lld MiB after %lldms",
               static_cast<long long>(cu_free / (1024 * 1024)),
               static_cast<long long>(need_b / (1024 * 1024)),
               static_cast<long long>(elapsed_ms));
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  leto_log("reclaim wait timeout: free still short of %lld MiB",
           static_cast<long long>(need_b / (1024 * 1024)));
  return false;
}

// Invoke the registered Python kill callback under the GIL. Returns false
// if no callback is registered or it threw. The callback returns
// (memory_freed, killed_pid); killed_pid is 0 when no specific pid is known
// (e.g. the standby's rank had not registered when the kill landed).
static bool run_kill_callback(bool* memory_freed_out = nullptr,
                              unsigned int* killed_pid_out = nullptr) {
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
    if (memory_freed_out != nullptr) {
      *memory_freed_out = py::cast<bool>(t[0]);
    }
    if (killed_pid_out != nullptr) {
      *killed_pid_out = py::cast<unsigned int>(t[1]);
    }
    return true;
  } catch (py::error_already_set& e) {
    e.discard_as_unraisable("leto_free_mem_callback");
    return false;
  } catch (const std::exception& e) {
    leto_log("kill callback threw: %s", e.what());
    return false;
  }
}

struct LetoFreeMemCallback final : public c10::FreeMemoryCallback {
  bool Execute() override {
    // Fires on every allocator cache miss, BEFORE the allocator tries
    // cudaMalloc or releases its own cache. Does exactly one NVML read
    // (plus the tail-reuse walk for large requests) OUTSIDE g_state_mutex,
    // then publishes the estimate and takes the keep/KILL decision in one
    // short critical section so (est, reserved) stay mutually consistent
    // with the broker's admission (see g_state_mutex discipline).
    if (!g_reservation_active.load(std::memory_order_acquire)) {
      return false;  // broker not installed: nothing to protect or reclaim
    }
    const int64_t request_b = static_cast<int64_t>(g_request_bytes);
    const int request_mb = static_cast<int>(request_b / (1024 * 1024));
    const int target_pid = g_standby_pid.load(std::memory_order_relaxed);

    int64_t est = 0, nvml_free = 0, total_b = 0, standby_actual = 0;
    nvmlDevice_t handle{};
    if (!resolve_nvml_handle(&handle) ||
        !compute_used_estimate(handle, target_pid, request_b, &est,
                               &nvml_free, &total_b, &standby_actual)) {
      // NVML unavailable: never kill blind (read_nvml_mem already logged);
      // PyTorch's native release-then-retry flow proceeds.
      return false;
    }

    // Decision ladder, per-miss fresh, on free_cap := capacity - est:
    //   free_cap >= granted + pending  -> keep
    //   free_cap in [granted, ..)      -> CANCEL the soft reservation (free:
    //     nothing was allocated; the pending phase-2 GRANT gets denied and
    //     the standby redoes both phases)
    //   starved                        -> KILL: this allocation cannot
    //     proceed without reclaimed memory — req_charge exceeds free minus
    //     the standby's committed-but-unallocated remainder. Guarded by
    //     something-to-reclaim (a kill of a memoryless standby frees
    //     nothing and can wedge the device) and the per-generation latch.
    const int64_t free_phys = g_pub_free.load(std::memory_order_relaxed);
    const int64_t req_charge =
        g_pub_req_charge.load(std::memory_order_relaxed);
    int64_t granted_now = 0;
    bool pressure = false, should_fire = false, canceled = false;
    {
      std::lock_guard<std::mutex> lk(g_state_mutex);
      g_used_estimate.store(est, std::memory_order_relaxed);
      granted_now = g_granted.load(std::memory_order_relaxed);
      const int64_t resv_pending =
          g_reserved_pending.load(std::memory_order_relaxed);
      const int64_t free_cap = total_b - est;
      if (resv_pending > 0 && free_cap < granted_now + resv_pending &&
          free_cap >= granted_now) {
        g_reserved_pending.store(0, std::memory_order_relaxed);
        canceled = true;
      }
      const bool starved =
          (free_cap < granted_now) &&
          req_charge >
              free_phys -
                  std::max<int64_t>(0, granted_now - standby_actual);
      pressure = starved && (standby_actual > 0 || granted_now > 0);
      should_fire =
          pressure &&
          g_kill_armed.exchange(false, std::memory_order_relaxed);
    }
    if (canceled) {
      // Always logged: each cancellation is a kill that didn't happen.
      leto_log("fmcb CANCEL reservation: free=%lld MiB < granted=%lld + "
               "pending reservation (est=%lld)",
               static_cast<long long>((total_b - est) / (1024 * 1024)),
               static_cast<long long>(granted_now / (1024 * 1024)),
               static_cast<long long>(est / (1024 * 1024)));
    }

    // Invariant (B): standby_actual <= granted. Rate-limited diagnostic.
    if (standby_actual > granted_now + RESV_ASSERT_TOL_BYTES) {
      const uint64_t n =
          g_assert_b_violations.fetch_add(1, std::memory_order_relaxed);
      if ((n & (n + 1)) == 0) {  // print at n = 0,1,3,7,15,...
        leto_log("ASSERT(B) VIOLATED: standby_actual=%lld MiB > "
                 "granted=%lld MiB (+tol)",
                 static_cast<long long>(standby_actual / (1024 * 1024)),
                 static_cast<long long>(granted_now / (1024 * 1024)));
      }
    }
    // Diagnostic: KILL decisions always; "keep" evaluations only with
    // LETO_FMCB_VERBOSE=1 (they dominate log volume and the fprintf sits
    // on the allocation path). One line per miss = the estimate the broker
    // consumes until this rank's next miss.
    if (should_fire || g_verbose_decisions) {
      int64_t alloc_reserved_b = -1, alloc_alloc_b = -1;
      if (g_verbose_decisions) {
        try {
          const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
              c10::cuda::current_device());
          constexpr size_t kAgg = static_cast<size_t>(
              c10::CachingDeviceAllocator::StatType::AGGREGATE);
          alloc_reserved_b = stats.reserved_bytes[kAgg].current;
          alloc_alloc_b = stats.allocated_bytes[kAgg].current;
        } catch (const std::exception&) {
        }
      }
      leto_log(
          "fmcb request=%d MiB -> %s | est=%lld = (total=%lld - free=%lld "
          "- stdby=%lld) + req_charge=%lld[reuse=%lld,self_release=%lld] "
          "| grant_reserved=%lld alloc_reserved=%lld alloc_allocated=%lld "
          "armed=%d pressure=%d",
          request_mb, should_fire ? "KILL" : "keep",
          static_cast<long long>(est / (1024 * 1024)),
          static_cast<long long>(
              g_pub_total.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(
              g_pub_free.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(
              g_pub_actual.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(
              g_pub_req_charge.load(std::memory_order_relaxed) /
              (1024 * 1024)),
          static_cast<long long>(
              g_pub_reuse.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(
              g_pub_released.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(granted_now / (1024 * 1024)),
          static_cast<long long>(alloc_reserved_b >= 0
                                     ? alloc_reserved_b / (1024 * 1024)
                                     : -1),
          static_cast<long long>(alloc_alloc_b >= 0
                                     ? alloc_alloc_b / (1024 * 1024)
                                     : -1),
          g_kill_armed.load(std::memory_order_relaxed) ? 1 : 0,
          pressure ? 1 : 0);
    }
    if (!should_fire) {
      return false;
    }
    g_num_kill_standby_called.fetch_add(1, std::memory_order_relaxed);

    // Synchronous by nature: the gate guarantees this malloc cannot proceed
    // until reclaimed memory lands, and Execute cannot race itself (it runs
    // under the device allocator lock).
    bool memory_freed = false;
    unsigned int killed_pid = 0;
    if (!run_kill_callback(&memory_freed, &killed_pid)) {
      return false;
    }
    if (!memory_freed) {
      return false;
    }
    // Wait for exactly what the retry needs (the release ladder supplies
    // the rest), not for full context teardown.
    leto_log("waiting for free >= %lld MiB after standby kill (pid=%u)",
             static_cast<long long>(req_charge / (1024 * 1024)), killed_pid);
    return wait_for_free_at_least(req_charge);
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

  // --- size-bearing entry points: stamp the per-thread pending size for
  // Execute(), then delegate. Pure recording, no locks. ---
  DataPtr allocate(size_t n) override {
    ::leto::record_request(n);
    return real_->allocate(n);
  }
  void* raw_alloc(size_t nbytes) override {
    ::leto::record_request(nbytes);
    return real_->raw_alloc(nbytes);
  }
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    ::leto::record_request(nbytes);
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
  m.doc() = "Leto standby memory reservation: recording allocator, "
            "reservation broker, and the reclaim (kill) FreeMemoryCallback. "
            "Execute() publishes used_estimate on every allocator cache "
            "miss; the broker grants standby reservations against it.";

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

  m.def("get_num_kill_standby_called", []() {
    return leto::g_num_kill_standby_called.load(std::memory_order_relaxed);
  });

  m.def("reset_num_kill_standby_called", []() {
    leto::g_num_kill_standby_called.store(0, std::memory_order_relaxed);
  });

  // --- recording allocator ---

  m.def("install_recording_allocator", []() {
    return leto::install_recording_allocator();
  });

  m.def("is_recording_allocator_installed", []() {
    return leto::recording_allocator_installed();
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
  // Current allocator memory fraction on the broker's device — observability
  // for the reservation budget (1.0/original when nothing is granted).
  m.def("get_memory_fraction", []() {
    return c10::cuda::CUDACachingAllocator::getMemoryFraction(
        static_cast<c10::DeviceIndex>(leto::g_broker_device));
  });

  m.def("get_granted_mb", []() {
    return static_cast<int>(
        leto::g_granted.load(std::memory_order_relaxed) / (1024 * 1024));
  });

  m.def("get_reserved_pending_mb", []() {
    return static_cast<int>(
        leto::g_reserved_pending.load(std::memory_order_relaxed) /
        (1024 * 1024));
  });

  // Current used_estimate (MiB) — the value the broker grants against.
  // INT64_MAX/2^20 until the first publish (deny-until-published).
  m.def("get_used_estimate_mb", []() {
    return static_cast<long long>(
        leto::g_used_estimate.load(std::memory_order_relaxed) /
        (1024 * 1024));
  });

  // Tests/observability: force one fresh NVML read into used_estimate. In
  // production only Execute() (per cache miss) and the start_broker seed
  // publish it, so an idle process's estimate ages by design. request=0:
  // this path has no allocation in flight.
  m.def("refresh_used_estimate", []() {
    py::gil_scoped_release nogil;  // NVML driver call; no Python touched
    nvmlDevice_t handle{};
    if (!leto::resolve_nvml_handle(&handle)) return false;
    int64_t est = 0;
    if (!leto::compute_used_estimate(
            handle, leto::g_standby_pid.load(std::memory_order_relaxed),
            /*request_bytes=*/0, &est, nullptr, nullptr, nullptr)) {
      return false;
    }
    std::lock_guard<std::mutex> lk(leto::g_state_mutex);
    leto::g_used_estimate.store(est, std::memory_order_relaxed);
    return true;
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
  m.attr("OFF_REQ_TYPE") =
      static_cast<int>(offsetof(leto::ReservationLedger, req_type));
  m.attr("REQ_RESERVE") = leto::RESV_REQ_RESERVE;
  m.attr("REQ_GRANT") = leto::RESV_REQ_GRANT;
  m.attr("REQ_ROLLBACK") = leto::RESV_REQ_ROLLBACK;
}
