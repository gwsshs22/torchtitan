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

// Rank tag resolved once from torchrun's env (RANK, falling back to
// LOCAL_RANK; "?" outside torchrun). Multiple ranks' stderr interleave in
// one per-node log file, so untagged lines leave forensics guessing which
// GPU's callback said what (the 2026-07 AWS OOM analysis took a detour
// over exactly this).
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
// return to the driver right now, for `device`. Replays the allocator's
// own release logic over snapshot():
//  - non-expandable segment: cudaFree-able iff nothing in it is active
//    (adjacent free blocks always merge, so active_size == 0 means the
//    segment is one whole free block);
//  - expandable segment: for each inactive block, only the whole pages
//    strictly inside it unmap (ExpandableSegment::unmap rounds begin up
//    and end down to the page); snapshot() reports one SegmentInfo per
//    contiguous *mapped* range, whose boundaries are page-aligned by
//    construction, so block offsets from `address` give exact page math.
// Do NOT use the stats formula reserved - allocated - inactive_split for
// this: PyTorch keeps no inactive_split stat for expandable segments
// (free_block explicitly skips them), so under expandable_segments it
// credits every cached byte -- including the sub-page fragments wedged
// between live tensors that no release path can ever return. On the
// 2026-07 AWS OOM that phantom credit was 945 MiB: used_estimate ran a
// full margin too low and the must-KILL evaluation kept instead.
// Event-pending blocks count as active in snapshot() although the release
// path would sync-and-free them, and private pools are skipped although
// freeable ones would be released: both make this an under-estimate, so
// the estimate errs toward "used" and never banks on memory that cannot
// actually be freed. snapshot() re-takes the (recursive) device allocator
// lock; safe from every caller (same reentrancy as getDeviceStats).
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

// The walk above is O(all blocks) -- measured 1.8 ms at ~18k blocks
// (mew2, 2026-07) -- far too heavy for every cache miss. Split its two
// consumers: the *published* estimate (broker grant admission) tolerates
// ~100 ms staleness (admission already races the standby's allocation by
// design), so it reuses a TTL-cached value clamped to the cache's current
// size; keep/KILL *decisions* never ride on a stale credit -- Execute()
// forces a fresh walk exactly when the zero-credit conservative check
// shows pressure (see there).
constexpr int64_t kLbTtlMs = 100;
static std::atomic<int64_t> g_lb_cached{0};
static std::atomic<int64_t> g_lb_cached_at_ms{-kLbTtlMs};

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


// ---------------------------------------------------------------------------
// Reservation broker — standby memory reservation against the active allocator
// ---------------------------------------------------------------------------
//
// The active process is the single broker for any GPU memory the co-located
// standby may use. A standby posts a Reserve(bytes) request into a per-rank
// shared-memory ledger; the broker thread (here, in the active process)
// grants iff the device can spare it, accounting the standby at its
// *reservation* rather than its current usage. The grant is enforced two
// ways: the allocator budget (apply_reservation_budget caps the active's
// reserved bytes so it physically cannot grow into the reservation), and the
// FreeMemCallback, which reclaims (kills) the standby when a request cannot
// fit even within the budget. See the plan: the no-OOM guarantee factors into
//   (A) active enforces:  others_used + granted <= capacity - margin
//       (allocator-budget-enforced; the callback is the reclaim backstop)
//   (B) protocol asserts:  standby_actual <= granted
// Memory truth flows one way: the Callback reads NVML (driver ground truth,
// never cudaMemGetInfo) on each cache miss and publishes an
// inflight-adjusted estimate; the broker only consumes it.

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
  int64_t effective_free;  // 40 : broker-published grantable bytes (observability)
  int64_t standby_actual;  // 48 : layout-reserved; no longer written
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

// Broker state — fully lock-free. `g_granted` (== reserved_to_standby) has
// two writers: the broker's grant raise (a CAS, so a concurrent reset is
// never overwritten) and reset_granted's zero-store (kill path / epoch).
// Every writer calls apply_reservation_budget() afterwards, which derives
// the cap from the CURRENT value, so any interleaving converges.
static std::atomic<bool> g_reservation_active{false};  // gates reservation path
static std::atomic<int64_t> g_granted{0};  // == reserved_to_standby
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
// The ONLY state shared between the broker and Execute() — four plain
// atomics, so the two sides never share a lock and the broker can never
// stall an allocation:
//   g_used_estimate  (Execute -> broker)  see definition below
//   g_granted        (broker -> Execute)  reserved_to_standby, declared above
//   g_standby_pid    (broker -> Execute)  declared above with broker state
//   g_kill_armed     (broker arms, Execute consumes)  see below
//
//   used_estimate := (physical used excl. standby − releasable cache)
//                    + in_flight_alloc_bytes + margin                [bytes]
//                    (i.e. LIVE memory: the active's reusable cache is
//                    creditable — the cap returns it to the device)
//
// Published by Execute() from its per-cache-miss NVML read (and seeded once
// at start_broker). The inflight term makes it post-malloc-correct — a raw
// pre-malloc reading over-credits free by the triggering allocation's size
// (2026-07 review). Initialized to INT64_MAX so the broker DENIES until the
// first publish — never grant blind. Broker admission:
//   GRANT iff used_estimate + new_reservation <= capacity.
static std::atomic<int64_t> g_used_estimate{INT64_MAX};

// Kill latch — at most ONE reclaim per standby generation. The broker arms
// it whenever it services a reservation request (a live standby is asking);
// Execute() consumes it on fire. used_estimate deliberately counts the
// active's cache as used, so on a cache-full device the kill condition
// (used_estimate + reserved > capacity) holds for essentially every cache
// miss; without the latch that would re-fire against the same dying standby
// throughout its teardown window (the 2026-06 churn). With it: one kill,
// reservation voided, and the replacement standby parks with ZERO GPU
// footprint (denied before its first grant, no CUDA context) until the
// device genuinely has room.
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

// Lock-free in-flight allocation accounting — the allocator wrapper is pure
// recording and never takes a lock on the allocation path. The broker's
// grant admission subtracts g_inflight_bytes from NVML-free: completed
// mallocs are in the NVML reading, mid-cudaMalloc ones are in this counter,
// and everything that starts after a grant is bounded by the allocator cap.
// Charged at request_physical_ub (page-rounded physical footprint), not raw
// bytes: a cache miss maps whole pages, and the raw charge under-counts by
// up to a page. Queued allocations that will be served from cache are
// overcounted — conservative (worst case a DENY the standby retries ~1s
// later).
static std::atomic<int64_t> g_inflight_bytes{0};

struct InflightRecord {
  int64_t n_;
  explicit InflightRecord(size_t n)
      : n_(request_physical_ub(static_cast<int64_t>(n))) {
    g_inflight_bytes.fetch_add(n_, std::memory_order_relaxed);
  }
  ~InflightRecord() {
    g_inflight_bytes.fetch_sub(n_, std::memory_order_relaxed);
  }
  InflightRecord(const InflightRecord&) = delete;
  InflightRecord& operator=(const InflightRecord&) = delete;
};

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
    leto_log("%s failed: %s", nvml_api_name, nvmlErrorString(nerr));
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

// Read device total/free bytes (NVML) and the GPU memory used by
// `target_pid` (0 if absent or NVML can't attribute per-process memory).
// NVML, not cudaMemGetInfo — the driver view is ground truth.
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
// cap the active's reserved bytes at (device_total - reserved - margin) via
// setMemoryFraction. PyTorch's malloc slow path then releases cached blocks
// incrementally (page-wise with expandable segments) whenever an allocation
// would cross the cap, so the active physically cannot grow into the
// standby's reservation — and, unlike the retired RELEASE tier's
// emptyCache-per-near-boundary-alloc (measured 2026-07: 5-16 full cache
// flushes per step, +15..60% step time), the cache is trimmed once and the
// cap stops it from re-growing. reserved==0 restores the pre-install
// fraction. Takes NO argument: it derives the fraction from the CURRENT
// g_granted, so concurrent grant/reset writers converge — whichever apply
// runs last reflects the latest reservation, no lock needed. Callable from
// the broker thread (no leto lock held) and the kill path (the allocator
// lock is recursive, same reentrancy as getDeviceStats).
static void apply_reservation_budget() {
  if (g_device_total_bytes <= 0) return;
  const int64_t reserved_b = g_granted.load(std::memory_order_relaxed);
  double frac = g_orig_fraction;
  if (reserved_b > 0) {
    const int64_t margin_b =
        static_cast<int64_t>(
            g_reservation_margin_mb.load(std::memory_order_relaxed)) *
        1024 * 1024;
    const int64_t cap_b = g_device_total_bytes - reserved_b - margin_b;
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
}

// One fresh NVML read -> recompute and publish g_used_estimate:
//   used_estimate = max(0, total - free - standby_actual   [used excl. standby]
//                          - releasable_cache)             [the active's own
//                                                           reusable cache]
//                   + g_inflight_bytes + margin
// Counting in-flight allocations makes the value post-malloc-correct (a raw
// pre-malloc reading over-credits free by the triggering allocation's size);
// in-flight allocations that end up served from cache leave it conservative
// until the next publish. Subtracting the releasable cache counts only LIVE
// memory as used: without it, a cache-full device makes the kill condition
// true on every miss (2026-06 futile-kill churn) and starves grant
// admission forever (2026-07 runs). The cache is legitimately creditable
// because the allocator cap forces it back to the device — but only lazily,
// on the active's next cache miss, so a grant admitted against cache races
// the standby's allocation for ~0.1-5s (accepted; the standby dies and
// rotates if it loses).
//
// The credit is the page-exact releasable_lower_bound (see there), NOT the
// stats formula reserved - allocated - inactive_split: under
// expandable_segments that formula credits stranded sub-page fragments no
// release path can return (945 MiB on the 2026-07 AWS OOM — a full margin
// of phantom credit, enough to flip a must-KILL into keep). The walk costs
// ~1.8 ms at 18k blocks, so `exact_lb` picks the freshness: false reuses a
// kLbTtlMs-cached value (grant admission; clamped to the cache's current
// size), true forces a fresh walk (decisions at the pressure boundary).
// Raw values go to the out-params for the caller's own decision math.
// Returns false (estimate untouched) on NVML failure.
static bool refresh_used_estimate(nvmlDevice_t handle, int target_pid,
                                  int64_t* free_b_out, int64_t* total_b_out,
                                  int64_t* standby_actual_out,
                                  bool exact_lb = false) {
  int64_t free_b = 0, total_b = 0, actual_b = 0;
  if (!read_nvml_mem(handle, target_pid, &free_b, &total_b, &actual_b)) {
    return false;
  }
  int64_t cache_b = 0;  // reserved - allocated: cheap cap on the credit
  try {
    // getDeviceStats re-takes the device allocator lock; safe from every
    // caller (recursive on the allocating thread inside Execute; briefly
    // contended from the start_broker seed / pybind refresh).
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        c10::cuda::current_device());
    constexpr size_t kAgg = static_cast<size_t>(
        c10::CachingDeviceAllocator::StatType::AGGREGATE);
    cache_b = std::max<int64_t>(
        0,
        stats.reserved_bytes[kAgg].current -
            stats.allocated_bytes[kAgg].current);
  } catch (const std::exception& e) {
    // Stats unavailable -> count the cache as used (conservative).
    leto_log("getDeviceStats failed: %s", e.what());
  }
  int64_t releasable_b = 0;
  if (cache_b > 0) {
    const int64_t now = mono_ms();
    if (exact_lb ||
        now - g_lb_cached_at_ms.load(std::memory_order_relaxed) >= kLbTtlMs) {
      try {
        releasable_b = releasable_lower_bound(c10::cuda::current_device());
      } catch (const std::exception& e) {
        // Snapshot unavailable -> count the cache as used (conservative).
        leto_log("releasable_lower_bound failed: %s", e.what());
      }
      g_lb_cached.store(releasable_b, std::memory_order_relaxed);
      g_lb_cached_at_ms.store(now, std::memory_order_relaxed);
    } else {
      // TTL reuse for the published (grant) estimate; never credit more
      // than the cache currently holds.
      releasable_b =
          std::min(g_lb_cached.load(std::memory_order_relaxed), cache_b);
    }
  } else {
    g_lb_cached.store(0, std::memory_order_relaxed);
  }
  const int64_t inflight_b = g_inflight_bytes.load(std::memory_order_relaxed);
  const int64_t margin_b =
      static_cast<int64_t>(
          g_reservation_margin_mb.load(std::memory_order_relaxed)) *
      1024 * 1024;
  const int64_t used_excl_b =
      std::max<int64_t>(0, total_b - free_b - actual_b - releasable_b);
  g_used_estimate.store(used_excl_b + inflight_b + margin_b,
                        std::memory_order_relaxed);
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

// Broker thread (active process). Polls the shm ledger, resets the grant
// ledger on a new standby epoch, and services reservation requests. It
// performs NO NVML/driver calls whatsoever: grants are decided against the
// Callback-published inflight-adjusted estimate, and allocations never wait
// on the broker (the wrapper is pure recording). History: a 15 ms per-rank
// NVML timer here serialized 8 ranks/host on the driver lock
// (~1000 calls/s/host), slowing training ~6x and nvidia-smi with it
// (2026-07 AWS measure_mem runs).
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
      g_granted.store(0, std::memory_order_relaxed);
      g_standby_epoch_seen = epoch;
      g_kill_armed.store(false, std::memory_order_relaxed);
      apply_reservation_budget();
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
      const int64_t used_est =
          g_used_estimate.load(std::memory_order_relaxed);
      const int64_t capacity =
          g_nvml_total_bytes.load(std::memory_order_relaxed);
      int verdict = RESV_VERDICT_DENY;
      const int64_t granted_before = g_granted.load(std::memory_order_relaxed);
      int64_t granted_after = granted_before;
      const int64_t needed = req_bytes - granted_before;
      if (needed <= 0) {
        verdict = RESV_VERDICT_GRANT;  // already covered; no change
      } else if (capacity > 0 && used_est <= capacity - req_bytes) {
        int64_t expected = granted_before;
        if (g_granted.compare_exchange_strong(expected, req_bytes,
                                              std::memory_order_relaxed)) {
          granted_after = req_bytes;
          apply_reservation_budget();
          verdict = RESV_VERDICT_GRANT;
        }
      }
      resv_store_rel(&L->effective_free,
                     capacity > 0 ? std::max<int64_t>(0, capacity - used_est)
                                  : int64_t{0});
      resv_store_rel(&L->granted, granted_after);
      resv_store_rel(&L->resp_verdict, verdict);
      resv_store_rel(&L->resp_seq, req_seq);  // bump last → signals done
      last_req_seq = req_seq;
      // Diagnostic: DENYs and grant-raising decisions always; idempotent
      // re-grants (needed<=0, one per peer-denied unanimity round) only
      // with LETO_FMCB_VERBOSE=1.
      if (verdict == RESV_VERDICT_DENY || needed > 0 || g_verbose_decisions)
        leto_log(
            "broker req_seq=%u request=%lld MiB -> %s | "
            "used_est=%lld capacity=%lld granted_before=%lld needed=%lld "
            "granted_after=%lld MiB",
            req_seq,
            static_cast<long long>(req_bytes / (1024 * 1024)),
            verdict == RESV_VERDICT_GRANT ? "GRANT" : "DENY",
            static_cast<long long>(used_est / (1024 * 1024)),
            static_cast<long long>(capacity / (1024 * 1024)),
            static_cast<long long>(granted_before / (1024 * 1024)),
            static_cast<long long>(needed / (1024 * 1024)),
            static_cast<long long>(granted_after / (1024 * 1024)));
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
  // starts at INT64_MAX (deny-until-published).
  {
    nvmlDevice_t handle{};
    int64_t total_b = 0;
    const int target_pid = resv_load_acq(&g_ledger->standby_pid);
    g_standby_pid.store(target_pid, std::memory_order_relaxed);
    if (resolve_nvml_handle(&handle) &&
        refresh_used_estimate(handle, target_pid, nullptr, &total_b,
                              nullptr)) {
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
// Lock-free: callable from the kill callback, which runs inside the
// FreeMemCallback deep in the allocator's malloc path. A concurrent broker
// raise cannot clobber this zero (the raise is a CAS), and the budget
// re-derives from the current value at apply time.
static void reset_granted() {
  g_granted.store(0, std::memory_order_relaxed);
  if (g_ledger != nullptr) resv_store_rel(&g_ledger->granted, int64_t{0});
  // Restore the allocator budget (kill path runs inside the malloc chain —
  // the device allocator lock is recursive, so re-entering is safe, and the
  // retried allocation must not stay capped by the dead standby's grant).
  apply_reservation_budget();
}

struct LetoFreeMemCallback final : public c10::FreeMemoryCallback {
  bool Execute() override {
    // Fires on every allocator cache miss, BEFORE the allocator tries
    // cudaMalloc or releases its own cache. Does exactly one NVML read,
    // publishes used_estimate for the broker, and decides keep/KILL.
    // No leto lock is ever taken here — everything shared with the broker
    // is a plain atomic.
    if (!g_reservation_active.load(std::memory_order_acquire)) {
      return false;  // broker not installed: nothing to protect or reclaim
    }
    const int request_mb =
        static_cast<int>(g_request_bytes / (1024ULL * 1024ULL));
    const int64_t reserved = g_granted.load(std::memory_order_relaxed);
    const int target_pid = g_standby_pid.load(std::memory_order_relaxed);

    int64_t nvml_free = 0, total_b = 0, standby_actual = 0;
    nvmlDevice_t handle{};
    if (!resolve_nvml_handle(&handle) ||
        !refresh_used_estimate(handle, target_pid, &nvml_free, &total_b,
                               &standby_actual)) {
      // NVML unavailable: never kill blind (read_nvml_mem already logged);
      // PyTorch's native release-then-retry flow proceeds.
      return false;
    }
    int64_t est =
        g_used_estimate.load(std::memory_order_relaxed);  // just published
    // The published estimate may credit the releasable cache from the TTL
    // cache (see refresh_used_estimate) — fine for grant admission, but a
    // keep/KILL decision must not ride on a stale credit in either
    // direction (stale-high: wrong keep, the 2026-07 AWS OOM; stale-low:
    // spurious kill against the one-shot latch). If even ZERO cache credit
    // shows no pressure, keep is provably safe with any credit; otherwise
    // re-refresh with a fresh page-exact walk and decide on that. This
    // bounds the O(blocks) walk to boundary evaluations.
    {
      const int64_t inflight_b =
          g_inflight_bytes.load(std::memory_order_relaxed);
      const int64_t margin_b =
          static_cast<int64_t>(
              g_reservation_margin_mb.load(std::memory_order_relaxed)) *
          1024 * 1024;
      const int64_t used_zero_credit =
          std::max<int64_t>(0, total_b - nvml_free - standby_actual) +
          inflight_b + margin_b;
      if (used_zero_credit + reserved > total_b &&
          (standby_actual > 0 || reserved > 0)) {
        if (refresh_used_estimate(handle, target_pid, &nvml_free, &total_b,
                                  &standby_actual, /*exact_lb=*/true)) {
          est = g_used_estimate.load(std::memory_order_relaxed);
        }
      }
    }

    // (B) standby_actual <= reserved. A breach means the standby physically
    // holds more than it reserved; loud, rate-limited diagnostic.
    if (standby_actual > reserved + RESV_ASSERT_TOL_BYTES) {
      const uint64_t n =
          g_assert_b_violations.fetch_add(1, std::memory_order_relaxed);
      if ((n & (n + 1)) == 0) {  // print at n = 0,1,3,7,15,...
        leto_log("ASSERT(B) VIOLATED: standby_actual=%lld MiB > "
                 "reserved=%lld MiB (+tol)",
                 static_cast<long long>(standby_actual / (1024 * 1024)),
                 static_cast<long long>(reserved / (1024 * 1024)));
      }
    }

    // Reclaim decision:
    //   KILL iff used_estimate + reserved > capacity
    // with something to take back (standby_actual > 0 or reserved > 0;
    // firing with nothing held and nothing reserved is a futile kill: a
    // SIGKILL'd mid-NCCL-init context lingers 10+ s on the GPU and can
    // wedge the device), AND the kill latch is armed. used_estimate counts
    // the active's cache as used, so on a cache-full device this condition
    // holds for most cache misses — the latch (one kill per standby
    // generation, re-armed only when a standby requests a grant) is what
    // bounds the reclaim rate instead of allocator introspection. After
    // the kill the reservation is voided by the kill callback
    // (reset_granted), and the replacement standby stays parked at zero
    // GPU footprint until the device genuinely has room for a grant.
    const bool pressure = (est + reserved > total_b) &&
                          (standby_actual > 0 || reserved > 0);
    const bool should_fire =
        pressure && g_kill_armed.exchange(false, std::memory_order_relaxed);
    // Diagnostic: KILL decisions always; "keep" evaluations only with
    // LETO_FMCB_VERBOSE=1 (they dominate log volume ~40:1 and the fprintf
    // sits on the allocation path).
    if (should_fire || g_verbose_decisions) {
      // releasable = last page-exact walk (fresh whenever this evaluation
      // was at the pressure boundary; TTL-aged otherwise).
      leto_log(
          "fmcb request=%d MiB -> %s | used_est=%lld "
          "reserved=%lld standby_actual=%lld nvml_free=%lld "
          "releasable=%lld capacity=%lld MiB",
          request_mb, should_fire ? "KILL" : "keep",
          static_cast<long long>(est / (1024 * 1024)),
          static_cast<long long>(reserved / (1024 * 1024)),
          static_cast<long long>(standby_actual / (1024 * 1024)),
          static_cast<long long>(nvml_free / (1024 * 1024)),
          static_cast<long long>(
              g_lb_cached.load(std::memory_order_relaxed) / (1024 * 1024)),
          static_cast<long long>(total_b / (1024 * 1024)));
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
        leto_log("kill callback threw: %s", e.what());
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
      leto_log("kill confirmed; no pid to track");
      return true;
    }

    // Poll NVML compute-running processes every 100ms for up to 10s,
    // waiting for `killed_pid` to disappear from this GPU (reuses the
    // handle resolved above). Mirrors the enumeration used by PyTorch's
    // CUDACachingAllocator::reportProcessMemoryInfo.
    leto_log("waiting for killed pid=%u to leave GPU", killed_pid);
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
        leto_log("nvmlDeviceGetComputeRunningProcesses failed: %s",
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
        leto_log("killed pid=%u gone from GPU; kill confirmed after %lldms",
                 killed_pid, static_cast<long long>(elapsed_ms));
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - poll_start)
            .count();
    leto_log("poll timeout after %lldms; pid=%u still present on GPU",
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
  // Pure recording, no locks: record_request stamps the per-thread pending
  // size for Execute(); InflightRecord maintains the global in-flight sum
  // the broker subtracts at grant time. That accounting (plus the allocator
  // cap) is what keeps grants consistent with concurrent allocations.
  DataPtr allocate(size_t n) override {
    ::leto::record_request(n);
    ::leto::InflightRecord infl(n);
    return real_->allocate(n);
  }
  void* raw_alloc(size_t nbytes) override {
    ::leto::record_request(nbytes);
    ::leto::InflightRecord infl(nbytes);
    return real_->raw_alloc(nbytes);
  }
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    ::leto::record_request(nbytes);
    ::leto::InflightRecord infl(nbytes);
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

  // Most recent allocation request observed by the RecordingAllocator, in
  // MiB, across all threads. Observability only (logging / tests); the kill
  // decision uses the per-thread value, not this.
  m.def("get_last_request_mb", []() {
    return static_cast<int>(
        leto::g_last_request_bytes.load(std::memory_order_relaxed) /
        (1024ULL * 1024ULL));
  });

  // Page-exact lower bound (bytes) on what emptyCache() can return to the
  // driver for the current device right now. Exposed for validation: the
  // used_estimate credit uses this same computation.
  m.def("compute_releasable_lb_bytes", []() {
    return leto::releasable_lower_bound(c10::cuda::current_device());
  });

  // Upper bound (bytes) on the physical memory a request of `req_bytes`
  // newly maps on a cache miss (round_size + whole-page rounding); the
  // charge InflightRecord applies per in-flight allocation.
  m.def(
      "request_physical_ub_bytes",
      [](int64_t req_bytes) { return leto::request_physical_ub(req_bytes); },
      py::arg("req_bytes"));

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

  // Current used_estimate (MiB) — the value the broker grants against.
  // INT64_MAX/2^20 until the first publish (deny-until-published).
  m.def("get_used_estimate_mb", []() {
    return static_cast<long long>(
        leto::g_used_estimate.load(std::memory_order_relaxed) /
        (1024 * 1024));
  });

  // Tests/observability: force one fresh NVML read into used_estimate. In
  // production only Execute() (per cache miss) and the start_broker seed
  // publish it, so an idle process's estimate ages by design.
  m.def("refresh_used_estimate", []() {
    py::gil_scoped_release nogil;  // NVML driver call; no Python touched
    nvmlDevice_t handle{};
    if (!leto::resolve_nvml_handle(&handle)) return false;
    // exact_lb: an explicit observability refresh wants the current truth,
    // not a TTL-aged credit (tests assert against this value).
    return leto::refresh_used_estimate(
        handle, leto::g_standby_pid.load(std::memory_order_relaxed), nullptr,
        nullptr, nullptr, /*exact_lb=*/true);
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
