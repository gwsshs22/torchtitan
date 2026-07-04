"""Tests for the standby GPU-memory reservation broker (leto_free_mem_callback).

Single-GPU, no torchrun. Run with:
    CUDA_VISIBLE_DEVICES=0 python -m torchtitan.components.mem.test_reservation

Covers:
  1. ledger layout — Python (progressive.py) vs C++ struct agree.
  2. broker logic — grant / deny / cumulative idempotence / reset / epoch.
  3. deadlock   — active reclaim path takes the GIL (inside the allocator
                  lock) while the broker thread services grants; concurrency
                  smoke for the lock-free allocation path.
  4. adversary  — (2 processes) a standby reserves a large target but only
                  partially allocates; the active then balloons and must
                  reclaim the standby BEFORE growing into the reservation, with
                  no active OOM.
  5. assert_b   — (2 processes) a standby allocates past its grant; the broker
                  counts the standby_actual > granted violation.
  6. cache      — (2 processes) RC-A regression: used_estimate counts only
                  live memory, so a self-satisfiable big allocation on a
                  cache-full device must NOT reclaim the standby, while
                  genuine pressure (exceeds live headroom) must — once
                  (kill latch), with the allocation succeeding after.
"""
import atexit
import mmap
import os
import signal
import struct
import subprocess
import sys
import time

# The deployment always runs with expandable segments (envs.sh), and the
# physical used_estimate charges Execute's own request net of the expandable
# tail reuse — set it before any torch import so the tests exercise the same
# allocator mode.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")


def _mk_ledger(m, path):
    with open(path, "wb") as f:
        f.write(b"\x00" * m.LEDGER_NBYTES)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, m.LEDGER_NBYTES)
    os.close(fd)
    return mm


def _reserve(m, mm, seq, cumulative_bytes, timeout=5.0):
    _I64.pack_into(mm, m.OFF_REQ_BYTES, int(cumulative_bytes))
    _U32.pack_into(mm, m.OFF_REQ_SEQ, seq)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _U32.unpack_from(mm, m.OFF_RESP_SEQ)[0] == seq:
            return _I32.unpack_from(mm, m.OFF_RESP_VERDICT)[0]
        time.sleep(0.002)
    raise TimeoutError("no broker response")


def test_layout():
    from torchtitan.components.init import progressive

    progressive.assert_ledger_layout()
    print("[1] ledger layout Python<->C++ MATCH")


def test_broker_logic(m):
    GB = 1024 ** 3
    path = "/dev/shm/leto_resvtest_logic"
    mm = _mk_ledger(m, path)
    assert m.attach_reservation_ledger(path)
    m.set_reservation_margin_mb(512)
    assert m.start_broker(512)
    atexit.register(m.stop_broker)
    _I32.pack_into(mm, m.OFF_STANDBY_PID, 999999)  # fake pid -> actual=0
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 1)
    # start_broker seeds used_estimate synchronously (it starts at INT64_MAX
    # = deny-until-published); a sane MiB value proves the seed read worked.
    est_mb = m.get_used_estimate_mb()
    assert est_mb < 10 ** 7, f"used_estimate not seeded (est={est_mb}MiB)"

    assert _reserve(m, mm, 1, 8 * GB) == m.VERDICT_GRANT
    assert 8000 <= m.get_granted_mb() <= 8400
    frac_8g = m.get_memory_fraction()
    assert frac_8g < 1.0, f"grant did not cap the allocator (frac={frac_8g})"
    assert _reserve(m, mm, 2, 16 * GB) == m.VERDICT_GRANT  # raise target
    assert 16000 <= m.get_granted_mb() <= 16800
    assert m.get_memory_fraction() < frac_8g  # bigger grant, tighter budget
    assert _reserve(m, mm, 3, 16 * GB) == m.VERDICT_GRANT  # idempotent retry
    assert 16000 <= m.get_granted_mb() <= 16800
    assert _reserve(m, mm, 4, 200 * GB) == m.VERDICT_DENY  # exceeds device
    assert 16000 <= m.get_granted_mb() <= 16800
    m.reset_granted()
    assert m.get_granted_mb() == 0
    assert m.get_memory_fraction() >= 0.999  # budget restored on reset
    assert _reserve(m, mm, 5, 4 * GB) == m.VERDICT_GRANT
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 2)  # new standby instance
    deadline = time.time() + 2  # epoch reset lands on the next broker tick
    while time.time() < deadline and m.get_granted_mb() != 0:
        time.sleep(0.02)
    assert m.get_granted_mb() == 0  # epoch bump voids grant
    m.stop_broker()
    os.unlink(path)
    print("[2] broker logic (grant/deny/cumulative/reset/epoch) OK")


def test_physical_admission(m):
    """[2b] Physical admission regression (the 2026-07 standby OOMs): a
    device whose free memory sits in the ACTIVE'S CACHE must DENY a request
    that exceeds physical free — the old estimate credited the cache and
    granted reservations nothing could physically back."""
    import torch

    GiB = 1024 ** 3
    path = "/dev/shm/leto_resvtest_phys"
    mm = _mk_ledger(m, path)
    assert m.attach_reservation_ledger(path)
    m.set_reservation_margin_mb(128)
    assert m.start_broker(128)
    _I32.pack_into(mm, m.OFF_STANDBY_PID, 999999)  # fake pid -> actual=0
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 1)

    total_b = torch.cuda.get_device_properties(0).total_memory
    # Fill most of the device with LIVE memory, then convert a large slab
    # to CACHE (freed but still mapped): physical free stays small while
    # the allocator holds a big reusable pool.
    hold_b = int(total_b * 0.60)
    cache_b = int(total_b * 0.30)
    hold = torch.empty(hold_b, dtype=torch.uint8, device="cuda:0")
    cache = torch.empty(cache_b, dtype=torch.uint8, device="cuda:0")
    del cache  # stays mapped in the allocator cache
    torch.cuda.synchronize()
    assert m.refresh_used_estimate()
    est_mb = m.get_used_estimate_mb()
    free_b, _ = torch.cuda.mem_get_info()

    # Request more than physical free (but far less than free+cache):
    # must DENY under physical admission.
    req_b = int(free_b + 2 * GiB)
    verdict = _reserve(m, mm, 1, req_b)
    print(f"[2b] physical admission: free={free_b // 2**20}MiB "
          f"cache~{cache_b // 2**20}MiB est={est_mb}MiB "
          f"request={req_b // 2**20}MiB -> "
          f"{'DENY' if verdict == m.VERDICT_DENY else 'GRANT'}")
    assert verdict == m.VERDICT_DENY, "cache credited as grantable again"

    # A request that fits physical free (minus margin) must still GRANT.
    req_ok_b = max(int(free_b - 2 * GiB), 1 << 30)
    assert _reserve(m, mm, 2, req_ok_b) == m.VERDICT_GRANT
    m.reset_granted()
    m.stop_broker()
    del hold
    torch.cuda.empty_cache()
    os.unlink(path)
    print("[2b] physical admission (cache-rich/free-poor DENY) OK")


def test_deadlock(m):
    import threading

    import torch

    path = "/dev/shm/leto_resvtest_deadlock"
    mm = _mk_ledger(m, path)
    assert m.attach_reservation_ledger(path)
    m.set_reservation_margin_mb(10 ** 7)  # huge -> reclaim fires every miss
    calls = {"n": 0}

    def kill_cb():
        calls["n"] += 1
        return (False, 0)

    m.set_kill_callback(kill_cb)
    assert m.start_broker(10 ** 7)
    atexit.register(m.clear_kill_callback)
    # Use our own pid as the "standby": the huge margin denies every grant
    # (granted stays 0), and the reclaim gate only fires when there is
    # something reclaimable (standby_actual > 0 or granted > 0). Our own
    # GPU usage gives standby_actual > 0 so the reclaim path is exercised;
    # this test only cares about concurrency (reclaim under the GIL vs the
    # broker thread servicing grants), not reclaim semantics.
    _I32.pack_into(mm, m.OFF_STANDBY_PID, os.getpid())
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 1)
    # The broker mirrors standby_pid out of the ledger once per tick (500ms);
    # the reclaim gate needs it (standby_actual > 0), so let one tick pass.
    time.sleep(0.7)
    stop = threading.Event()

    def requester():
        seq = 0
        while not stop.is_set():
            seq += 1
            _I64.pack_into(mm, m.OFF_REQ_BYTES, 4 * 1024 ** 3)
            _U32.pack_into(mm, m.OFF_REQ_SEQ, seq)
            time.sleep(0.003)

    t = threading.Thread(target=requester, daemon=True)
    t.start()
    for i in range(60):
        torch.cuda.empty_cache()
        x = torch.empty((5_000_000 + i * 137) % 50_000_000 + 1_000_000,
                        dtype=torch.float32, device="cuda:0")
        del x
        torch.cuda.synchronize()
        time.sleep(0.05)
    stop.set()
    t.join(timeout=2)
    # The kill latch limits fires to one per broker-serviced request
    # (500ms tick), so a few seconds of concurrent reclaim+grant cycles
    # yields a handful of fires — enough to exercise the interleaving.
    assert calls["n"] >= 2, f"reclaim path under-exercised ({calls['n']})"
    m.stop_broker()
    m.clear_kill_callback()
    os.unlink(path)
    print(f"[3] deadlock: {calls['n']} concurrent reclaim+grant cycles, no hang")


# --- two-process adversary / assertion(B) ---

_PREFIX = "/dev/shm/leto_resvtest_e2e_"
_LEDGER = _PREFIX + "0"


def _run_child(mode):
    os.environ["LETO_PROGRESSIVE_SHM_PREFIX"] = _PREFIX
    os.environ["LETO_PROCESS_GROUP_ID"] = "1"
    import torch
    from torchtitan.components.init import progressive

    torch.cuda.init()
    torch.empty(1, device="cuda:0")
    progressive.standby_register(0, epoch=1)
    total_mb = torch.cuda.get_device_properties(0).total_memory // 2 ** 20
    if mode == "cache":
        reserve_mb, alloc_mb = 3600, 3000
    elif mode == "adversary":
        # ~85% of the device (70000 MiB on the original ~80GiB dev machine):
        # large enough that the active's balloon must hit the reservation
        # boundary within a few GiB on any capacity.
        reserve_mb, alloc_mb = int(total_mb * 0.854), 2000
    else:
        reserve_mb, alloc_mb = 1000, 3000
    kind, _ = progressive._reserve(0, reserve_mb * 1024 * 1024, 0.005, None)
    print(f"CHILD reserve({reserve_mb}) -> {kind}", flush=True)
    # Every mode's premise needs the reservation granted; fail loudly here
    # (the parent then times out on CHILD_READY with this in child output)
    # rather than silently testing nothing.
    assert kind == "grant", f"child reserve({reserve_mb}MB) denied"
    buf = torch.empty(alloc_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda:0")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    print(f"CHILD_READY allocated={alloc_mb}", flush=True)
    time.sleep(120)


def _run_e2e(m, mode):
    import torch
    import torchtitan.components.mem as mem

    mm = _mk_ledger(m, _LEDGER)

    def kill_cb():
        pid = _I32.unpack_from(mm, m.OFF_STANDBY_PID)[0]
        if pid > 0:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Mirror the production _on_oom (train.py): voiding the grant also
        # restores the allocator budget — without this the dead standby's
        # cap would block the retried allocation forever.
        m.reset_granted()
        return (True, pid)

    mem.install_reservation_broker(_LEDGER, margin_mb=1024, kill_callback=kill_cb)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    child = subprocess.Popen(
        [sys.executable, __file__, "child", mode],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    ready = False
    deadline = time.time() + 90
    while time.time() < deadline:
        line = child.stdout.readline()
        if not line:
            break
        if "CHILD_READY" in line:
            ready = True
            break
    assert ready, "child never became ready"
    time.sleep(0.5)

    if mode == "adversary":
        m.reset_num_kill_standby_called()
        held, oom = [], False
        try:
            for i in range(20):  # up to 20 GiB; 80 GiB device never real-OOMs
                held.append(torch.empty(256 * 1024 * 1024, dtype=torch.float32,
                                        device="cuda:0"))
                torch.cuda.synchronize()
                if m.get_num_kill_standby_called() > 0:
                    break
        except torch.cuda.OutOfMemoryError:
            oom = True
        nkill = m.get_num_kill_standby_called()
        ok = (not oom) and nkill >= 1
        print(f"[4] adversary: active_OOM={oom} num_kill={nkill} -> "
              f"{'OK' if ok else 'FAIL'}")
    else:
        m.reset_assert_b_violations()
        for i in range(8):
            torch.cuda.empty_cache()
            _ = torch.empty(40_000_000 + i * 1000, dtype=torch.float32, device="cuda:0")
            torch.cuda.synchronize()
        v = m.get_assert_b_violations()
        ok = v >= 1
        print(f"[5] assert_b: violations={v} -> {'OK' if ok else 'FAIL'}")

    child.send_signal(signal.SIGKILL)
    child.wait(timeout=10)
    m.stop_broker()
    m.clear_kill_callback()
    try:
        os.unlink(_LEDGER)
    except FileNotFoundError:
        pass
    return ok


def _run_cache_test(m):
    """[6] RC-A regression: used_estimate counts only LIVE memory (the
    active's releasable cache is subtracted at publish), so a big
    cache-missing allocation the allocator can self-satisfy must NOT
    reclaim the standby — while genuine pressure (the request exceeds
    live headroom even after a full cache release) MUST, exactly once
    (kill latch), with the allocation succeeding after the reclaim.

    Geometry scaled to device capacity T (H=0.375T live, C=0.3125T
    freed-but-cached, standby holds ~3GiB granted):
      S2: fresh R1=H on a cache-full device -> NO kill (est credits the
          cache; pre-RC-A this fired the futile churn kill).
      S3: empty_cache, fresh R2 ~= free'+1GiB -> MUST kill (live memory
          alone cannot fit R2), and the allocation succeeds.
    """
    import torch
    import torchtitan.components.mem as mem

    GiB = 1024 ** 3
    total_mb = torch.cuda.get_device_properties(0).total_memory // 2 ** 20
    h_gib = int(total_mb * 0.375) // 1024
    c_gib = int(total_mb * 0.3125) // 1024
    r1_gib = h_gib
    mm = _mk_ledger(m, _LEDGER)

    def kill_cb():
        pid = _I32.unpack_from(mm, m.OFF_STANDBY_PID)[0]
        if pid > 0:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Mirror the production _on_oom (train.py): a confirmed kill voids
        # the cumulative grant so the dead standby's reservation stops
        # poisoning effective_free.
        m.reset_granted()
        return (True, pid)

    mem.install_reservation_broker(_LEDGER, margin_mb=128, kill_callback=kill_cb)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    child = subprocess.Popen(
        [sys.executable, __file__, "child", "cache"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    ready = False
    deadline = time.time() + 90
    while time.time() < deadline:
        line = child.stdout.readline()
        if not line:
            break
        if "CHILD_READY" in line:
            ready = True
            break
    assert ready, "child never became ready"
    time.sleep(0.5)

    ok = True
    try:
        # --- build the geometry ---
        held = [torch.empty(GiB, dtype=torch.uint8, device="cuda:0")
                for _ in range(h_gib)]
        cached = [torch.empty(GiB, dtype=torch.uint8, device="cuda:0")
                  for _ in range(c_gib)]
        del cached  # C GiB now freed-but-cached (whole releasable segments)
        torch.cuda.synchronize()
        time.sleep(0.2)  # let the kernel settle per-process accounting
        # Device free for the test's own geometry math (the guard exposes
        # only used_estimate; cudaMemGetInfo tracks NVML free within MiBs).
        free_mb = torch.cuda.mem_get_info(0)[0] // 2 ** 20
        stats = torch.cuda.memory_stats(0)
        reserved_mb = stats["reserved_bytes.all.current"] // 2 ** 20
        alloc_mb = stats["allocated_bytes.all.current"] // 2 ** 20
        print(f"[6] setup: H={h_gib}GiB C={c_gib}GiB R1={r1_gib}GiB "
              f"nvml_free={free_mb}MiB reserved={reserved_mb}MiB "
              f"allocated={alloc_mb}MiB cached={(reserved_mb - alloc_mb)}MiB")
        assert free_mb < r1_gib * 1024, "geometry broken: too much NVML-free"

        # --- S2: self-satisfiable big alloc must NOT reclaim the standby ---
        m.reset_num_kill_standby_called()
        big = torch.empty(r1_gib * GiB, dtype=torch.uint8, device="cuda:0")
        torch.cuda.synchronize()
        nkill = m.get_num_kill_standby_called()
        s2_ok = nkill == 0 and child.poll() is None
        print(f"[6] S2 (self-satisfiable {r1_gib}GiB alloc): num_kill={nkill} "
              f"standby_alive={child.poll() is None} -> "
              f"{'OK' if s2_ok else 'FAIL (futile churn kill)'}")
        ok = ok and s2_ok

        # --- S3: genuine pressure must still reclaim (once) ---
        del big
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.3)
        free_mb = torch.cuda.mem_get_info(0)[0] // 2 ** 20
        r2_mb = free_mb + 1024  # ~1GiB more than device-free; standby has 3GiB
        m.reset_num_kill_standby_called()
        oom = False
        try:
            big2 = torch.empty(r2_mb * 2 ** 20, dtype=torch.uint8,
                               device="cuda:0")
            torch.cuda.synchronize()
            del big2
        except torch.cuda.OutOfMemoryError:
            oom = True
        nkill = m.get_num_kill_standby_called()
        s3_ok = (not oom) and nkill >= 1
        print(f"[6] S3 (genuine pressure {r2_mb}MiB vs free={free_mb}MiB): "
              f"num_kill={nkill} OOM={oom} -> {'OK' if s3_ok else 'FAIL'}")
        ok = ok and s3_ok
        del held
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGKILL)
        child.wait(timeout=10)
        m.stop_broker()
        m.clear_kill_callback()
        torch.cuda.empty_cache()
        try:
            os.unlink(_LEDGER)
        except FileNotFoundError:
            pass
    return ok


def main():
    import torch
    import torchtitan.components.mem as mem

    assert torch.cuda.is_available(), "need a CUDA device"
    m = mem._load_module()
    torch.cuda.init()
    torch.empty(1, device="cuda:0")

    test_layout()
    test_broker_logic(m)
    test_physical_admission(m)
    test_deadlock(m)
    ok_adv = _run_e2e(m, "adversary")
    ok_b = _run_e2e(m, "assert_b")
    ok_cache = _run_cache_test(m)
    if ok_adv and ok_b and ok_cache:
        print("ALL RESERVATION TESTS PASSED")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "child":
        _run_child(sys.argv[2])
    else:
        main()
