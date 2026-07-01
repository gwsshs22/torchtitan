"""Tests for the standby GPU-memory reservation broker (leto_free_mem_callback).

Single-GPU, no torchrun. Run with:
    CUDA_VISIBLE_DEVICES=0 python -m torchtitan.components.mem.test_reservation

Covers:
  1. ledger layout — Python (progressive.py) vs C++ struct agree.
  2. broker logic — grant / deny / cumulative idempotence / reset / epoch.
  3. deadlock   — active reclaim path takes the GIL under broker_mutex while
                  the broker thread services grants; the pure-C++ broker must
                  not AB-BA deadlock.
  4. adversary  — (2 processes) a standby reserves a large target but only
                  partially allocates; the active then balloons and must
                  reclaim the standby BEFORE growing into the reservation, with
                  no active OOM.
  5. assert_b   — (2 processes) a standby allocates past its grant; the broker
                  counts the standby_actual > granted violation.
"""
import atexit
import mmap
import os
import signal
import struct
import subprocess
import sys
import time

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
    for _ in range(200):
        if m.get_cached_free_mb() > 0:
            break
        time.sleep(0.01)
    assert m.get_cached_free_mb() > 1000, "broker NVML cache not populated"

    assert _reserve(m, mm, 1, 8 * GB) == m.VERDICT_GRANT
    assert 8000 <= m.get_granted_mb() <= 8400
    assert _reserve(m, mm, 2, 16 * GB) == m.VERDICT_GRANT  # raise target
    assert 16000 <= m.get_granted_mb() <= 16800
    assert _reserve(m, mm, 3, 16 * GB) == m.VERDICT_GRANT  # idempotent retry
    assert 16000 <= m.get_granted_mb() <= 16800
    assert _reserve(m, mm, 4, 200 * GB) == m.VERDICT_DENY  # exceeds device
    assert 16000 <= m.get_granted_mb() <= 16800
    m.reset_granted()
    assert m.get_granted_mb() == 0
    assert _reserve(m, mm, 5, 4 * GB) == m.VERDICT_GRANT
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 2)  # new standby instance
    time.sleep(0.05)
    assert m.get_granted_mb() == 0  # epoch bump voids grant
    m.stop_broker()
    os.unlink(path)
    print("[2] broker logic (grant/deny/cumulative/reset/epoch) OK")


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
    _I32.pack_into(mm, m.OFF_STANDBY_PID, 999999)
    _U32.pack_into(mm, m.OFF_STANDBY_EPOCH, 1)
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
    stop.set()
    t.join(timeout=2)
    assert calls["n"] >= 30, "reclaim path under-exercised"
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
    reserve_mb = 70000 if mode == "adversary" else 1000
    alloc_mb = 2000 if mode == "adversary" else 3000
    kind, _ = progressive._reserve(0, reserve_mb * 1024 * 1024, 0.005, None)
    print(f"CHILD reserve({reserve_mb}) -> {kind}", flush=True)
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


def main():
    import torch
    import torchtitan.components.mem as mem

    assert torch.cuda.is_available(), "need a CUDA device"
    m = mem._load_module()
    torch.cuda.init()
    torch.empty(1, device="cuda:0")

    test_layout()
    test_broker_logic(m)
    test_deadlock(m)
    ok_adv = _run_e2e(m, "adversary")
    ok_b = _run_e2e(m, "assert_b")
    if ok_adv and ok_b:
        print("ALL RESERVATION TESTS PASSED")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "child":
        _run_child(sys.argv[2])
    else:
        main()
