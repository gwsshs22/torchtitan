"""Correctness tests for resilient optimizer variants.

Tests:
  1. Manual fault recovery: snapshot → step → corrupt → restore → replay → verify
  2. ResilientOptimizer (exp bootstrap): chunked step matches optimizer.step()
  3. Exhaustive fault injection: fault at every marker.fill_ call, with
     memcpy corruption and adam corruption variants, verify recovery.

Note on hooks: ResilientOptimizer no longer invokes the optimizer's
``_optimizer_step_pre_hooks`` / ``_optimizer_step_post_hooks``.  The MoE
expert-bias balancing that used to be wired up as a pre-hook is now
implemented in-line inside ``ResilientOptimizer.step`` (via
``_atomic_update_expert_bias``) so a fault mid-update is rollback-able.
These tests build a synthetic optimizer with no MoE module attached, so
that path is a no-op here; correctness of the chunked AdamW step is the
remaining property under test.

Usage:
    uv run python -m torchtitan.experiments.resilient_opt.tests.test_correctness \
        --optimizer-info /path/to/optimizer_info.json
"""

import sys

import torch

from torchtitan.experiments.resilient_opt.resilient_opt import ResilientOptimizer
from torchtitan.experiments.resilient_opt.tests.common import (
    MockRmpClient,
    OptimizerList,
    create_optimizer_from_info,
    get_local,
    load_optimizer_info,
    make_parser,
    populate_grads,
    restore,
    snapshot,
)


# ---------------------------------------------------------------------------
# Fault injection helpers
# ---------------------------------------------------------------------------


class FaultInjected(Exception):
    pass


class _CountingMarker:
    """Wraps marker tensor to count fill_ calls during a normal step."""

    def __init__(self, real):
        self._real = real
        self.count = 0

    def fill_(self, v):
        self.count += 1
        return self._real.fill_(v)

    def item(self):
        return self._real.item()


class _FaultingMarker:
    """Wraps marker tensor; raises FaultInjected at the N-th fill_ call."""

    def __init__(self, real, fault_at):
        self._real = real
        self._fault_at = fault_at
        self._count = 0

    def fill_(self, v):
        if self._count == self._fault_at:
            self._count += 1
            raise FaultInjected(f"fault at fill_ #{self._fault_at}")
        self._count += 1
        return self._real.fill_(v)

    def item(self):
        return self._real.item()


def _states_match(params, optimizer, gt_params, gt_exp_avg, gt_exp_avg_sq, gt_steps):
    """Check if current optimizer state matches ground truth."""
    for i, p in enumerate(params):
        if not torch.equal(gt_params[i], p.data):
            return False
        if not torch.equal(gt_exp_avg[i], optimizer.state[p]["exp_avg"]):
            return False
        if not torch.equal(gt_exp_avg_sq[i], optimizer.state[p]["exp_avg_sq"]):
            return False
        if not torch.equal(gt_steps[i], optimizer.state[p]["step"]):
            return False
    return True


def test_manual_fault_recovery(params, optimizer):
    """Simulate mid-step fault and verify snapshot+replay recovery."""
    pre_params, pre_optim = snapshot(params, optimizer)

    populate_grads(params, seed=200)
    saved_grads = [p.grad.clone() for p in params]

    # Ground truth
    optimizer.step()
    optimizer.zero_grad()
    correct_params = [p.data.clone() for p in params]

    # Restore, re-step, then corrupt first half
    restore(params, optimizer, pre_params, pre_optim)
    populate_grads(params, seed=200)
    optimizer.step()

    n_corrupt = len(params) // 2
    for i in range(n_corrupt):
        params[i].data.copy_(pre_params[i])
        if id(params[i]) in pre_optim:
            for k, v in pre_optim[id(params[i])].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[params[i]][k].copy_(v)

    corrupted = any(
        not torch.equal(correct_params[i], params[i].data)
        for i in range(len(params))
    )
    print(f"  State corrupted after fault: {corrupted}")
    assert corrupted, "Fault simulation failed to corrupt state"

    # Recovery
    restore(params, optimizer, pre_params, pre_optim)
    for p, g in zip(params, saved_grads):
        p.grad = g
    optimizer.step()
    optimizer.zero_grad()

    all_match = True
    for i, (correct, p) in enumerate(zip(correct_params, params)):
        if not torch.equal(correct, p.data):
            diff = (correct - p.data).abs().max().item()
            print(f"  Param {i}: MISMATCH max_diff={diff:.6e}")
            all_match = False

    print(f"  Recovery matches ground truth: {all_match}")
    return all_match


def test_resilient_optimizer(params, optimizer, device):
    """Test ResilientOptimizer (exp bootstrap) step matches optimizer.step()."""
    pre_params, pre_optim = snapshot(params, optimizer)
    populate_grads(params, seed=300)
    saved_grads = [p.grad.clone() for p in params]

    # Ground truth
    optimizer.step()
    correct_params = [p.data.clone() for p in params]
    correct_exp_avg = [optimizer.state[p]["exp_avg"].clone() for p in params]
    correct_exp_avg_sq = [optimizer.state[p]["exp_avg_sq"].clone() for p in params]
    correct_steps = [optimizer.state[p]["step"].clone() for p in params]

    # Resilient step
    restore(params, optimizer, pre_params, pre_optim)
    for p, g in zip(params, saved_grads):
        p.grad = g.clone()

    rmp = MockRmpClient()
    resilient = ResilientOptimizer(
        OptimizerList(optimizer),
        rmp,
        torch.device(device),
        init_chunk_size_mb=1,
        max_chunk_size_mb=256,
    )
    resilient.bind()
    resilient.step()
    rmp.close()

    step_match = True
    for i, p in enumerate(params):
        if not torch.equal(correct_params[i], p.data):
            diff = (correct_params[i] - p.data).abs().max().item()
            print(f"  [step] Param {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_exp_avg[i], optimizer.state[p]["exp_avg"]):
            diff = (correct_exp_avg[i] - optimizer.state[p]["exp_avg"]).abs().max().item()
            print(f"  [step] exp_avg {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_exp_avg_sq[i], optimizer.state[p]["exp_avg_sq"]):
            diff = (correct_exp_avg_sq[i] - optimizer.state[p]["exp_avg_sq"]).abs().max().item()
            print(f"  [step] exp_avg_sq {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_steps[i], optimizer.state[p]["step"]):
            print(
                f"  [step] step {i}: {correct_steps[i].item()} "
                f"vs {optimizer.state[p]['step'].item()}"
            )
            step_match = False

    print(f"  Chunked step matches optimizer.step(): {step_match}")
    return step_match


def test_multi_step(params, optimizer, device, num_steps=5):
    """Test that N consecutive resilient steps match N vanilla optimizer steps."""
    pre_params, pre_optim = snapshot(params, optimizer)

    # Ground truth: N vanilla steps with different grads each iteration
    for step_i in range(num_steps):
        populate_grads(params, seed=500 + step_i)
        optimizer.step()
    gt_params = [p.data.clone() for p in params]
    gt_exp_avg = [optimizer.state[p]["exp_avg"].clone() for p in params]
    gt_exp_avg_sq = [optimizer.state[p]["exp_avg_sq"].clone() for p in params]
    gt_steps = [optimizer.state[p]["step"].clone() for p in params]

    # Resilient: N steps with the same grad sequence
    restore(params, optimizer, pre_params, pre_optim)
    rmp = MockRmpClient()
    resilient = ResilientOptimizer(
        OptimizerList(optimizer), rmp, torch.device(device),
    )
    resilient.bind()
    for step_i in range(num_steps):
        populate_grads(params, seed=500 + step_i)
        resilient.step()
    rmp.close()

    all_match = True
    for i, p in enumerate(params):
        if not torch.equal(gt_params[i], p.data):
            diff = (gt_params[i] - p.data).abs().max().item()
            print(f"  param {i}: MISMATCH max_diff={diff:.6e}")
            all_match = False
        if not torch.equal(gt_exp_avg[i], optimizer.state[p]["exp_avg"]):
            diff = (gt_exp_avg[i] - optimizer.state[p]["exp_avg"]).abs().max().item()
            print(f"  exp_avg {i}: MISMATCH max_diff={diff:.6e}")
            all_match = False
        if not torch.equal(gt_exp_avg_sq[i], optimizer.state[p]["exp_avg_sq"]):
            diff = (gt_exp_avg_sq[i] - optimizer.state[p]["exp_avg_sq"]).abs().max().item()
            print(f"  exp_avg_sq {i}: MISMATCH max_diff={diff:.6e}")
            all_match = False
        if not torch.equal(gt_steps[i], optimizer.state[p]["step"]):
            print(
                f"  step {i}: {gt_steps[i].item()} "
                f"vs {optimizer.state[p]['step'].item()}"
            )
            all_match = False

    print(f"  {num_steps} steps match: {all_match}")
    return all_match


def test_exhaustive_fault_recovery(params, optimizer, device):
    """Inject fault at every marker.fill_ call point and verify recovery.

    For each fault point, tests up to three fault types:
      - "marker":  fault during marker.fill_ itself (just raise)
      - "memcpy":  fault during backup memcpy (raise at next fill_, corrupt
                   the partially-written backup buffer)
      - "adam":    fault during fused adam kernel (raise at next fill_,
                   corrupt the partially-updated chunk's slices)

    The step() timeline:
        fill_(PRE_RUNNING) ──counter+=1──► [if MoE: EB_BACKUP/UPDATE]
        ──fill_(PRE_DONE)──► chunks ──fill_(POST_RUNNING)──► fill_(IDLE)

    Chunk timeline:
        fill_(K*2)  ──backup──► fill_(K*2+1)  ──adam──► harvest

    Total fills = 4 (PRE_RUNNING, PRE_DONE, POST_RUNNING, IDLE) + 2*num_chunks.
    For tests without MoE the EB_BACKUP/EB_UPDATE markers are skipped.

    Mapping `fault_at` to the marker-at-fault (FaultingMarker raises BEFORE
    the fill, so marker stays at the value set by the previous successful
    fill):
        fault_at = 0 → marker at fault = IDLE (initial state)
        fault_at = 1 → marker at fault = PRE_RUNNING
        fault_at = 2 → marker at fault = PRE_DONE
        fault_at = 3 + 2K (K=0..N-1) → marker at fault = chunk K backup_running
                                       ⇒ "memcpy" corruption applicable
        fault_at = 4 + 2K (K=0..N-1) → marker at fault = chunk K step_running
                                       ⇒ "adam" corruption applicable
        fault_at = 2 + 2N → marker at fault = chunk N-1 step_running (last)
        fault_at = 3 + 2N → marker at fault = POST_RUNNING (final = IDLE fill)
    """
    dev = torch.device(device)

    pre_p, pre_o = snapshot(params, optimizer)
    populate_grads(params, seed=400)
    saved_grads = [p.grad.clone() for p in params]

    # The step value the caller expects after completion.
    # This is what the caller knows — not derived from ground truth.
    pre_step_val = int(optimizer.state[params[0]]["step"].item())
    resume_step = pre_step_val + 1

    # -- Ground truth --
    optimizer.step()
    gt_params = [p.data.clone() for p in params]
    gt_exp_avg = [optimizer.state[p]["exp_avg"].clone() for p in params]
    gt_exp_avg_sq = [optimizer.state[p]["exp_avg_sq"].clone() for p in params]
    gt_steps = [optimizer.state[p]["step"].clone() for p in params]

    # -- Counting + recording pass --
    # Record which (param_id, start, end) slices belong to each chunk.
    restore(params, optimizer, pre_p, pre_o)
    for p, g in zip(params, saved_grads):
        p.grad = g.clone()
    rmp = MockRmpClient()
    r = ResilientOptimizer(OptimizerList(optimizer), rmp, dev)
    r.bind()
    counter = _CountingMarker(r._marker)
    r._marker = counter

    # Use the precomputed schedule to record chunk-to-slice mapping.
    chunk_slices: list[list[tuple[int, int, int]]] = []
    for chunk in r._schedule:
        chunk_slices.append([(id(sl.param_ref), sl.start, sl.end) for sl in chunk])

    r.step()
    total_fills = counter.count
    num_chunks = len(chunk_slices)
    rmp.close()

    # Map param id → param index for targeted corruption
    param_id_to_idx = {id(p): i for i, p in enumerate(params)}

    print(f"  Total marker.fill_ calls: {total_fills}, chunks: {num_chunks}")

    # -- Test every (fault_at, fault_type) combination --
    all_passed = True
    for fault_at in range(total_fills):
        is_last = fault_at == total_fills - 1  # IDLE fill (no chunks left)

        # Map fault_at to the chunk whose backup or step was in progress at
        # the time of fault — see the docstring's mapping.  "memcpy" is
        # applicable when backup was running for a chunk (marker even);
        # "adam" is applicable when the fused step was running (marker
        # odd).  PRE/POST/IDLE markers and faults outside [3, 3+2N) only
        # exercise the "marker" case.
        fault_types = ["marker"]
        memcpy_chunk: int | None = None
        adam_chunk: int | None = None
        if fault_at >= 3 and fault_at % 2 == 1:
            K = (fault_at - 3) // 2
            if K < num_chunks:
                fault_types.append("memcpy")
                memcpy_chunk = K
        elif fault_at >= 4 and fault_at % 2 == 0:
            K = (fault_at - 4) // 2
            if K < num_chunks:
                fault_types.append("adam")
                adam_chunk = K

        for ft in fault_types:
            label = f"fill_#{fault_at}/{ft}"

            # -- 1. Setup: restore pre-step state + grads --
            restore(params, optimizer, pre_p, pre_o)
            for p, g in zip(params, saved_grads):
                p.grad = g.clone()

            rmp = MockRmpClient()
            r = ResilientOptimizer(OptimizerList(optimizer), rmp, dev)
            r.bind()
            real_marker = r._marker
            r._marker = _FaultingMarker(real_marker, fault_at)

            # -- 2. Run step until fault --
            try:
                r.step()
                print(f"  FAIL [{label}]: no exception raised")
                all_passed = False
                rmp.close()
                continue
            except FaultInjected:
                pass

            # -- 3. Apply targeted corruption for the faulted chunk --
            if ft == "memcpy":
                # Backup of memcpy_chunk was partial: corrupt its backup
                # destination so that any code that reads it sees garbage.
                # Recovery should overwrite the backup destination with
                # fresh data (the chunk's live state is intact at this
                # point, and the bug-free recovery path is to redo the
                # backup + step).
                if memcpy_chunk == 0:
                    # Chunk 0's backup lives in cpu_buffer.
                    r._cpu_buffer.fill_(0x42)
                else:
                    # Chunk K (>0) backs up into freed_grad_segments
                    # collected from chunks 0..K-1's grad memory.
                    freed: list[torch.Tensor] = []
                    for i in range(memcpy_chunk):
                        for sl in r._schedule[i]:
                            grad_local = get_local(sl.param_ref.grad)
                            seg = (
                                grad_local.view(-1)[sl.start : sl.end]
                                .view(torch.uint8)
                                .reshape(-1)
                            )
                            freed.append(seg)
                    for seg in freed:
                        seg.fill_(0x42)
            elif ft == "adam":
                # Adam for adam_chunk was interrupted mid-kernel: corrupt
                # that chunk's slices in live state with NaN to force the
                # recovery path to restore from the (correct) CPU/freed
                # backup before redoing the step.
                for pid, start, end in chunk_slices[adam_chunk]:
                    pidx = param_id_to_idx.get(pid)
                    if pidx is None:
                        continue
                    p = params[pidx]
                    state = optimizer.state[p]
                    with torch.no_grad():
                        get_local(p).view(-1)[start:end].fill_(float("nan"))
                        get_local(state["exp_avg"]).view(-1)[start:end].fill_(
                            float("nan")
                        )
                        get_local(state["exp_avg_sq"]).view(-1)[start:end].fill_(
                            float("nan")
                        )
                # Also corrupt step tensors for the faulted chunk.
                for sl in r._schedule[adam_chunk]:
                    if sl.is_first_slice:
                        sl.step.fill_(999999)

            # -- 4. Verify state diverged from ground truth --
            # For "marker" with no actual corruption applied, the live state
            # may still match GT (specifically when the fault landed AFTER
            # all chunks completed but before IDLE was set, or before
            # PRE_RUNNING was set).  Don't require divergence in those
            # boundary cases.
            corruption_applied = ft != "marker"
            if corruption_applied:
                if _states_match(
                    params, optimizer, gt_params, gt_exp_avg, gt_exp_avg_sq, gt_steps
                ):
                    print(f"  FAIL [{label}]: state matches GT before recovery")
                    all_passed = False
                    rmp.close()
                    continue

            # -- 5. Recovery: new optimizer with SAME rmp --
            # Optimizer states + grads persist in RMP. No checkpoint restore.
            # maybe_recover() resumes from the interrupted chunk.
            r2 = ResilientOptimizer(OptimizerList(optimizer), rmp, dev)
            r2.bind()

            recovered = r2.maybe_recover(resume_step=resume_step)

            if not recovered:
                assert ft == "marker" and all_done
                print(f"  SKIP [{label}] (maybe_recover returned False)")
                rmp.close()
                continue

            # -- 6. Verify recovery produced correct results --
            if _states_match(params, optimizer, gt_params, gt_exp_avg, gt_exp_avg_sq, gt_steps):
                print(f"  PASS [{label}]")
            else:
                # Print first mismatch for debugging
                for i, p in enumerate(params):
                    if not torch.equal(gt_params[i], p.data):
                        diff = (gt_params[i] - p.data).abs().max().item()
                        print(f"  FAIL [{label}]: param {i} max_diff={diff:.6e}")
                        break
                    if not torch.equal(gt_exp_avg[i], optimizer.state[p]["exp_avg"]):
                        diff = (gt_exp_avg[i] - optimizer.state[p]["exp_avg"]).abs().max().item()
                        print(f"  FAIL [{label}]: exp_avg {i} max_diff={diff:.6e}")
                        break
                    if not torch.equal(gt_exp_avg_sq[i], optimizer.state[p]["exp_avg_sq"]):
                        diff = (gt_exp_avg_sq[i] - optimizer.state[p]["exp_avg_sq"]).abs().max().item()
                        print(f"  FAIL [{label}]: exp_avg_sq {i} max_diff={diff:.6e}")
                        break
                    if not torch.equal(gt_steps[i], optimizer.state[p]["step"]):
                        print(
                            f"  FAIL [{label}]: step {i}: "
                            f"{gt_steps[i].item()} vs {optimizer.state[p]['step'].item()}"
                        )
                        break
                all_passed = False

            rmp.close()

    return all_passed


def main():
    parser = make_parser("Resilient optimizer correctness tests")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    info = load_optimizer_info(args.optimizer_info)
    params, optimizer = create_optimizer_from_info(info, device=args.device)
    print(f"Created {len(params)} params on {args.device}")

    print("\n=== Manual Fault Recovery Test ===")
    passed_manual = test_manual_fault_recovery(params, optimizer)

    # Re-create for resilient optimizer test (clean state)
    params, optimizer = create_optimizer_from_info(info, device=args.device)

    print("\n=== ResilientOptimizer Test ===")
    passed_resilient = test_resilient_optimizer(params, optimizer, args.device)

    # Re-create for multi-step test (clean state)
    params, optimizer = create_optimizer_from_info(info, device=args.device)

    print("\n=== Multi-Step Test (5 steps) ===")
    passed_multi = test_multi_step(params, optimizer, args.device)

    # Re-create for exhaustive fault test (clean state)
    params, optimizer = create_optimizer_from_info(info, device=args.device)

    print("\n=== Exhaustive Fault Recovery Test ===")
    passed_fault = test_exhaustive_fault_recovery(params, optimizer, args.device)

    passed = passed_manual and passed_resilient and passed_multi and passed_fault
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
