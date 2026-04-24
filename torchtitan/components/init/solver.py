import json
from pathlib import Path

import pulp


def solve_schedule(durations, memories, edges, time_limit=None, gap=None, verbose=True):
    """
    Args:
        durations: list of N positive numbers, dur[i]
        memories:  list of N non-negative numbers, mem[i]
        edges:     list of (u, v) pairs meaning task u must finish before task v
        time_limit: optional seconds for the solver
        gap:        optional relative MIP gap (e.g. 0.01 = stop at 1% from optimal)

    Returns:
        order: list of task indices in scheduled order
        cost:  the prefix-memory objective value
    """
    N = len(durations)
    assert len(memories) == N

    # Forced-precedence set: x[i,j] = 1 means i is scheduled before j.
    # We compute the transitive closure so the solver gets every implied edge fixed up front.
    forced = [[False] * N for _ in range(N)]
    for (u, v) in edges:
        forced[u][v] = True
    # Floyd-Warshall on the reachability matrix
    for k in range(N):
        for i in range(N):
            if forced[i][k]:
                row_k = forced[k]
                row_i = forced[i]
                for j in range(N):
                    if row_k[j]:
                        row_i[j] = True

    prob = pulp.LpProblem("prefix_memory_scheduling", pulp.LpMinimize)

    # Create x[i,j] for all ordered pairs i != j.
    # Pin to 1 if forced by transitive precedence, 0 if reverse is forced.
    x = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if forced[i][j]:
                x[i, j] = 1  # constant, not a variable
            elif forced[j][i]:
                x[i, j] = 0
            else:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Objective: prefix-memory cost.
    # Each task t contributes mem[t] to cum-memory of every task scheduled at-or-after t.
    # So total cost = sum_t mem[t] * (dur[t] + sum_{u: t before u} dur[u])
    #              = sum_t mem[t] * (dur[t] + sum_{u != t} dur[u] * x[t,u])
    prob += pulp.lpSum(
        memories[t] * (durations[t] + pulp.lpSum(durations[u] * x[t, u]
                                                 for u in range(N) if u != t))
        for t in range(N)
    )

    # Antisymmetry: exactly one of (i,j), (j,i) is 1, for free pairs.
    for i in range(N):
        for j in range(i + 1, N):
            if isinstance(x[i, j], pulp.LpVariable) and isinstance(x[j, i], pulp.LpVariable):
                prob += x[i, j] + x[j, i] == 1

    # Transitivity: forbid 3-cycles. Only needed when at least one of the three
    # variables is free; if all three are pinned, the constraint is either
    # automatically satisfied or the inputs are inconsistent.
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                terms = [x[i, j], x[j, k], x[k, i]]
                if any(isinstance(t, pulp.LpVariable) for t in terms):
                    prob += pulp.lpSum(terms) <= 2

    # Solve
    solver_kwargs = {"msg": 1 if verbose else 0}
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit
    if gap is not None:
        solver_kwargs["gapRel"] = gap
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"):
        # "Not Solved" can appear when stopped by time limit with a feasible solution
        if prob.status != 1:  # 1 = Optimal in PuLP
            print(f"Warning: solver status = {status}")

    # Recover the order: position of task i = number of tasks scheduled before it
    pos = [0] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = x[j, i] if not isinstance(x[j, i], pulp.LpVariable) else x[j, i].value()
            if val is not None and val > 0.5:
                pos[i] += 1
    order = sorted(range(N), key=lambda i: pos[i])

    # Compute objective value directly from the order (sanity check)
    cost = 0.0
    cum_mem = 0.0
    for i in order:
        cum_mem += memories[i]
        cost += cum_mem * durations[i]

    return order, cost


def solve(profile_path, dependency_path=None, time_limit=None, gap=None, verbose=False):
    """Schedule init tasks using a profile JSON and the shipped dependency DAG.

    Args:
        profile_path: path to a per-rank profile JSON produced by
            ``torchtitan.init.run_init_sequence`` (fields: ``tasks``, each with
            ``task``, ``duration_s``, ``gpu_mem_before_mb``, ``gpu_mem_after_mb``).
        dependency_path: path to dependency.json. Defaults to the sibling
            ``dependency.json`` next to this file.

    Returns:
        dict with keys:
          names             — task names in declaration order (dependency.json order)
          order             — indices into ``names`` in scheduled order
          cost              — prefix-memory objective value for the solver's order
          durations         — per-task duration (seconds), indexed like ``names``
          memories          — per-task memory delta (MiB, clamped to >= 0)
          original_order    — indices into ``names`` matching the profile's recorded order
    """
    if dependency_path is None:
        dependency_path = Path(__file__).parent / "dependency.json"
    dep = json.loads(Path(dependency_path).read_text())
    profile = json.loads(Path(profile_path).read_text())

    profile_map = {t["task"]: t for t in profile["tasks"]}
    names = [t["name"] for t in dep["tasks"]]

    missing = [n for n in names if n not in profile_map]
    if missing:
        raise ValueError(f"tasks in dependency.json missing from profile: {missing}")
    extra = [t for t in profile_map if t not in set(names)]
    if extra:
        raise ValueError(f"tasks in profile not declared in dependency.json: {extra}")

    durations = [profile_map[n]["duration_s"] + 0.0001 for n in names]
    memories = [
        max(0.0, profile_map[n]["gpu_mem_after_mb"] - profile_map[n]["gpu_mem_before_mb"])
        for n in names
    ]
    name_to_i = {n: i for i, n in enumerate(names)}
    edges = [
        (name_to_i[p], name_to_i[t["name"]])
        for t in dep["tasks"]
        for p in t["depends_on"]
    ]

    order, cost = solve_schedule(
        durations, memories, edges,
        time_limit=time_limit, gap=gap, verbose=verbose,
    )
    original_order = [name_to_i[t["task"]] for t in profile["tasks"]]
    return {
        "names": names,
        "order": order,
        "cost": cost,
        "durations": durations,
        "memories": memories,
        "original_order": original_order,
    }


def _cost_of(order_idx, durations, memories):
    c = 0.0
    m = 0.0
    for i in order_idx:
        m += memories[i]
        c += m * durations[i]
    return c


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <profile.json>", file=sys.stderr)
        sys.exit(2)

    result = solve(sys.argv[1])
    names = result["names"]
    order = result["order"]
    durations = result["durations"]
    memories = result["memories"]
    original_order = result["original_order"]

    print(f"N={len(names)}  solver_cost={result['cost']:.4f}")
    print()
    print(f"{'#':>3} {'task':<40} {'dur_s':>8} {'mem_mb':>8}")
    print("-" * 62)
    for k, i in enumerate(order):
        print(f"{k:>3} {names[i]:<40} {durations[i]:>8.4f} {memories[i]:>8.1f}")

    orig_cost = _cost_of(original_order, durations, memories)
    solv_cost = _cost_of(order, durations, memories)
    print()
    print(f"Original order cost : {orig_cost:.4f}")
    print(f"Solver order cost   : {solv_cost:.4f}")
    print(f"Improvement         : {orig_cost - solv_cost:.4f}")

    # Verify against brute force
    # from itertools import permutations
    # def is_topo(perm, edges):
    #     pos = {t: i for i, t in enumerate(perm)}
    #     return all(pos[u] < pos[v] for u, v in edges)
    # def cost_of(perm, dur, mem):
    #     c, m = 0, 0
    #     for t in perm:
    #         m += mem[t]
    #         c += m * dur[t]
    #     return c
    # best = min((cost_of(p, durations, memories), p)
    #            for p in permutations(range(len(durations))) if is_topo(p, edges))
    # print(f"Brute-force optimum: order={list(best[1])}, cost={best[0]}")