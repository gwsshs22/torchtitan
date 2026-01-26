def get_snapshot_strategy(
    gap_times: dict[int, float],
    block_sizes: list[int],
    dtype_size: int,
    bandwidth_gbps: float,
    gap_threshold_ms: float,
    alpha: float,
    max_blocks_per_gap: int,
) -> dict[int, int]:
    """Compute snapshot strategy from gap times using actual block sizes.

    Gemini's algorithm (snapshot_strategy.py:get_snapshot_strategy):
    For each gap, compute how many blocks can fit based on bandwidth and gap duration.

    Args:
        gap_times: Map of gap_id -> minimum gap time in ms.
        block_sizes: Actual size of each block in elements.
        dtype_size: Element size in bytes.
        bandwidth_gbps: Network bandwidth in Gbps.
        gap_threshold_ms: Minimum gap to consider (ms).
        alpha: Utilization factor.
        max_blocks_per_gap: Cap on blocks per gap.

    Returns:
        Strategy map: gap_id -> number of blocks to snapshot.
    """
    # Filter gaps above threshold (Gemini: get_valid_comm_idle_time)
    valid_gaps = {
        k: v for k, v in gap_times.items() if v > gap_threshold_ms and k > 0
    }

    strategy: dict[int, int] = {}
    cur_block_id = 0
    total_blocks = len(block_sizes)

    for gap_id, gap_ms in sorted(valid_gaps.items()):
        # Compute max elements that can be transferred in this gap
        # Formula: gap_time_sec * bandwidth_bytes_per_sec / dtype_size * alpha
        # Note: Gemini divides by local_rank_size for intra-node NVLink sharing,
        # but FSDP P2P may span nodes, so we don't divide by fsdp_size here.
        # User can adjust bandwidth_gbps to account for shared bandwidth if needed.
        max_elements = (
            gap_ms / 1000
            * bandwidth_gbps
            * 1e9
            / 8
            / dtype_size
            * alpha
        )

        blocks = 0
        cur_elements = 0

        # Gemini: while cur_block_id < total_blocks_num and
        # cur_comm_size + block_sizes[cur_block_id] <= max_comm_size
        while cur_block_id < total_blocks:
            block_size = block_sizes[cur_block_id]
            if cur_elements + block_size > max_elements:
                break
            cur_elements += block_size
            cur_block_id += 1
            blocks += 1
            # Gemini: if blocks >= max_blocks: break
            if blocks >= max_blocks_per_gap:
                break

        if blocks > 0:
            strategy[gap_id] = blocks

        if cur_block_id >= total_blocks:
            break

    # If not all blocks fit, mark last gap to snapshot remaining
    # Gemini: strategy[last_gap] = -1
    if cur_block_id < total_blocks:
        last_gap_id = len(gap_times) - 1  # Gemini uses len(self.get_comm_idle_time())
        strategy[last_gap_id] = -1  # -1 means "all remaining"

    return strategy
