def get_snapshot_strategy(
    gap_times: dict[int, float],
    block_sizes: list[int],
    dtype_size: int,
    bandwidth_gbps: float,
    gap_threshold_ms: float,
    min_p2p_time_ms: float,
) -> dict[int, int]:
    # Filter gaps above threshold (Gemini: get_valid_comm_idle_time)
    valid_gaps = {
        k: v for k, v in gap_times.items() if v > gap_threshold_ms and k > 0
    }

    strategy: dict[int, int] = {}
    cur_block_id = 0
    total_blocks = len(block_sizes)

    def get_alpha_for_block_size(block_bytes: int) -> float:
        """Get alpha based on block size thresholds."""
        # Small messages can use less effective BW in p2p kernels.
        if block_bytes <= 64 * 1024 * 1024:  # 64 MiB
            return 0.4
        if block_bytes <= 128 * 1024 * 1024:  # 128 MiB
            return 0.5
        elif block_bytes <= 256 * 1024 * 1024:  # 256 MiB
            return 0.6
        elif block_bytes <= 1024 * 1024 * 1024:  # 512 MiB
            return 0.7
        else:
            return 0.8

    def estimate_p2p_time_ms(block_bytes: int, bandwidth_gbps: float, alpha: float, min_p2p_time_ms: float) -> float:
        """
        Estimate P2P transfer time for a block.

        The estimated time is adjusted by 1/alpha to account for overhead.
        The function ensures that the mean estimated time equals min_p2p_time_ms.
        """
        # Base transfer time: bytes / (bandwidth_gbps * 1e9 / 8) * 1000 to get ms
        base_time_ms = block_bytes / (bandwidth_gbps * 1e9 / 8) * 1000
        # Adjust by 1/alpha for overhead
        adjusted_time_ms = base_time_ms / alpha
        # Scale to ensure mean equals min_p2p_time_ms
        return max(adjusted_time_ms, min_p2p_time_ms)

    # Fill gaps from end to beginning to avoid CPU P2P overhead causing GPU stalls
    # at the beginning of training steps
    for gap_id in sorted(valid_gaps.keys(), reverse=True):
        gap_ms = valid_gaps[gap_id]
        blocks = 0
        total_time_ms = 0.0

        # Fit blocks into the gap based on per-block P2P time estimates
        while cur_block_id < total_blocks:
            block_size = block_sizes[cur_block_id]
            block_bytes = block_size * dtype_size

            # Get adaptive alpha based on block size
            alpha = get_alpha_for_block_size(block_bytes)

            # Estimate P2P time for this block
            p2p_time_ms = estimate_p2p_time_ms(block_bytes, bandwidth_gbps, alpha, min_p2p_time_ms)

            # Check if this block fits in the remaining gap time
            if total_time_ms + p2p_time_ms > gap_ms:
                break

            total_time_ms += p2p_time_ms
            cur_block_id += 1
            blocks += 1

        if blocks > 0:
            strategy[gap_id] = blocks

        if cur_block_id >= total_blocks:
            break

    # If not all blocks fit, mark last gap to snapshot remaining
    # Gemini: strategy[last_gap] = -1
    if cur_block_id < total_blocks:
        last_gap_id = len(gap_times) - 1
        strategy[last_gap_id] = -1  # -1 means "all remaining"

    # print(f"bandwidth_gbps={bandwidth_gbps}, gap_threshold_ms={gap_threshold_ms}")

    # # Print per gap what block size (in bytes) are scheduled
    # print("Snapshot strategy per gap:")
    # block_start = 0
    # for gap_id in sorted(strategy.keys()):
    #     num_blocks = strategy[gap_id]
    #     if num_blocks == -1:
    #         # All remaining blocks
    #         remaining_blocks = block_sizes[block_start:]
    #         block_bytes = [size * dtype_size for size in remaining_blocks]
    #         total_bytes = sum(block_bytes)
    #         print(f"  Gap {gap_id}: {len(remaining_blocks)} blocks (remaining), total {total_bytes:,} bytes ({total_bytes / 1e9:.3f} GB)")
    #         for i, bytes_val in enumerate(block_bytes):
    #             print(f"    Block {block_start + i}: {bytes_val:,} bytes ({bytes_val / 1e9:.3f} GB)")
    #     else:
    #         # Specific number of blocks
    #         scheduled_blocks = block_sizes[block_start:block_start + num_blocks]
    #         block_bytes = [size * dtype_size for size in scheduled_blocks]
    #         total_bytes = sum(block_bytes)
    #         print(f"  Gap {gap_id}: {num_blocks} blocks, total {total_bytes:,} bytes ({total_bytes / 1e9:.3f} GB)")
    #         for i, bytes_val in enumerate(block_bytes):
    #             print(f"    Block {block_start + i}: {bytes_val:,} bytes ({bytes_val / 1e9:.3f} GB)")
    #         block_start += num_blocks

    return strategy
