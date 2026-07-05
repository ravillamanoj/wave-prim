/*
 * Blelloch work-efficient parallel prefix scan.
 *
 * Each work item loads two elements into local memory. The algorithm
 * runs in two phases within each block:
 *
 *   Up-sweep (reduce): build a sum tree bottom-up
 *   Down-sweep:        propagate prefix values from the root down
 *
 * This produces an exclusive scan within each block. For arrays spanning
 * multiple blocks, the host:
 *   1. Collects per-block sums written to block_sums[]
 *   2. Computes an exclusive scan of block_sums (CPU for small counts)
 *   3. Calls scan_add_offsets to add block offsets back to output[]
 *
 * Inclusive scan: caller adds input[i] to each output[i] after the fact.
 */

__kernel void scan_exclusive(
    __global       float* output,
    __global const float* input,
    __global       float* block_sums,
    __local        float* local_buf,
    int n
) {
    int lid        = get_local_id(0);
    int wg_size    = get_local_size(0);
    int block_size = wg_size << 1;          /* 2 * wg_size elements per block */
    int block_id   = get_group_id(0);
    int base       = block_id * block_size;

    /* Load two elements per work item; pad out-of-bounds with identity (0). */
    int idx0 = base + (lid << 1);
    int idx1 = base + (lid << 1) + 1;
    local_buf[lid << 1]       = (idx0 < n) ? input[idx0] : 0.0f;
    local_buf[(lid << 1) + 1] = (idx1 < n) ? input[idx1] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Up-sweep: build reduction tree. */
    int offset = 1;
    for (int d = block_size >> 1; d > 0; d >>= 1) {
        if (lid < d) {
            int ai = offset * (2 * lid + 1) - 1;
            int bi = offset * (2 * lid + 2) - 1;
            local_buf[bi] += local_buf[ai];
        }
        offset <<= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Save block total; zero the root for exclusive scan. */
    if (lid == 0) {
        block_sums[block_id]       = local_buf[block_size - 1];
        local_buf[block_size - 1]  = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Down-sweep: distribute prefix sums. */
    for (int d = 1; d < block_size; d <<= 1) {
        offset >>= 1;
        if (lid < d) {
            int ai    = offset * (2 * lid + 1) - 1;
            int bi    = offset * (2 * lid + 2) - 1;
            float tmp = local_buf[ai];
            local_buf[ai]  = local_buf[bi];
            local_buf[bi] += tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Write exclusive-scan results back to global memory. */
    if (idx0 < n) output[idx0] = local_buf[lid << 1];
    if (idx1 < n) output[idx1] = local_buf[(lid << 1) + 1];
}


/*
 * Add per-block prefix offsets to make the scan global.
 * Each element looks up its block's offset using its global index.
 */
__kernel void scan_add_offsets(
    __global float*       output,
    __global const float* offsets,
    int n,
    int block_size
) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] += offsets[gid / block_size];
    }
}
