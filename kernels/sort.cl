/*
 * Batcher bitonic sort — one kernel launch per comparison-swap step.
 *
 * The host drives the nested loop:
 *
 *   for stage in 0 .. log2(n)-1:
 *     for step in stage .. 0:
 *       launch bitonic_sort_step(data, stage, step)
 *
 * The input array must be padded to a power-of-two length before the
 * first launch; padding slots are filled with MAXFLOAT so they sort to
 * the tail and can be discarded afterwards.  Final order is ascending.
 *
 * Each work item handles one element; it locates its comparison partner
 * via XOR and processes the pair only from the lower-index side, so
 * every pair is swapped at most once per step with no data races.
 */

__kernel void bitonic_sort_step(
    __global float* data,
    int stage,
    int step
) {
    int tid     = get_global_id(0);
    int k       = 2 << stage;   /* block size at this stage: 2^(stage+1) */
    int j       = 1 << step;    /* comparison stride:        2^step       */
    int partner = tid ^ j;

    /* Process each pair exactly once from the lower-index side. */
    if (partner > tid) {
        bool ascending = (tid & k) == 0;
        float a = data[tid];
        float b = data[partner];
        if ((a > b) == ascending) {
            data[tid]     = b;
            data[partner] = a;
        }
    }
}
