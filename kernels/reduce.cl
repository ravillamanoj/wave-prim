/*
 * Two-pass parallel reduction over float arrays.
 *
 * Each workgroup reduces its assigned chunk into one partial result using
 * local (shared) memory. The host performs the second-pass aggregation
 * over the partial results.
 *
 * Identity elements used for out-of-bounds padding:
 *   sum ->  0.0f
 *   min ->  MAXFLOAT
 *   max -> -MAXFLOAT
 */

__kernel void reduce_sum(
    __global const float* restrict input,
    __global       float* restrict partial,
    __local        float*          local_data,
    int n
) {
    int gid     = get_global_id(0);
    int lid     = get_local_id(0);
    int wg_size = get_local_size(0);

    local_data[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) local_data[lid] += local_data[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial[get_group_id(0)] = local_data[0];
}

__kernel void reduce_min(
    __global const float* restrict input,
    __global       float* restrict partial,
    __local        float*          local_data,
    int n
) {
    int gid     = get_global_id(0);
    int lid     = get_local_id(0);
    int wg_size = get_local_size(0);

    local_data[lid] = (gid < n) ? input[gid] : MAXFLOAT;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) local_data[lid] = fmin(local_data[lid], local_data[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial[get_group_id(0)] = local_data[0];
}

__kernel void reduce_max(
    __global const float* restrict input,
    __global       float* restrict partial,
    __local        float*          local_data,
    int n
) {
    int gid     = get_global_id(0);
    int lid     = get_local_id(0);
    int wg_size = get_local_size(0);

    local_data[lid] = (gid < n) ? input[gid] : -MAXFLOAT;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) local_data[lid] = fmax(local_data[lid], local_data[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial[get_group_id(0)] = local_data[0];
}
