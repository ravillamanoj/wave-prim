/*
 * Tiled matrix multiply: C = A * B
 *
 *   A: M x K  (row-major)
 *   B: K x N  (row-major)
 *   C: M x N  (row-major)
 *
 * Each workgroup computes one TILE_SIZE x TILE_SIZE output tile.
 * Work item (lr, lc) within the group handles element C[r][c] where
 * r = get_global_id(0), c = get_global_id(1).
 *
 * The K dimension is traversed in TILE_SIZE-wide strips. At each strip
 * every work item cooperatively loads one element into tileA and one
 * into tileB, then all TILE_SIZE^2 work items compute their partial dot
 * product from local memory before advancing to the next strip.
 *
 * Out-of-bounds loads are replaced with 0, so callers do not need to
 * pad matrices to multiples of TILE_SIZE.
 */

#define TILE_SIZE 16

__kernel void gemm(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int K, int N,
    __local float* tileA,
    __local float* tileB
) {
    int r  = get_global_id(0);
    int c  = get_global_id(1);
    int lr = get_local_id(0);
    int lc = get_local_id(1);

    float acc = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;

        tileA[lr * TILE_SIZE + lc] = (r < M && aCol < K) ? A[r * K + aCol] : 0.0f;
        tileB[lr * TILE_SIZE + lc] = (bRow < K && c < N) ? B[bRow * N + c] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[lr * TILE_SIZE + k] * tileB[k * TILE_SIZE + lc];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (r < M && c < N)
        C[r * N + c] = acc;
}
