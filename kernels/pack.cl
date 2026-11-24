// Inputs of the Alchemist big-tile products (kernels/gemm_big.cl) in the DPAS operands' layout: word 8 c + lane of a
// [group][K / 16 step][64] block holds component c of lane l. pack_b: a matrix's rows (the products' B); pack_a: the
// tokens (A). Rows outside the matrix read as 0, so the group's tail is padded.
#include "builtins.h"

#ifdef ENGINE_DG2
#include "dg2.h"

// words of 16 values: [row][k] pairs at k0 + 2 j of row r.
static inline uint pk_pair(__global const ushort *m, size_t at) { return *( __global const uint *)&m[at]; }

// One sub-group a block: 8 rows and one 16-value step. Rows are the lanes (B) or the components (A).
__attribute__((intel_reqd_sub_group_size(SG8))) __kernel void pack_b(
    __global const ushort *restrict M, uint rows, uint K, uint ksteps, __global uint *restrict out) {
  const uint gid = get_global_id(0), lane = get_sub_group_local_id();
  const uint sg = gid / SG8, rg = sg / ksteps, kb = sg % ksteps, r = rg * SG8 + lane;
  uint w[8];
#pragma unroll
  for (uint c = 0; c < 8; c++) w[c] = r < rows ? pk_pair(M, (size_t)r * K + kb * 16 + 2 * c) : 0u;
  intel_sub_group_block_write8(out + (size_t)sg * 64, as_uint8((int8)(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])));
}

__attribute__((intel_reqd_sub_group_size(SG8))) __kernel void pack_a(
    __global const ushort *restrict M, uint rows, uint K, uint ldx, uint ksteps, __global uint *restrict out) {
  const uint gid = get_global_id(0), lane = get_sub_group_local_id();
  const uint sg = gid / SG8, tg = sg / ksteps, kb = sg % ksteps;
  uint w[8];
#pragma unroll
  for (uint c = 0; c < 8; c++) {
    const uint r = tg * SG8 + c;
    w[c] = r < rows ? pk_pair(M, (size_t)r * ldx + kb * 16 + 2 * lane) : 0u;
  }
  intel_sub_group_block_write8(out + (size_t)sg * 64, as_uint8((int8)(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])));
}
#endif

#ifndef ENGINE_DG2
__kernel void pack_absent(void) {}
#endif
