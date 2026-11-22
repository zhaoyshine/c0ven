// The prompt path's products on the Alchemist XMX units at a work-group's tile of 128 tokens by 128 rows: a
// sub-group of 8 computes 32 tokens by 32 rows (2 column groups of 16), and 4 column sub-groups by 4 token
// sub-groups make the work-group. The work-group's operands go to local memory by all its sub-groups together and
// double-buffered, so each byte of them is read from L2 once a work-group; they are written there in the DPAS
// operands' layout (component c of a lane at word 8 c + lane), so each operand is one block read.
// Structurally inferzero's gemm_xw8 (zero/kernels/gemm_xw8.cl) at 16-bit products.
//
// M = *ntok tokens, N rows of W, K columns; W and X are `ldw` / `ldx` elements apart. Y: f32, or fp16 when yhalf.
#include "builtins.h"

#ifdef ENGINE_DG2
#include "dg2.h"
#ifdef BIG_BF16
#define BIG_DPAS dpas8_bf16
#else
#define BIG_DPAS dpas8_f16
#endif

#define BTM 32                       // tokens a sub-group
#define BNC 2                        // 16-row groups a sub-group (32 rows)
#define BTG (BTM / 8)                // 8-token DPAS groups a sub-group
#define BNH (2 * BNC)                // 8-row DPAS blocks a sub-group
#define BKC 32                       // K values a stage of local memory
#define BAX 128                      // tokens a work-group
#define BBX 128                      // rows a work-group
#define BKA (BKC / 16)               // 16-value K steps a stage
#define BAG (BAX / 8)                // 8-token groups a work-group
#define BCG (BBX / 8)                // 8-row groups a work-group
#define BAW (BAG * BKA * 64)         // words of A a stage
#define BBW (BCG * BKA * 64)         // words of B a stage

static inline float post(float x, uint mode) {
  if (mode == 1) return x / (1.0f + exp(-x));
  if (mode == 2) return 1.0f / (1.0f + exp(-x));
  return x;
}

__attribute__((intel_reqd_sub_group_size(SG8))) __kernel void gemm_big(
    __global const ushort *restrict W, uint N, uint K, uint ldw, __global const ushort *restrict X, uint ldx,
    __global const uint *restrict ntok, __global ushort *restrict Y, uint ldy, uint yhalf, uint mode, float scale,
    uint nblk) {
  const uint lane = get_sub_group_local_id();
  const uint sx = get_local_id(0) / SG8;
  const uint sy = get_local_id(1);
  const uint wi = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const uint nwi = get_local_size(0) * get_local_size(1);
  const uint g = get_group_id(0);
  const uint cb = g % nblk, row = g / nblk;
  const uint T = *ntok;
  const uint m0 = row * BAX;              // the work-group's first token
  const uint n0 = cb * BBX;               // its first row of W
  const uint t0 = m0 + sy * BTM;          // this sub-group's first token
  const uint c0 = n0 + sx * (16 * BNC);   // its first row
  __local uint la[2][BAW];
  __local uint lb[2][BBW];
  float8 acc[BTG][BNH];
#pragma unroll
  for (uint t = 0; t < BTG; t++)
#pragma unroll
    for (uint h = 0; h < BNH; h++) acc[t][h] = (float8)0.0f;

  // One stage: the work-group's operands for K values [k0, k0 + BKC). Word i of A: component c = (i >> 3) & 7 of
  // lane l = i & 7, K step (i >> 6) & (BKA - 1), token group i >> 7. B: lane l = column, component c = K pair.
#define BIG_STAGE(buf, k0)                                                                                   \
  {                                                                                                          \
    for (uint i = wi; i < BAW; i += nwi) {                                                                   \
      const uint l_ = i & 7, c_ = (i >> 3) & 7, kb_ = (i / 64) % BKA, tg_ = i / (64 * BKA);                  \
      const uint t_ = m0 + tg_ * 8 + c_, k_ = (k0) + kb_ * 16 + 2 * l_;                                      \
      la[buf][i] = (t_ < T && k_ + 1 < K) ? *(__global const uint *)&X[(size_t)t_ * ldx + k_] : 0u;           \
    }                                                                                                        \
    for (uint i = wi; i < BBW; i += nwi) {                                                                   \
      const uint l_ = i & 7, c_ = (i >> 3) & 7, kb_ = (i / 64) % BKA, cg_ = i / (64 * BKA);                  \
      const uint n_ = n0 + cg_ * 8 + l_, k_ = (k0) + kb_ * 16 + 2 * c_;                                      \
      lb[buf][i] = (n_ < N && k_ + 1 < K) ? *(__global const uint *)&W[(size_t)n_ * ldw + k_] : 0u;           \
    }                                                                                                        \
  }
  const uint stages = (K + BKC - 1) / BKC;
  BIG_STAGE(0, 0)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = 0; s < stages; s++) {
    const uint cur = s & 1;
    if (s + 1 < stages) BIG_STAGE(cur ^ 1, (s + 1) * BKC)
    // A lane reads one 16-value DPAS operand a block; B's four.
#pragma unroll
    for (uint kb = 0; kb < BKA; kb++) {
      int8 b[BNH];
#pragma unroll
      for (uint h = 0; h < BNH; h++)
        b[h] = as_int8(intel_sub_group_block_read8(lb[cur] + ((sx * BNH + h) * BKA + kb) * 64));
#pragma unroll
      for (uint t = 0; t < BTG; t++) {
        const int8 a = as_int8(intel_sub_group_block_read8(la[cur] + ((sy * BTG + t) * BKA + kb) * 64));
#pragma unroll
        for (uint h = 0; h < BNH; h++) acc[t][h] = BIG_DPAS(a, b[h], acc[t][h]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // The epilogue: lane l of a block holds rows c0 + 8 h + l of the 8 tokens t0 + 8 t .. + 7.
#pragma unroll
  for (uint t = 0; t < BTG; t++) {
#pragma unroll
    for (uint h = 0; h < BNH; h++) {
      const uint n = c0 + h * SG8 + lane;
      float r[8];
      vstore8(acc[t][h], 0, r);
      for (uint m = 0; m < 8; m++) {
        r[m] = post(r[m] * scale, mode);
        const uint tt = t0 + t * 8 + m;
        if (tt < T && n < N) {
          if (yhalf)
            vstore_half(r[m], (ulong)tt * ldy + n, (__global half *)Y);
          else
            ((__global float *)Y)[(ulong)tt * ldy + n] = r[m];
        }
      }
    }
  }
}
#endif

#ifndef ENGINE_DG2
// The 16-lane cards run the prompt products on gemm_f16 (kernels/gemm.cl); this module exists for the AOT build.
__kernel void gemm_big_absent(void) {}
#endif

#ifdef ENGINE_DG2
// As gemm_big, with the operands already in the DPAS layout (word 8 c + lane: component c of lane l), so a stage's
// copy is a block read a unit: inferzero's PACKA / PACKW. A: [token group][K / 16 step][64]; B: [row group][step][64].
__attribute__((intel_reqd_sub_group_size(SG8))) __kernel void gemm_big_p(
    __global const uint *restrict Wp, __global const uint *restrict Xp, uint N, uint K, __global const uint *restrict ntok,
    __global ushort *restrict Y, uint ldy, uint yhalf, uint mode, float scale, uint nblk,
    __global ushort *restrict Y2, uint mrows) {
  const uint lane = get_sub_group_local_id();
  const uint sx = get_local_id(0) / SG8;
  const uint sy = get_local_id(1);
  const uint sid = sy * (get_local_size(0) / SG8) + sx;
  const uint nsg = (get_local_size(0) / SG8) * get_local_size(1);
  const uint g = get_group_id(0);
  const uint cb = g % nblk, row = g / nblk;
  const uint T = *ntok;
  const uint m0 = row * BAX, n0 = cb * BBX;
  const uint ksteps = (K + 15) / 16;
  const uint mag_a = (mrows + 7) / 8, mag_b = (N + 7) / 8;  // the groups packed
  __local uint la[2][BAW];
  __local uint lb[2][BBW];
  float8 acc[BTG][BNH];
#pragma unroll
  for (uint t = 0; t < BTG; t++)
#pragma unroll
    for (uint h = 0; h < BNH; h++) acc[t][h] = (float8)0.0f;
#define BIGP_STAGE(buf, k0)                                                                       \
  {                                                                                               \
    for (uint u = sid; u < BAW / 64; u += nsg) {                                                  \
      const uint tg = u / BKA, kb = u % BKA;                                                      \
      const uint8 v = intel_sub_group_block_read8(Xp + ((size_t)(m0 / 8 + tg) * ksteps + (k0) / 16 + kb) * 64); \
      intel_sub_group_block_write8(la[buf] + u * 64, v);                                \
    }                                                                                             \
    for (uint u = sid; u < BBW / 64; u += nsg) {                                                  \
      const uint cg = u / BKA, kb = u % BKA;                                                      \
      const uint8 v = intel_sub_group_block_read8(Wp + ((size_t)(n0 / 8 + cg) * ksteps + (k0) / 16 + kb) * 64); \
      intel_sub_group_block_write8(lb[buf] + u * 64, v);                                \
    }                                                                                             \
  }
  const uint stages = (K + BKC - 1) / BKC;
  BIGP_STAGE(0, 0)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = 0; s < stages; s++) {
    const uint cur = s & 1;
    if (s + 1 < stages) BIGP_STAGE(cur ^ 1, (s + 1) * BKC)
#pragma unroll
    for (uint kb = 0; kb < BKA; kb++) {
      int8 b[BNH];
#pragma unroll
      for (uint h = 0; h < BNH; h++)
        b[h] = as_int8(intel_sub_group_block_read8(lb[cur] + ((sx * BNH + h) * BKA + kb) * 64));
#pragma unroll
      for (uint t = 0; t < BTG; t++) {
        const int8 a = as_int8(intel_sub_group_block_read8(la[cur] + ((sy * BTG + t) * BKA + kb) * 64));
#pragma unroll
        for (uint h = 0; h < BNH; h++) acc[t][h] = BIG_DPAS(a, b[h], acc[t][h]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (uint t = 0; t < BTG; t++) {
#pragma unroll
    for (uint h = 0; h < BNH; h++) {
      const uint nb = n0 + sx * (16 * BNC) + h * SG8;  // the block's first row
      const uint n = nb + lane;
      float r[8];
      vstore8(acc[t][h], 0, r);
      for (uint m = 0; m < 8; m++) {
        r[m] = post(r[m] * scale, mode);
        const uint tt = m0 + sy * BTM + t * 8 + m;
        if (tt < T && n < N) {
          if (yhalf)
            vstore_half(r[m], (ulong)tt * ldy + n, (__global half *)Y);
          else
            ((__global float *)Y)[(ulong)tt * ldy + n] = r[m];
        }
      }
      // The other card's copy: 8 lanes of 2 bytes a block write when the rows and the buffer allow it (gemm's rule),
      // a value a lane otherwise.
      if (Y2 && yhalf) {
        if (ldy % SG8 == 0 && ((ulong)Y2 % 16) == 0 && nb + SG8 <= N) {
          for (uint m = 0; m < 8; m++)
            if (m0 + sy * BTM + t * 8 + m < T)
              intel_sub_group_block_write_us(Y2 + (ulong)(m0 + sy * BTM + t * 8 + m) * ldy + nb,
                                             (ushort)as_ushort(convert_half(r[m])));
        } else if (n < N) {
          for (uint m = 0; m < 8; m++)
            if (m0 + sy * BTM + t * 8 + m < T)
              vstore_half(r[m], (ulong)(m0 + sy * BTM + t * 8 + m) * ldy + n, (__global half *)Y2);
        }
      }
    }
  }
}
#endif
