// Sparse attention over a prompt chunk on the XMX units (docs/spec/MODEL_LAYER.md, 3.5 and 3.6), built with 256
// registers, for every run longer than a short run, at any context. One sub-group takes one row and 8 of its query
// heads (the A operand's 8 rows), and walks the row's cells 32 at a time with an online softmax: S = Q K^T by dpas
// calls over the 256 dimensions, then O += P V, one call per 16 dimensions and 16 cells. The queries (16 A operands)
// and the output, 8 heads x 256, stay in registers.
#include "builtins.h"
#include "layout.h"
#include "step.h"

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// The online softmax keeps the maximum it scales by until a score passes it by more than this, so P stays under
// e^8 in fp16 and the sub-group skips the maximum over its lanes and the rescaling of O on most steps.
#define ATT_MAX_SLACK 8.0f

#if Q_IDX_BLOCK != 4
#error "p_attn_xmx reads V in runs of 4 cells"
#endif
#if Q_ATT_SLOTS % 8 || Q_ATT_SLOTS < Q_ATT_H / Q_ATT_KV
#error "Q_ATT_SLOTS holds a KV head's query heads in whole sub-groups of 8"
#endif

// qt: each row's queries as fp16 scaled by 1/16, dimension-major, qt[t][d][slot], hp = nkv Q_ATT_SLOTS slots a
// dimension: KV head c's heads in slots c Q_ATT_SLOTS on, the slots past its 12 heads zero. So a sub-group's 8 slots
// never span two KV heads. kh, vh: p_kv16's fp16 K and V, [position][nkv][256]. cells: p_idx_merge's, whose first
// 2,048 cells are 512 whole blocks of 4 (the dense rows' cells are 0, 1, 2, ...), so a tile of 16 cells is mostly 4
// runs of 4 positions. A work-group holds `rows` consecutive rows of hp / 8 sub-groups each, which share many cells
// through the core's cache. Groups (1, ceil(T / rows)), 16 hp / 8 rows work-items (prompt.cpp: 16 sub-groups).
#ifdef ENGINE_DG2
// The Alchemist form: sub-group 8. A pair of sub-groups takes one row's 8 heads. Each computes the 16 cells' scores
// over the full K in two 8-cell halves, so both hold the same probabilities, and takes half the output's dimensions:
// 16 groups of 8, dim = 8 * (part * 16 + j) + lane. The V operand is gathered a half a lane: B[p] pairs cells 2p and
// 2p + 1 for the lane's dimension.
#include "dg2.h"
REQD_SG8 __kernel void p_attn_xmx(__global const half *qt, __global const float *Q, __global const half *kh,
                                  __global const half *vh, uint nq, uint hp, uint nkv, __global const uint *cells_all,
                                  __global const uint *ncells, __global const PromptPlan *pp, __global half *o16) {
  const uint lane = sg_lane(), sgi = get_sub_group_id(), part = sgi & 1, sg = sgi >> 1;
  const uint hg = hp / 8, rows = get_num_sub_groups() / 2 / hg;
  const uint t = rows * get_group_id(1) + sg / hg, h0 = 8 * (sg % hg);
  if (t >= pp->n || pp->n <= PROMPT_SHORT_RUN || !hg) return;
  const ulong r0 = pp->row0;
  qt += r0 * Q_ATT_D * hp;
  Q += r0 * nq * 2 * Q_ATT_D;
  cells_all += r0 * Q_IDX_CELLS;
  ncells += r0;
  o16 += r0 * nq * Q_ATT_D;
  const uint c = h0 / Q_ATT_SLOTS, hk0 = h0 % Q_ATT_SLOTS;
  const uint n = ncells[t];
  __global const uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
  __global const ushort *q = (__global const ushort *)qt + (ulong)t * Q_ATT_D * hp + h0;
  int8 qa[16];
  for (uint k = 0; k < 16; k++) {
    __global const ushort *qp = q + (ulong)(16 * k + 2 * lane) * hp;
    const ushort8 v0 = vload8(0, qp), v1 = vload8(0, qp + hp);
    qa[k] = (int8)((uint)v0.s0 | ((uint)v1.s0 << 16), (uint)v0.s1 | ((uint)v1.s1 << 16),
                   (uint)v0.s2 | ((uint)v1.s2 << 16), (uint)v0.s3 | ((uint)v1.s3 << 16),
                   (uint)v0.s4 | ((uint)v1.s4 << 16), (uint)v0.s5 | ((uint)v1.s5 << 16),
                   (uint)v0.s6 | ((uint)v1.s6 << 16), (uint)v0.s7 | ((uint)v1.s7 << 16));
  }
  float8 o[16];
  for (uint j = 0; j < 16; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;
  for (uint i0 = 0; i0 < n; i0 += 16) {
    const uint i1 = i0 + 8;
    const bool val0 = i0 + lane < n, val1 = i1 + lane < n;
    const uint c0 = cells[val0 ? i0 + lane : n - 1], c1 = cells[val1 ? i1 + lane : n - 1];
    __global const uint *kr0 = (__global const uint *)(kh + ((ulong)c0 * nkv + c) * Q_ATT_D);
    __global const uint *kr1 = (__global const uint *)(kh + ((ulong)c1 * nkv + c) * Q_ATT_D);
    float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
    __attribute__((opencl_unroll_hint)) for (uint k = 0; k < Q_ATT_D / 16; k++) {
      s0 = dpas8_f16(qa[k], as_int8(vload8(0, kr0 + 8 * k)), s0);
      s1 = dpas8_f16(qa[k], as_int8(vload8(0, kr1 + 8 * k)), s1);
    }
    if (!val0) s0 = (float8)(-INFINITY);
    if (!val1) s1 = (float8)(-INFINITY);
    const float8 bm = fmax(s0, s1);
    if (sub_group_any(any(bm > mrow + ATT_MAX_SLACK))) {
      float b8[8];
      vstore8(bm, 0, b8);
      for (uint m = 0; m < 8; m++) b8[m] = sub_group_reduce_max(b8[m]);
      const float8 mn = fmax(mrow, vload8(0, b8)), al = exp(mrow - mn);
      for (uint j = 0; j < 16; j++) o[j] *= al;
      lrow *= al;
      mrow = mn;
    }
    const float8 e0 = exp(s0 - mrow), e1 = exp(s1 - mrow);
    lrow += e0 + e1;
    const short8 P0 = as_short8(convert_half8(e0)), P1 = as_short8(convert_half8(e1));
    // The A operand's cell pairs: lane l holds cells 2l and 2l + 1, the first from the lower half for l < 4.
    int8 Pc;
    const uint src = lane < 4 ? 2 * lane : 2 * lane - 8;
    __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) {
      const int lo0 = intel_sub_group_shuffle((int)as_ushort(P0[m]), src);
      const int lo1 = intel_sub_group_shuffle((int)as_ushort(P1[m]), src);
      const int hi0 = intel_sub_group_shuffle((int)as_ushort(P0[m]), src + 1);
      const int hi1 = intel_sub_group_shuffle((int)as_ushort(P1[m]), src + 1);
      const int lo = lane < 4 ? lo0 : lo1, hi = lane < 4 ? hi0 : hi1;
      Pc[m] = (lo & 0xffff) | (hi << 16);
    }
    for (uint j = 0; j < 16; j++) {
      const uint d = 8 * (part * 16 + j) + lane;
      int8 B;
      __attribute__((opencl_unroll_hint)) for (uint p = 0; p < 8; p++) {
        const uint ce = cells[min(i0 + 2 * p, n - 1)], co = cells[min(i0 + 2 * p + 1, n - 1)];
        const ushort a = as_ushort(vh[((ulong)ce * nkv + c) * Q_ATT_D + d]);
        const ushort b = as_ushort(vh[((ulong)co * nkv + c) * Q_ATT_D + d]);
        B[p] = (int)((uint)a | ((uint)b << 16));
      }
      o[j] = dpas8_f16(Pc, B, o[j]);
    }
  }
  for (uint x = 1; x < 8; x *= 2) lrow += intel_sub_group_shuffle_xor(lrow, x);
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    const uint d = 8 * (part * 16 + j) + lane;
    for (uint m = 0; m < 8; m++) {
      if (hk0 + m >= Q_ATT_H / Q_ATT_KV) break;
      const uint h = c * (Q_ATT_H / Q_ATT_KV) + hk0 + m;
      const float gate = Q[((ulong)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
      o16[((ulong)t * nq + h) * Q_ATT_D + d] = convert_half(ov[m] / lr[m] * sigmoid(gate));
    }
  }
}
#else
REQD_SG __kernel void p_attn_xmx(__global const half *qt, __global const float *Q, __global const half *kh,
                                 __global const half *vh, uint nq, uint hp, uint nkv, __global const uint *cells_all,
                                 __global const uint *ncells, __global const PromptPlan *pp, __global half *o16) {
  const uint hg = hp / 8, sg = get_sub_group_id(), rows = get_num_sub_groups() / hg;
  const uint t = rows * get_group_id(1) + sg / hg, h0 = 8 * (sg % hg), lane = sg_lane();
  if (t >= pp->n || pp->n <= PROMPT_SHORT_RUN) return;
  const ulong r0 = pp->row0;
  qt += r0 * Q_ATT_D * hp;
  Q += r0 * nq * 2 * Q_ATT_D;
  cells_all += r0 * Q_IDX_CELLS;
  ncells += r0;
  o16 += r0 * nq * Q_ATT_D;
  // KV head c; the sub-group's slots are its heads hk0 .. hk0 + 7, those under 12 real.
  const uint c = h0 / Q_ATT_SLOTS, hk0 = h0 % Q_ATT_SLOTS;
  const uint n = ncells[t];
  __global const uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
  __global const ushort *q = (__global const ushort *)qt + (ulong)t * Q_ATT_D * hp + h0;
  // V as a 2D surface for the block reads: one row a position, this KV head's 256 values.
  const uint height = pp->pos0 + pp->n, pitch = nkv * Q_ATT_D * 2;
  __global const ushort *vb = (__global const ushort *)(vh + c * Q_ATT_D);
  short8 qa[16];
  for (uint k = 0; k < 16; k++) qa[k] = as_short8(vload8(0, q + (ulong)(16 * k + lane) * hp));
  float8 o[16];
  for (uint j = 0; j < 16; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;  // lrow: each lane's own part of the sums
  for (uint i0 = 0; i0 < n; i0 += 32) {
    // Two tiles of 16 cells: lane = cell, component = head. A tile is "runs" when its cells are 4 runs of 4
    // consecutive positions and every lane has one; g0 holds each run's first position.
    uint cell[2], g0[2][4];
    bool valid[2], runs[2];
    float8 s[2];
    for (uint u = 0; u < 2; u++) {
      valid[u] = i0 + 16 * u + lane < n;
      cell[u] = cells[valid[u] ? i0 + 16 * u + lane : i0];
      runs[u] = sub_group_all(valid[u]);
      for (uint g = 0; g < 4; g++) {
        g0[u][g] = sub_group_broadcast(cell[u], 4 * g);
        runs[u] = runs[u] && sub_group_broadcast(cell[u], 4 * g + 3) == g0[u][g] + 3;
      }
      __global const uint *krow = (__global const uint *)(kh + ((ulong)cell[u] * nkv + c) * Q_ATT_D);
      float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
      __attribute__((opencl_unroll_hint)) for (uint k = 0; k < Q_ATT_D / 16; k += 2) {
        s0 = dpas_f16(qa[k], as_int8(vload8(0, krow + 8 * k)), s0);
        s1 = dpas_f16(qa[k + 1], as_int8(vload8(0, krow + 8 * k + 8)), s1);
      }
      s[u] = valid[u] ? s0 + s1 : (float8)(-INFINITY);
    }
    // The online softmax, per head (component). Each lane checks its own scores against the maximum in use; only
    // when one passes it by more than ATT_MAX_SLACK is the maximum over the lanes taken and O rescaled.
    float8 bm = fmax(s[0], s[1]);
    if (sub_group_any(any(bm > mrow + ATT_MAX_SLACK))) {
      float b8[8];
      vstore8(bm, 0, b8);
      for (uint m = 0; m < 8; m++) b8[m] = sub_group_reduce_max(b8[m]);
      const float8 mn = fmax(mrow, vload8(0, b8)), al = exp(mrow - mn);
      for (uint j = 0; j < 16; j++) o[j] *= al;
      lrow *= al;
      mrow = mn;
    }
    const float8 ea = exp(s[0] - mrow), eb = exp(s[1] - mrow);
    lrow += ea + eb;
    short8 P[2] = {as_short8(convert_half8(ea)), as_short8(convert_half8(eb))};
    // O += P V, a quarter of the dimensions at a time: B[k][n] = V[cell k][d + n], cells paired in each 32-bit word.
    __attribute__((opencl_unroll_hint)) for (uint qd = 0; qd < 4; qd++) {
      __attribute__((opencl_unroll_hint)) for (uint u = 0; u < 2; u++) {
        if (runs[u]) {
          // A run's 4 positions by 32 dimensions in one 2D block read: lane n gets dimension n, then n + 16, of the
          // run's 4 positions in turn.
          __attribute__((opencl_unroll_hint)) for (uint j2 = 0; j2 < 2; j2++) {
            uint ba[8], bb[8];
            __attribute__((opencl_unroll_hint)) for (uint g = 0; g < 4; g++) {
              ushort r8[8];
              intel_sub_group_2d_block_read_16b_4r16x2c((__global void *)vb, Q_ATT_D * 2, height, pitch,
                                                        (int2)(64 * qd + 32 * j2, g0[u][g]), r8);
              ba[2 * g] = (uint)r8[0] | ((uint)r8[1] << 16);
              ba[2 * g + 1] = (uint)r8[2] | ((uint)r8[3] << 16);
              bb[2 * g] = (uint)r8[4] | ((uint)r8[5] << 16);
              bb[2 * g + 1] = (uint)r8[6] | ((uint)r8[7] << 16);
            }
            o[4 * qd + 2 * j2] = dpas_f16(P[u], as_int8(vload8(0, ba)), o[4 * qd + 2 * j2]);
            o[4 * qd + 2 * j2 + 1] = dpas_f16(P[u], as_int8(vload8(0, bb)), o[4 * qd + 2 * j2 + 1]);
          }
          continue;
        }
        // Any other tile (the partial block and the tail, a row's last cells): each cell's row by block reads (lane
        // n gets dimensions n, n + 16, n + 32, n + 48 of the quarter), paired afterwards.
        ushort r[16][4];
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < 16; k++) {
          uint ck = sub_group_broadcast(cell[u], k);
          ushort4 v = intel_sub_group_block_read_us4((__global const ushort *)(vh + ((ulong)ck * nkv + c) * Q_ATT_D) +
                                                     64 * qd);
          r[k][0] = v.s0;
          r[k][1] = v.s1;
          r[k][2] = v.s2;
          r[k][3] = v.s3;
        }
        __attribute__((opencl_unroll_hint)) for (uint j = 0; j < 4; j++) {
          uint pr[8];
          __attribute__((opencl_unroll_hint)) for (uint q2 = 0; q2 < 8; q2++)
            pr[q2] = (uint)r[2 * q2][j] | ((uint)r[2 * q2 + 1][j] << 16);
          o[4 * qd + j] = dpas_f16(P[u], as_int8(vload8(0, pr)), o[4 * qd + j]);
        }
      }
    }
  }
  // Out: head c 12 + hk0 + m, dimension 64 (j / 4) + 16 (j % 4) + lane.
  for (uint x = 1; x < 16; x *= 2) lrow += intel_sub_group_shuffle_xor(lrow, x);
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    uint d = 64 * (j / 4) + 16 * (j % 4) + lane;
    for (uint m = 0; m < 8; m++) {
      if (hk0 + m >= Q_ATT_H / Q_ATT_KV) break;
      uint h = c * (Q_ATT_H / Q_ATT_KV) + hk0 + m;
      float gate = Q[((ulong)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
      o16[((ulong)t * nq + h) * Q_ATT_D + d] = convert_half(ov[m] / lr[m] * sigmoid(gate));
    }
  }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// A short run's rows (at most PROMPT_SHORT_RUN: a decode row and its drafts) attend as the decode step's attn_split
// and attn_merge (layer.cl) do: over the cells p_idx_merge chose, read from the 8-bit cache through the page table,
// with no fp16 copy of the sequence. p_attn_rows: work-group (KV head c, row y, split s) takes cells
// [s ATT_SPLIT, +ATT_SPLIT) of row y % PROMPT_SHORT_RUN of run y / PROMPT_SHORT_RUN, for the ATT_GQA query heads of
// KV head c, and writes each head's unnormalised output, max and sum into part (ATT_PART floats a head and split, rows
// PROMPT_MAX_RUNS PROMPT_SHORT_RUN of them). Groups (nkv, PROMPT_MAX_RUNS PROMPT_SHORT_RUN, ATT_SPLITS), 256.
#define ATT_SPLIT 64
#define ATT_SPLITS ((Q_IDX_CELLS + ATT_SPLIT - 1) / ATT_SPLIT)
#define ATT_PART (Q_ATT_D + 2)
#define ATT_GQA (Q_ATT_H / Q_ATT_KV)

// Row y of the short runs' grid: its run's plan, or null when the run is long or holds no row y.
static inline __global const PromptPlan *short_row(__global const PromptPlan *runs, uint y) {
  __global const PromptPlan *pp = runs + y / PROMPT_SHORT_RUN;
  return pp->n > PROMPT_SHORT_RUN || y % PROMPT_SHORT_RUN >= pp->n ? 0 : pp;
}

REQD_SG __kernel void p_attn_rows(__global const float *Q, __global const uchar *kc, __global const uchar *vc,
                                  __global const uint *pt, uint npp, uint nq, uint nkv, __global const uint *cells_all,
                                  __global const uint *ncells, __global const PromptPlan *runs, __global float *part,
                                  __global const uchar *hk, __global const uchar *hv) {
  // Streamed (hk, hv given): the cells are read in place from the host copy, which holds every page; the 8-bit
  // cache then holds only the resident pages, and a page index into it would read past its end.
  __local float q[ATT_GQA][Q_ATT_D], p[ATT_GQA][ATT_SPLIT];
  __local uint cl[ATT_SPLIT];
  uint c = get_group_id(0), y = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp) return;
  const uint r = pp->row0 + y % PROMPT_SHORT_RUN;  // the chunk's row
  uint n = ncells[r], i0 = s * ATT_SPLIT;
  if (i0 >= n) return;
  uint m = min((uint)ATT_SPLIT, n - i0);
  for (uint h = 0; h < ATT_GQA; h++) q[h][d] = Q[((ulong)r * nq + c * ATT_GQA + h) * 2 * Q_ATT_D + d];
  __global const uint *cells = cells_all + (ulong)r * Q_IDX_CELLS + i0;
  if (d < m) {
    const uint pos = cells[d];
    cl[d] = (uint)kv_cell(kv_page(pt, npp, pp->slot, pos), pos);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uchar *K = (hk ? hk : kc) + c * KV_HEAD_BYTES;
  __global const uchar *V = (hv ? hv : vc) + c * KV_HEAD_BYTES;
  ulong cell_bytes = (ulong)nkv * KV_HEAD_BYTES;
  uint lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();
  for (uint i = sg; i < m; i += nsg) {
    // Lane l holds values 16l .. 16l+15: half of block l/2.
    __global const uchar *blk = K + cl[i] * cell_bytes + (lane / 2) * Q8_BLOCK;
    float db = vload_half(0, (__global const half *)blk);
    char16 v = vload16(0, (__global const char *)(blk + 2 + 16 * (lane % 2)));
    float kv[16] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7, v.s8, v.s9, v.sa, v.sb, v.sc, v.sd, v.se, v.sf};
    for (uint h = 0; h < ATT_GQA; h++) {
      __local const float *qh = &q[h][16 * lane];
      float part_ = 0.0f;
      for (uint t = 0; t < 16; t++) part_ += qh[t] * kv[t];
      float dot = sub_group_reduce_add(db * part_);
      if (lane == 0) p[h][i] = dot * Q_ATT_SCALE;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float *po = part + (((ulong)y * nq + c * ATT_GQA) * ATT_SPLITS + s) * ATT_PART;
  if (sg < ATT_GQA) {
    float mx = -INFINITY;
    for (uint i = lane; i < m; i += 16) mx = fmax(mx, p[sg][i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < m; i += 16) {
      float e = exp(p[sg][i] - mx);
      p[sg][i] = e;
      sum += e;
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
      po[(ulong)sg * ATT_SPLITS * ATT_PART + Q_ATT_D] = mx;
      po[(ulong)sg * ATT_SPLITS * ATT_PART + Q_ATT_D + 1] = sum;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uchar *vb = V + (d / 32) * Q8_BLOCK;
  float acc[ATT_GQA];
  for (uint h = 0; h < ATT_GQA; h++) acc[h] = 0.0f;
  // Eight cells' loads issued together, then summed in order.
  uint i = 0;
  for (; i + 8 <= m; i += 8) {
    float v[8];
    for (uint j = 0; j < 8; j++) {
      __global const uchar *b = vb + cl[i + j] * cell_bytes;
      v[j] = vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
    }
    for (uint j = 0; j < 8; j++)
      for (uint h = 0; h < ATT_GQA; h++) acc[h] += p[h][i + j] * v[j];
  }
  for (; i < m; i++) {
    __global const uchar *b = vb + cl[i] * cell_bytes;
    float v = vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
    for (uint h = 0; h < ATT_GQA; h++) acc[h] += p[h][i] * v;
  }
  for (uint h = 0; h < ATT_GQA; h++) po[(ulong)h * ATT_SPLITS * ATT_PART + d] = acc[h];
}

// p_attn_rows' splits combined in order, then the output gate, into o16 as the long runs' kernels write it. Groups
// (nq, PROMPT_MAX_RUNS PROMPT_SHORT_RUN), 256.
__kernel void p_attn_rows_merge(__global const float *Q, uint nq, __global const uint *ncells,
                                __global const PromptPlan *runs, __global const float *part, __global half *o16) {
  uint lh = get_group_id(0), y = get_group_id(1), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp) return;
  const uint r = pp->row0 + y % PROMPT_SHORT_RUN;
  uint ns = (ncells[r] + ATT_SPLIT - 1) / ATT_SPLIT;
  __global const float *pt = part + ((ulong)y * nq + lh) * ATT_SPLITS * ATT_PART;
  float mx = -INFINITY;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[s * ATT_PART + Q_ATT_D]);
  float acc = 0.0f, sum = 0.0f;
  for (uint s = 0; s < ns; s++) {
    float f = exp(pt[s * ATT_PART + Q_ATT_D] - mx);
    acc += f * pt[s * ATT_PART + d];
    sum += f * pt[s * ATT_PART + Q_ATT_D + 1];
  }
  float gate = Q[((ulong)r * nq + lh) * 2 * Q_ATT_D + Q_ATT_D + d];
  o16[((ulong)r * nq + lh) * Q_ATT_D + d] = convert_half(acc / sum * sigmoid(gate));
}

// The prompt indexer's scores on the XMX units: scores[t][lb] = sum_h relu(iq_h . K_b) for this card's blocks complete
// at row t's position, by local index lb (block lb C + c), rows nb_max apart. Each operand is split in two fp16 terms,
// x = hi + lo, and a score is hi hi + hi lo + lo hi summed in f32 (three DPAS calls): plain fp16 changes a row's chosen
// blocks one row in six, the split as often as summing the f32 products in another order (on long-256k's queries and
// keys of layers 3, 23 and 47: 0 to 2 rows of 4,352 differ, by one block of 512).
// A work-group takes IDX_XMX_ROWS rows of a run, as 8 tiles of 2 rows by 4 heads (the queries split by p_idx_qsplit:
// qh, ql [chunk row][Q_IDX_D][Q_IDX_H]); each sub-group takes 16 IDX_XMX_TB blocks, their keys split by p_idx_ksplit
// (kh, kl: [local page][Q_IDX_D]). Groups
// (ceil(nb_max / (16 IDX_XMX_TB kIdxXmxSg)), ceil(T / IDX_XMX_ROWS), runs), 16 kIdxXmxSg (model_impl.h).
#define IDX_XMX_ROWS 16
#define IDX_XMX_TB 2
#if Q_IDX_H != 4 || Q_IDX_D % 16
#error "p_idx_score_xmx tiles 2 rows by the indexer's 4 heads, and 16 dimensions a step"
#endif
#ifdef ENGINE_DG2
// The Alchemist form: sub-group 8, two K values a lane (the queries split by dimension), 8 blocks a tile.
#include "dg2.h"
#define IDX_SG REQD_SG8
#define IDX_LANES SG8
#define IDX_DPAS dpas8_f16
#else
#define IDX_SG REQD_SG
#define IDX_LANES 16
#define IDX_DPAS dpas_f16
#endif
static inline uint idx_own_x(uint nb, uint c, uint C) { return nb > c ? (nb - c + C - 1) / C : 0; }
IDX_SG __kernel void p_idx_score_xmx(__global const ushort *qh, __global const ushort *ql, __global const half *kh,
                                     __global const half *kl, __global const uint *pt, uint npp, uint nb_max,
                                     __global const PromptPlan *runs, __global float *scores, uint c, uint C) {
  __global const PromptPlan *pp = runs + get_group_id(2);
  const uint lane = sg_lane(), n = pp->n, t0 = IDX_XMX_ROWS * get_group_id(1);
  if (t0 >= n) return;
  scores += (ulong)pp->row0 * nb_max;
  const uint rows = min((uint)IDX_XMX_ROWS, n - t0);
  const uint nkv_last = pp->pos0 + t0 + rows;
  if (nkv_last <= Q_IDX_CELLS) return;
  const uint nown = idx_own_x(nkv_last / Q_IDX_BLOCK, c, C);
  const uint b0 = (get_group_id(0) * get_num_sub_groups() + get_sub_group_id()) * IDX_LANES * IDX_XMX_TB;
  if (b0 >= nown) return;
  ulong K[IDX_XMX_TB];
  for (uint j = 0; j < IDX_XMX_TB; j++) {
    const uint lb = min(b0 + IDX_LANES * j + lane, nown - 1);
    K[j] = (ulong)(kv_page(pt, npp, pp->slot, (lb * C + c) * Q_IDX_BLOCK) / C) * Q_IDX_D;
  }
  float8 acc[IDX_XMX_ROWS / 2][IDX_XMX_TB];
  for (uint i = 0; i < IDX_XMX_ROWS / 2; i++)
    for (uint j = 0; j < IDX_XMX_TB; j++) acc[i][j] = (float8)0.0f;
  for (uint s = 0; s < Q_IDX_D / 16; s++) {
    int8 bh[IDX_XMX_TB], bl[IDX_XMX_TB];
    for (uint j = 0; j < IDX_XMX_TB; j++) {
      bh[j] = as_int8(vload8(0, (__global const uint *)(kh + K[j]) + 8 * s));
      bl[j] = as_int8(vload8(0, (__global const uint *)(kl + K[j]) + 8 * s));
    }
    for (uint i = 0; i < IDX_XMX_ROWS / 2; i++) {
#ifdef ENGINE_DG2
      // The tile's rows 2i and 2i + 1 (the last row again past the run's end), 4 heads each, at dimensions
      // 16 s + 2 lane and 16 s + 2 lane + 1: a[m] pairs the two dimensions of row m / 4's head m % 4.
      const ulong r0 = ((ulong)(pp->row0 + t0 + min(2 * i, rows - 1)) * Q_IDX_D + 16 * s + 2 * lane) * Q_IDX_H;
      const ulong r1 = ((ulong)(pp->row0 + t0 + min(2 * i + 1, rows - 1)) * Q_IDX_D + 16 * s + 2 * lane) * Q_IDX_H;
      const uint st = Q_IDX_H;
      const ushort4 h00 = vload4(0, qh + r0), h01 = vload4(0, qh + r0 + st);
      const ushort4 h10 = vload4(0, qh + r1), h11 = vload4(0, qh + r1 + st);
      const ushort4 l00 = vload4(0, ql + r0), l01 = vload4(0, ql + r0 + st);
      const ushort4 l10 = vload4(0, ql + r1), l11 = vload4(0, ql + r1 + st);
      const int8 a = (int8)((uint)h00.s0 | ((uint)h01.s0 << 16), (uint)h00.s1 | ((uint)h01.s1 << 16),
                            (uint)h00.s2 | ((uint)h01.s2 << 16), (uint)h00.s3 | ((uint)h01.s3 << 16),
                            (uint)h10.s0 | ((uint)h11.s0 << 16), (uint)h10.s1 | ((uint)h11.s1 << 16),
                            (uint)h10.s2 | ((uint)h11.s2 << 16), (uint)h10.s3 | ((uint)h11.s3 << 16));
      const int8 b = (int8)((uint)l00.s0 | ((uint)l01.s0 << 16), (uint)l00.s1 | ((uint)l01.s1 << 16),
                            (uint)l00.s2 | ((uint)l01.s2 << 16), (uint)l00.s3 | ((uint)l01.s3 << 16),
                            (uint)l10.s0 | ((uint)l11.s0 << 16), (uint)l10.s1 | ((uint)l11.s1 << 16),
                            (uint)l10.s2 | ((uint)l11.s2 << 16), (uint)l10.s3 | ((uint)l11.s3 << 16));
      for (uint j = 0; j < IDX_XMX_TB; j++) {
        acc[i][j] = IDX_DPAS(a, bh[j], acc[i][j]);
        acc[i][j] = IDX_DPAS(a, bl[j], acc[i][j]);
        acc[i][j] = IDX_DPAS(b, bh[j], acc[i][j]);
      }
#else
      // The tile's rows 2i and 2i + 1 (the last row again past the run's end), 4 heads each, at dimension 16 s + lane.
      const ulong r0 = ((ulong)(pp->row0 + t0 + min(2 * i, rows - 1)) * Q_IDX_D + 16 * s + lane) * Q_IDX_H;
      const ulong r1 = ((ulong)(pp->row0 + t0 + min(2 * i + 1, rows - 1)) * Q_IDX_D + 16 * s + lane) * Q_IDX_H;
      const short8 a = as_short8((ushort8)(vload4(0, qh + r0), vload4(0, qh + r1)));
      const short8 b = as_short8((ushort8)(vload4(0, ql + r0), vload4(0, ql + r1)));
      for (uint j = 0; j < IDX_XMX_TB; j++) {
        acc[i][j] = dpas_f16(a, bh[j], acc[i][j]);
        acc[i][j] = dpas_f16(a, bl[j], acc[i][j]);
        acc[i][j] = dpas_f16(b, bh[j], acc[i][j]);
      }
#endif
    }
  }
  uint lim[IDX_XMX_ROWS];  // row r's blocks on this card, those complete at its position (none below the width)
  for (uint r = 0; r < IDX_XMX_ROWS; r++) {
    const uint nkv = pp->pos0 + t0 + r + 1;
    lim[r] = r < rows && nkv > Q_IDX_CELLS ? idx_own_x(nkv / Q_IDX_BLOCK, c, C) : 0;
  }
  for (uint i = 0; i < IDX_XMX_ROWS / 2; i++)
    for (uint j = 0; j < IDX_XMX_TB; j++) {
      const uint lb = b0 + IDX_LANES * j + lane;
      const float8 a = acc[i][j];
      for (uint u = 0; u < 2; u++) {
        const uint r = 2 * i + u, t = t0 + r;
        if (lb >= lim[r]) continue;
        float score = 0.0f;
        score += fmax(u ? a.s4 : a.s0, 0.0f);
        score += fmax(u ? a.s5 : a.s1, 0.0f);
        score += fmax(u ? a.s6 : a.s2, 0.0f);
        score += fmax(u ? a.s7 : a.s3, 0.0f);
        scores[(ulong)t * nb_max + lb] = score;
      }
    }
}
