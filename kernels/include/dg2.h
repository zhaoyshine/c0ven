// Alchemist (Xe-HPG) matrix units: the 8-lane form of the XMX products. A sub-group of 8 computes M = 8 rows by
// N = 8 columns by K = 16 (intel_sub_group_{f16,bf16}_matrix_mad_k16, int a, int8 b, float8 c). Lane i holds column i
// of B and C, and A[m][2i], A[m][2i + 1] packed low half first in a[m]. The 16-lane kernels' B packing and
// accumulators are unchanged; only A is repacked. These functions are defined only where the device's minimum
// sub-group size is 8, and in a kernel with a required sub-group size of 8 (cl_intel_subgroup_matrix_multiply_accumulate
// 1.1.0, section on devices whose minimum sub-group size is 8).
#ifndef KERNELS_DG2_H
#define KERNELS_DG2_H

#define SG8 8
#define REQD_SG8 __attribute__((intel_reqd_sub_group_size(SG8)))

static inline float8 dpas8_f16(int8 a, int8 b, float8 c) { return intel_sub_group_f16_f16_matrix_mad_k16(a, b, c); }
static inline float8 dpas8_bf16(int8 a, int8 b, float8 c) { return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c); }

// Lane l's two K values of row `row`, from column k0 + 2l: the A operand's packing.
static inline int sg_pair(__global const ushort *base, ulong row, uint pitch, uint k0, uint lane) {
  const ushort2 v = vload2(0, base + row * pitch + k0 + 2 * lane);
  return (int)((uint)v.x | ((uint)v.y << 16));
}

// A tile of 8 rows from row0, zero past `height` (the 2D block reads' surface bounds).
static inline int8 sg_load_a8(__global const ushort *base, ulong row0, uint pitch, uint k0, uint height, uint lane) {
  int8 a;
  __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++)
    a[m] = row0 + m < height ? sg_pair(base, row0 + m, pitch, k0, lane) : 0;
  return a;
}

// Lane l's column `row + l` of B: 16 16-bit values, pairs low half first. Rows past `rows` read as 0.
static inline int8 sg_load_b8(__global const ushort *base, ulong row, uint pitch, uint k0, uint rows, uint lane) {
  if (row + lane >= rows) return (int8)0;
  return as_int8(vload8(0, (__global const int *)(base + (row + lane) * pitch + k0)));
}

// a8.h's a8_store for a sub-group of 8: unit u of the row at a, 4 values a lane (32 in all).
static inline void a8_store8(float4 v, __global uchar *a, uint K, uint u) {
  const uint lane = get_sub_group_local_id();
  const float amax = sub_group_reduce_max(fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  vstore4((char4)((char)round(v.x * id), (char)round(v.y * id), (char)round(v.z * id), (char)round(v.w * id)), lane,
          (__global char *)(a + 32 * u));
  if (lane == 0) ((__global float *)(a + K))[u] = d;
}

#endif
