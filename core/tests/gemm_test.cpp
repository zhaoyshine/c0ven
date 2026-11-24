// Level 2 and 3 (docs/TESTING.md): the XMX products of kernels/gemm.cl, fp16 and bf16, against the same 16-bit
// inputs multiplied in double precision. This also checks the register layout of the DPAS builtins that
// kernels/include/builtins.h states. Tolerance 2e-3 of the largest |reference| (TESTING.md, "Tolerances"). The bf16
// modules with other tiles (gemm_bf16_t4n<TN>) and the banded order must give the plain launch's bits, and the second
// output Y2 of an fp16 product the first's.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "load/runs.h"
#include "quant.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}
float from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = std::getenv("GEMM_CARD") ? (uint32_t)std::atoi(std::getenv("GEMM_CARD")) : 0;
  ze::List L(rt, card, ze::List::immediate);
  if (std::getenv("GEMM_BENCH")) {
    const uint32_t T = 4352, N = 5120, K = 2560;
    auto *dW = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)N * K * 2));
    auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * K * 2));
    auto *dT = static_cast<uint32_t *>(rt.alloc_host(4));
    auto *dY = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * N * 2));
    auto *dY2 = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * N * 2));
    std::vector<uint16_t> W((size_t)N * K), X((size_t)T * K);
    for (size_t i = 0; i < W.size(); i++) W[i] = ggml_fp32_to_fp16(0.5f + (float)(i % 13) * 0.01f);
    for (size_t i = 0; i < X.size(); i++) X[i] = ggml_fp32_to_fp16((float)(i % 11) * 0.1f);
    *dT = T;
    L.copy(dW, W.data(), W.size() * 2);
    L.copy(dX, X.data(), X.size() * 2);
    L.synchronize();
    auto k = lib.get(card, "gemm_f16", "gemm");
    L.launch(k, {dW, N, K, K, dX, K, dT, (float *)dY, N, 1u, 0u, 1.0f, nullptr, 0u}, (N + 127) / 128,
             (T + 63) / 64, 1, 64);
    L.synchronize();
    auto kb = lib.get(card, "gemm_big", "gemm_big");
    const uint32_t nblk = (N + 127) / 128, mblk = (T + 127) / 128;
    L.launch(kb, {dW, N, K, K, dX, K, dT, dY2, N, 1u, 0u, 1.0f, nblk}, nblk * mblk, 1, 1, 32, 4, 1);
    L.synchronize();
    {
      auto *h1 = static_cast<uint16_t *>(rt.alloc_host((size_t)T * N * 2));
      auto *h2 = static_cast<uint16_t *>(rt.alloc_host((size_t)T * N * 2));
      L.copy(h1, dY, (size_t)T * N * 2);
      L.copy(h2, dY2, (size_t)T * N * 2);
      L.synchronize();
      size_t bad = 0;
      float worst = 0;
      int shown = 0;
      for (size_t i = 0; i < (size_t)T * N; i++)
        if (h1[i] != h2[i]) {
          bad++;
          worst = std::max(worst, std::fabs(ggml_fp16_to_fp32(h1[i]) - ggml_fp16_to_fp32(h2[i])));
          if (shown < 10) {
            std::printf("  differ t %zu n %zu: %f vs %f\n", i / N, i % N, ggml_fp16_to_fp32(h1[i]),
                        ggml_fp16_to_fp32(h2[i]));
            shown++;
          }
        }
      std::printf("big vs plain: %zu of %zu differ, worst %.4f\n", bad, (size_t)T * N, worst);
      rt.free(h1);
      rt.free(h2);
    }
    for (uint32_t band : {0u}) {
      (void)band;
      for (int i = 0; i < 2; i++)
        L.launch(k, {dW, N, K, K, dX, K, dT, (float *)dY, N, 1u, 0u, 1.0f, nullptr, band}, (N + 127) / 128,
                 (T + 63) / 64, 1, 64);
      L.synchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 5; i++)
        L.launch(k, {dW, N, K, K, dX, K, dT, (float *)dY, N, 1u, 0u, 1.0f, nullptr, band}, (N + 127) / 128,
                 (T + 63) / 64, 1, 64);
      L.synchronize();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
      std::printf("bench plain fp16 T %u N %u K %u: %.1f ms, %.2f TFLOPS\n", T, N, K, s * 1e3,
                  2.0 * T * N * K / s / 1e12);
    }
    {
      for (int i = 0; i < 2; i++)
        L.launch(kb, {dW, N, K, K, dX, K, dT, dY2, N, 1u, 0u, 1.0f, nblk}, nblk * mblk, 1, 1, 32, 4, 1);
      L.synchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 5; i++)
        L.launch(kb, {dW, N, K, K, dX, K, dT, dY2, N, 1u, 0u, 1.0f, nblk}, nblk * mblk, 1, 1, 32, 4, 1);
      L.synchronize();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
      std::printf("bench big fp16 T %u N %u K %u: %.1f ms, %.2f TFLOPS\n", T, N, K, s * 1e3,
                  2.0 * T * N * K / s / 1e12);
    }
    {
      const uint32_t ksteps = K / 16;
      std::vector<uint32_t> Wp((size_t)(N / 8) * ksteps * 64), Xp((size_t)(T / 8) * ksteps * 64);
      auto pair = [](const std::vector<uint16_t> &m, size_t at) {
        return (uint32_t)m[at] | ((uint32_t)m[at + 1] << 16);
      };
      for (uint32_t cg = 0; cg < N / 8; cg++)
        for (uint32_t kb = 0; kb < ksteps; kb++)
          for (uint32_t c = 0; c < 8; c++)
            for (uint32_t l = 0; l < 8; l++)
              Wp[((size_t)(cg * ksteps + kb) * 64) + c * 8 + l] =
                  pair(W, (size_t)(cg * 8 + l) * K + kb * 16 + 2 * c);
      for (uint32_t tg = 0; tg < T / 8; tg++)
        for (uint32_t kb = 0; kb < ksteps; kb++)
          for (uint32_t c = 0; c < 8; c++)
            for (uint32_t l = 0; l < 8; l++)
              Xp[((size_t)(tg * ksteps + kb) * 64) + c * 8 + l] =
                  pair(X, (size_t)(tg * 8 + c) * K + kb * 16 + 2 * l);
      auto *dWp = static_cast<uint32_t *>(rt.alloc_device(card, Wp.size() * 4));
      auto *dXp = static_cast<uint32_t *>(rt.alloc_device(card, Xp.size() * 4));
      L.copy(dWp, Wp.data(), Wp.size() * 4);
      L.copy(dXp, Xp.data(), Xp.size() * 4);
      L.synchronize();
      auto kp = lib.get(card, "gemm_big", "gemm_big_p");
      L.launch(kp, {dWp, dXp, N, K, dT, dY2, N, 1u, 0u, 1.0f, nblk, (uint16_t *)nullptr}, nblk * mblk, 1, 1, 32, 4, 1);
      L.synchronize();
      {
        auto *h1 = static_cast<uint16_t *>(rt.alloc_host((size_t)T * N * 2));
        auto *h2 = static_cast<uint16_t *>(rt.alloc_host((size_t)T * N * 2));
        L.copy(h1, dY, (size_t)T * N * 2);
        L.copy(h2, dY2, (size_t)T * N * 2);
        L.synchronize();
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)T * N; i++) bad += h1[i] != h2[i];
        std::printf("packed vs plain: %zu of %zu differ\n", bad, (size_t)T * N);
        rt.free(h1);
        rt.free(h2);
      }
      for (int i = 0; i < 2; i++)
        L.launch(kp, {dWp, dXp, N, K, dT, dY2, N, 1u, 0u, 1.0f, nblk, (uint16_t *)nullptr}, nblk * mblk, 1, 1, 32, 4, 1);
      L.synchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 5; i++)
        L.launch(kp, {dWp, dXp, N, K, dT, dY2, N, 1u, 0u, 1.0f, nblk, (uint16_t *)nullptr}, nblk * mblk, 1, 1, 32, 4, 1);
      L.synchronize();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
      std::printf("bench packed fp16 T %u N %u K %u: %.1f ms, %.2f TFLOPS\n", T, N, K, s * 1e3,
                  2.0 * T * N * K / s / 1e12);
      rt.free(dWp);
      rt.free(dXp);
    }
    rt.free(dW);
    rt.free(dX);
    rt.free(dT);
    rt.free(dY);
    rt.free(dY2);
    return 0;
  }
  if (std::getenv("BIG_STRESS")) {
    // The engine's chain at its largest dense shape, one round at a time, each line on disk before the round runs:
    // BIG_STEPS 1 dequant_f16_p, 2 pack_a, 4 gemm_big_p; BIG_ROUNDS rounds.
    const uint32_t T = std::getenv("BIG_T") ? (uint32_t)std::atoi(std::getenv("BIG_T")) : 2048;
    const uint32_t N = std::getenv("BIG_N") ? (uint32_t)std::atoi(std::getenv("BIG_N")) : 5120;
    const uint32_t K = std::getenv("BIG_K") ? (uint32_t)std::atoi(std::getenv("BIG_K")) : 2560;
    const uint32_t qt = GGML_TYPE_IQ3_XXS;
    // BIG_MIX: the engine's three dense shapes in turn on one set of buffers (sized for the largest), a chunk's order.
    struct Mix { uint32_t N, K; int f32, y2; };
    static const Mix mixes[3] = {{5120, 2560, 0, 1}, {2560, 3072, 0, 1}, {256, 2560, 1, 0}};
    const bool mix = std::getenv("BIG_MIX") != nullptr;
    const uint32_t bpr = K / quant_block_values(qt), units = K / 32, ks = K / 16;
    const bool y2_on = std::getenv("BIG_NO_Y2") == nullptr;
    const bool f32y = std::getenv("BIG_F32") != nullptr;
    const uint32_t nb = (N + 7) / 8, nbx = (T + 7) / 8, nblk = (N + 127) / 128, mblk = (T + 127) / 128;
    void *pad = nullptr;
    if (const char *pg = std::getenv("BIG_PAD_GB")) {
      const uint64_t bytes = (uint64_t)std::atoi(pg) << 30;
      pad = rt.alloc_device(card, bytes);
      std::printf("pad %llu GiB at %p\n", (unsigned long long)(bytes >> 30), pad);
      std::fflush(stdout);
    }
    const uint32_t Nmax = mix ? 5120 : N, Kmax = mix ? 3072 : K;
    std::vector<uint8_t> runs(runs_bytes(qt, (uint64_t)Nmax * (Kmax / quant_block_values(qt))), 0x5a);
    std::vector<uint16_t> X((size_t)T * K);
    for (size_t i = 0; i < X.size(); i++) X[i] = (uint16_t)(0x3c00 + (i % 11));
    auto *grids = static_cast<uint8_t *>(rt.alloc_device(card, QGRID_BYTES));
    auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, runs.size()));
    auto *dPk = static_cast<uint32_t *>(rt.alloc_device(card, (size_t)(Nmax / 8) * (Kmax / 16) * 64 * 4));
    auto *dAct = static_cast<uint32_t *>(rt.alloc_device(card, (size_t)nbx * (Kmax / 16) * 64 * 4));
    auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * Kmax * 2));
    auto *dT = static_cast<uint32_t *>(rt.alloc_host(4));
    auto *dY = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * 5120 * 4));
    const bool y2_host = std::getenv("BIG_Y2_HOST") != nullptr;
    auto *dY2 = y2_host ? static_cast<uint16_t *>(rt.alloc_host((size_t)T * 5120 * 2))
                        : static_cast<uint16_t *>(rt.alloc_device(card, (size_t)T * 5120 * 2));
    *dT = T;
    L.copy(grids, codebooks().data(), QGRID_BYTES);
    L.copy(dW, runs.data(), runs.size());
    L.copy(dX, X.data(), X.size() * 2);
    L.synchronize();
    const int steps = std::getenv("BIG_STEPS") ? std::atoi(std::getenv("BIG_STEPS")) : 7;
    const int rounds = std::getenv("BIG_ROUNDS") ? std::atoi(std::getenv("BIG_ROUNDS")) : 20;
    auto kd = lib.get(card, "qgemm_iq3_xxs", "dequant_f16_p");
    auto kp = lib.get(card, "pack", "pack_a");
    auto kg = lib.get(card, "gemm_big", "gemm_big_p");
    std::printf("stress card %u y2host %d steps %d rounds %d: T %u N %u K %u bpr %u units %u ks %u nb %u nbx %u nblk %u mblk %u"
                " pk_words %zu act_words %zu\n",
                card, y2_host ? 1 : 0, steps, rounds, T, N, K, bpr, units, ks, nb, nbx, nblk, mblk, (size_t)nb * ks * 64,
                (size_t)nbx * ks * 64);
    std::fflush(stdout);
    auto *hS = static_cast<uint16_t *>(rt.alloc_host(128 * 2));
    for (int r = 0; r < rounds; r++) {
      const Mix &mx = mixes[r % 3];
      const uint32_t mN = mix ? mx.N : N, mK = mix ? mx.K : K;
      const bool mf32 = mix ? mx.f32 != 0 : f32y, my2 = mix ? mx.y2 != 0 : y2_on;
      const uint32_t mbpr = mK / quant_block_values(qt), munits = mK / 32, mks = mK / 16;
      const uint32_t mnb = (mN + 7) / 8, mnblk = (mN + 127) / 128, mmblk = (T + 127) / 128;
      std::printf("round %d begin: N %u K %u T %u f32 %d y2 %d\n", r, mN, mK, T, mf32 ? 1 : 0, my2 ? 1 : 0);
      std::fflush(stdout);
      if (steps & 1) {
        const auto t0 = std::chrono::steady_clock::now();
        L.launch(kd, {dW, mN, mbpr, 0u, munits, dPk, mks, grids}, mnb * munits * 8, 1, 1, 8, 1, 1);
        L.synchronize();
        std::printf("  dequant_f16_p %.2f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
      }
      if (steps & 2) {
        const auto t0 = std::chrono::steady_clock::now();
        L.launch(kp, {dX, T, mK, mK, mks, dAct}, nbx * mks * 8, 1, 1, 8, 1, 1);
        L.synchronize();
        std::printf("  pack_a %.2f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
      }
      if (steps & 4) {
        const auto t0 = std::chrono::steady_clock::now();
        L.launch(kg, {dPk, dAct, mN, mK, dT, dY, mN, mf32 ? 0u : 1u, 0u, 1.0f, mnblk, my2 ? dY2 : nullptr, T},
                 mnblk * mmblk, 1, 1, 32, 4, 1);
        L.synchronize();
        std::printf("  gemm_big_p %.2f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
      }
      // The output: the same 64 values of row 0 each round, and a sum over sampled rows.
      if (!mf32) L.copy(hS, dY, 64 * 2);
      L.copy(hS + 64, dY2, 64 * 2);
      L.synchronize();
      std::printf("  y[0..3] %u %u %u %u  y[60] %u  y2[0..3] %u %u %u %u  f32y %d\n", hS[0], hS[1], hS[2], hS[3], hS[60],
                  hS[64], hS[65], hS[66], hS[67], mf32 ? 1 : 0);
      std::fflush(stdout);
    }
    rt.free(grids);
    if (pad) rt.free(pad);
    rt.free(dW);
    rt.free(dPk);
    rt.free(dAct);
    rt.free(dX);
    rt.free(dT);
    rt.free(dY);
    rt.free(dY2);
    return 0;
  }

  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  struct Shape {
    uint32_t T, N, K;
  };
  // The big tile (kernels/gemm_big.cl) on the operands kernels/pack.cl makes: the same bits as gemm's, f16 and
  // bf16, at the aligned shapes and at the tails of N and rows (130 tokens, 300 rows).
  {
    int big_failures = 0;
    for (bool bf : {false, true})
      for (Shape s : {Shape{1, 33, 16}, Shape{37, 300, 320}, Shape{100, 513, 2560}, Shape{130, 300, 2560},
                      Shape{64, 4, 10240}}) {
        if (s.K % 16) continue;
        const uint32_t ks = s.K / 16, nbw = (s.N + 7) / 8, nbx = (s.T + 7) / 8;
        std::vector<uint16_t> W((size_t)s.N * s.K), X((size_t)s.T * s.K);
        for (size_t i = 0; i < W.size(); i++)
          W[i] = bf ? to_bf16(0.5f + (float)(i % 13) * 0.01f) : ggml_fp32_to_fp16(0.5f + (float)(i % 13) * 0.01f);
        for (size_t i = 0; i < X.size(); i++)
          X[i] = bf ? to_bf16((float)(i % 11) * 0.1f) : ggml_fp32_to_fp16((float)(i % 11) * 0.1f);
        auto *dW = static_cast<uint16_t *>(rt.alloc_device(card, W.size() * 2));
        auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
        auto *dT = static_cast<uint32_t *>(rt.alloc_host(4));
        auto *dRef = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)s.T * s.N * 2));
        auto *dY = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)s.T * s.N * 2));
        auto *dY2 = static_cast<uint16_t *>(rt.alloc_device(card, (size_t)s.T * s.N * 2));
        const size_t wg = (size_t)((s.N + 127) / 128) * 16, xg = (size_t)((s.T + 127) / 128) * 16;
        auto *dWp = static_cast<uint32_t *>(rt.alloc_device(card, wg * ks * 64 * 4 + (64u << 10)));
        auto *dXp = static_cast<uint32_t *>(rt.alloc_device(card, xg * ks * 64 * 4 + (64u << 10)));
        *dT = s.T;
        L.copy(dW, W.data(), W.size() * 2);
        L.copy(dX, X.data(), X.size() * 2);
        L.synchronize();
        const uint32_t zero = 0, one = 1, mode = 0;
        const float scale = 1.0f;
        L.launch(lib.get(card, bf ? "gemm_bf16" : "gemm_f16", "gemm"),
                 {dW, s.N, s.K, s.K, dX, s.K, dT, (float *)dRef, s.N, one, mode, scale, (uint16_t *)nullptr, zero},
                 (s.N + 127) / 128, (s.T + 63) / 64, 1, 64);
        L.launch(lib.get(card, "pack", "pack_b"), {dW, s.N, s.K, ks, dWp}, nbw * ks * 8, 1, 1, 8);
        L.launch(lib.get(card, "pack", "pack_a"), {dX, s.T, s.K, s.K, ks, dXp}, nbx * ks * 8, 1, 1, 8);
        const uint32_t nblk = (s.N + 127) / 128, mblk = (s.T + 127) / 128;
        L.launch(lib.get(card, bf ? "gemm_big_bf16" : "gemm_big", "gemm_big_p"),
                 {dWp, dXp, s.N, s.K, dT, dY, s.N, one, mode, scale, nblk, dY2, s.T}, nblk * mblk, 1, 1, 32, 4, 1);
        L.synchronize();
        auto *h0 = static_cast<uint16_t *>(rt.alloc_host((size_t)s.T * s.N * 2));
        auto *h1 = static_cast<uint16_t *>(rt.alloc_host((size_t)s.T * s.N * 2));
        auto *h2 = static_cast<uint16_t *>(rt.alloc_host((size_t)s.T * s.N * 2));
        L.copy(h0, dRef, (size_t)s.T * s.N * 2);
        L.copy(h1, dY, (size_t)s.T * s.N * 2);
        L.copy(h2, dY2, (size_t)s.T * s.N * 2);
        L.synchronize();
        size_t bad = 0, bad2 = 0, z1 = 0, z2 = 0;
        for (size_t i = 0; i < (size_t)s.T * s.N; i++) {
          bad += h0[i] != h1[i];
          bad2 += h1[i] != h2[i];
          z1 += h1[i] == 0;
          z2 += h2[i] == 0;
        }
        big_failures += bad != 0;
        big_failures += bad2 != 0;
        std::printf("big %s T %4u N %4u K %5u: %zu differ from gemm, %zu in Y2 (zeros %zu / %zu)\n",
                    bf ? "bf16" : "fp16", s.T, s.N, s.K, bad, bad2, z1, z2);
        rt.free(h0);
        rt.free(h1);
        rt.free(h2);
        rt.free(dW);
        rt.free(dX);
        rt.free(dT);
        rt.free(dRef);
        rt.free(dY);
        rt.free(dY2);
        rt.free(dWp);
        rt.free(dXp);
      }
    EXPECT(big_failures == 0, "%d big-tile case(s) differ", big_failures);
  }

  for (bool bf : {false, true})
    for (Shape s : {Shape{1, 33, 16}, Shape{37, 300, 320}, Shape{100, 513, 2560}, Shape{64, 4, 10240},
                    Shape{37, 2560, 320}}) {
      std::vector<uint16_t> W((size_t)s.N * s.K), X((size_t)s.T * s.K);
      std::vector<double> wf(W.size()), xf(X.size());
      auto fill = [&](std::vector<uint16_t> &v, std::vector<double> &f, float sc) {
        for (size_t i = 0; i < v.size(); i++) {
          float x = normal(rng) * sc;
          v[i] = bf ? to_bf16(x) : ggml_fp32_to_fp16(x);
          f[i] = bf ? from_bf16(v[i]) : ggml_fp16_to_fp32(v[i]);
        }
      };
      fill(W, wf, 0.05f);
      fill(X, xf, 1.0f);
      auto *dW = static_cast<uint16_t *>(rt.alloc_device(card, W.size() * 2));
      auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
      auto *dT = static_cast<uint32_t *>(rt.alloc_host(4));
      auto *dY = static_cast<float *>(rt.alloc_host((size_t)s.T * s.N * 4));
      *dT = s.T;
      L.copy(dW, W.data(), W.size() * 2);
      L.copy(dX, X.data(), X.size() * 2);
      L.synchronize();
      for (uint32_t mode : {0u, 1u}) {
        std::fill(dY, dY + (size_t)s.T * s.N, NAN);
        auto k = lib.get(card, bf ? "gemm_bf16" : "gemm_f16", "gemm");
        uint32_t zero = 0;
        float scale = 0.5f;
        uint16_t *null = nullptr;
        L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, zero}, (s.N + 127) / 128,
                 (s.T + 63) / 64, 1, 64);
        L.synchronize();
        std::vector<float> y0(dY, dY + (size_t)s.T * s.N);
        for (uint32_t band : {1u, 4u}) {
          std::fill(dY, dY + (size_t)s.T * s.N, NAN);
          L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, band}, (s.N + 127) / 128,
                   (s.T + 63) / 64, 1, 64);
          L.synchronize();
          bool same = std::memcmp(y0.data(), dY, y0.size() * 4) == 0;
          failures += !same;
          std::printf("%s T %4u N %4u K %5u mode %u: band %u %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode, band,
                      same ? "same bits" : "FAIL, not the plain order's bits");
        }
        if (bf) {
          for (uint32_t tn : {2u, 1u}) {
            std::fill(dY, dY + (size_t)s.T * s.N, NAN);
            std::string mod = "gemm_bf16_t4n" + std::to_string(tn);
            L.launch(lib.get(card, mod, "gemm"),
                     {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, zero},
                     (s.N + 64 * tn - 1) / (64 * tn), (s.T + 31) / 32, 1, 64);
            L.synchronize();
            bool same = std::memcmp(y0.data(), dY, y0.size() * 4) == 0;
            failures += !same;
            std::printf("%s T %4u N %4u K %5u mode %u: %s\n", mod.c_str(), s.T, s.N, s.K, mode,
                        same ? "same bits" : "FAIL, not gemm_bf16's bits");
          }
        }
        std::memcpy(dY, y0.data(), y0.size() * 4);
        if (s.N % 32 == 0) {
          // fp16 into Y (rows N apart) and Y2: the same bits.
          const size_t n16 = (size_t)s.T * s.N;
          auto *h1 = static_cast<uint16_t *>(rt.alloc_host(n16 * 2));
          auto *h2 = static_cast<uint16_t *>(rt.alloc_host(n16 * 2));
          std::fill(h1, h1 + n16, 0xffff);
          std::fill(h2, h2 + n16, 0);
          L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, (float *)h1, s.N, 1u, mode, scale, h2, zero}, (s.N + 127) / 128,
                   (s.T + 63) / 64, 1, 64);
          L.synchronize();
          bool same = std::memcmp(h1, h2, n16 * 2) == 0;
          failures += !same;
          std::printf("%s T %4u N %4u K %5u mode %u: fp16 Y2 %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode,
                      same ? "same bits as Y" : "FAIL, not Y's bits");
          rt.free(h1);
          rt.free(h2);
        }
        double maxerr = 0, maxref = 0;
        for (uint32_t t = 0; t < s.T; t++)
          for (uint32_t n = 0; n < s.N; n++) {
            double r = 0;
            for (uint32_t i = 0; i < s.K; i++) r += wf[(size_t)n * s.K + i] * xf[(size_t)t * s.K + i];
            r *= scale;
            if (mode == 1) r = r / (1 + std::exp(-r));
            maxref = std::max(maxref, std::fabs(r));
            double e = std::fabs(r - dY[(size_t)t * s.N + n]);
            maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
          }
        double rel = maxerr / maxref;
        bool ok = rel <= 2e-3;
        failures += !ok;
        std::printf("%s T %4u N %4u K %5u mode %u: max error %.2e %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode, rel,
                    ok ? "ok" : "FAIL");
      }
      rt.free(dW);
      rt.free(dX);
      rt.free(dT);
      rt.free(dY);
    }
  EXPECT(failures == 0, "%d case(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
