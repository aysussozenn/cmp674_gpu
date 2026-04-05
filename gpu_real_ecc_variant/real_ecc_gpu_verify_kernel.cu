#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// Minimal CGBN bootstrap without host GMP dependency.
typedef enum {
  cgbn_no_error = 0,
  cgbn_unsupported_threads_per_instance = 1,
  cgbn_unsupported_size = 2,
  cgbn_unsupported_limbs_per_thread = 3,
  cgbn_unsupported_operation = 4,
  cgbn_threads_per_block_mismatch = 5,
  cgbn_threads_per_instance_mismatch = 6,
  cgbn_division_by_zero_error = 7,
  cgbn_division_overflow_error = 8,
  cgbn_invalid_montgomery_modulus_error = 9,
  cgbn_modulus_not_odd_error = 10,
  cgbn_inverse_does_not_exist_error = 11,
} cgbn_error_t;

typedef struct {
  volatile cgbn_error_t _error;
  uint32_t _instance;
  dim3 _threadIdx;
  dim3 _blockIdx;
} cgbn_error_report_t;

typedef enum {
  cgbn_no_checks,
  cgbn_report_monitor,
  cgbn_print_monitor,
  cgbn_halt_monitor,
} cgbn_monitor_t;

cudaError_t cgbn_error_report_alloc(cgbn_error_report_t** report);
cudaError_t cgbn_error_report_free(cgbn_error_report_t* report);
bool cgbn_error_report_check(cgbn_error_report_t* report);
const char* cgbn_error_string(cgbn_error_report_t* report);

#include "cgbn/cgbn.cu"
#include "cgbn/cgbn_cuda.h"

namespace {

constexpr int BITS = 256;
constexpr int TPI = 32;
constexpr int TPB = 128;

struct EcPointMem {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  uint32_t inf;
};

struct VerifyInstance {
  EcPointMem QS;
  cgbn_mem_t<BITS> w;
  cgbn_mem_t<BITS> v;
  cgbn_mem_t<BITS> alpha;
  int expected;
  int result;
};

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__device__ __forceinline__ void bn_set_zero(env_t env, env_t::cgbn_t& x) {
  env.set_ui32(x, 0);
}

__device__ __forceinline__ bool bn_is_zero(env_t env, const env_t::cgbn_t& x) {
  env_t::cgbn_t z;
  env.set_ui32(z, 0);
  return env.compare(x, z) == 0;
}

__device__ __forceinline__ void load_p(env_t env, env_t::cgbn_t& p) {
  env.set_ui32(p, 0);
  env.insert_bits_ui32(p, p, 0, 32, 0xffffffffu);
  env.insert_bits_ui32(p, p, 32, 32, 0xffffffffu);
  env.insert_bits_ui32(p, p, 64, 32, 0xffffffffu);
  env.insert_bits_ui32(p, p, 96, 32, 0x00000000u);
  env.insert_bits_ui32(p, p, 128, 32, 0x00000000u);
  env.insert_bits_ui32(p, p, 160, 32, 0x00000000u);
  env.insert_bits_ui32(p, p, 192, 32, 0x00000001u);
  env.insert_bits_ui32(p, p, 224, 32, 0xffffffffu);
}

__device__ __forceinline__ void load_a(env_t env, env_t::cgbn_t& a, const env_t::cgbn_t& p) {
  // a = -3 mod p = p - 3
  env_t::cgbn_t three;
  env.set_ui32(three, 3);
  env.sub(a, p, three);
}

__device__ __forceinline__ void load_generator(env_t env, env_t::cgbn_t& gx, env_t::cgbn_t& gy) {
  env.set_ui32(gx, 0);
  env.set_ui32(gy, 0);

  // Gx = 6b17...c296
  env.insert_bits_ui32(gx, gx, 0, 32, 0xd898c296u);
  env.insert_bits_ui32(gx, gx, 32, 32, 0xf4a13945u);
  env.insert_bits_ui32(gx, gx, 64, 32, 0x2deb33a0u);
  env.insert_bits_ui32(gx, gx, 96, 32, 0x77037d81u);
  env.insert_bits_ui32(gx, gx, 128, 32, 0x63a440f2u);
  env.insert_bits_ui32(gx, gx, 160, 32, 0xf8bce6e5u);
  env.insert_bits_ui32(gx, gx, 192, 32, 0xe12c4247u);
  env.insert_bits_ui32(gx, gx, 224, 32, 0x6b17d1f2u);

  // Gy = 4fe3...51f5
  env.insert_bits_ui32(gy, gy, 0, 32, 0x37bf51f5u);
  env.insert_bits_ui32(gy, gy, 32, 32, 0xcbb64068u);
  env.insert_bits_ui32(gy, gy, 64, 32, 0x6b315eceu);
  env.insert_bits_ui32(gy, gy, 96, 32, 0x2bce3357u);
  env.insert_bits_ui32(gy, gy, 128, 32, 0x7c0f9e16u);
  env.insert_bits_ui32(gy, gy, 160, 32, 0x8ee7eb4au);
  env.insert_bits_ui32(gy, gy, 192, 32, 0xfe1a7f9bu);
  env.insert_bits_ui32(gy, gy, 224, 32, 0x4fe342e2u);
}

__device__ __forceinline__ void mod_add(env_t env, env_t::cgbn_t& r, const env_t::cgbn_t& a,
                                         const env_t::cgbn_t& b, const env_t::cgbn_t& p) {
  env.add(r, a, b);
  if (env.compare(r, p) >= 0) env.sub(r, r, p);
}

__device__ __forceinline__ void mod_sub(env_t env, env_t::cgbn_t& r, const env_t::cgbn_t& a,
                                         const env_t::cgbn_t& b, const env_t::cgbn_t& p) {
  if (env.compare(a, b) >= 0) {
    env.sub(r, a, b);
  } else {
    env_t::cgbn_t t;
    env.sub(t, b, a);
    env.sub(r, p, t);
  }
}

__device__ __forceinline__ void mod_mul(env_t env, env_t::cgbn_t& r, const env_t::cgbn_t& a,
                                         const env_t::cgbn_t& b, const env_t::cgbn_t& p) {
  env_t::cgbn_wide_t w;
  env.mul_wide(w, a, b);
  env.rem_wide(r, w, p);
}

__device__ __forceinline__ bool mod_inv(env_t env, env_t::cgbn_t& r, const env_t::cgbn_t& a,
                                         const env_t::cgbn_t& p) {
  return env.modular_inverse(r, a, p);
}

struct Point {
  env_t::cgbn_t x;
  env_t::cgbn_t y;
  int inf;
};

__device__ __forceinline__ void point_set_inf(Point& P) { P.inf = 1; }

__device__ __forceinline__ bool point_equal(env_t env, const Point& A, const Point& B) {
  if (A.inf && B.inf) return true;
  if (A.inf != B.inf) return false;
  return env.compare(A.x, B.x) == 0 && env.compare(A.y, B.y) == 0;
}

__device__ __forceinline__ void point_double(env_t env, Point& R, const Point& P,
                                             const env_t::cgbn_t& p, const env_t::cgbn_t& a_curve) {
  if (P.inf || bn_is_zero(env, P.y)) {
    point_set_inf(R);
    return;
  }

  env_t::cgbn_t x2, three_x2, num, den, inv_den, lambda, lambda2, two_x, t;

  mod_mul(env, x2, P.x, P.x, p);
  env.set_ui32(three_x2, 0);
  mod_add(env, three_x2, x2, x2, p);
  mod_add(env, three_x2, three_x2, x2, p);
  mod_add(env, num, three_x2, a_curve, p);

  mod_add(env, den, P.y, P.y, p);
  if (!mod_inv(env, inv_den, den, p)) {
    point_set_inf(R);
    return;
  }

  mod_mul(env, lambda, num, inv_den, p);
  mod_mul(env, lambda2, lambda, lambda, p);
  mod_add(env, two_x, P.x, P.x, p);
  mod_sub(env, R.x, lambda2, two_x, p);

  mod_sub(env, t, P.x, R.x, p);
  mod_mul(env, t, lambda, t, p);
  mod_sub(env, R.y, t, P.y, p);
  R.inf = 0;
}

__device__ __forceinline__ void point_add(env_t env, Point& R, const Point& P, const Point& Q,
                                          const env_t::cgbn_t& p, const env_t::cgbn_t& a_curve) {
  if (P.inf) {
    R = Q;
    return;
  }
  if (Q.inf) {
    R = P;
    return;
  }

  if (env.compare(P.x, Q.x) == 0) {
    env_t::cgbn_t ysum;
    mod_add(env, ysum, P.y, Q.y, p);
    if (bn_is_zero(env, ysum)) {
      point_set_inf(R);
      return;
    }
    point_double(env, R, P, p, a_curve);
    return;
  }

  env_t::cgbn_t num, den, inv_den, lambda, lambda2, t;
  mod_sub(env, num, Q.y, P.y, p);
  mod_sub(env, den, Q.x, P.x, p);
  if (!mod_inv(env, inv_den, den, p)) {
    point_set_inf(R);
    return;
  }

  mod_mul(env, lambda, num, inv_den, p);
  mod_mul(env, lambda2, lambda, lambda, p);
  mod_sub(env, t, lambda2, P.x, p);
  mod_sub(env, R.x, t, Q.x, p);

  mod_sub(env, t, P.x, R.x, p);
  mod_mul(env, t, lambda, t, p);
  mod_sub(env, R.y, t, P.y, p);
  R.inf = 0;
}

__device__ __forceinline__ void point_scalar_mul(env_t env, Point& R, const env_t::cgbn_t& k,
                                                 const Point& P, const env_t::cgbn_t& p,
                                                 const env_t::cgbn_t& a_curve) {
  Point acc;
  point_set_inf(acc);
  Point cur = P;

  for (int bit = 0; bit < 256; bit++) {
    const uint32_t b = env.extract_bits_ui32(k, bit, 1);
    if (b == 1u) {
      Point tmp;
      point_add(env, tmp, acc, cur, p, a_curve);
      acc = tmp;
    }
    Point dblp;
    point_double(env, dblp, cur, p, a_curve);
    cur = dblp;
  }
  R = acc;
}

__global__ void kernel_real_verify(cgbn_error_report_t* report, VerifyInstance* inst, int count) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= count) return;

  context_t bn_context(cgbn_report_monitor, report, idx);
  env_t env(bn_context.env<env_t>());

  env_t::cgbn_t p, a_curve;
  load_p(env, p);
  load_a(env, a_curve, p);

  Point QS;
  env.load(QS.x, &inst[idx].QS.x);
  env.load(QS.y, &inst[idx].QS.y);
  QS.inf = static_cast<int>(inst[idx].QS.inf);

  env_t::cgbn_t w, v, alpha;
  env.load(w, &inst[idx].w);
  env.load(v, &inst[idx].v);
  env.load(alpha, &inst[idx].alpha);

  // Ppub1 = O, Ppub2 = O for this minimum real-ECC GPU verify stage.
  Point Ppub1, Ppub2;
  point_set_inf(Ppub1);
  point_set_inf(Ppub2);

  Point t, vP1, tmp;
  point_scalar_mul(env, t, w, QS, p, a_curve);
  point_scalar_mul(env, vP1, v, Ppub1, p, a_curve);
  point_add(env, tmp, t, Ppub2, p, a_curve);
  point_add(env, t, tmp, vP1, p, a_curve);

  Point lhs;
  point_scalar_mul(env, lhs, alpha, t, p, a_curve);

  inst[idx].result = point_equal(env, lhs, QS) ? 1 : 0;
}

void check_cuda(cudaError_t code, const char* expr) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error at " << expr << ": " << cudaGetErrorString(code) << '\n';
    std::exit(1);
  }
}

void set_u32_bn(cgbn_mem_t<BITS>& x, uint32_t v) {
  std::memset(&x, 0, sizeof(x));
  x._limbs[0] = v;
}

void set_generator(EcPointMem& g) {
  std::memset(&g, 0, sizeof(g));
  g.x._limbs[0] = 0xd898c296u;
  g.x._limbs[1] = 0xf4a13945u;
  g.x._limbs[2] = 0x2deb33a0u;
  g.x._limbs[3] = 0x77037d81u;
  g.x._limbs[4] = 0x63a440f2u;
  g.x._limbs[5] = 0xf8bce6e5u;
  g.x._limbs[6] = 0xe12c4247u;
  g.x._limbs[7] = 0x6b17d1f2u;

  g.y._limbs[0] = 0x37bf51f5u;
  g.y._limbs[1] = 0xcbb64068u;
  g.y._limbs[2] = 0x6b315eceu;
  g.y._limbs[3] = 0x2bce3357u;
  g.y._limbs[4] = 0x7c0f9e16u;
  g.y._limbs[5] = 0x8ee7eb4au;
  g.y._limbs[6] = 0xfe1a7f9bu;
  g.y._limbs[7] = 0x4fe342e2u;
  g.inf = 0;
}

}  // namespace

int main(int argc, char** argv) {
  int N = 2048;
  int repeats = 10;
  if (argc >= 2) {
    N = std::atoi(argv[1]);
    if (N <= 0) N = 2048;
  }
  if (argc >= 3) {
    repeats = std::atoi(argv[2]);
    if (repeats <= 0) repeats = 10;
  }

  std::vector<VerifyInstance> h(N);
  EcPointMem G;
  set_generator(G);

  for (int i = 0; i < N; i++) {
    h[i].QS = G;
    set_u32_bn(h[i].w, 1u);
    set_u32_bn(h[i].v, 0u);
    set_u32_bn(h[i].alpha, (i % 10 == 0) ? 2u : 1u);
    h[i].expected = (i % 10 == 0) ? 0 : 1;
    h[i].result = -1;
  }

  VerifyInstance* d = nullptr;
  cgbn_error_report_t* report = nullptr;
  check_cuda(cudaMalloc(&d, sizeof(VerifyInstance) * N), "cudaMalloc(inst)");
  check_cuda(cudaMemcpy(d, h.data(), sizeof(VerifyInstance) * N, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  check_cuda(cgbn_error_report_alloc(&report), "cgbn_error_report_alloc");

  cudaEvent_t s, e;
  check_cuda(cudaEventCreate(&s), "cudaEventCreate(s)");
  check_cuda(cudaEventCreate(&e), "cudaEventCreate(e)");

  const int blocks = (N * TPI + TPB - 1) / TPB;
  // Warmup.
  kernel_real_verify<<<blocks, TPB>>>(report, d, N);
  check_cuda(cudaGetLastError(), "kernel warmup launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

  check_cuda(cudaEventRecord(s), "cudaEventRecord(start)");
  for (int i = 0; i < repeats; i++) {
    kernel_real_verify<<<blocks, TPB>>>(report, d, N);
  }
  check_cuda(cudaEventRecord(e), "cudaEventRecord(end)");
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  if (cgbn_error_report_check(report)) {
    std::cerr << "CGBN error: " << cgbn_error_string(report) << '\n';
    return 1;
  }

  float kernel_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&kernel_ms, s, e), "cudaEventElapsedTime");

  check_cuda(cudaMemcpy(h.data(), d, sizeof(VerifyInstance) * N, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

  int mismatches = 0;
  int valid = 0;
  for (int i = 0; i < N; i++) {
    valid += (h[i].result == 1);
    if (h[i].result != h[i].expected) mismatches++;
  }

  std::cout << "Real ECC GPU verify kernel (minimum P-256 stage)\n";
  std::cout << "Instances: " << N << '\n';
  const double avg_ms = static_cast<double>(kernel_ms) / static_cast<double>(repeats);
  std::cout << "Kernel total time over " << repeats << " runs (ms): " << kernel_ms << '\n';
  std::cout << "Kernel avg time (ms): " << avg_ms << '\n';
  std::cout << "Throughput (verifies/s): " << ((static_cast<double>(N) * 1000.0) / avg_ms) << '\n';
  std::cout << "Valid: " << valid << ", Invalid: " << (N - valid) << '\n';
  std::cout << "Mismatches vs expected: " << mismatches << '\n';

  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cgbn_error_report_free(report);
  cudaFree(d);
  return 0;
}
