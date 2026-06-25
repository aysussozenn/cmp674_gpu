#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <chrono>
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
constexpr int RS_N = 70;
constexpr int RS_K = 43;
constexpr int RS_PARITY = RS_N - RS_K;

struct EcPointMem {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  uint32_t inf;
};

struct VerifyInstance {
  EcPointMem QS;
  EcPointMem Ppub1;
  EcPointMem Ppub2;
  cgbn_mem_t<BITS> w;
  cgbn_mem_t<BITS> v;
  cgbn_mem_t<BITS> alpha;
  int expected;
  int result;
};

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

struct RsContext {
  uint8_t exp_table[512];
  uint8_t log_table[256];
  std::array<std::array<uint8_t, RS_K>, RS_PARITY> parity_matrix{};
};

inline uint8_t gf_add(uint8_t a, uint8_t b) { return static_cast<uint8_t>(a ^ b); }

uint8_t gf_mul(const RsContext& ctx, uint8_t a, uint8_t b) {
  if (a == 0 || b == 0) return 0;
  const int idx = static_cast<int>(ctx.log_table[a]) + static_cast<int>(ctx.log_table[b]);
  return ctx.exp_table[idx];
}

uint8_t gf_inv(const RsContext& ctx, uint8_t a) {
  if (a == 0) return 0;
  return ctx.exp_table[255 - ctx.log_table[a]];
}

void init_rs_tables(RsContext& ctx) {
  uint16_t x = 1;
  for (int i = 0; i < 255; i++) {
    ctx.exp_table[i] = static_cast<uint8_t>(x);
    ctx.log_table[static_cast<uint8_t>(x)] = static_cast<uint8_t>(i);
    x <<= 1;
    if (x & 0x100u) x ^= 0x11du;
  }
  for (int i = 255; i < 512; i++) ctx.exp_table[i] = ctx.exp_table[i - 255];
  ctx.log_table[0] = 0;
}

bool invert_matrix_gf(const RsContext& ctx,
                      const std::array<std::array<uint8_t, RS_K>, RS_K>& in,
                      std::array<std::array<uint8_t, RS_K>, RS_K>& out) {
  std::array<std::array<uint8_t, RS_K * 2>, RS_K> aug{};
  for (int r = 0; r < RS_K; r++) {
    for (int c = 0; c < RS_K; c++) aug[r][c] = in[r][c];
    for (int c = 0; c < RS_K; c++) aug[r][RS_K + c] = (r == c) ? 1u : 0u;
  }
  for (int col = 0; col < RS_K; col++) {
    int pivot = col;
    while (pivot < RS_K && aug[pivot][col] == 0) pivot++;
    if (pivot == RS_K) return false;
    if (pivot != col) std::swap(aug[pivot], aug[col]);
    const uint8_t inv_pivot = gf_inv(ctx, aug[col][col]);
    for (int c = 0; c < RS_K * 2; c++) aug[col][c] = gf_mul(ctx, aug[col][c], inv_pivot);
    for (int r = 0; r < RS_K; r++) {
      if (r == col) continue;
      const uint8_t factor = aug[r][col];
      if (factor == 0) continue;
      for (int c = 0; c < RS_K * 2; c++) aug[r][c] = gf_add(aug[r][c], gf_mul(ctx, factor, aug[col][c]));
    }
  }
  for (int r = 0; r < RS_K; r++) for (int c = 0; c < RS_K; c++) out[r][c] = aug[r][RS_K + c];
  return true;
}

void init_rs_context(RsContext& ctx) {
  init_rs_tables(ctx);
  std::array<std::array<uint8_t, RS_K>, RS_K> vand_left{};
  for (int i = 0; i < RS_K; i++) {
    const uint8_t x = ctx.exp_table[i];
    uint8_t pow = 1;
    for (int j = 0; j < RS_K; j++) {
      vand_left[i][j] = pow;
      pow = gf_mul(ctx, pow, x);
    }
  }
  std::array<std::array<uint8_t, RS_K>, RS_K> inv_left{};
  if (!invert_matrix_gf(ctx, vand_left, inv_left)) {
    std::cerr << "RS matrix init failed\n";
    std::exit(1);
  }
  for (int r = 0; r < RS_PARITY; r++) {
    const uint8_t x = ctx.exp_table[RS_K + r];
    uint8_t vand_row[RS_K];
    uint8_t pow = 1;
    for (int j = 0; j < RS_K; j++) {
      vand_row[j] = pow;
      pow = gf_mul(ctx, pow, x);
    }
    for (int c = 0; c < RS_K; c++) {
      uint8_t acc = 0;
      for (int t = 0; t < RS_K; t++) acc = gf_add(acc, gf_mul(ctx, vand_row[t], inv_left[t][c]));
      ctx.parity_matrix[r][c] = acc;
    }
  }
}

void rs_encode_systematic(const RsContext& ctx, const std::array<uint8_t, RS_K>& data, std::array<uint8_t, RS_N>& codeword) {
  for (int i = 0; i < RS_K; i++) codeword[i] = data[i];
  for (int r = 0; r < RS_PARITY; r++) {
    uint8_t acc = 0;
    for (int c = 0; c < RS_K; c++) acc = gf_add(acc, gf_mul(ctx, ctx.parity_matrix[r][c], data[c]));
    codeword[RS_K + r] = acc;
  }
}

bool rs_decode_erasures(const RsContext& ctx, std::array<uint8_t, RS_N>& rx, const std::array<uint8_t, RS_N>& erased) {
  int received_positions[RS_K];
  int count = 0;
  for (int i = 0; i < RS_N && count < RS_K; i++) if (!erased[i]) received_positions[count++] = i;
  if (count < RS_K) return false;
  std::array<std::array<uint8_t, RS_K>, RS_K> mat{};
  std::array<uint8_t, RS_K> rhs{};
  for (int row = 0; row < RS_K; row++) {
    const int pos = received_positions[row];
    if (pos < RS_K) for (int col = 0; col < RS_K; col++) mat[row][col] = (col == pos) ? 1u : 0u;
    else for (int col = 0; col < RS_K; col++) mat[row][col] = ctx.parity_matrix[pos - RS_K][col];
    rhs[row] = rx[pos];
  }
  std::array<std::array<uint8_t, RS_K>, RS_K> inv{};
  if (!invert_matrix_gf(ctx, mat, inv)) return false;
  std::array<uint8_t, RS_K> msg{};
  for (int r = 0; r < RS_K; r++) {
    uint8_t acc = 0;
    for (int c = 0; c < RS_K; c++) acc = gf_add(acc, gf_mul(ctx, inv[r][c], rhs[c]));
    msg[r] = acc;
  }
  std::array<uint8_t, RS_N> rebuilt{};
  rs_encode_systematic(ctx, msg, rebuilt);
  rx = rebuilt;
  return true;
}

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

  Point Ppub1, Ppub2;
  env.load(Ppub1.x, &inst[idx].Ppub1.x);
  env.load(Ppub1.y, &inst[idx].Ppub1.y);
  Ppub1.inf = static_cast<int>(inst[idx].Ppub1.inf);

  env.load(Ppub2.x, &inst[idx].Ppub2.x);
  env.load(Ppub2.y, &inst[idx].Ppub2.y);
  Ppub2.inf = static_cast<int>(inst[idx].Ppub2.inf);

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

void set_neg_generator(EcPointMem& ng) {
  std::memset(&ng, 0, sizeof(ng));
  ng.x._limbs[0] = 0xd898c296u;
  ng.x._limbs[1] = 0xf4a13945u;
  ng.x._limbs[2] = 0x2deb33a0u;
  ng.x._limbs[3] = 0x77037d81u;
  ng.x._limbs[4] = 0x63a440f2u;
  ng.x._limbs[5] = 0xf8bce6e5u;
  ng.x._limbs[6] = 0xe12c4247u;
  ng.x._limbs[7] = 0x6b17d1f2u;

  // -Gy mod p for secp256r1.
  ng.y._limbs[0] = 0xc840ae0au;
  ng.y._limbs[1] = 0x3449bf97u;
  ng.y._limbs[2] = 0x94cea131u;
  ng.y._limbs[3] = 0xd431cca9u;
  ng.y._limbs[4] = 0x83f061e9u;
  ng.y._limbs[5] = 0x711814b5u;
  ng.y._limbs[6] = 0x01e58065u;
  ng.y._limbs[7] = 0xb01cbd1cu;
  ng.inf = 0;
}

void serialize_instance_rs(const VerifyInstance& in, std::array<uint8_t, RS_K>& data) {
  for (int i = 0; i < RS_K; i++) data[i] = 0;
  auto put_u32 = [&](int off, uint32_t v) {
    data[off + 0] = static_cast<uint8_t>((v >> 24) & 0xffu);
    data[off + 1] = static_cast<uint8_t>((v >> 16) & 0xffu);
    data[off + 2] = static_cast<uint8_t>((v >> 8) & 0xffu);
    data[off + 3] = static_cast<uint8_t>(v & 0xffu);
  };
  put_u32(0, in.w._limbs[0]);
  put_u32(4, in.v._limbs[0]);
  put_u32(8, in.alpha._limbs[0]);
  put_u32(12, in.expected ? 1u : 0u);
  put_u32(16, in.QS.x._limbs[0]);
  put_u32(20, in.QS.y._limbs[0]);
  put_u32(24, in.Ppub1.x._limbs[0]);
  put_u32(28, in.Ppub2.y._limbs[0]);
}

void deserialize_instance_rs(const std::array<uint8_t, RS_K>& data, VerifyInstance& out) {
  auto get_u32 = [&](int off) -> uint32_t {
    return (static_cast<uint32_t>(data[off + 0]) << 24) |
           (static_cast<uint32_t>(data[off + 1]) << 16) |
           (static_cast<uint32_t>(data[off + 2]) << 8) |
           static_cast<uint32_t>(data[off + 3]);
  };
  std::memset(&out.w, 0, sizeof(out.w));
  std::memset(&out.v, 0, sizeof(out.v));
  std::memset(&out.alpha, 0, sizeof(out.alpha));
  out.w._limbs[0] = get_u32(0);
  out.v._limbs[0] = get_u32(4);
  out.alpha._limbs[0] = get_u32(8);
  out.expected = static_cast<int>(get_u32(12) & 1u);
  out.QS.x._limbs[0] = get_u32(16);
  out.QS.y._limbs[0] = get_u32(20);
  out.Ppub1.x._limbs[0] = get_u32(24);
  out.Ppub2.y._limbs[0] = get_u32(28);
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
  RsContext rs_ctx{};
  init_rs_context(rs_ctx);
  int rs_recovered = 0;
  int rs_decode_failures = 0;
  EcPointMem G, negG;
  set_generator(G);
  set_neg_generator(negG);

  for (int i = 0; i < N; i++) {
    h[i].QS = G;
    h[i].Ppub1 = G;
    h[i].Ppub2 = negG;
    set_u32_bn(h[i].w, 1u);
    set_u32_bn(h[i].v, 1u);
    set_u32_bn(h[i].alpha, (i % 10 == 0) ? 2u : 1u);
    h[i].expected = (i % 10 == 0) ? 0 : 1;
    h[i].result = -1;
  }

  const int rs_instances = (N < 1024) ? N : 1024;
  constexpr int rs_erasures = 8;
  for (int i = 0; i < rs_instances; i++) {
    std::array<uint8_t, RS_K> data{};
    std::array<uint8_t, RS_N> codeword{};
    std::array<uint8_t, RS_N> received{};
    std::array<uint8_t, RS_N> erased{};
    serialize_instance_rs(h[i], data);
    rs_encode_systematic(rs_ctx, data, codeword);
    received = codeword;
    for (int j = 0; j < RS_N; j++) erased[j] = 0;
    for (int e = 0; e < rs_erasures; e++) {
      const int pos = (i * 11 + e * 7) % RS_N;
      received[pos] = 0;
      erased[pos] = 1;
    }
    if (!rs_decode_erasures(rs_ctx, received, erased)) {
      rs_decode_failures++;
      continue;
    }
    std::array<uint8_t, RS_K> recovered{};
    for (int j = 0; j < RS_K; j++) recovered[j] = received[j];
    deserialize_instance_rs(recovered, h[i]);
    rs_recovered++;
  }

  const auto e2e_start = std::chrono::high_resolution_clock::now();
  VerifyInstance* d = nullptr;
  cgbn_error_report_t* report = nullptr;
  check_cuda(cudaMalloc(&d, sizeof(VerifyInstance) * N), "cudaMalloc(inst)");
  check_cuda(cgbn_error_report_alloc(&report), "cgbn_error_report_alloc");

  constexpr int BATCH_SIZE = 8192;
  constexpr int NUM_STREAMS = 3;
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
    check_cuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
  }

  cudaEvent_t s, e, h2d_s, h2d_e, d2h_s, d2h_e;
  cudaEvent_t stream_kernel_starts[NUM_STREAMS], stream_kernel_ends[NUM_STREAMS];
  check_cuda(cudaEventCreate(&s), "cudaEventCreate(s)");
  check_cuda(cudaEventCreate(&e), "cudaEventCreate(e)");
  check_cuda(cudaEventCreate(&h2d_s), "cudaEventCreate(h2d_s)");
  check_cuda(cudaEventCreate(&h2d_e), "cudaEventCreate(h2d_e)");
  check_cuda(cudaEventCreate(&d2h_s), "cudaEventCreate(d2h_s)");
  check_cuda(cudaEventCreate(&d2h_e), "cudaEventCreate(d2h_e)");
  for (int i = 0; i < NUM_STREAMS; i++) {
    check_cuda(cudaEventCreate(&stream_kernel_starts[i]), "cudaEventCreate(stream_kernel_starts[i])");
    check_cuda(cudaEventCreate(&stream_kernel_ends[i]), "cudaEventCreate(stream_kernel_ends[i])");
  }

  check_cuda(cudaEventRecord(h2d_s), "cudaEventRecord(h2d_s)");
  constexpr int PROFILE_BATCHES[] = {0, 2, 4};
  constexpr int NUM_PROFILE_BATCHES = sizeof(PROFILE_BATCHES) / sizeof(PROFILE_BATCHES[0]);
  float profile_times_ms[NUM_PROFILE_BATCHES] = {0.0f};
  int profile_collected = 0;

  int batch_idx = 0;
  for (int batch_start = 0; batch_start < N; batch_start += BATCH_SIZE, batch_idx++) {
    const int batch_end = (batch_start + BATCH_SIZE < N) ? (batch_start + BATCH_SIZE) : N;
    const int batch_size = batch_end - batch_start;
    const int blocks = (batch_size * TPI + TPB - 1) / TPB;
    const int stream_idx = batch_idx % NUM_STREAMS;
    cudaStream_t current_stream = streams[stream_idx];

    check_cuda(cudaMemcpyAsync(d + batch_start, h.data() + batch_start, sizeof(VerifyInstance) * batch_size,
                               cudaMemcpyHostToDevice, current_stream),
               "cudaMemcpyAsync H2D");

    // Warmup.
    kernel_real_verify<<<blocks, TPB, 0, current_stream>>>(report, d + batch_start, batch_size);
    check_cuda(cudaGetLastError(), "kernel warmup launch");

    bool is_profile_batch = false;
    for (int i = 0; i < NUM_PROFILE_BATCHES; i++) {
      if (batch_idx == PROFILE_BATCHES[i]) {
        is_profile_batch = true;
        break;
      }
    }
    if (is_profile_batch) {
      check_cuda(cudaEventRecord(stream_kernel_starts[stream_idx], current_stream), "cudaEventRecord(kernel_start)");
      for (int i = 0; i < repeats; i++) {
        kernel_real_verify<<<blocks, TPB, 0, current_stream>>>(report, d + batch_start, batch_size);
      }
      check_cuda(cudaEventRecord(stream_kernel_ends[stream_idx], current_stream), "cudaEventRecord(kernel_end)");
      check_cuda(cudaGetLastError(), "kernel profile launch");
      profile_collected++;
    }

    kernel_real_verify<<<blocks, TPB, 0, current_stream>>>(report, d + batch_start, batch_size);
    check_cuda(cudaGetLastError(), "kernel final launch");

    check_cuda(cudaMemcpyAsync(h.data() + batch_start, d + batch_start, sizeof(VerifyInstance) * batch_size,
                               cudaMemcpyDeviceToHost, current_stream),
               "cudaMemcpyAsync D2H");
  }

  check_cuda(cudaEventRecord(h2d_e), "cudaEventRecord(h2d_e)");
  check_cuda(cudaEventRecord(d2h_s), "cudaEventRecord(d2h_s)");
  for (int i = 0; i < NUM_STREAMS; i++) {
    check_cuda(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
  }
  check_cuda(cudaEventRecord(d2h_e), "cudaEventRecord(d2h_e)");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  if (cgbn_error_report_check(report)) {
    std::cerr << "CGBN error: " << cgbn_error_string(report) << '\n';
    return 1;
  }

  float kernel_ms = 0.0f, h2d_ms = 0.0f, d2h_ms = 0.0f;
  int valid_profile_count = 0;
  for (int i = 0; i < NUM_PROFILE_BATCHES; i++) {
    if (PROFILE_BATCHES[i] >= batch_idx) continue;
    float elapsed = 0.0f;
    const int sidx = PROFILE_BATCHES[i] % NUM_STREAMS;
    check_cuda(cudaEventElapsedTime(&elapsed, stream_kernel_starts[sidx], stream_kernel_ends[sidx]),
               "cudaEventElapsedTime profile");
    profile_times_ms[valid_profile_count++] = elapsed;
  }
  if (valid_profile_count > 0) {
    std::sort(profile_times_ms, profile_times_ms + valid_profile_count);
    kernel_ms = profile_times_ms[valid_profile_count / 2];
  } else {
    kernel_ms = 0.0f;
  }
  if (kernel_ms <= 0.0f) {
    const int blocks_all = (N * TPI + TPB - 1) / TPB;
    check_cuda(cudaEventRecord(s), "cudaEventRecord(fallback_start)");
    for (int i = 0; i < repeats; i++) {
      kernel_real_verify<<<blocks_all, TPB>>>(report, d, N);
    }
    check_cuda(cudaEventRecord(e), "cudaEventRecord(fallback_end)");
    check_cuda(cudaGetLastError(), "kernel fallback launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fallback");
    check_cuda(cudaEventElapsedTime(&kernel_ms, s, e), "cudaEventElapsedTime fallback");
  }
  check_cuda(cudaEventElapsedTime(&h2d_ms, h2d_s, h2d_e), "cudaEventElapsedTime h2d");
  check_cuda(cudaEventElapsedTime(&d2h_ms, d2h_s, d2h_e), "cudaEventElapsedTime d2h");
  const auto e2e_end = std::chrono::high_resolution_clock::now();
  const double e2e_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();

  int mismatches = 0;
  int valid = 0;
  for (int i = 0; i < N; i++) {
    valid += (h[i].result == 1);
    if (h[i].result != h[i].expected) mismatches++;
  }

  std::cout << "Real ECC GPU verify kernel (full Q~ terms: wQS + Ppub2 + vPpub1)\n";
  std::cout << "Instances: " << N << '\n';
  std::cout << "RS simulated instances: " << rs_instances << ", erasures/instance: " << rs_erasures << '\n';
  std::cout << "RS recoveries: " << rs_recovered << ", RS decode failures: " << rs_decode_failures << '\n';
  const double avg_ms = (kernel_ms > 0.0f) ? (static_cast<double>(kernel_ms) / static_cast<double>(repeats)) : 0.0;
  std::cout << "Kernel total time over " << repeats << " runs (ms): " << kernel_ms << '\n';
  std::cout << "Kernel avg time (ms): " << avg_ms << '\n';
  std::cout << "GPU H2D time (ms): " << h2d_ms << '\n';
  std::cout << "GPU D2H time (ms): " << d2h_ms << '\n';
  std::cout << "GPU end-to-end time (ms): " << e2e_ms << '\n';
  std::cout << "Throughput (verifies/s): " << ((avg_ms > 0.0) ? ((static_cast<double>(N) * 1000.0) / avg_ms) : 0.0) << '\n';
  std::cout << "Valid: " << valid << ", Invalid: " << (N - valid) << '\n';
  std::cout << "Mismatches vs expected: " << mismatches << '\n';

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cudaEventDestroy(h2d_s);
  cudaEventDestroy(h2d_e);
  cudaEventDestroy(d2h_s);
  cudaEventDestroy(d2h_e);
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaEventDestroy(stream_kernel_starts[i]);
    cudaEventDestroy(stream_kernel_ends[i]);
  }
  cgbn_error_report_free(report);
  cudaFree(d);
  return 0;
}
