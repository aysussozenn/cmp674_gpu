#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
#define HAS_BOOST_CPP_INT 1
#else
#define HAS_BOOST_CPP_INT 0
#endif

namespace {
#if HAS_BOOST_CPP_INT
using boost::multiprecision::cpp_int;
#endif

constexpr uint32_t ADSB_MESSAGE_BYTES = 14;  // 112-bit ADS-B payload
constexpr int THREADS_PER_BLOCK = 256;
constexpr int GPU_PROFILE_RUNS = 20;
constexpr uint32_t FIELD_P = 65521u;         // Toy curve field prime
constexpr uint32_t SCALAR_N = 4111u;         // Order of chosen generator
constexpr uint32_t CURVE_A = 2u;
constexpr uint32_t CURVE_B = 3u;
constexpr uint32_t GEN_X = 3u;
constexpr uint32_t GEN_Y = 65515u;

struct EcPoint {
  uint32_t x;
  uint32_t y;
  uint8_t inf;
};

struct SignerPublicKey {
  uint32_t id;
  EcPoint RS;
  EcPoint QS;
};

struct SignerPrivateState {
  uint32_t id;
  uint32_t k;
  uint32_t x;
  EcPoint RS;
  EcPoint QS;
};

struct ADSB_Packet {
  uint8_t payload[ADSB_MESSAGE_BYTES];
  uint32_t timestamp;
  uint32_t signer_index;
  uint32_t alpha;
};

#if HAS_BOOST_CPP_INT
struct RealEcPoint {
  cpp_int x;
  cpp_int y;
  bool inf;
};

struct RealSignerPublicKey {
  uint32_t id;
  RealEcPoint RS;
  RealEcPoint QS;
};

struct RealSignerPrivateState {
  uint32_t id;
  cpp_int k;
  cpp_int x;
  RealEcPoint RS;
  RealEcPoint QS;
};

struct RealADSBPacket {
  uint8_t payload[ADSB_MESSAGE_BYTES];
  uint32_t timestamp;
  uint32_t signer_index;
  cpp_int alpha;
};
#endif

// SHA-256 constants.
__constant__ uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__host__ __device__ inline uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

__host__ __device__ inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

__host__ __device__ inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

__host__ __device__ inline uint32_t big_sigma0(uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__host__ __device__ inline uint32_t big_sigma1(uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__host__ __device__ inline uint32_t small_sigma0(uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__host__ __device__ inline uint32_t small_sigma1(uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// Single-block SHA-256, valid for inputs <=55 bytes.
__host__ __device__ void sha256_single_block(const uint8_t* message, int len, uint32_t digest[8]) {
  uint8_t block[64];
  uint32_t w[64];

  for (int i = 0; i < 64; i++) block[i] = 0;
  for (int i = 0; i < len; i++) block[i] = message[i];
  block[len] = 0x80;

  const uint64_t bit_len = static_cast<uint64_t>(len) * 8ull;
  for (int i = 0; i < 8; i++) {
    block[63 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xffu);
  }

  for (int t = 0; t < 16; t++) {
    w[t] = (static_cast<uint32_t>(block[t * 4]) << 24) |
           (static_cast<uint32_t>(block[t * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[t * 4 + 2]) << 8) |
           (static_cast<uint32_t>(block[t * 4 + 3]));
  }

  for (int t = 16; t < 64; t++) {
    w[t] = small_sigma1(w[t - 2]) + w[t - 7] + small_sigma0(w[t - 15]) + w[t - 16];
  }

  uint32_t a = 0x6a09e667u;
  uint32_t b = 0xbb67ae85u;
  uint32_t c = 0x3c6ef372u;
  uint32_t d = 0xa54ff53au;
  uint32_t e = 0x510e527fu;
  uint32_t f = 0x9b05688cu;
  uint32_t g = 0x1f83d9abu;
  uint32_t h = 0x5be0cd19u;

  for (int t = 0; t < 64; t++) {
#ifdef __CUDA_ARCH__
    const uint32_t k = kSha256K[t];
#else
    static constexpr uint32_t kHost[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    const uint32_t k = kHost[t];
#endif
    const uint32_t t1 = h + big_sigma1(e) + ch(e, f, g) + k + w[t];
    const uint32_t t2 = big_sigma0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  digest[0] = 0x6a09e667u + a;
  digest[1] = 0xbb67ae85u + b;
  digest[2] = 0x3c6ef372u + c;
  digest[3] = 0xa54ff53au + d;
  digest[4] = 0x510e527fu + e;
  digest[5] = 0x9b05688cu + f;
  digest[6] = 0x1f83d9abu + g;
  digest[7] = 0x5be0cd19u + h;
}

__host__ __device__ inline EcPoint point_infinity() {
  EcPoint p{};
  p.x = 0;
  p.y = 0;
  p.inf = 1;
  return p;
}

__host__ __device__ inline EcPoint generator_point() {
  EcPoint p{};
  p.x = GEN_X;
  p.y = GEN_Y;
  p.inf = 0;
  return p;
}

__host__ __device__ inline bool point_equal(const EcPoint& a, const EcPoint& b) {
  if (a.inf && b.inf) return true;
  if (a.inf != b.inf) return false;
  return (a.x == b.x) && (a.y == b.y);
}

__host__ __device__ inline uint32_t add_mod(uint32_t a, uint32_t b) {
  const uint64_t t = static_cast<uint64_t>(a) + b;
  return static_cast<uint32_t>(t >= FIELD_P ? (t - FIELD_P) : t);
}

__host__ __device__ inline uint32_t sub_mod(uint32_t a, uint32_t b) {
  return (a >= b) ? (a - b) : static_cast<uint32_t>(FIELD_P - (b - a));
}

__host__ __device__ inline uint32_t mul_mod(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) % FIELD_P);
}

__host__ __device__ uint32_t inv_mod_p(uint32_t a) {
  if (a == 0) return 0;
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = static_cast<int64_t>(FIELD_P);
  int64_t new_r = static_cast<int64_t>(a);

  while (new_r != 0) {
    const int64_t q = r / new_r;
    const int64_t tmp_t = t - q * new_t;
    t = new_t;
    new_t = tmp_t;
    const int64_t tmp_r = r - q * new_r;
    r = new_r;
    new_r = tmp_r;
  }

  if (r != 1) return 0;
  if (t < 0) t += FIELD_P;
  return static_cast<uint32_t>(t);
}

__host__ __device__ inline uint32_t add_mod_n(uint32_t a, uint32_t b) {
  const uint32_t t = a + b;
  return (t >= SCALAR_N) ? (t - SCALAR_N) : t;
}

__host__ __device__ inline uint32_t mul_mod_n(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) % SCALAR_N);
}

__host__ __device__ uint32_t inv_mod_n(uint32_t a) {
  if (a == 0) return 0;
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = static_cast<int64_t>(SCALAR_N);
  int64_t new_r = static_cast<int64_t>(a);

  while (new_r != 0) {
    const int64_t q = r / new_r;
    const int64_t tmp_t = t - q * new_t;
    t = new_t;
    new_t = tmp_t;
    const int64_t tmp_r = r - q * new_r;
    r = new_r;
    new_r = tmp_r;
  }

  if (r != 1) return 0;
  if (t < 0) t += SCALAR_N;
  return static_cast<uint32_t>(t);
}

__host__ __device__ EcPoint point_double(const EcPoint& p) {
  if (p.inf || p.y == 0) return point_infinity();

  const uint32_t x2 = mul_mod(p.x, p.x);
  const uint32_t num = add_mod(mul_mod(3u, x2), CURVE_A);
  const uint32_t den = mul_mod(2u, p.y);
  const uint32_t inv_den = inv_mod_p(den);
  if (inv_den == 0) return point_infinity();

  const uint32_t lambda = mul_mod(num, inv_den);
  const uint32_t x3 = sub_mod(sub_mod(mul_mod(lambda, lambda), p.x), p.x);
  const uint32_t y3 = sub_mod(mul_mod(lambda, sub_mod(p.x, x3)), p.y);

  EcPoint r{};
  r.x = x3;
  r.y = y3;
  r.inf = 0;
  return r;
}

__host__ __device__ EcPoint point_add(const EcPoint& p, const EcPoint& q) {
  if (p.inf) return q;
  if (q.inf) return p;

  if (p.x == q.x) {
    if (add_mod(p.y, q.y) == 0) return point_infinity();
    return point_double(p);
  }

  const uint32_t num = sub_mod(q.y, p.y);
  const uint32_t den = sub_mod(q.x, p.x);
  const uint32_t inv_den = inv_mod_p(den);
  if (inv_den == 0) return point_infinity();

  const uint32_t lambda = mul_mod(num, inv_den);
  const uint32_t x3 = sub_mod(sub_mod(mul_mod(lambda, lambda), p.x), q.x);
  const uint32_t y3 = sub_mod(mul_mod(lambda, sub_mod(p.x, x3)), p.y);

  EcPoint r{};
  r.x = x3;
  r.y = y3;
  r.inf = 0;
  return r;
}

__host__ __device__ EcPoint point_mul(uint32_t scalar, EcPoint base) {
  EcPoint result = point_infinity();
  while (scalar != 0) {
    if (scalar & 1u) {
      result = point_add(result, base);
    }
    base = point_double(base);
    scalar >>= 1u;
  }
  return result;
}

__host__ __device__ inline void write_u32_be(uint8_t* out, uint32_t v) {
  out[0] = static_cast<uint8_t>((v >> 24) & 0xffu);
  out[1] = static_cast<uint8_t>((v >> 16) & 0xffu);
  out[2] = static_cast<uint8_t>((v >> 8) & 0xffu);
  out[3] = static_cast<uint8_t>(v & 0xffu);
}

__host__ __device__ uint32_t hash_to_scalar(const uint8_t* msg, int len) {
  uint32_t digest[8];
  sha256_single_block(msg, len, digest);
  uint32_t v = digest[0] % SCALAR_N;
  if (v == 0) v = 1;
  return v;
}

__host__ __device__ uint32_t H1(uint32_t id, const EcPoint& P) {
  uint8_t buf[12];
  write_u32_be(buf + 0, id);
  write_u32_be(buf + 4, P.x);
  write_u32_be(buf + 8, P.y);
  return hash_to_scalar(buf, 12);
}

__host__ __device__ uint32_t H2(const uint8_t payload[ADSB_MESSAGE_BYTES], uint32_t id, uint32_t ts,
                                const EcPoint& RS, const EcPoint& P) {
  uint8_t buf[38];
  for (int i = 0; i < static_cast<int>(ADSB_MESSAGE_BYTES); i++) {
    buf[i] = payload[i];
  }
  write_u32_be(buf + 14, id);
  write_u32_be(buf + 18, ts);
  write_u32_be(buf + 22, RS.x);
  write_u32_be(buf + 26, RS.y);
  write_u32_be(buf + 30, P.x);
  write_u32_be(buf + 34, P.y);
  return hash_to_scalar(buf, 38);
}

__host__ __device__ bool verify_one(const ADSB_Packet& packet, const SignerPublicKey& pub,
                                    const EcPoint& Ppub1, const EcPoint& Ppub2, const EcPoint& P) {
  const uint32_t v = H1(pub.id, P);
  const uint32_t w = H2(packet.payload, pub.id, packet.timestamp, pub.RS, P);

  EcPoint t = point_mul(w, pub.QS);
  t = point_add(t, Ppub2);
  t = point_add(t, point_mul(v, Ppub1));

  const EcPoint lhs = point_mul(packet.alpha, t);
  return point_equal(lhs, pub.QS);
}

__global__ void verify_kernel(const ADSB_Packet* packets, const SignerPublicKey* pubkeys,
                              const EcPoint Ppub1, const EcPoint Ppub2, const EcPoint P,
                              int* results, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const ADSB_Packet pkt = packets[tid];
  const SignerPublicKey pub = pubkeys[pkt.signer_index];
  results[tid] = verify_one(pkt, pub, Ppub1, Ppub2, P) ? 1 : 0;
}

void verify_cpu(const ADSB_Packet* packets, const SignerPublicKey* pubkeys,
                const EcPoint& Ppub1, const EcPoint& Ppub2, const EcPoint& P,
                int* results, int n) {
  for (int i = 0; i < n; i++) {
    const SignerPublicKey& pub = pubkeys[packets[i].signer_index];
    results[i] = verify_one(packets[i], pub, Ppub1, Ppub2, P) ? 1 : 0;
  }
}

bool batch_verify_cpu(const ADSB_Packet* packets, const SignerPublicKey* pubkeys,
                      const EcPoint& Ppub1, const EcPoint& Ppub2, const EcPoint& P,
                      const std::vector<int>& indices) {
  EcPoint left_sum = point_infinity();
  EcPoint right_sum = point_infinity();

  for (int idx : indices) {
    const ADSB_Packet& pkt = packets[idx];
    const SignerPublicKey& pub = pubkeys[pkt.signer_index];
    const uint32_t v = H1(pub.id, P);
    const uint32_t w = H2(pkt.payload, pub.id, pkt.timestamp, pub.RS, P);

    EcPoint q_tilde = point_mul(w, pub.QS);
    q_tilde = point_add(q_tilde, Ppub2);
    q_tilde = point_add(q_tilde, point_mul(v, Ppub1));

    left_sum = point_add(left_sum, point_mul(pkt.alpha, q_tilde));
    right_sum = point_add(right_sum, pub.QS);
  }

  return point_equal(left_sum, right_sum);
}

void check_cuda(cudaError_t code, const char* expr) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error at " << expr << ": " << cudaGetErrorString(code) << '\n';
    std::exit(1);
  }
}

uint32_t next_nonzero_lcg(uint32_t& s) {
  s = 1664525u * s + 1013904223u;
  uint32_t out = (s % (SCALAR_N - 1u)) + 1u;
  return out;
}

#if HAS_BOOST_CPP_INT
cpp_int hex_to_cpp_int(const char* hex) {
  cpp_int out = 0;
  for (const char* p = hex; *p != '\0'; ++p) {
    const char c = *p;
    uint32_t v = 0;
    if (c >= '0' && c <= '9') v = static_cast<uint32_t>(c - '0');
    else if (c >= 'a' && c <= 'f') v = static_cast<uint32_t>(10 + c - 'a');
    else if (c >= 'A' && c <= 'F') v = static_cast<uint32_t>(10 + c - 'A');
    else continue;
    out = (out << 4) + v;
  }
  return out;
}

const cpp_int& real_p() {
  static const cpp_int p = hex_to_cpp_int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF");
  return p;
}

const cpp_int& real_n() {
  static const cpp_int n = hex_to_cpp_int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551");
  return n;
}

const cpp_int& real_a() {
  static const cpp_int a = hex_to_cpp_int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC");
  return a;
}

const cpp_int& real_b() {
  static const cpp_int b = hex_to_cpp_int("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B");
  return b;
}

RealEcPoint real_generator() {
  RealEcPoint g{};
  g.x = hex_to_cpp_int("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
  g.y = hex_to_cpp_int("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");
  g.inf = false;
  return g;
}

inline cpp_int mod_real_p(const cpp_int& x) {
  cpp_int r = x % real_p();
  if (r < 0) r += real_p();
  return r;
}

inline cpp_int mod_real_n(const cpp_int& x) {
  cpp_int r = x % real_n();
  if (r < 0) r += real_n();
  return r;
}

cpp_int inv_mod_big(cpp_int a, const cpp_int& mod) {
  a %= mod;
  if (a < 0) a += mod;
  cpp_int t = 0, new_t = 1;
  cpp_int r = mod, new_r = a;
  while (new_r != 0) {
    const cpp_int q = r / new_r;
    cpp_int tmp_t = t - q * new_t;
    t = new_t;
    new_t = tmp_t;
    cpp_int tmp_r = r - q * new_r;
    r = new_r;
    new_r = tmp_r;
  }
  if (r != 1) return 0;
  if (t < 0) t += mod;
  return t;
}

RealEcPoint real_inf() {
  RealEcPoint p{};
  p.x = 0;
  p.y = 0;
  p.inf = true;
  return p;
}

bool real_point_equal(const RealEcPoint& a, const RealEcPoint& b) {
  if (a.inf && b.inf) return true;
  if (a.inf != b.inf) return false;
  return a.x == b.x && a.y == b.y;
}

RealEcPoint real_point_double(const RealEcPoint& p) {
  if (p.inf) return real_inf();
  if (mod_real_p(p.y) == 0) return real_inf();
  const cpp_int num = mod_real_p(3 * p.x * p.x + real_a());
  const cpp_int den = mod_real_p(2 * p.y);
  const cpp_int inv_den = inv_mod_big(den, real_p());
  if (inv_den == 0) return real_inf();
  const cpp_int lambda = mod_real_p(num * inv_den);
  RealEcPoint r{};
  r.x = mod_real_p(lambda * lambda - p.x - p.x);
  r.y = mod_real_p(lambda * (p.x - r.x) - p.y);
  r.inf = false;
  return r;
}

RealEcPoint real_point_add(const RealEcPoint& p, const RealEcPoint& q) {
  if (p.inf) return q;
  if (q.inf) return p;
  if (p.x == q.x) {
    if (mod_real_p(p.y + q.y) == 0) return real_inf();
    return real_point_double(p);
  }
  const cpp_int num = mod_real_p(q.y - p.y);
  const cpp_int den = mod_real_p(q.x - p.x);
  const cpp_int inv_den = inv_mod_big(den, real_p());
  if (inv_den == 0) return real_inf();
  const cpp_int lambda = mod_real_p(num * inv_den);
  RealEcPoint r{};
  r.x = mod_real_p(lambda * lambda - p.x - q.x);
  r.y = mod_real_p(lambda * (p.x - r.x) - p.y);
  r.inf = false;
  return r;
}

RealEcPoint real_point_mul(cpp_int k, RealEcPoint base) {
  k = mod_real_n(k);
  RealEcPoint r = real_inf();
  while (k > 0) {
    if ((k & 1) != 0) r = real_point_add(r, base);
    base = real_point_double(base);
    k >>= 1;
  }
  return r;
}

uint32_t low32(const cpp_int& x) {
  cpp_int m = x & cpp_int(0xffffffffu);
  return m.convert_to<uint32_t>();
}

cpp_int digest_to_scalar_mod_n(const uint32_t digest[8]) {
  cpp_int x = 0;
  for (int i = 0; i < 8; i++) {
    x <<= 32;
    x += digest[i];
  }
  x = mod_real_n(x);
  if (x == 0) x = 1;
  return x;
}

cpp_int H1_real(uint32_t id, const RealEcPoint& P) {
  uint8_t buf[12];
  write_u32_be(buf + 0, id);
  write_u32_be(buf + 4, low32(P.x));
  write_u32_be(buf + 8, low32(P.y));
  uint32_t digest[8];
  sha256_single_block(buf, 12, digest);
  return digest_to_scalar_mod_n(digest);
}

cpp_int H2_real(const uint8_t payload[ADSB_MESSAGE_BYTES], uint32_t id, uint32_t ts,
                const RealEcPoint& RS, const RealEcPoint& P) {
  uint8_t buf[38];
  for (int i = 0; i < static_cast<int>(ADSB_MESSAGE_BYTES); i++) {
    buf[i] = payload[i];
  }
  write_u32_be(buf + 14, id);
  write_u32_be(buf + 18, ts);
  write_u32_be(buf + 22, low32(RS.x));
  write_u32_be(buf + 26, low32(RS.y));
  write_u32_be(buf + 30, low32(P.x));
  write_u32_be(buf + 34, low32(P.y));
  uint32_t digest[8];
  sha256_single_block(buf, 38, digest);
  return digest_to_scalar_mod_n(digest);
}

bool verify_one_real_cpu(const RealADSBPacket& packet, const RealSignerPublicKey& pub,
                         const RealEcPoint& Ppub1, const RealEcPoint& Ppub2, const RealEcPoint& P) {
  const cpp_int v = H1_real(pub.id, P);
  const cpp_int w = H2_real(packet.payload, pub.id, packet.timestamp, pub.RS, P);
  RealEcPoint t = real_point_mul(w, pub.QS);
  t = real_point_add(t, Ppub2);
  t = real_point_add(t, real_point_mul(v, Ppub1));
  const RealEcPoint lhs = real_point_mul(packet.alpha, t);
  return real_point_equal(lhs, pub.QS);
}

void run_real_ecc_cpu_reference_profile() {
  constexpr int N = 4096;
  constexpr int SIGNER_COUNT = 32;

  const RealEcPoint P = real_generator();
  const cpp_int master_a = mod_real_n(cpp_int(0x10293847u));
  const cpp_int master_b = mod_real_n(cpp_int(0x55667788u));
  const RealEcPoint Ppub1 = real_point_mul(master_a, P);
  const RealEcPoint Ppub2 = real_point_mul(master_b, P);

  std::vector<RealSignerPrivateState> signer_priv(SIGNER_COUNT);
  std::vector<RealSignerPublicKey> signer_pub(SIGNER_COUNT);
  std::vector<RealADSBPacket> packets(N);
  std::vector<int> verify_results(N, 0);

  uint32_t seed = 0x12345678u;
  for (int i = 0; i < SIGNER_COUNT; i++) {
    RealSignerPrivateState st{};
    st.id = 0xFACE0000u + static_cast<uint32_t>(i);
    const cpp_int u = next_nonzero_lcg(seed);
    st.RS = real_point_mul(u, P);

    const cpp_int v = H1_real(st.id, P);
    const cpp_int inv_u = inv_mod_big(u, real_n());
    st.k = mod_real_n((master_b + master_a * v) * inv_u);

    const RealEcPoint lhs = real_point_mul(st.k, st.RS);
    const RealEcPoint rhs = real_point_add(real_point_mul(v, Ppub1), Ppub2);
    if (!real_point_equal(lhs, rhs)) {
      std::cout << "Real ECC setup failed at signer " << i << '\n';
      return;
    }

    st.x = next_nonzero_lcg(seed);
    st.QS = real_point_mul(st.x, st.RS);
    signer_priv[i] = st;

    RealSignerPublicKey pk{};
    pk.id = st.id;
    pk.RS = st.RS;
    pk.QS = st.QS;
    signer_pub[i] = pk;
  }

  for (int i = 0; i < N; i++) {
    RealADSBPacket pkt{};
    pkt.signer_index = static_cast<uint32_t>(i % SIGNER_COUNT);
    pkt.timestamp = 1800000000u + static_cast<uint32_t>(i);
    for (int j = 0; j < static_cast<int>(ADSB_MESSAGE_BYTES); j++) {
      pkt.payload[j] = static_cast<uint8_t>((i * 13 + j * 7) & 0xff);
    }

    const RealSignerPrivateState& st = signer_priv[pkt.signer_index];
    cpp_int w = H2_real(pkt.payload, st.id, pkt.timestamp, st.RS, P);
    cpp_int denom = mod_real_n(w * st.x + st.k);
    while (denom == 0) {
      pkt.timestamp += 1u;
      w = H2_real(pkt.payload, st.id, pkt.timestamp, st.RS, P);
      denom = mod_real_n(w * st.x + st.k);
    }
    pkt.alpha = mod_real_n(st.x * inv_mod_big(denom, real_n()));
    if ((i % 10) == 0) pkt.alpha ^= 0x1u;
    packets[i] = pkt;
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    const auto& pub = signer_pub[packets[i].signer_index];
    verify_results[i] = verify_one_real_cpu(packets[i], pub, Ppub1, Ppub2, P) ? 1 : 0;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  int valid = 0;
  for (int x : verify_results) valid += x;
  std::cout << "Real ECC CPU (secp256r1) verify profile\n";
  std::cout << "Packets: " << N << ", Signers: " << SIGNER_COUNT << '\n';
  std::cout << "Verify time (ms): " << ms << '\n';
  std::cout << "Throughput (pkt/s): " << ((static_cast<double>(N) * 1000.0) / ms) << '\n';
  std::cout << "Valid packets: " << valid << ", Invalid packets: " << (N - valid) << '\n';
}
#endif

}  // namespace

int main() {
  constexpr int N = 1 << 16;
  constexpr int SIGNER_COUNT = 64;

  const size_t packet_bytes = sizeof(ADSB_Packet) * N;
  const size_t result_bytes = sizeof(int) * N;
  const size_t pubkey_bytes = sizeof(SignerPublicKey) * SIGNER_COUNT;

  const EcPoint P = generator_point();
  const uint32_t master_a = 0x10293847u % SCALAR_N;
  const uint32_t master_b = 0x55667788u % SCALAR_N;
  const EcPoint Ppub1 = point_mul(master_a, P);
  const EcPoint Ppub2 = point_mul(master_b, P);

  std::vector<SignerPrivateState> signer_priv(SIGNER_COUNT);
  std::vector<SignerPublicKey> signer_pub(SIGNER_COUNT);
  std::vector<ADSB_Packet> h_packets(N);
  std::vector<int> h_cpu_results(N, 0);
  std::vector<int> h_gpu_results(N, 0);

  uint32_t seed = 0xC0FFEE11u;

  // Setup + Set Partial Private Key + Set Public Key from Algorithm 1.
  const auto keygen_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < SIGNER_COUNT; i++) {
    SignerPrivateState st{};
    st.id = 0xABCD0000u + static_cast<uint32_t>(i);

    const uint32_t u = next_nonzero_lcg(seed);
    st.RS = point_mul(u, P);

    const uint32_t v = H1(st.id, P);
    const uint32_t inv_u = inv_mod_n(u);
    const uint32_t term = add_mod_n(master_b, mul_mod_n(master_a, v));
    st.k = mul_mod_n(term, inv_u);

    const EcPoint lhs = point_mul(st.k, st.RS);
    const EcPoint rhs = point_add(point_mul(v, Ppub1), Ppub2);
    if (!point_equal(lhs, rhs)) {
      std::cerr << "Partial private key verification failed for signer " << i << '\n';
      return 1;
    }

    st.x = next_nonzero_lcg(seed);
    st.QS = point_mul(st.x, st.RS);

    signer_priv[i] = st;

    SignerPublicKey pk{};
    pk.id = st.id;
    pk.RS = st.RS;
    pk.QS = st.QS;
    signer_pub[i] = pk;
  }
  const auto keygen_end = std::chrono::high_resolution_clock::now();
  const auto keygen_ms = std::chrono::duration<double, std::milli>(keygen_end - keygen_start).count();

  std::vector<int> valid_indices;
  valid_indices.reserve(N);
  std::vector<int> all_indices;
  all_indices.reserve(N);

  // ECB-CLS Sign from Algorithm 1.
  const auto sign_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    ADSB_Packet pkt{};
    pkt.signer_index = static_cast<uint32_t>(i % SIGNER_COUNT);
    pkt.timestamp = 1700000000u + static_cast<uint32_t>(i);

    for (int j = 0; j < static_cast<int>(ADSB_MESSAGE_BYTES); j++) {
      pkt.payload[j] = static_cast<uint8_t>((i * 17 + j * 29) & 0xff);
    }

    const SignerPrivateState& st = signer_priv[pkt.signer_index];
    uint32_t w = H2(pkt.payload, st.id, pkt.timestamp, st.RS, P);
    uint32_t denom = add_mod_n(mul_mod_n(w, st.x), st.k);
    while (denom == 0) {
      pkt.timestamp += 1u;
      w = H2(pkt.payload, st.id, pkt.timestamp, st.RS, P);
      denom = add_mod_n(mul_mod_n(w, st.x), st.k);
    }
    pkt.alpha = mul_mod_n(st.x, inv_mod_n(denom));

    if ((i % 10) == 0) {
      pkt.alpha ^= 0x1u;
    } else {
      valid_indices.push_back(i);
    }

    all_indices.push_back(i);
    h_packets[i] = pkt;
  }
  const auto sign_end = std::chrono::high_resolution_clock::now();
  const auto sign_ms = std::chrono::duration<double, std::milli>(sign_end - sign_start).count();

  ADSB_Packet* d_packets = nullptr;
  SignerPublicKey* d_pubkeys = nullptr;
  int* d_results = nullptr;

  const auto cpu_start = std::chrono::high_resolution_clock::now();
  verify_cpu(h_packets.data(), signer_pub.data(), Ppub1, Ppub2, P, h_cpu_results.data(), N);
  const auto cpu_end = std::chrono::high_resolution_clock::now();
  const auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  const bool batch_valid_ok = batch_verify_cpu(h_packets.data(), signer_pub.data(), Ppub1, Ppub2, P, valid_indices);
  const bool batch_all_ok = batch_verify_cpu(h_packets.data(), signer_pub.data(), Ppub1, Ppub2, P, all_indices);

  check_cuda(cudaMalloc(&d_packets, packet_bytes), "cudaMalloc(d_packets)");
  check_cuda(cudaMalloc(&d_pubkeys, pubkey_bytes), "cudaMalloc(d_pubkeys)");
  check_cuda(cudaMalloc(&d_results, result_bytes), "cudaMalloc(d_results)");

  cudaEvent_t kernel_start, kernel_end, h2d_start, h2d_end, d2h_start, d2h_end;
  check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
  check_cuda(cudaEventCreate(&kernel_end), "cudaEventCreate(kernel_end)");
  check_cuda(cudaEventCreate(&h2d_start), "cudaEventCreate(h2d_start)");
  check_cuda(cudaEventCreate(&h2d_end), "cudaEventCreate(h2d_end)");
  check_cuda(cudaEventCreate(&d2h_start), "cudaEventCreate(d2h_start)");
  check_cuda(cudaEventCreate(&d2h_end), "cudaEventCreate(d2h_end)");

  const auto gpu_total_start = std::chrono::high_resolution_clock::now();
  check_cuda(cudaEventRecord(h2d_start), "cudaEventRecord(h2d_start)");
  check_cuda(cudaMemcpy(d_packets, h_packets.data(), packet_bytes, cudaMemcpyHostToDevice), "cudaMemcpy packets H2D");
  check_cuda(cudaMemcpy(d_pubkeys, signer_pub.data(), pubkey_bytes, cudaMemcpyHostToDevice), "cudaMemcpy pubkeys H2D");
  check_cuda(cudaEventRecord(h2d_end), "cudaEventRecord(h2d_end)");

  const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  verify_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_packets, d_pubkeys, Ppub1, Ppub2, P, d_results, N);
  check_cuda(cudaGetLastError(), "verify_kernel warmup launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

  check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord(kernel_start)");
  for (int run = 0; run < GPU_PROFILE_RUNS; run++) {
    verify_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_packets, d_pubkeys, Ppub1, Ppub2, P, d_results, N);
  }
  verify_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_packets, d_pubkeys, Ppub1, Ppub2, P, d_results, N);
  check_cuda(cudaEventRecord(kernel_end), "cudaEventRecord(kernel_end)");
  check_cuda(cudaGetLastError(), "verify_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check_cuda(cudaEventRecord(d2h_start), "cudaEventRecord(d2h_start)");
  check_cuda(cudaMemcpy(h_gpu_results.data(), d_results, result_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy results D2H");
  check_cuda(cudaEventRecord(d2h_end), "cudaEventRecord(d2h_end)");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize post D2H");
  const auto gpu_total_end = std::chrono::high_resolution_clock::now();

  float gpu_kernel_ms = 0.0f, gpu_h2d_ms = 0.0f, gpu_d2h_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start, kernel_end), "cudaEventElapsedTime");
  check_cuda(cudaEventElapsedTime(&gpu_h2d_ms, h2d_start, h2d_end), "cudaEventElapsedTime h2d");
  check_cuda(cudaEventElapsedTime(&gpu_d2h_ms, d2h_start, d2h_end), "cudaEventElapsedTime d2h");
  const auto gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_total_end - gpu_total_start).count();
  const double gpu_kernel_avg_ms = static_cast<double>(gpu_kernel_ms) / static_cast<double>(GPU_PROFILE_RUNS + 1);

  int valid_count = 0;
  for (int i = 0; i < N; i++) valid_count += h_gpu_results[i];
  const int invalid_count = N - valid_count;

  int mismatches = 0;
  for (int i = 0; i < N; i++) {
    if (h_cpu_results[i] != h_gpu_results[i]) mismatches++;
  }

  const double gpu_kernel_speedup = cpu_ms / gpu_kernel_avg_ms;
  const double gpu_total_speedup = cpu_ms / gpu_total_ms;
  const double cpu_pps = (static_cast<double>(N) * 1000.0) / cpu_ms;
  const double gpu_pps = (static_cast<double>(N) * 1000.0) / gpu_kernel_avg_ms;

  std::cout << "ECB-CLS signature model (Algorithm 1 style)\n";
  std::cout << "Curve: y^2 = x^3 + " << CURVE_A << "x + " << CURVE_B << " over F_p, p=" << FIELD_P
            << ", scalar n=" << SCALAR_N << '\n';
  std::cout << "Packets: " << N << ", Signers: " << SIGNER_COUNT << '\n';
  std::cout << "Keygen time (ms): " << keygen_ms << '\n';
  std::cout << "Sign time (ms): " << sign_ms << '\n';
  std::cout << "CPU verify time (ms): " << cpu_ms << '\n';
  std::cout << "GPU H2D time (ms): " << gpu_h2d_ms << '\n';
  std::cout << "GPU kernel total time over " << (GPU_PROFILE_RUNS + 1) << " runs (ms): " << gpu_kernel_ms << '\n';
  std::cout << "GPU kernel avg time (ms): " << gpu_kernel_avg_ms << '\n';
  std::cout << "GPU D2H time (ms): " << gpu_d2h_ms << '\n';
  std::cout << "GPU end-to-end time (ms): " << gpu_total_ms << '\n';
  std::cout << "CPU throughput (pkt/s): " << cpu_pps << '\n';
  std::cout << "GPU kernel throughput (pkt/s): " << gpu_pps << '\n';
  std::cout << "Speedup CPU vs GPU kernel(avg): " << gpu_kernel_speedup << "x\n";
  std::cout << "Speedup CPU vs GPU end-to-end: " << gpu_total_speedup << "x\n";
  std::cout << "CPU/GPU mismatches: " << mismatches << '\n';
  std::cout << "Valid packets: " << valid_count << '\n';
  std::cout << "Invalid packets: " << invalid_count << '\n';
  std::cout << "Batch verify (valid-only set): " << (batch_valid_ok ? "pass" : "fail") << '\n';
  std::cout << "Batch verify (all packets incl. tampered): " << (batch_all_ok ? "pass" : "fail") << '\n';
  std::cout << "First 8 GPU results: ";
  for (int i = 0; i < 8; i++) std::cout << h_gpu_results[i] << ' ';
  std::cout << '\n';

  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_end);
  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_end);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_end);
  cudaFree(d_packets);
  cudaFree(d_pubkeys);
  cudaFree(d_results);

  std::cout << '\n';
#if HAS_BOOST_CPP_INT
  run_real_ecc_cpu_reference_profile();
#else
  std::cout << "Real ECC CPU (secp256r1) profile skipped: boost::multiprecision not available in this toolchain.\n";
#endif
  return 0;
}
