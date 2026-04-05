#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

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
  return 0;
}
