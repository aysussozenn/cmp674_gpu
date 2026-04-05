#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <intrin.h>

#include <cuda.h>

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
constexpr int INSTANCES = 1 << 15;

struct Instance {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> p;
  cgbn_mem_t<BITS> add_mod;
  cgbn_mem_t<BITS> mul_mod;
};

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__device__ __forceinline__ void load_prime(env_t bn_env, env_t::cgbn_t& p) {
  bn_env.set_ui32(p, 0);
  bn_env.insert_bits_ui32(p, p, 0, 32, 0xffffffffu);
  bn_env.insert_bits_ui32(p, p, 32, 32, 0xffffffffu);
  bn_env.insert_bits_ui32(p, p, 64, 32, 0xffffffffu);
  bn_env.insert_bits_ui32(p, p, 96, 32, 0x00000000u);
  bn_env.insert_bits_ui32(p, p, 128, 32, 0x00000000u);
  bn_env.insert_bits_ui32(p, p, 160, 32, 0x00000000u);
  bn_env.insert_bits_ui32(p, p, 192, 32, 0x00000001u);
  bn_env.insert_bits_ui32(p, p, 224, 32, 0xffffffffu);
}

__global__ void kernel_mod_ops(cgbn_error_report_t* report, Instance* instances, int count) {
  int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t bn_env(bn_context.env<env_t>());

  env_t::cgbn_t a, b, p, add_res, mul_res;
  env_t::cgbn_wide_t wide;

  bn_env.load(a, &instances[instance].a);
  bn_env.load(b, &instances[instance].b);
  load_prime(bn_env, p);

  bn_env.add(add_res, a, b);
  if (bn_env.compare(add_res, p) >= 0) {
    bn_env.sub(add_res, add_res, p);
  }

  bn_env.mul_wide(wide, a, b);
  bn_env.rem_wide(mul_res, wide, p);

  bn_env.store(&instances[instance].add_mod, add_res);
  bn_env.store(&instances[instance].mul_mod, mul_res);

  bn_env.store(&instances[instance].p, p);
}

void check_cuda(cudaError_t code, const char* expr) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error at " << expr << ": " << cudaGetErrorString(code) << '\n';
    std::exit(1);
  }
}

void fill_prime(cgbn_mem_t<BITS>& p) {
  p._limbs[0] = 0xffffffffu;
  p._limbs[1] = 0xffffffffu;
  p._limbs[2] = 0xffffffffu;
  p._limbs[3] = 0x00000000u;
  p._limbs[4] = 0x00000000u;
  p._limbs[5] = 0x00000000u;
  p._limbs[6] = 0x00000001u;
  p._limbs[7] = 0xffffffffu;
}

uint64_t lcg64(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1ULL;
  return s;
}

void add_mod_p_cpu(const uint32_t* a, const uint32_t* b, const uint32_t* p, uint32_t* out) {
  uint64_t carry = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t t = static_cast<uint64_t>(a[i]) + static_cast<uint64_t>(b[i]) + carry;
    out[i] = static_cast<uint32_t>(t & 0xffffffffu);
    carry = t >> 32;
  }

  bool ge = true;
  for (int i = 7; i >= 0; i--) {
    if (out[i] > p[i]) {
      ge = true;
      break;
    }
    if (out[i] < p[i]) {
      ge = false;
      break;
    }
  }

  if (carry != 0 || ge) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
      uint64_t ai = out[i];
      uint64_t bi = static_cast<uint64_t>(p[i]) + borrow;
      if (ai >= bi) {
        out[i] = static_cast<uint32_t>(ai - bi);
        borrow = 0;
      } else {
        out[i] = static_cast<uint32_t>((ai + (1ULL << 32)) - bi);
        borrow = 1;
      }
    }
  }
}

void mul_small_cpu(const uint32_t* a, const uint32_t* b, uint32_t* out) {
  // a,b are generated as <=63-bit values so product fits in 126 bits.
  const uint64_t a64 = (static_cast<uint64_t>(a[1]) << 32) | a[0];
  const uint64_t b64 = (static_cast<uint64_t>(b[1]) << 32) | b[0];
  uint64_t high = 0;
  const uint64_t low = _umul128(a64, b64, &high);
  out[0] = static_cast<uint32_t>(low & 0xffffffffu);
  out[1] = static_cast<uint32_t>((low >> 32) & 0xffffffffu);
  out[2] = static_cast<uint32_t>(high & 0xffffffffu);
  out[3] = static_cast<uint32_t>((high >> 32) & 0xffffffffu);
  out[4] = out[5] = out[6] = out[7] = 0;
}

bool equal8(const uint32_t* a, const uint32_t* b) {
  for (int i = 0; i < 8; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

}  // namespace

int main() {
  std::vector<Instance> h(INSTANCES);

  uint64_t seed = 0xC0FFEE1234567890ULL;
  for (int i = 0; i < INSTANCES; i++) {
    std::memset(&h[i], 0, sizeof(Instance));
    fill_prime(h[i].p);

    uint64_t ra = lcg64(seed) & 0x7fffffffffffffffULL;
    uint64_t rb = lcg64(seed) & 0x7fffffffffffffffULL;

    h[i].a._limbs[0] = static_cast<uint32_t>(ra & 0xffffffffu);
    h[i].a._limbs[1] = static_cast<uint32_t>((ra >> 32) & 0xffffffffu);

    h[i].b._limbs[0] = static_cast<uint32_t>(rb & 0xffffffffu);
    h[i].b._limbs[1] = static_cast<uint32_t>((rb >> 32) & 0xffffffffu);
  }

  Instance* d = nullptr;
  cgbn_error_report_t* report = nullptr;

  check_cuda(cudaMalloc(&d, sizeof(Instance) * INSTANCES), "cudaMalloc(instances)");
  check_cuda(cudaMemcpy(d, h.data(), sizeof(Instance) * INSTANCES, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  check_cuda(cgbn_error_report_alloc(&report), "cgbn_error_report_alloc");

  cudaEvent_t s, e;
  check_cuda(cudaEventCreate(&s), "cudaEventCreate(s)");
  check_cuda(cudaEventCreate(&e), "cudaEventCreate(e)");

  const int blocks = (INSTANCES * TPI + TPB - 1) / TPB;

  check_cuda(cudaEventRecord(s), "cudaEventRecord(start)");
  kernel_mod_ops<<<blocks, TPB>>>(report, d, INSTANCES);
  check_cuda(cudaEventRecord(e), "cudaEventRecord(end)");
  check_cuda(cudaGetLastError(), "kernel_mod_ops launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  if (cgbn_error_report_check(report)) {
    std::cerr << "CGBN error: " << cgbn_error_string(report) << "\\n";
    return 1;
  }

  float kernel_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&kernel_ms, s, e), "cudaEventElapsedTime");

  check_cuda(cudaMemcpy(h.data(), d, sizeof(Instance) * INSTANCES, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

  int add_mismatch = 0;
  int mul_mismatch = 0;
  for (int i = 0; i < INSTANCES; i++) {
    uint32_t exp_add[8];
    uint32_t exp_mul[8];
    add_mod_p_cpu(h[i].a._limbs, h[i].b._limbs, h[i].p._limbs, exp_add);
    mul_small_cpu(h[i].a._limbs, h[i].b._limbs, exp_mul);

    if (!equal8(exp_add, h[i].add_mod._limbs)) add_mismatch++;
    if (!equal8(exp_mul, h[i].mul_mod._limbs)) mul_mismatch++;
  }

  const double ops = static_cast<double>(INSTANCES);
  const double modops_per_sec = (ops * 1000.0) / static_cast<double>(kernel_ms);

  std::cout << "CGBN secp256r1 field mod-op benchmark (GPU groundwork)\n";
  std::cout << "Instances: " << INSTANCES << " (256-bit)\n";
  std::cout << "Kernel time (ms): " << kernel_ms << '\n';
  std::cout << "Instance throughput (ops/s): " << modops_per_sec << '\n';
  std::cout << "Add mismatches: " << add_mismatch << '\n';
  std::cout << "Mul mismatches (small-input check): " << mul_mismatch << '\n';

  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cgbn_error_report_free(report);
  cudaFree(d);
  return 0;
}
