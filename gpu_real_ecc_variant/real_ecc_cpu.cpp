#define BOOST_NO_EXCEPTIONS
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <exception>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/assert/source_location.hpp>

namespace boost {
void throw_exception(std::exception const&) { std::terminate(); }
void throw_exception(std::exception const&, const boost::source_location&) { std::terminate(); }
}  // namespace boost

using boost::multiprecision::cpp_int;

namespace {

constexpr uint32_t ADSB_MESSAGE_BYTES = 14;

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

inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t big_sigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
inline uint32_t big_sigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
inline uint32_t small_sigma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
inline uint32_t small_sigma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

void sha256_single_block(const uint8_t* message, int len, uint32_t digest[8]) {
  static constexpr uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint8_t block[64]{};
  uint32_t w[64]{};
  for (int i = 0; i < len; i++) block[i] = message[i];
  block[len] = 0x80;
  const uint64_t bit_len = static_cast<uint64_t>(len) * 8ull;
  for (int i = 0; i < 8; i++) block[63 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xffu);
  for (int t = 0; t < 16; t++) {
    w[t] = (static_cast<uint32_t>(block[t * 4]) << 24) |
           (static_cast<uint32_t>(block[t * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[t * 4 + 2]) << 8) |
           (static_cast<uint32_t>(block[t * 4 + 3]));
  }
  for (int t = 16; t < 64; t++) w[t] = small_sigma1(w[t - 2]) + w[t - 7] + small_sigma0(w[t - 15]) + w[t - 16];
  uint32_t a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
  uint32_t e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;
  for (int t = 0; t < 64; t++) {
    const uint32_t t1 = h + big_sigma1(e) + ch(e, f, g) + k[t] + w[t];
    const uint32_t t2 = big_sigma0(a) + maj(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  digest[0] = 0x6a09e667u + a; digest[1] = 0xbb67ae85u + b; digest[2] = 0x3c6ef372u + c; digest[3] = 0xa54ff53au + d;
  digest[4] = 0x510e527fu + e; digest[5] = 0x9b05688cu + f; digest[6] = 0x1f83d9abu + g; digest[7] = 0x5be0cd19u + h;
}

void write_u32_be(uint8_t* out, uint32_t v) {
  out[0] = static_cast<uint8_t>((v >> 24) & 0xffu);
  out[1] = static_cast<uint8_t>((v >> 16) & 0xffu);
  out[2] = static_cast<uint8_t>((v >> 8) & 0xffu);
  out[3] = static_cast<uint8_t>(v & 0xffu);
}

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

const cpp_int& p() { static const cpp_int v = hex_to_cpp_int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF"); return v; }
const cpp_int& n() { static const cpp_int v = hex_to_cpp_int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551"); return v; }
const cpp_int& a() { static const cpp_int v = hex_to_cpp_int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC"); return v; }

RealEcPoint G() {
  RealEcPoint g{};
  g.x = hex_to_cpp_int("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
  g.y = hex_to_cpp_int("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");
  g.inf = false;
  return g;
}

inline cpp_int modp(cpp_int x){ x%=p(); if(x<0)x+=p(); return x; }
inline cpp_int modn(cpp_int x){ x%=n(); if(x<0)x+=n(); return x; }

cpp_int inv(cpp_int a, const cpp_int& m){
  a%=m; if(a<0)a+=m;
  cpp_int t=0, nt=1, r=m, nr=a;
  while(nr!=0){ cpp_int q=r/nr; cpp_int tt=t-q*nt; t=nt; nt=tt; cpp_int rr=r-q*nr; r=nr; nr=rr; }
  if(r!=1) return 0; if(t<0) t+=m; return t;
}

RealEcPoint INF(){ return {0,0,true}; }

bool eq(const RealEcPoint& x,const RealEcPoint& y){ if(x.inf&&y.inf) return true; if(x.inf!=y.inf) return false; return x.x==y.x && x.y==y.y; }

RealEcPoint dbl(const RealEcPoint& q){
  if(q.inf || modp(q.y)==0) return INF();
  cpp_int lam = modp((3*q.x*q.x + a()) * inv(modp(2*q.y), p()));
  RealEcPoint r{};
  r.x = modp(lam*lam - q.x - q.x);
  r.y = modp(lam*(q.x-r.x) - q.y);
  r.inf = false;
  return r;
}

RealEcPoint add(const RealEcPoint& q,const RealEcPoint& w){
  if(q.inf) return w; if(w.inf) return q;
  if(q.x==w.x){ if(modp(q.y+w.y)==0) return INF(); return dbl(q); }
  cpp_int lam = modp(modp(w.y-q.y) * inv(modp(w.x-q.x), p()));
  RealEcPoint r{};
  r.x = modp(lam*lam - q.x - w.x);
  r.y = modp(lam*(q.x-r.x) - q.y);
  r.inf = false;
  return r;
}

RealEcPoint mul(cpp_int k, RealEcPoint base){
  k=modn(k); RealEcPoint r=INF();
  while(k>0){ if((k&1)!=0) r=add(r,base); base=dbl(base); k>>=1; }
  return r;
}

uint32_t low32(const cpp_int& x){ return (x & cpp_int(0xffffffffu)).convert_to<uint32_t>(); }

cpp_int digest_scalar(const uint32_t d[8]){ cpp_int x=0; for(int i=0;i<8;i++){ x<<=32; x+=d[i]; } x=modn(x); if(x==0) x=1; return x; }

cpp_int H1(uint32_t id,const RealEcPoint& P){ uint8_t b[12]; write_u32_be(b,id); write_u32_be(b+4,low32(P.x)); write_u32_be(b+8,low32(P.y)); uint32_t d[8]; sha256_single_block(b,12,d); return digest_scalar(d); }
cpp_int H2(const uint8_t payload[ADSB_MESSAGE_BYTES], uint32_t id,uint32_t ts,const RealEcPoint& RS,const RealEcPoint& P){
  uint8_t b[38]; for(int i=0;i<14;i++) b[i]=payload[i]; write_u32_be(b+14,id); write_u32_be(b+18,ts); write_u32_be(b+22,low32(RS.x)); write_u32_be(b+26,low32(RS.y)); write_u32_be(b+30,low32(P.x)); write_u32_be(b+34,low32(P.y)); uint32_t d[8]; sha256_single_block(b,38,d); return digest_scalar(d);
}

bool verify_one(const RealADSBPacket& pkt, const RealSignerPublicKey& pub, const RealEcPoint& Ppub1, const RealEcPoint& Ppub2, const RealEcPoint& P){
  cpp_int v=H1(pub.id,P); cpp_int w=H2(pkt.payload,pub.id,pkt.timestamp,pub.RS,P);
  RealEcPoint qt = add(add(mul(w,pub.QS),Ppub2),mul(v,Ppub1));
  return eq(mul(pkt.alpha,qt), pub.QS);
}

uint32_t lcg(uint32_t& s){ s = 1664525u*s + 1013904223u; return (s % 0xffffffffu) + 1u; }

} // namespace

int main(int argc, char** argv){
  int N = 512;
  if(argc >= 2) {
    N = std::atoi(argv[1]);
    if(N <= 0) N = 512;
  }
  constexpr int SIGNERS=32;
  RealEcPoint P=G();
  cpp_int ma=modn(0x10293847u), mb=modn(0x55667788u);
  RealEcPoint Ppub1=mul(ma,P), Ppub2=mul(mb,P);

  std::vector<RealSignerPrivateState> priv(SIGNERS);
  std::vector<RealSignerPublicKey> pub(SIGNERS);
  std::vector<RealADSBPacket> packets(N);
  std::vector<int> out(N,0);

  uint32_t seed=0x12345678u;
  for(int i=0;i<SIGNERS;i++){
    RealSignerPrivateState st{}; st.id=0xFACE0000u + static_cast<uint32_t>(i);
    cpp_int u = modn(lcg(seed)); if(u==0) u=1;
    st.RS = mul(u,P);
    cpp_int v = H1(st.id,P);
    st.k = modn((mb + ma*v) * inv(u,n()));
    if(!eq(mul(st.k,st.RS), add(mul(v,Ppub1),Ppub2))){ std::cout<<"real setup failed\n"; return 1; }
    st.x = modn(lcg(seed)); if(st.x==0) st.x=1;
    st.QS = mul(st.x, st.RS);
    priv[i]=st;
    pub[i] = {st.id, st.RS, st.QS};
  }

  for(int i=0;i<N;i++){
    RealADSBPacket pkt{}; pkt.signer_index = static_cast<uint32_t>(i % SIGNERS); pkt.timestamp = 1800000000u + static_cast<uint32_t>(i);
    for(int j=0;j<14;j++) pkt.payload[j] = static_cast<uint8_t>((i*13 + j*7) & 0xff);
    const auto& st = priv[pkt.signer_index];
    cpp_int w = H2(pkt.payload, st.id, pkt.timestamp, st.RS, P);
    cpp_int den = modn(w*st.x + st.k);
    while(den==0){ pkt.timestamp++; w = H2(pkt.payload, st.id, pkt.timestamp, st.RS, P); den = modn(w*st.x + st.k); }
    pkt.alpha = modn(st.x * inv(den, n()));
    if((i%10)==0) pkt.alpha ^= 1;
    packets[i]=pkt;
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<N;i++) out[i] = verify_one(packets[i], pub[packets[i].signer_index], Ppub1, Ppub2, P) ? 1 : 0;
  const auto t1 = std::chrono::high_resolution_clock::now();

  int valid=0; for(int x:out) valid+=x;
  const double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
  std::cout << "Real ECC CPU (secp256r1-like arithmetic)\n";
  std::cout << "Packets: " << N << ", Signers: " << SIGNERS << '\n';
  std::cout << "Verify time (ms): " << ms << '\n';
  std::cout << "Throughput (verifies/s): " << ((static_cast<double>(N)*1000.0)/ms) << '\n';
  std::cout << "Valid packets: " << valid << "\n";
  std::cout << "Invalid packets: " << (N-valid) << "\n";
  return 0;
}
