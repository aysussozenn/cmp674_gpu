# GPU Real ECC Variant

This folder keeps the newer ECC/GPU benchmarking work in isolation, without changing the original root flow.

Included files:
- `main.cu`: current integrated CUDA experiment code
- `real_ecc_cpu.cpp`: CPU-side real-ECC reference benchmark
- `real_ecc_gpu_cgbn.cu`: CGBN arithmetic kernel benchmark
- `real_ecc_gpu_verify_kernel.cu`: CGBN-based ECC verify kernel
- `fair_benchmark_real_ecc.ps1`: fair same-`N` CPU vs GPU benchmark script
- `compare_real_ecc.ps1`: quick throughput/speedup comparison script

Run (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File .\gpu_real_ecc_variant\fair_benchmark_real_ecc.ps1
```
