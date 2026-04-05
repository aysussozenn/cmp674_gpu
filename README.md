# CMP674 - ECB-CLS CUDA Demo

This repository contains a CUDA-based demonstration of an ECB-CLS-style ADS-B signature verification pipeline.

## What is included

- `main.cu`: End-to-end demo implementation
  - Toy elliptic-curve arithmetic (for demonstration only)
  - ECB-CLS-like key generation, signing, and verification flow
  - CPU single verification
  - GPU parallel verification kernel (`one thread per packet`)
  - CPU batch-verification equation check
  - Profiling outputs (keygen, sign, H2D, kernel avg, D2H, end-to-end)

## Build (Windows + NVCC)

```powershell
nvcc -O2 -std=c++17 main.cu -o adsb_verify_ecbcls.exe
```

If `nvcc` cannot find `cl.exe`, prepend your MSVC compiler path in the current terminal:

```powershell
$env:Path="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64;$env:Path"
```

## Run

```powershell
.\adsb_verify_ecbcls.exe
```

## Important note

This is a **research/demo prototype**. It uses toy ECC parameters and is **not** production-grade cryptography.

