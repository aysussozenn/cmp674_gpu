param(
  [int[]]$Sizes = @(256, 512, 1024),
  [int]$GpuRepeats = 10,
  [string]$CpuExe = "c:\cudaimp\real_ecc_cpu.exe",
  [string]$GpuExe = "c:\cudaimp\real_ecc_gpu_verify_kernel.exe"
)

$results = @()

foreach($n in $Sizes) {
  Write-Host "=== N=$n ===" -ForegroundColor Cyan

  $cpuOut = & $CpuExe $n 2>&1 | Out-String
  $gpuOut = & $GpuExe $n $GpuRepeats 2>&1 | Out-String

  $cpuThr = [double]([regex]::Match($cpuOut, "Throughput \(verifies/s\):\s*([0-9eE\+\.-]+)").Groups[1].Value)
  $gpuThr = [double]([regex]::Match($gpuOut, "Throughput \(verifies/s\):\s*([0-9eE\+\.-]+)").Groups[1].Value)
  $cpuMs  = [double]([regex]::Match($cpuOut, "Verify time \(ms\):\s*([0-9eE\+\.-]+)").Groups[1].Value)
  $gpuMs  = [double]([regex]::Match($gpuOut, "Kernel avg time \(ms\):\s*([0-9eE\+\.-]+)").Groups[1].Value)

  $speedup = if($cpuThr -gt 0){ $gpuThr / $cpuThr } else { 0 }

  $results += [pscustomobject]@{
    N = $n
    CPU_ms = [math]::Round($cpuMs, 3)
    GPU_ms = [math]::Round($gpuMs, 3)
    CPU_verifies_s = [math]::Round($cpuThr, 2)
    GPU_verifies_s = [math]::Round($gpuThr, 2)
    Speedup_x = [math]::Round($speedup, 2)
  }
}

Write-Host "`n=== Fair Benchmark Summary (same N) ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize
