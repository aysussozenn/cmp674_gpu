param(
  [string]$CpuExe = "c:\cudaimp\real_ecc_cpu.exe",
  [string]$GpuExe = "c:\cudaimp\real_ecc_gpu_verify_kernel.exe"
)

if (!(Test-Path $CpuExe)) { throw "CPU executable not found: $CpuExe" }
if (!(Test-Path $GpuExe)) { throw "GPU executable not found: $GpuExe" }

Write-Host "Running real ECC CPU reference..." -ForegroundColor Cyan
$cpuOut = & $CpuExe 2>&1
$cpuText = ($cpuOut | Out-String)
Write-Host $cpuText

Write-Host "Running real ECC GPU verify kernel..." -ForegroundColor Cyan
$gpuOut = & $GpuExe 2>&1
$gpuText = ($gpuOut | Out-String)
Write-Host $gpuText

$cpuPps = $null
$gpuPps = $null

$cpuMatch = [regex]::Match($cpuText, "Throughput \(pkt/s\):\s*([0-9eE\+\.-]+)")
if ($cpuMatch.Success) {
  $cpuPps = [double]$cpuMatch.Groups[1].Value
}

$gpuMatch = [regex]::Match($gpuText, "Throughput \(verifies/s\):\s*([0-9eE\+\.-]+)")
if ($gpuMatch.Success) {
  $gpuPps = [double]$gpuMatch.Groups[1].Value
}

Write-Host "----- Summary -----" -ForegroundColor Yellow
if ($null -ne $cpuPps) { Write-Host ("CPU throughput: {0:N2} verifies/s" -f $cpuPps) }
if ($null -ne $gpuPps) { Write-Host ("GPU throughput: {0:N2} verifies/s" -f $gpuPps) }
if (($null -ne $cpuPps) -and ($null -ne $gpuPps) -and ($cpuPps -gt 0)) {
  $speedup = $gpuPps / $cpuPps
  Write-Host ("GPU/CPU speedup: {0:N2}x" -f $speedup) -ForegroundColor Green
} else {
  Write-Host "Could not compute speedup automatically from output." -ForegroundColor DarkYellow
}
