# ============================================================================
# Blood Smear Domain Expert — one-click launcher
# ============================================================================
# Starts the FastAPI backend (port 8767) and the Vite frontend (port 5173) in
# two separate PowerShell windows so you can see logs from each. Idempotent:
# if either port is already in use, the existing process is killed first so
# you never end up in the "backend unreachable" state again.
#
# Usage from the repo root:
#     .\start.ps1
#
# Then open http://localhost:5173 in your browser.
# ============================================================================

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Stop-PortListener {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Write-Host "[start] Port $Port is busy (PID $($conn.OwningProcess)) - terminating..." -ForegroundColor Yellow
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

# 1. Free the ports
Stop-PortListener -Port 8767
Stop-PortListener -Port 5173

# 2. Sanity checks
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[start] ERROR: .venv not found at $venvPython" -ForegroundColor Red
    Write-Host "[start] Create it first: python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot "frontend\node_modules"))) {
    Write-Host "[start] frontend\node_modules missing - running npm install..." -ForegroundColor Yellow
    Push-Location (Join-Path $RepoRoot "frontend")
    npm install
    Pop-Location
}

# 3. Launch backend in a new window (cwd = backend/, so `uvicorn main:app`
#    works the natural way; relative paths like data/pdfs are resolved against
#    the repo root inside the code, so cwd doesn't matter for correctness).
Write-Host "[start] Launching backend on http://127.0.0.1:8767 ..." -ForegroundColor Cyan
$backendCmd = "Set-Location '$RepoRoot\backend'; `$env:PYTHONIOENCODING='utf-8'; & '$venvPython' -m uvicorn main:app --port 8767 --log-level info"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# 4. Launch frontend in a new window
Write-Host "[start] Launching frontend on http://localhost:5173 ..." -ForegroundColor Cyan
$frontendCmd = "Set-Location '$RepoRoot\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

# 5. Wait for backend to come up, then open the browser
Write-Host "[start] Waiting for backend health check..." -ForegroundColor Cyan
$maxAttempts = 30
for ($i = 1; $i -le $maxAttempts; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:8767/api/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Host "[start] Backend is up ($($r.Content))" -ForegroundColor Green
            break
        }
    } catch { }
    if ($i -eq $maxAttempts) {
        Write-Host "[start] WARNING: backend did not respond within ${maxAttempts}s. Check the backend window for errors." -ForegroundColor Red
    }
}

Write-Host "[start] Opening http://localhost:5173 ..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host " Blood Smear Domain Expert is running." -ForegroundColor Green
Write-Host "   Frontend : http://localhost:5173" -ForegroundColor Green
Write-Host "   Backend  : http://127.0.0.1:8767  (Swagger: /docs)" -ForegroundColor Green
Write-Host " To stop: close the two new PowerShell windows." -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
