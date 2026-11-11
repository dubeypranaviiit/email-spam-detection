Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " Starting Spam Detection System with Docker Compose" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI not found. Please install Docker Desktop."
    exit 1
}

$dockerReady = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerReady = $true }
} catch {}

if (-not $dockerReady) {
    Write-Host "[INFO] Docker Desktop is not running. Launching Docker Desktop..." -ForegroundColor Yellow
    $dockerExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerExe) {
        Start-Process $dockerExe
        Write-Host "Waiting for Docker daemon to initialize..." -ForegroundColor Yellow
        $retries = 0
        while (-not $dockerReady -and $retries -lt 15) {
            Start-Sleep -Seconds 4
            $retries++
            try {
                $null = docker info 2>&1
                if ($LASTEXITCODE -eq 0) { $dockerReady = $true }
            } catch {}
        }
    } else {
        Write-Error "Could not find Docker Desktop.exe. Please start Docker Desktop manually."
        exit 1
    }
}

if (-not $dockerReady) {
    Write-Error "Docker daemon did not become ready in time. Please verify Docker Desktop is running."
    exit 1
}

Write-Host "[OK] Docker engine is ready." -ForegroundColor Green
Write-Host "Building and launching containers..." -ForegroundColor Cyan
docker compose up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Green
    Write-Host " [OK] Applications running successfully!" -ForegroundColor Green
    Write-Host "=========================================================" -ForegroundColor Green
    Write-Host "  - Next.js Dashboard:  http://localhost:3000" -ForegroundColor White
    Write-Host "  - FastAPI Swagger UI: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  - API Health Check:   http://localhost:8000/health" -ForegroundColor White
    Write-Host ""
    Write-Host " View logs: docker compose logs -f" -ForegroundColor Gray
    Write-Host " Stop:      docker compose down" -ForegroundColor Gray
    Write-Host "=========================================================" -ForegroundColor Green
} else {
    Write-Error "Docker Compose encountered an error."
}
