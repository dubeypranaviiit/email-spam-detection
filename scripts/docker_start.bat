@echo off
echo =========================================================
echo  Starting Spam Detection System with Docker Compose
echo =========================================================
echo.

where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker CLI not found in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker info >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Docker daemon is not running. Attempting to start Docker Desktop...
    if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        echo Waiting for Docker engine to initialize...
        :wait_loop
        timeout /t 5 /nobreak >nul
        docker info >nul 2>nul
        if %errorlevel% neq 0 (
            echo Waiting for Docker daemon...
            goto wait_loop
        )
        echo [OK] Docker engine is ready!
    ) else (
        echo [ERROR] Docker Desktop.exe not found at default path.
        echo Please launch Docker Desktop manually, then re-run this script.
        pause
        exit /b 1
    )
)

echo.
echo Building and launching containers (FastAPI on :8000, Next.js on :3000)...
docker compose up --build -d

if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose failed to launch.
    pause
    exit /b %errorlevel%
)

echo.
echo =========================================================
echo  [OK] System successfully started in background!
echo =========================================================
echo.
echo   - Next.js Dashboard:  http://localhost:3000
echo   - FastAPI Swagger UI: http://localhost:8000/docs
echo   - API Health Probe:   http://localhost:8000/health
echo.
echo   To view live logs:    docker compose logs -f
echo   To stop containers:   docker compose down (or run scripts\docker_stop.bat)
echo =========================================================
echo.
pause
