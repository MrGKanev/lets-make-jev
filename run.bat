@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (set "JEV_MODE=full") else (set "JEV_MODE=%~1")
if /i "%JEV_MODE%"=="full" goto start
if /i "%JEV_MODE%"=="smoke" goto start
if /i "%JEV_MODE%"=="test" goto start
echo Usage: run.bat [full^|smoke^|test]
exit /b 2
:start
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Start Docker Desktop with the WSL2 backend and Linux containers.
    goto failed
)
docker compose version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose v2 is required. Update Docker Desktop.
    goto failed
)
echo Running Jev: %JEV_MODE%
docker compose run --rm --build jev
if errorlevel 1 goto failed
echo.
echo Completed. Results: bench/results/
if not defined JEV_NO_PAUSE pause
exit /b 0
:failed
echo.
echo Jev failed. See the error above. GPU runs require an NVIDIA GPU and a WSL-compatible driver.
if not defined JEV_NO_PAUSE pause
exit /b 1
