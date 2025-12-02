@echo off

REM Bayesian Optimization System Launcher
REM =========================================
echo =========================================
echo Bayesian Optimization System Launcher
echo =========================================
echo.

REM 检查Python环境
echo 【INFO】Checking Python environment...
python --version
echo.

REM 启动后端API服务
echo 【INFO】Starting Backend API Server...
echo =========================================
start "Backend API Server" cmd /k "cd MyAiProj && python api_server.py"
echo Backend API Server started
echo API URL: http://localhost:8000
echo.

REM 等待API服务启动
echo 【INFO】Waiting for API server to start...
timeout /t 3 /nobreak >nul

REM 启动前端应用
echo 【INFO】Starting Frontend Application...
echo =========================================
start "Frontend Application" cmd /k "cd bayesadhesion-optimize && npm install && npm run dev"
echo Frontend Application started
echo.

echo 【INFO】Launch Complete!
echo =========================================
echo Backend API: http://localhost:8000
echo Frontend App: http://localhost:5173
echo.
echo Press any key to exit
pause >nul
