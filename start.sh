#!/bin/bash

# 启动入口脚本

echo "========================================"
echo "Bayesian Optimization System Launcher"
echo "========================================"
echo ""

# 检查Python环境
echo "【INFO】Checking Python environment..."
python --version

# 检查Node.js环境
echo "【INFO】Checking Node.js environment..."
node --version

# 启动后端API服务
echo ""
echo "【INFO】Starting Backend API Server..."
echo "========================================"
cd MyAiProj
python api_server.py &
API_PID=$!
echo "Backend API Server started with PID: $API_PID"
echo "API URL: http://localhost:8000"
echo ""

# 等待API服务启动
echo "【INFO】Waiting for API server to start..."
sleep 3

# 启动前端应用
echo "【INFO】Starting Frontend Application..."
echo "========================================"
cd ../bayesadhesion-optimize
npm run dev &
FRONTEND_PID=$!
echo "Frontend Application started with PID: $FRONTEND_PID"
echo ""

echo "【INFO】Launch Complete!"
echo "========================================"
echo "Backend API: http://localhost:8000"
echo "Frontend App: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# 等待用户中断
wait $API_PID $FRONTEND_PID
