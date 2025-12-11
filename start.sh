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
echo "【INFO】Starting Backend API Server (MetalBayes)..."
echo "========================================"
cd MetalBayes

# 检查并激活 conda 环境
if command -v conda &> /dev/null; then
    echo "【INFO】Activating conda environment: bayes"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bayes
    if [ $? -eq 0 ]; then
        echo "【INFO】Conda environment 'bayes' activated successfully"
        python api_server.py &
        API_PID=$!
    else
        echo "【ERROR】Failed to activate conda environment 'bayes'"
        echo "【WARNING】Trying to use conda run instead..."
        conda run -n bayes --no-capture-output python api_server.py &
        API_PID=$!
    fi
else
    # 如果没有 conda，直接使用 python
    echo "【WARNING】Conda not found, using system python"
    python api_server.py &
    API_PID=$!
fi
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

# 检查并安装依赖
if [ ! -d "node_modules" ]; then
    echo "【INFO】Installing frontend dependencies..."
    npm install
fi

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
