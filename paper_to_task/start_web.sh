#!/bin/bash

echo "============================================================"
echo "🌐 启动 Paper-to-Task Web应用"
echo "============================================================"
echo ""

# 检查是否在项目根目录
if [ ! -f "paper_to_task/__init__.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    echo "   当前目录: $(pwd)"
    exit 1
fi

# 检查Flask是否安装
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 正在安装Flask..."
    pip install flask -q
fi

echo "✅ 环境检查完成"
echo ""
echo "🚀 启动Web服务器..."
echo "📱 访问地址: http://localhost:5000"
echo "📄 请在浏览器中打开上述地址"
echo ""
echo "按 Ctrl+C 停止服务"
echo "============================================================"
echo ""

# 启动Flask应用
python -m paper_to_task.interaction.web_app
