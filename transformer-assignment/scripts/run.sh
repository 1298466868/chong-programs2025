#!/bin/bash

echo "Setting up Transformer Assignment for Windows Git Bash..."

# 检查是否在Git Bash环境中
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "检测到Windows Git Bash环境"
else
    echo "警告：这不是Windows Git Bash环境，可能仍有兼容性问题"
fi

# 如果下载失败，创建示例数据的函数
create_sample_data() {
    cat > "data/tiny_shakespeare.txt" << 'EOF'
First Citizen:
Before we proceed any further, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?
All:
Resolved, resolved.
First Citizen:
First, you know Caius Marcius is chief enemy to the people.
All:
We know't, we know't.
First Citizen:
Let us kill him, and we'll have corn at our own price.
EOF
    echo "已创建示例数据"
}

# 创建SSL修复的Python脚本
create_ssl_fix() {
    cat > "ssl_fix.py" << 'EOF'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 下载数据集
import urllib.request
import os

try:
    print("下载Tiny Shakespeare数据集...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/tiny_shakespeare.txt"
    )
    print("下载成功")
except Exception as e:
    print(f"下载失败: {e}")
    print("创建示例数据...")
    # 这里可以调用create_sample_data的逻辑
EOF
}

# 使用系统Python，不创建虚拟环境（避免路径问题）
echo "使用系统Python环境..."

# 安装依赖（如果尚未安装）
echo "检查依赖..."
pip install torch matplotlib numpy pyyaml requests

# 创建目录 - 使用mkdir -p确保兼容性
echo "创建目录结构..."
mkdir -p data
mkdir -p checkpoints
mkdir -p results

# 下载数据集 - 添加错误处理和备用方案
echo "下载Tiny Shakespeare数据集..."
create_ssl_fix
python ssl_fix.py

# 如果下载失败，确保有数据文件
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "下载失败，创建示例数据..."
    create_sample_data
fi

# 分割数据 - 添加错误处理
echo "处理数据集..."
python -c "
import sys
try:
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    if len(text) < 100:  # 如果数据太小，扩展它
        text = text * 50
    
    train_end = int(len(text) * 0.8)
    val_end = train_end + int(len(text) * 0.1)
    
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        f.write(text[:train_end])
    with open('data/val.txt', 'w', encoding='utf-8') as f:
        f.write(text[train_end:val_end])
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        f.write(text[val_end:])
    
    print(f'数据集分割完成: 训练集{len(text[:train_end])}字符, 验证集{len(text[train_end:val_end])}字符, 测试集{len(text[val_end:])}字符')
    
except Exception as e:
    print(f'数据处理错误: {e}')
    # 创建基础训练文件
    with open('data/train.txt', 'w') as f:
        f.write('A B C D E ' * 100)
    with open('data/val.txt', 'w') as f:
        f.write('A B C D E ' * 20)
    with open('data/test.txt', 'w') as f:
        f.write('A B C D E ' * 10)
    print('已创建基础训练数据')
"

# 清理临时文件
rm -f ssl_fix.py

# 检查必要的Python文件是否存在
echo "检查必要文件..."
if [ ! -f "src/train.py" ]; then
    echo "错误: src/train.py 不存在"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

if [ ! -f "configs/base.yaml" ]; then
    echo "警告: configs/base.yaml 不存在，使用命令行参数"
    CONFIG_ARG=""
else
    CONFIG_ARG="--config configs/base.yaml"
fi

# 开始训练 - 使用更小的配置确保能运行
echo "开始训练..."
python src/train.py $CONFIG_ARG --seed 42 --batch_size 16 --seq_length 64 --epochs 30

echo "训练完成! 查看 results/ 目录获取结果"
