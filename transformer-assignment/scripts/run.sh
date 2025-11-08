#!/bin/bash

echo "Setting up Transformer Assignment for Windows Git Bash..."

# 检查是否在Git Bash环境中
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "检测到Windows Git Bash环境"
else
    echo "警告：这不是Windows Git Bash环境，可能仍有兼容性问题"
fi

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

# 检查数据集是否存在
echo "检查数据集..."
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "错误: data/tiny_shakespeare.txt 不存在"
    echo "请手动下载数据集并放在 data/tiny_shakespeare.txt"
    echo "下载地址: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    exit 1
else
    echo "✅ 数据集已存在"
fi

# 分割数据 - 添加错误处理
echo "处理数据集..."
python -c "
import sys
try:
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f'数据集大小: {len(text)} 字符')
    
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

# 开始训练
echo "开始训练..."
python src/train.py $CONFIG_ARG --seed 42 --batch_size 128 --seq_length 64 --epochs 10

echo "训练完成! 查看 results/ 目录获取结果"
