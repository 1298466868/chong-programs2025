#!/bin/bash

echo "Setting up Transformer Assignment for Windows Git Bash..."

# 检查是否在Git Bash环境中
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "检测到Windows Git Bash环境"
else
    echo "警告：这不是Windows Git Bash环境，可能仍有兼容性问题"
fi

# 专门针对GitHub原始文件的下载函数
download_github_raw_file() {
    local url="$1"
    local output="$2"
    
    echo "正在从GitHub下载: $url"
    
    # 方法1: 使用curl (Git Bash通常自带)
    if command -v curl &> /dev/null; then
        echo "使用curl下载..."
        curl -s -L -o "$output" "$url" && return 0
    fi
    
    # 方法2: 使用wget
    if command -v wget &> /dev/null; then
        echo "使用wget下载..."
        wget -q -O "$output" "$url" && return 0
    fi
    
    # 方法3: 使用Python (最可靠)
    echo "使用Python下载..."
    python -c "
import urllib.request
import ssl
try:
    # 修复SSL证书问题
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve('$url', '$output')
    print('Python下载成功')
except Exception as e:
    print(f'Python下载失败: {e}')
"
}

# 创建示例数据的函数
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
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

All:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
EOF
    echo "已创建示例莎士比亚数据"
}

# 创建目录
echo "创建目录结构..."
mkdir -p data
mkdir -p checkpoints
mkdir -p results

# 下载数据集 - 专门处理GitHub原始文件
echo "下载Tiny Shakespeare数据集..."
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

download_github_raw_file "$URL" "data/tiny_shakespeare.txt"

# 检查下载是否成功
if [ -f "data/tiny_shakespeare.txt" ] && [ -s "data/tiny_shakespeare.txt" ]; then
    file_size=$(wc -c < "data/tiny_shakespeare.txt")
    if [ "$file_size" -gt 1000 ]; then
        echo "✅ 数据集下载成功! 文件大小: $file_size 字节"
    else
        echo "⚠️ 下载的文件太小，使用示例数据"
        create_sample_data
    fi
else
    echo "❌ 下载失败，使用示例数据"
    create_sample_data
fi

# 显示数据样本
echo "数据预览 (前500字符):"
head -c 500 "data/tiny_shakespeare.txt"
echo -e "\n...\n"

# 分割数据
echo "处理数据集..."
python -c "
import sys
try:
    with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f'原始数据总字符数: {len(text)}')
    
    # 分割数据集
    train_end = int(len(text) * 0.8)
    val_end = train_end + int(len(text) * 0.1)
    
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        f.write(text[:train_end])
    with open('data/val.txt', 'w', encoding='utf-8') as f:
        f.write(text[train_end:val_end])
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        f.write(text[val_end:])
    
    print(f'✅ 数据集分割完成:')
    print(f'   训练集: {len(text[:train_end])} 字符')
    print(f'   验证集: {len(text[train_end:val_end])} 字符') 
    print(f'   测试集: {len(text[val_end:])} 字符')
    
except Exception as e:
    print(f'❌ 数据处理错误: {e}')
    # 创建基础训练文件
    sample_text = 'The quick brown fox jumps over the lazy dog. ' * 100
    with open('data/train.txt', 'w') as f:
        f.write(sample_text)
    with open('data/val.txt', 'w') as f:
        f.write(sample_text[:len(sample_text)//5])
    with open('data/test.txt', 'w') as f:
        f.write(sample_text[:len(sample_text)//10])
    print('⚠️ 已创建基础训练数据')
"

# 安装依赖
echo "安装Python依赖..."
pip install torch matplotlib numpy pyyaml requests

# 检查必要的Python文件
echo "检查必要文件..."
if [ ! -f "src/train.py" ]; then
    echo "❌ 错误: src/train.py 不存在"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

if [ ! -f "configs/base.yaml" ]; then
    echo "⚠️ 警告: configs/base.yaml 不存在，使用默认参数"
    CONFIG_ARG=""
else
    CONFIG_ARG="--config configs/base.yaml"
fi

# 开始训练
echo "开始训练..."
python src/train.py $CONFIG_ARG --seed 42 --batch_size 16 --seq_length 64 --epochs 30

echo "✅ 训练完成! 查看 results/ 目录获取结果"
