#!/bin/bash

echo "Setting up Transformer Assignment..."

# Create virtual environment
python -m venv transformer_env
# 尝试Windows方式的虚拟环境激活
if [ -f "transformer_env/Scripts/activate" ]; then
    source transformer_env/Scripts/activate
else
    echo "虚拟环境不存在，使用系统Python环境"
    echo "当前Python: $(which python)"
fi

# Install dependencies
pip install torch matplotlib numpy pyyaml requests

# Create directories
mkdir -p data checkpoints results

# Download and prepare tiny shakespeare dataset
echo "Downloading Tiny Shakespeare dataset..."
python -c "
import requests
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url)
with open('data/tiny_shakespeare.txt', 'w') as f:
    f.write(response.text)

# Split data
text = response.text
train_ratio = 0.8
val_ratio = 0.1
train_end = int(len(text) * train_ratio)
val_end = train_end + int(len(text) * val_ratio)

with open('data/train.txt', 'w') as f:
    f.write(text[:train_end])
with open('data/val.txt', 'w') as f:
    f.write(text[train_end:val_end])
with open('data/test.txt', 'w') as f:
    f.write(text[val_end:])

print(f'Dataset sizes - Train: {len(text[:train_end])}, Val: {len(text[train_end:val_end])}, Test: {len(text[val_end:])}')
"

# Train the model
echo "Starting training..."
python src/train.py --config configs/base.yaml --seed 42

echo "Training completed! Check results/ directory for outputs."

deactivate
