# Transformer from Scratch

本作业实现了从零开始的Transformer模型，包含完整的训练流程和实验分析。

## 项目结构
transformer-assignment/
├── src/ # 源代码
│ ├── model.py # Transformer模型实现
│ ├── train.py # 训练脚本
│ ├── data_utils.py # 数据预处理工具
│ └── config.py # 配置管理
├── configs/ # 配置文件
│ └── base.yaml # 基础配置
├── scripts/ # 运行脚本
│ └── run.sh # 自动运行脚本
├── data/ # 数据集
├── checkpoints/ # 模型检查点
├── results/ # 实验结果
├── requirements.txt # Python依赖
└── README.md # 说明文档

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- 至少4GB RAM
- GPU推荐但不必须

### 运行命令
```bash
# 方式1: 使用自动脚本
chmod +x scripts/run.sh
./scripts/run.sh

# 方式2: 手动运行
pip install -r requirements.txt

### 复现实验
# 设置随机种子确保可复现性
python src/train.py --config configs/base.yaml --seed 42

## 模型特性
✅ Multi-Head Self-Attention

✅ Position-wise Feed-Forward Networks

✅ 残差连接 + Layer Normalization

✅ 正弦位置编码

✅ 编码器-解码器架构

✅ 训练稳定性技巧(学习率调度、梯度裁剪)

## 实验结果
训练完成后，在results/目录下查看:

训练/验证损失曲线

困惑度变化图

生成文本示例

## 扩展功能
相对位置编码

稀疏注意力

模型蒸馏

多GPU训练

mkdir -p data checkpoints results
# 准备数据后运行
python src/train.py --config configs/base.yaml
