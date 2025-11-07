# Transformer from Scratch

从零实现的Transformer模型，包含完整的训练流程和消融实验。

## 项目结构
transformer-assignment/
├── src/ # 源代码
│ ├── model.py # Transformer模型实现
│ ├── train.py # 训练脚本
│ ├── data_utils.py # 数据预处理
│ └── config.py # 配置管理
├── configs/ # 配置文件
│ └── base.yaml # 基础配置
├── scripts/ # 运行脚本
│ ├── run.sh # 一键运行
│ └── run_ablation.py # 消融实验
├── data/ # 数据集
├── checkpoints/ # 模型检查点
├── results/ # 实验结果
├── requirements.txt # 依赖
└── README.md # 说明文档


## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM
- GPU推荐

### 运行命令
```bash
1.一键运行（推荐）
chmod +x scripts/run.sh
./scripts/run.sh

2.手动运行
pip install -r requirements.txt
python src/train.py --config configs/base.yaml

### 消融实验
1.运行所有消融实验
python scripts/run_ablation.py

2.运行单个消融实验
python src/train.py --num_heads 1 --epochs 30

## 模型特性
✅ Multi-Head Self-Attention

✅ Position-wise Feed-Forward Networks

✅ 残差连接 + Layer Normalization

✅ 正弦位置编码

✅ 完整的消融实验支持

## 实验结果
训练完成后查看：

results/training_curves.png - 训练曲线

results/generated_text.txt - 生成文本

results/ablation_comparison.png - 消融实验结果
✅ 训练稳定性技巧
