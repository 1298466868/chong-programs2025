# Transformer from Scratch

从零实现的Transformer模型，包含完整的训练流程和消融实验。

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM
- GPU推荐

### 数据集位置
data/tiny_shakespeare.txt    # 原始数据集
data/train.txt              # 训练集
data/val.txt                # 验证集  
data/test.txt               # 测试集

### 超参数配置
configs/base.yaml：
model_type: "lm"              # 模型类型: lm(语言模型) / transformer(编码器-解码器)
d_model: 128                  # 模型维度
num_heads: 4                  # 注意力头数
d_ff: 512                     # 前馈网络维度
num_layers: 2                 # Transformer层数
seq_length: 128               # 序列长度
batch_size: 32                # 批大小
learning_rate: 0.0003         # 学习率
epochs: 50                    # 训练轮数

### 运行命令
```bash
1.一键运行（推荐）
chmod +x scripts/run.sh
./scripts/run.sh

2.手动运行
pip install -r requirements.txt
python src/train.py --config configs/base.yaml
```


### 消融实验
1.运行所有消融实验
python scripts/run_ablation.py

2.运行单个消融实验
python src/train.py --num_heads 1 --epochs 30

## 模型特性
1.核心组件
✅ Multi-Head Self-Attention: 缩放点积注意力机制

✅ Position-wise FFN: 位置前馈网络

✅ 残差连接 + LayerNorm: 训练稳定性

✅ 正弦位置编码: 位置信息注入

✅ 编码器-解码器架构: 完整Transformer支持

2.训练特性
✅ 学习率调度: Cosine annealing + warmup

✅ 梯度裁剪: 防止梯度爆炸

✅ 早停机制: 基于验证集性能

✅ 模型保存: 定期保存检查点

✅ 可视化: 训练曲线实时绘制

## 实验结果
训练完成后查看：

results/training_curves.png - 训练曲线

results/generated_text.txt - 生成文本

results/ablation_comparison.png - 消融实验结果

## 许可证
本项目仅用于教育目的，基于MIT许可证。
