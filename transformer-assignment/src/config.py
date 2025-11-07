# src/config.py
import yaml
import argparse
import torch

class Config:
    def __init__(self, config_path=None, **kwargs):
        # 默认配置
        self.defaults = {
            'model_type': 'lm',  # 'lm' for language model, 'transformer' for encoder-decoder
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 512,
            'num_layers': 2,
            'seq_length': 128,
            'batch_size': 32,
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'epochs': 50,
            'dropout': 0.1,
            'seed': 42,
            'warmup_steps': 4000,
            'grad_clip': 1.0,
            'save_dir': 'checkpoints',
            'results_dir': 'results',
            'data_dir': 'data',
            'log_interval': 100,
            'eval_interval': 500,
            'save_interval': 10
        }
        
        # 从YAML文件加载配置（如果提供）
        if config_path:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self.defaults.update(file_config)
        
        # 更新命令行参数
        self.defaults.update(kwargs)
        
        # 设置属性
        for key, value in self.defaults.items():
            setattr(self, key, value)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0 if self.device.type == 'cuda' else 4
    
    def __str__(self):
        config_str = "Configuration:\n"
        for key, value in self.defaults.items():
            config_str += f"  {key}: {value}\n"
        config_str += f"  device: {self.device}\n"
        config_str += f"  num_workers: {self.num_workers}\n"
        return config_str
    
    def to_dict(self):
        return self.defaults.copy()
    
    def save(self, path):
        """保存配置到YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self.defaults, f, default_flow_style=False)
    
    @classmethod
    def from_args(cls):
        """从命令行参数创建配置"""
        parser = argparse.ArgumentParser(description='Transformer Training Configuration')
        
        # 模型架构参数
        parser.add_argument('--model_type', type=str, default='lm', 
                          choices=['lm', 'transformer'],
                          help='Model type: lm (language model) or transformer (encoder-decoder)')
        parser.add_argument('--d_model', type=int, default=128, 
                          help='Model dimension')
        parser.add_argument('--num_heads', type=int, default=4, 
                          help='Number of attention heads')
        parser.add_argument('--d_ff', type=int, default=512, 
                          help='Feed-forward dimension')
        parser.add_argument('--num_layers', type=int, default=2, 
                          help='Number of transformer layers')
        parser.add_argument('--dropout', type=float, default=0.1, 
                          help='Dropout rate')
        
        # 训练参数
        parser.add_argument('--batch_size', type=int, default=32, 
                          help='Batch size')
        parser.add_argument('--seq_length', type=int, default=128, 
                          help='Sequence length')
        parser.add_argument('--learning_rate', type=float, default=0.0003, 
                          help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, 
                          help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50, 
                          help='Number of epochs')
        parser.add_argument('--warmup_steps', type=int, default=4000, 
                          help='Warmup steps for learning rate scheduler')
        parser.add_argument('--grad_clip', type=float, default=1.0, 
                          help='Gradient clipping value')
        
        # 实验设置
        parser.add_argument('--seed', type=int, default=42, 
                          help='Random seed')
        parser.add_argument('--config', type=str, default=None, 
                          help='Path to config YAML file')
        parser.add_argument('--save_dir', type=str, default='checkpoints', 
                          help='Directory to save checkpoints')
        parser.add_argument('--results_dir', type=str, default='results', 
                          help='Directory to save results')
        parser.add_argument('--data_dir', type=str, default='data', 
                          help='Directory containing data')
        
        # 日志和保存
        parser.add_argument('--log_interval', type=int, default=100, 
                          help='Log interval in steps')
        parser.add_argument('--eval_interval', type=int, default=500, 
                          help='Evaluation interval in steps')
        parser.add_argument('--save_interval', type=int, default=10, 
                          help='Save interval in epochs')
        
        args = parser.parse_args()
        
        # 从YAML文件加载配置（如果提供）
        config_dict = {}
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        # 更新命令行参数
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
        
        return cls(**config_dict)

# 实验配置预设
def get_preset_config(preset_name):
    """获取预设配置"""
    presets = {
        'tiny': {
            'd_model': 64,
            'num_heads': 2,
            'd_ff': 256,
            'num_layers': 2,
            'batch_size': 16,
            'seq_length': 64
        },
        'small': {
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 512,
            'num_layers': 4,
            'batch_size': 32,
            'seq_length': 128
        },
        'base': {
            'd_model': 256,
            'num_heads': 8,
            'd_ff': 1024,
            'num_layers': 6,
            'batch_size': 32,
            'seq_length': 256
        },
        'debug': {
            'd_model': 32,
            'num_heads': 2,
            'd_ff': 128,
            'num_layers': 1,
            'batch_size': 8,
            'seq_length': 32,
            'epochs': 5
        }
    }
    
    if preset_name in presets:
        return Config(**presets[preset_name])
    else:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

# 用于消融实验的配置
def get_ablation_configs():
    """获取消融实验配置"""
    base_config = {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 512,
        'num_layers': 2,
        'batch_size': 32,
        'seq_length': 128,
        'epochs': 30
    }
    
    ablations = {
        'base': base_config,
        'no_positional_encoding': {**base_config, 'use_positional_encoding': False},
        'single_head': {**base_config, 'num_heads': 1},
        'no_residual': {**base_config, 'use_residual': False},
        'no_layer_norm': {**base_config, 'use_layer_norm': False},
        'small_ffn': {**base_config, 'd_ff': 128},
        'shallow': {**base_config, 'num_layers': 1}
    }
    
    return {name: Config(**config) for name, config in ablations.items()}

if __name__ == '__main__':
    # 测试配置类
    config = Config.from_args()
    print(config)
