import yaml
import argparse
import torch

class Config:
    def __init__(self, config_path=None, **kwargs):
        # 默认配置
        self.defaults = {
            'model_type': 'lm',
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
            'save_interval': 10,
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layer_norm': True
        }
        
        if config_path:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self.defaults.update(file_config)
        
        self.defaults.update(kwargs)
        
        for key, value in self.defaults.items():
            setattr(self, key, value)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0 if self.device.type == 'cuda' else 4
    
    def __str__(self):
        config_str = "Configuration:\n"
        for key, value in self.defaults.items():
            config_str += f"  {key}: {value}\n"
        config_str += f"  device: {self.device}\n"
        return config_str
    
    def to_dict(self):
        return self.defaults.copy()
    
    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.defaults, f, default_flow_style=False)
    
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='Transformer Training Configuration')
        
        # 模型架构参数
        parser.add_argument('--model_type', type=str, default='lm', choices=['lm', 'transformer'])
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--d_ff', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--use_positional_encoding', type=bool, default=True)
        parser.add_argument('--use_residual', type=bool, default=True)
        parser.add_argument('--use_layer_norm', type=bool, default=True)
        
        # 训练参数
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--seq_length', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0003)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--warmup_steps', type=int, default=4000)
        parser.add_argument('--grad_clip', type=float, default=1.0)
        
        # 实验设置
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--config', type=str, default=None)
        parser.add_argument('--save_dir', type=str, default='checkpoints')
        parser.add_argument('--results_dir', type=str, default='results')
        parser.add_argument('--data_dir', type=str, default='data')
        
        args = parser.parse_args()
        
        config_dict = {}
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
        
        return cls(**config_dict)

def get_ablation_configs():
    """获取消融实验配置"""
    base_config = {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 512,
        'num_layers': 2,
        'batch_size': 32,
        'seq_length': 128,
        'epochs': 30,
        'use_positional_encoding': True,
        'use_residual': True,
        'use_layer_norm': True
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
