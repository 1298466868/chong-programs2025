import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from model import TransformerLM, Transformer
from data_utils import TextDataset
from config import Config

# ===== CUDA修复方案 =====
import os
# 禁用CUDA图和相关优化
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("✅ 已应用CUDA修复")
# ===== 修复结束 =====

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 修复调度器逻辑
        if hasattr(config, 'warmup_steps') and config.warmup_steps > 0:
            self.scheduler = self.get_cosine_schedule_with_warmup()
            self.scheduler_type = 'step'  # 按step调整
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.epochs
            )
            self.scheduler_type = 'epoch'  # 按epoch调整
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.device = config.device
        self.model.to(self.device)
        
        # 预计算掩码以提高效率
        self._cached_masks = {}
        
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.learning_rates = []
        
    def get_cosine_schedule_with_warmup(self):
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, self.config.total_training_steps - self.config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_mask(self, seq_len):
        """缓存掩码以提高效率"""
        if seq_len not in self._cached_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            self._cached_masks[seq_len] = mask.to(self.device)
        return self._cached_masks[seq_len]
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            seq_len = data.size(1)
            mask = self.get_mask(seq_len)
            
            self.optimizer.zero_grad()
            
            output = self.model(data, mask)
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            loss.backward()
            
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
            
            self.optimizer.step()
            
            # 修复：只在step类型调度器时每个batch更新
            if self.scheduler_type == 'step':
                self.scheduler.step()
            
            total_loss += loss.item() * data.size(0)
            total_tokens += data.size(0)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                
        return total_loss / total_tokens if total_tokens > 0 else 0
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                seq_len = data.size(1)
                mask = self.get_mask(seq_len)
                
                output = self.model(data, mask)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                
                total_loss += loss.item() * data.size(0)
                total_tokens += data.size(0)
                
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')  # 防止溢出
        
        return avg_loss, perplexity
    
    def train(self):
        print("Starting training...")
        print(f"Training on: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 计算总训练步数
        if hasattr(self.config, 'warmup_steps') and self.config.warmup_steps > 0:
            self.config.total_training_steps = self.config.epochs * len(self.train_loader)
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss, perplexity = self.validate()
            
            # 修复：只在epoch类型调度器时每个epoch更新
            if self.scheduler_type == 'epoch':
                self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.perplexities.append(perplexity)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Perplexity: {perplexity:.2f}')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 50)
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
                
        self.plot_training_curves()
        self.generate_sample_text()
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'perplexities': self.perplexities,
            'config': self.config.__dict__
        }
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(checkpoint, f'{self.config.save_dir}/model_epoch_{epoch+1}.pt')
        print(f"✅ Checkpoint saved: {self.config.save_dir}/model_epoch_{epoch+1}.pt")
    
    def plot_training_curves(self):
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.perplexities)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.config.results_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_sample_text(self):
        self.model.eval()
        dataset = self.train_loader.dataset
        
        start_text = "First Citizen:"
        start_tokens = [dataset.char_to_idx.get(ch, 1) for ch in start_text]
        
        with torch.no_grad():
            input_seq = torch.tensor(start_tokens).unsqueeze(0).to(self.device)
            generated = start_text
            
            for _ in range(100):
                seq_len = input_seq.size(1)
                mask = self.get_mask(seq_len)
                
                output = self.model(input_seq, mask)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token == 0 or len(generated) > 200:
                    break
                    
                generated += dataset.idx_to_char.get(next_token, '')
                input_seq = torch.cat([input_seq, torch.tensor([[next_token]]).to(self.device)], dim=1)
                
                if input_seq.size(1) > self.config.seq_length:
                    input_seq = input_seq[:, -self.config.seq_length:]
        
        with open(f'{self.config.results_dir}/generated_text.txt', 'w', encoding='utf-8') as f:
            f.write(generated)
        
        print("Generated text sample:")
        print(generated)

def main():
    # 确保所有必要目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    config = Config.from_args()
    print(config)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    train_dataset = TextDataset(f'{config.data_dir}/train.txt', config.seq_length)
    val_dataset = TextDataset(f'{config.data_dir}/val.txt', config.seq_length)
    
    # Windows下num_workers设为0避免问题
    num_workers = 0 if os.name == 'nt' else config.num_workers
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    if config.model_type == 'lm':
        model = TransformerLM(
            vocab_size=train_dataset.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            max_seq_length=config.seq_length,
            dropout=config.dropout,
            use_positional_encoding=config.use_positional_encoding,
            use_residual=config.use_residual,
            use_layer_norm=config.use_layer_norm
        )
    else:
        model = Transformer(
            src_vocab_size=train_dataset.vocab_size,
            tgt_vocab_size=train_dataset.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            max_seq_length=config.seq_length,
            dropout=config.dropout,
            use_positional_encoding=config.use_positional_encoding,
            use_residual=config.use_residual,
            use_layer_norm=config.use_layer_norm
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main()
