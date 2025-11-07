# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import yaml
from model import TransformerLM, Transformer
from data_utils import TextDataset, create_masks
import math

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Create mask for language modeling
            seq_len = data.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data, mask)
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_tokens += data.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        return total_loss / total_tokens
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                seq_len = data.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
                mask = mask.to(self.device)
                
                output = self.model(data, mask)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                
                total_loss += loss.item() * data.size(0)
                total_tokens += data.size(0)
                
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, perplexity = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.perplexities.append(perplexity)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Perplexity: {perplexity:.2f}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  LR: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Save model checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
                
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'perplexities': self.perplexities
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pt')
        
    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.perplexities)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load config
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create datasets and dataloaders
    train_dataset = TextDataset('data/train.txt', config['seq_length'])
    val_dataset = TextDataset('data/val.txt', config['seq_length'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Initialize model
    if config['model_type'] == 'lm':
        model = TransformerLM(
            vocab_size=train_dataset.vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_length=config['seq_length'],
            dropout=config.get('dropout', 0.1)
        )
    else:
        model = Transformer(
            src_vocab_size=train_dataset.vocab_size,
            tgt_vocab_size=train_dataset.vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_length=config['seq_length'],
            dropout=config.get('dropout', 0.1)
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main()
