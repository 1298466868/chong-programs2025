import os
import json
import torch
import matplotlib.pyplot as plt
from src.config import get_ablation_configs
from src.train import Trainer
from src.data_utils import TextDataset
from src.model import TransformerLM
from torch.utils.data import DataLoader

def run_single_ablation(config, exp_name):
    print(f"Running ablation: {exp_name}")
    
    config.save_dir = f"checkpoints/ablation_{exp_name}"
    config.results_dir = f"results/ablation_{exp_name}"
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    torch.manual_seed(config.seed)
    
    train_dataset = TextDataset('data/train.txt', config.seq_length)
    val_dataset = TextDataset('data/val.txt', config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = TransformerLM(
        vocab_size=train_dataset.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        max_seq_length=config.seq_length,
        dropout=config.dropout,
        use_positional_encoding=getattr(config, 'use_positional_encoding', True),
        use_residual=getattr(config, 'use_residual', True),
        use_layer_norm=getattr(config, 'use_layer_norm', True)
    )
    
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    final_loss, final_ppl = trainer.validate()
    
    return {
        'final_loss': final_loss,
        'final_perplexity': final_ppl,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'perplexities': trainer.perplexities
    }

def main():
    ablation_configs = get_ablation_configs()
    results = {}
    
    for exp_name, config in ablation_configs.items():
        try:
            result = run_single_ablation(config, exp_name)
            results[exp_name] = result
            print(f"✅ {exp_name}: Loss={result['final_loss']:.4f}, PPL={result['final_perplexity']:.2f}")
        except Exception as e:
            print(f"❌ {exp_name} failed: {e}")
    
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_comparison(results)
    print("Ablation study completed!")

def plot_comparison(results):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    names = list(results.keys())
    perplexities = [results[name]['final_perplexity'] for name in names]
    bars = plt.bar(names, perplexities, color=['skyblue' if 'base' in name else 'lightcoral' for name in names])
    plt.xticks(rotation=45)
    plt.ylabel('Final Perplexity')
    plt.title('Ablation Study - Final Performance')
    
    for bar, ppl in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{ppl:.1f}', 
                ha='center', va='bottom')
    
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=name, linewidth=2 if 'base' in name else 1)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Validation Loss Curves')
    
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name, linewidth=2 if 'base' in name else 1)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Training Loss Curves')
    
    plt.tight_layout()
    plt.savefig('results/ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
