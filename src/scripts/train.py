import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import get_dataset
from src.models.model import LSTM
from src.trainer.trainer import LSTMTrainer

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" 
                        if config['train']['device'] == "cuda" 
                        and torch.cuda.is_available() 
                        else "cpu")
    
    train_dataset, val_dataset, _ = get_dataset(
        data_path=config['data']['path'],
        sequence_length=config['data']['sequence_length'],
        prediction_steps=config['data']['prediction_steps'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        prediction_targets=config['model']['prediction_targets']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    
    model = LSTM(config)
    model = model.to(device)
    model.train()
    
    trainer = LSTMTrainer(model, config)
    
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
if __name__ == "__main__":
    main()