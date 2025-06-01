import yaml
import torch
from src.data.dataset import EVTOLDataset, get_dataset
from src.models.model import LSTM
from src.trainer.trainer import Trainer

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" 
                        if config['train']['device'] == "cuda" 
                        and torch.cuda.is_available() 
                        else "cpu")
    
    train_dataset = get_dataset(config['data']['train_path'], "train")
    val_dataset = get_dataset(config['data']['val_path'], "val")
    
    model = LSTM(config['model']).to(device)
    trainer = Trainer(model, config)
    trainer.train(train_dataset, val_dataset)
    
if __name__ == "__main__":
    main()