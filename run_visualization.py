from pathlib import Path
from visualize_predictions import main as vis_main
import sys

def main():
    model_path = Path("checkpoints/best_model.pth")
    config_path = Path("configs/config.yaml")
    data_path = Path("Datasets/test")
    sys.argv = [
        'visualize_predictions.py',
        '--config', str(config_path),
        '--model', str(model_path),
        '--data', str(data_path),
        '--device', 'auto',
        '--output_dir', 'visualizations'
    ]
    vis_main()

if __name__ == '__main__':
    main() 