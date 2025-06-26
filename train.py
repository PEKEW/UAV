import torch
import argparse
import logging
from pathlib import Path
import yaml

from src.models import BatteryAnomalyNet, FlightAnomalyNet
from src.data import DualDomainDataLoader
from src.training import DualDomainTrainer


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return {
            'data': {
                'battery_h5_path': 'processed/evtol_anomaly_dataset_3d.h5',
                'flight_h5_path': 'processed/flight_anomaly_dataset_3d.h5',
                'battery_data_path': 'src/data/evtol_anomaly_dataset.csv',
                'flight_data_path': 'src/data/flight_anomaly_dataset.csv',
                'batch_size': 32,
                'num_workers': 4,
                'battery_augmentation': {
                    'enabled': True,
                    'time_jitter_prob': 0.4,
                    'gaussian_noise_prob': 0.6,
                    'voltage_fluctuation_prob': 0.4,
                    'noise_std': 0.01
                },
                'flight_augmentation': {
                    'enabled': True,
                    'trajectory_warp_prob': 0.3,
                    'attitude_noise_prob': 0.4,
                    'velocity_perturb_prob': 0.3
                }
            },
            'battery_model': {
                'sequence_length': 30,
                'input_features': 7,
                'num_classes': 2,
                'cnn_channels': [16, 32, 64],
                'lstm_hidden': 64,
                'attention_heads': 2,
                'classifier_hidden': [32],
                'dropout_rate': 0.5
            },
            'flight_model': {
                'sequence_length': 30,
                'input_features': 9,
                'num_classes': 2,
                'cnn_channels': [96, 192, 384],
                'lstm_hidden': 256,
                'attention_heads': 8,
                'classifier_hidden': [128, 64]
            },
            'training': {
                'battery_config': {
                    'epochs': 100,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-4,
                    'loss_function': 'crossentropy',
                    'focal_alpha': 0.25,
                    'focal_gamma': 2.0,
                    'scheduler': 'cosine',
                    'warmup_epochs': 15,
                    'early_stopping': True,
                    'patience': 20,
                    'use_amp': True,
                    'gradient_clip': 0.5,
                    'accumulation_steps': 2,
                    'min_lr': 1e-6,
                    'save_dir': 'checkpoints/battery',
                    'use_tensorboard': True  # 启用TensorBoard
                },
                'flight_config': {
                    'epochs': 100,
                    'learning_rate': 8e-4,
                    'weight_decay': 5e-4,
                    'loss_function': 'label_smoothing',
                    'label_smoothing': 0.1,
                    'scheduler': 'step',
                    'step_size': 10,
                    'gamma': 0.7,
                    'early_stopping': True,
                    'patience': 12,
                    'use_amp': True,
                    'gradient_clip': 1.0,
                    'save_dir': 'checkpoints/flight',
                    'use_tensorboard': True 
                }
            }
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双域异常检测模型训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--wandb', action='store_true', help='使用wandb记录')
    parser.add_argument('--tensorboard', action='store_true', help='使用TensorBoard记录')
    parser.add_argument('--no-tensorboard', action='store_true', help='禁用TensorBoard记录')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = load_config(args.config)
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    
    config['training']['battery_config']['device'] = device
    config['training']['flight_config']['device'] = device
    config['training']['battery_config']['use_wandb'] = args.wandb
    config['training']['flight_config']['use_wandb'] = args.wandb
    
    if args.tensorboard:
        config['training']['battery_config']['use_tensorboard'] = True
        config['training']['flight_config']['use_tensorboard'] = True
        logger.info("启用TensorBoard监控")
    elif args.no_tensorboard:
        config['training']['battery_config']['use_tensorboard'] = False
        config['training']['flight_config']['use_tensorboard'] = False
        logger.info("禁用TensorBoard监控")
    
    try:
        logger.info("初始化数据加载器...")
        data_loader = DualDomainDataLoader(config['data'])
        
        battery_loaders = data_loader.get_battery_loaders()
        
        # flight_loaders = data_loader.get_flight_loaders()
        flight_loaders = None
        
        if battery_loaders[0] is None:
            logger.warning("电池数据不可用，跳过电池模型训练")
        if flight_loaders is None or flight_loaders[0] is None:
            logger.warning("飞行数据不可用，跳过飞行模型训练")
        
        if battery_loaders[0] is None and flight_loaders[0] is None:
            logger.error("没有可用的数据，退出训练")
            return
        
        data_info = data_loader.get_data_info()
        logger.info(f"数据信息: {data_info}")
        
        logger.info("初始化模型...")
        
        battery_model = None
        flight_model = None
        
        if battery_loaders and battery_loaders[0] is not None:
            battery_model = BatteryAnomalyNet(config['battery_model'])
            logger.info(f"电池模型参数数量: {sum(p.numel() for p in battery_model.parameters()):,}")
        
        if flight_loaders and flight_loaders[0] is not None:
            flight_model = FlightAnomalyNet(config['flight_model'])
            logger.info(f"飞行模型参数数量: {sum(p.numel() for p in flight_model.parameters()):,}")
        
        logger.info("初始化训练器...")
        trainer = DualDomainTrainer(
            battery_model=battery_model,
            flight_model=flight_model,
            config=config['training']
        )
        
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            # TODO: 实现检查点恢复逻辑
        
        logger.info("开始训练...")
        trainer.train(battery_loaders, flight_loaders)
        
        summary = trainer.get_training_summary()
        logger.info("训练完成！")
        logger.info(f"训练总结: {summary}")
        
        if args.config != 'config.yaml':
            final_config_path = Path(config['training']['battery_config']['save_dir']).parent / 'final_config.yaml'
            final_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {final_config_path}")
        
        if config['training']['battery_config'].get('use_tensorboard', False):
            tensorboard_dir = Path(config['training']['battery_config']['save_dir']).parent / 'tensorboard_logs'
            logger.info(f"TensorBoard日志已保存到: {tensorboard_dir}")
            logger.info(f"启动TensorBoard: tensorboard --logdir={tensorboard_dir}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise
    finally:
        if 'data_loader' in locals():
            data_loader.cleanup()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()