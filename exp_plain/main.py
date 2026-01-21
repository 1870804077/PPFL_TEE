import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import gc
import random
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.LSH_proj_extra import SuperBitLSH
from _utils_.poison_loader import PoisonLoader
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import check_result_exists, save_result_with_config, plot_comparison_curves
from entity.Server import Server
from entity.Client import Client

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. 攻击策略配置 (MESAS 对齐)
# =============================================================================
# 这里定义各种攻击的具体参数，修改此处即可调整实验设置
ATTACK_PARAMS_CONFIG = {
    "random_poison": {
        "noise_std": 0.5
    },
    "label_flip": {
        # MESAS 设置: Source-to-Target Flip
        "source_class": 5,  # 将所有 5 (例如 Dog)
        "target_class": 7,  # 翻转为 7 (例如 Horse)
        "scale_update": True,
        "scale_factor": 2.0 # 尝试放大更新以对抗聚合
    },
    "backdoor": {
        # MESAS 设置: Pixel Trigger, PDR=0.1
        "backdoor_ratio": 0.1,
        "backdoor_target": 0,
        "trigger_size": 3,
        "scale_update": True,
        "scale_factor": 3.0 # Train-and-Scale 策略
    },
    "model_compress": {
        "compress_ratio": 0.95
    },
    "gradient_amplify": {
        "amplify_factor": 5.0
    },
    "gradient_inversion": {
        "inversion_strength": 1.0
    },
    "feature_poison": {
        "poison_strength": 0.3,
        "perturb_dim": 100
    },
    "batch_poison": {
        "poison_ratio": 0.2,
        "batch_noise_std": 0.1
    }
}

# =============================================================================
# 2. 客户端类 (Client)
# =============================================================================


# =============================================================================
# 3. 服务端类 (Server)
# =============================================================================


# =============================================================================
# 4. 单模式训练流程 (Core Logic)
# =============================================================================
def run_single_mode(model_type, dataset_type, config, mode_name, detection_method, seed):
    """
    运行单个实验模式（例如：有攻击无防御、有攻击有防御）
    """
    # 1. 检查结果是否存在
    exists, acc_history = check_result_exists(
        save_dir="results",
        mode_name=mode_name,
        model_type=model_type,
        dataset_type=dataset_type,
        detection_method=detection_method,
        config=config
    )
    if exists:
        print(f"模式 {mode_name} 结果已存在，跳过训练。")
        return np.array(acc_history)

    # 2. 数据准备
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=dataset_type,
        num_clients=config['total_clients'],
        batch_size=config['batch_size'],
        if_noniid=config['if_noniid'],
        alpha=config['alpha'],
        data_dir="./data"
    )

    # 3. 模型与服务端初始化
    model_class = LeNet5 if model_type == 'lenet5' else CIFAR10Net
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    server = Server(init_model, detection_method=detection_method, seed=seed)
    matrix_path = f"proj/projection_matrix_{dataset_type}_{model_type}.pt"
    # LSH 维度通常设为 1024 或更小
    server.generate_projection_matrix(model_param_dim, min(1024, model_param_dim), matrix_path)

    # 4. 客户端初始化与角色分配
    poison_client_ids = []
    # 只有在非 pure_training 模式且 poison_ratio > 0 时才设置恶意客户端
    if config['poison_ratio'] > 0:
        poison_client_ids = random.sample(
            range(config['total_clients']), 
            int(config['total_clients'] * config['poison_ratio'])
        )
    
    clients = []
    attack_idx = 0
    attack_types = config.get('attack_types', [])

    for cid in range(config['total_clients']):
        poison_loader = None
        
        if cid in poison_client_ids and attack_types:
            # 轮询分配攻击类型
            a_type = attack_types[attack_idx % len(attack_types)]
            attack_idx += 1
            
            # 从全局配置获取具体参数 (Scale, SourceClass 等)
            a_params = ATTACK_PARAMS_CONFIG.get(a_type, {})
            poison_loader = PoisonLoader([a_type], a_params)
            
            print(f"  [Client {cid}] 设为恶意: {a_type} (参数: {a_params})")
        else:
            # 正常客户端 (默认 PoisonLoader 为空)
            poison_loader = PoisonLoader([], {})

        clients.append(Client(cid, all_client_dataloaders[cid], model_class, poison_loader))

    # 5. 联邦学习主循环
    accuracy_history = []
    total_rounds = config['comm_rounds']
    
    print(f"\n开始训练: {mode_name} | 总轮数: {total_rounds}")
    
    for r in range(1, total_rounds + 1):
        # 5.1 参数分发
        global_params, proj_path = server.get_global_params_and_proj()

        # 5.2 客户端选择
        active_ids = random.sample(range(config['total_clients']), config['active_clients'])
        
        # 5.3 客户端本地训练
        for cid in active_ids:
            client = clients[cid]
            client.receive_model_and_proj(global_params, proj_path)
            
            # 训练并获取参数和梯度特征
            trained_params, grad_flat = client.local_train()
            feature = client.extract_gradient_feature(grad_flat)
            data_size = len(client.dataloader.dataset)
            
            # 上传至服务器
            server.receive_client_upload(trained_params, data_size, feature, cid)
            
            # 及时释放本地显存
            del grad_flat

        # 5.4 服务端聚合
        server.aggregate()

        # 5.5 评估与记录
        acc = server.evaluate(test_loader)
        accuracy_history.append(acc)
        
        if r % 10 == 0 or r == 1:
            print(f"  Round {r}/{total_rounds} | Accuracy: {acc:.2f}%")
        
        # 垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 保存结果
    save_result_with_config(
        save_dir="results",
        mode_name=mode_name,
        model_type=model_type,
        dataset_type=dataset_type,
        detection_method=detection_method,
        config=config,
        accuracy_history=accuracy_history
    )

    return np.array(accuracy_history)

# =============================================================================
# 5. 实验入口 (Entry Point)
# =============================================================================
def main_train(
    model_type='lenet5',
    dataset_type='mnist',
    detection_method="lsh_score_kickout",
    save_dir="results",
    **kwargs
):
    """统一的主训练入口"""
    # 基础配置
    config = {
        'lr': 0.01,
        'local_epochs': 5,
        'comm_rounds': 100,
        'total_clients': 20,
        'active_clients': 20,
        'poison_ratio': 0.2,
        'batch_size': 64,
        'if_noniid': False,
        'alpha': 0.1,
        'detection_method': detection_method,
        'model_type': model_type,
        'dataset_type': dataset_type,
        'attack_types': ["random_poison"], 
        'seed': 42
    }
    config.update(kwargs)

    print("="*60)
    print(f"PPFL_TEE")
    print(f"  Model: {model_type} | Dataset: {dataset_type}")
    print(f"  Detection: {detection_method}")
    print(f"  Attacks: {config['attack_types']}")
    print("="*60)

    # 定义三个对比组
    modes = [
        # 1. 纯净训练 (基准)
        {
            'name': 'pure_training',
            'config': {**config, 'poison_ratio': 0.0, 'attack_types': []},
            'detection_method': 'none'
        },
        # 2. 投毒 + 无防御
        {
            'name': 'poison_no_detection',
            'config': {**config},
            'detection_method': 'none'
        },
        # 3. 投毒 + 有防御 (本方案)
        {
            'name': 'poison_with_detection',
            'config': {**config},
            'detection_method': detection_method
        }
    ]

    all_results = {}
    for mode in modes:
        print(f"\n>>> Running Mode: {mode['name']}")
        hist = run_single_mode(
            model_type=model_type,
            dataset_type=dataset_type,
            config=mode['config'],
            mode_name=mode['name'],
            detection_method=mode['detection_method'],
            seed=config['seed']
        )
        all_results[mode['name']] = hist

    # 绘图
    plot_path = os.path.join(save_dir, f"comparison_{detection_method}_{dataset_type}.png")
    plot_comparison_curves(config, save_dir, plot_path)
    print(f"\n实验完成! 对比图已保存至: {plot_path}")

if __name__ == "__main__":
    # 在这里配置想要运行的实验参数
    
    # 1. 选择数据集和模型
    model_type='lenet5', dataset_type='mnist'
    # model_type='cifar10', dataset_type='cifar10'
    
    # 2. 选择攻击组合
    attacks = ["label_flip"] 
    
    # 3. 运行
    main_train(
        model_type='cifar10',
        dataset_type='cifar10',
        detection_method="lsh_score_kickout", 
        comm_rounds=100,                      
        poison_ratio=0.2,                    
        attack_types=attacks,
        if_noniid=False,
        alpha=0.5
    )