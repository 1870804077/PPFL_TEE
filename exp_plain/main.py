import torch
import numpy as np
import random
import os
import yaml
import argparse
import sys
import gc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.poison_loader import PoisonLoader
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import check_result_exists, save_result_with_config, plot_comparison_curves
from entity.Server import Server
from entity.Client import Client

def get_device(config_device):
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)

def run_single_mode(full_config, mode_name, current_mode_config):
    device = get_device(full_config['experiment'].get('device', 'auto'))
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    attack_conf = full_config['attack']
    
    # --- 结果检查 ---
    exists, acc_history = check_result_exists(
        save_dir=full_config['experiment']['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config
    )
    if exists:
        return np.array(acc_history)

    # --- 数据加载 ---
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )

    # --- 步骤 2: Server 启动并生成初始化模型 ---
    model_class = LeNet5 if data_conf['model'] == 'lenet5' else CIFAR10Net
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    server = Server(
        init_model, 
        detection_method=current_mode_config['defense_method'], 
        seed=full_config['experiment']['seed']
    )
    
    # --- 步骤 1: 生成投影矩阵 ---
    matrix_path = f"proj/projection_matrix_{data_conf['dataset']}_{data_conf['model']}.pt"
    # 这里默认生成 1024 维，如果需要分层，底层 SuperBitLSH 还是加载这个全量矩阵，
    # 客户端 extract_feature 时通过 start_idx 访问
    server.generate_projection_matrix(model_param_dim, min(1024, model_param_dim), matrix_path)

    # --- 客户端初始化 ---
    poison_client_ids = []
    current_poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    if current_poison_ratio > 0:
        poison_client_ids = random.sample(
            range(fed_conf['total_clients']), 
            int(fed_conf['total_clients'] * current_poison_ratio)
        )
    
    clients = []
    active_attacks = attack_conf.get('active_attacks', [])
    attack_params_dict = attack_conf.get('params', {})
    attack_idx = 0

    for cid in range(fed_conf['total_clients']):
        poison_loader = None
        if cid in poison_client_ids and active_attacks:
            a_type = active_attacks[attack_idx % len(active_attacks)]
            attack_idx += 1
            a_params = attack_params_dict.get(a_type, {})
            poison_loader = PoisonLoader([a_type], a_params)
        else:
            poison_loader = PoisonLoader([], {})
        clients.append(Client(cid, all_client_dataloaders[cid], model_class, poison_loader))

    # --- 训练主循环 ---
    accuracy_history = []
    total_rounds = fed_conf['comm_rounds']
    
    print(f"\n开始训练: {mode_name} | 轮数: {total_rounds}")
    
    for r in range(1, total_rounds + 1):
        # --- 步骤 3: Server 发送模型参数 ---
        global_params, proj_path = server.get_global_params_and_proj()
        
        # 客户端选择
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        
        # 暂存本轮数据
        round_features = []
        round_data_sizes = []
        round_weighted_models = []
        
        # >>> Phase 1: 训练与检测 <<<
        for cid in active_ids:
            client = clients[cid]
            client.receive_model_and_proj(global_params, proj_path)
            
            # --- 步骤 4: 客户端执行本地训练 ---
            # 返回梯度/update用于投影，但不立即上传参数
            _ = client.local_train() 
            
            # --- 步骤 5: 客户端执行投影 ---
            # 这里 start_idx=0 表示对全量参数投影。
            # 如果需要只对某一层投影，可以在这里修改 start_idx 和 LSH 内部逻辑
            feature = client.generate_gradient_projection(start_idx=0)
            
            # --- 步骤 6: 上传投影至 Server (暂存) ---
            round_features.append(feature)
            round_data_sizes.append(len(client.dataloader.dataset))
        
        # --- 步骤 7: Server 返回检测结果(权重) ---
        # Server 根据收集到的所有 feature 计算权重
        weights_map = server.calculate_weights(active_ids, round_features, round_data_sizes)
        
        # >>> Phase 2: 加权上传与聚合 <<<
        for cid in active_ids:
            client = clients[cid]
            w = weights_map.get(cid, 0.0)
            
            # --- 步骤 8: 客户端计算全量参数 * 权重并上传 ---
            # 如果被剔除 (w=0)，理论上可以不传以节省带宽，但这里逻辑统一处理
            weighted_params = client.prepare_upload_weighted_params(w)
            round_weighted_models.append(weighted_params)
            
        # --- 步骤 9: Server 聚合更新 ---
        server.update_global_model(round_weighted_models, active_ids)
        
        # --- 步骤 10: Server 准备下一轮 (Loop continues) ---
        
        # 评估
        acc = server.evaluate(test_loader)
        accuracy_history.append(acc)
        
        if r % 10 == 0 or r == 1:
            print(f"  Round {r} | Acc: {acc:.2f}%")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存
    save_result_with_config(
        save_dir=full_config['experiment']['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config,
        accuracy_history=accuracy_history
    )
    return np.array(accuracy_history)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 构建对比模式配置 (保持之前的逻辑)
    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed']
    }
    
    default_poison_ratio = config['attack']['poison_ratio']
    default_defense = config['defense']['method']

    modes = [
        {'name': 'pure_training', 'mode_config': {**base_flat_config, 'poison_ratio': 0.0, 'defense_method': 'none'}},
        {'name': 'poison_no_detection', 'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': 'none'}},
        {'name': 'poison_with_detection', 'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': default_defense}}
    ]

    for mode in modes:
        print(f"\n>>> Mode: {mode['name']}")
        run_single_mode(config, mode['name'], mode['mode_config'])

    # 绘图
    plot_path = os.path.join(config['experiment']['save_dir'], f"comparison_{default_defense}_{config['data']['dataset']}.png")
    plot_config = {**base_flat_config, 'poison_ratio': default_poison_ratio}
    plot_comparison_curves(plot_config, config['experiment']['save_dir'], plot_path)

if __name__ == "__main__":
    main()