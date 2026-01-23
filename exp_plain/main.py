import torch
import numpy as np
import random
import os
import yaml
import argparse
import sys
import gc
import time

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
    # 提取常用参数
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    attack_conf = full_config['attack']
    
    # 提取日志配置
    verbose = full_config['experiment'].get('verbose', False)
    log_interval = full_config['experiment'].get('log_interval', 100)

    # 1. 结果存在性检查
    exists, acc_history = check_result_exists(
        save_dir=full_config['experiment']['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config
    )
    if exists:
        print(f"模式 {mode_name} 已完成，直接返回结果。")
        return np.array(acc_history)

    device = get_device(full_config['experiment'].get('device', 'auto'))

    # 2. 数据准备
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )

    # 3. 初始化 Server
    model_class = LeNet5 if data_conf['model'] == 'lenet5' else CIFAR10Net
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    server = Server(
        init_model, 
        detection_method=current_mode_config['defense_method'], 
        seed=full_config['experiment']['seed'],
        verbose=verbose
    )
    
    matrix_path = f"proj/projection_matrix_{data_conf['dataset']}_{data_conf['model']}.pt"
    server.generate_projection_matrix(model_param_dim, min(1024, model_param_dim), matrix_path)

    # 4. 初始化 Client
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

    # 记录实际使用的攻击配置用于ASR检测
    primary_attack_type = None
    primary_attack_params = {}

    for cid in range(fed_conf['total_clients']):
        poison_loader = None
        if cid in poison_client_ids and active_attacks:
            a_type = active_attacks[attack_idx % len(active_attacks)]
            attack_idx += 1
            a_params = attack_params_dict.get(a_type, {})
            poison_loader = PoisonLoader([a_type], a_params)
            
            if primary_attack_type is None:
                primary_attack_type = a_type
                primary_attack_params = a_params
            
            if verbose:
                print(f"  [Init] Client {cid} set as Malicious ({a_type})")
        else:
            poison_loader = PoisonLoader([], {})

        clients.append(Client(
            cid, 
            all_client_dataloaders[cid], 
            model_class, 
            poison_loader,
            verbose=verbose,
            log_interval=log_interval
        ))

    # 5. 训练循环
    accuracy_history = []
    asr_history = []
    total_rounds = fed_conf['comm_rounds']
    target_layers_config = full_config['defense'].get('target_layers', [])
    
    print(f"\n>>> 开始训练: {mode_name} | 恶意比例: {current_poison_ratio} | 防御: {current_mode_config['defense_method']}")
    
    start_time = time.time()
    
    for r in range(1, total_rounds + 1):
        global_params, proj_path = server.get_global_params_and_proj()
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        
        round_features = []
        round_data_sizes = []
        round_weighted_models = []
        
        # Phase 1: 本地训练 & 投影
        for cid in active_ids:
            client = clients[cid]
            client.receive_model_and_proj(global_params, proj_path)
            _ = client.local_train()
            feature_dict = client.generate_gradient_projection(target_layers=target_layers_config)
            round_features.append(feature_dict)
            round_data_sizes.append(len(client.dataloader.dataset))
        
        # Phase 2: 检测 & 聚合
        weights_map = server.calculate_weights(active_ids, round_features, round_data_sizes, current_round=r)
        
        for cid in active_ids:
            client = clients[cid]
            w = weights_map.get(cid, 0.0)
            if w > 0:
                weighted_params = client.prepare_upload_weighted_params(w)
                round_weighted_models.append(weighted_params)
            else:
                round_weighted_models.append(None)
                
        valid_models = [m for m in round_weighted_models if m is not None]
        valid_ids = [cid for i, cid in enumerate(active_ids) if round_weighted_models[i] is not None]
        
        server.update_global_model(valid_models, valid_ids)
        
        # 评估
        acc = server.evaluate(test_loader)
        accuracy_history.append(acc)
        
        # ASR 评估
        asr_str = ""
        if current_poison_ratio > 0 and primary_attack_type in ["label_flip", "backdoor"]:
            asr = server.evaluate_asr(test_loader, primary_attack_type, primary_attack_params)
            asr_history.append(asr)
            asr_str = f" | ASR: {asr:.2f}%"
        
        print(f"  Round {r}/{total_rounds} | Accuracy: {acc:.2f}%{asr_str}")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"  实验总结报告 ({mode_name})")
    print("="*50)
    print(f"  总耗时     : {total_time:.2f} seconds ({total_time/60:.2f} mins)")
    print(f"  最终准确率 : {accuracy_history[-1]:.2f}%")
    if asr_history:
        print(f"  最终 ASR   : {asr_history[-1]:.2f}%")
        
    if server.detection_history:
        print("\n  [防御拦截统计]")
        print(f"  {'Client ID':<10} {'Suspect Count':<15} {'Kicked Count':<15} {'Details'}")
        print("  " + "-"*60)
        for cid, stats in sorted(server.detection_history.items()):
            if stats['suspect_cnt'] > 0 or stats['kicked_cnt'] > 0:
                events_str = ",".join(stats['events'][-5:])
                print(f"  {cid:<10} {stats['suspect_cnt']:<15} {stats['kicked_cnt']:<15} {events_str}...")
    print("="*50 + "\n")

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
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    parser.add_argument('--mode', type=str, default=None, help='指定只运行某个模式(逗号分隔)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # [修改点] 补充 model_type 和 dataset_type，确保绘图时能匹配到文件
    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed'],
        'model_type': config['data']['model'],      # 新增
        'dataset_type': config['data']['dataset']   # 新增
    }
    
    default_poison_ratio = config['attack']['poison_ratio']
    default_defense = config['defense']['method']

    all_modes = [
        {
            'name': 'pure_training', 
            'mode_config': {**base_flat_config, 'poison_ratio': 0.0, 'defense_method': 'none'}
        },
        {
            'name': 'poison_no_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': 'none'}
        },
        {
            'name': 'poison_with_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': default_defense}
        }
    ]
    
    target_modes_str = args.mode
    if target_modes_str is None:
        target_modes_str = config['experiment'].get('modes', 'all')
    
    modes_to_run = []
    if target_modes_str == 'all' or not target_modes_str:
        modes_to_run = all_modes
    else:
        target_names = [m.strip() for m in target_modes_str.split(',')]
        modes_to_run = [m for m in all_modes if m['name'] in target_names]

    print(f"计划运行模式: {[m['name'] for m in modes_to_run]}")

    for mode in modes_to_run:
        run_single_mode(config, mode['name'], mode['mode_config'])

    if modes_to_run:
        plot_path = os.path.join(config['experiment']['save_dir'], f"comparison_{default_defense}_{config['data']['dataset']}.png")
        plot_config = {**base_flat_config, 'poison_ratio': default_poison_ratio}
        plot_comparison_curves(plot_config, config['experiment']['save_dir'], plot_path)

if __name__ == "__main__":
    main()