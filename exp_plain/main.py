import torch
import torch.multiprocessing as mp
import numpy as np
import random
import os
import yaml
import argparse
import sys
import gc
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.poison_loader import PoisonLoader
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import check_result_exists, save_result_with_config, plot_comparison_curves, get_result_filename
from entity.Server import Server
from entity.Client import Client

# 强制设置多进程启动方式
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def get_device(config_device):
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)

# =======================================================
# 配置打印辅助函数
# =======================================================
def print_configuration_summary(mode_name, current_mode_config, full_config):
    """打印详细的实验配置清单"""
    print("\n" + "=" * 65)
    print(f"  Configuration Summary | Mode: {mode_name}")
    print("=" * 65)
    
    exp = full_config['experiment']
    print(f" [Experiment]")
    print(f"    Device         : {exp.get('device', 'auto')}")
    print(f"    Seed           : {exp['seed']}")
    concurrency_type = "Multiprocessing" if exp.get('use_multiprocessing', False) else "Threading"
    print(f"    Concurrency    : {concurrency_type} (Workers: {exp.get('thread_count', 1)})")
    
    fed = full_config['federated']
    print(f" [Federated]")
    print(f"    Rounds         : {fed['comm_rounds']}")
    print(f"    Clients        : {fed['total_clients']} (Active: {fed['active_clients']})")
    print(f"    Local Epochs   : {fed['local_epochs']}")
    print(f"    Batch Size     : {fed['batch_size']}")
    print(f"    Learning Rate  : {fed.get('lr', 0.01)}")
    
    data = full_config['data']
    print(f" [Data]")
    print(f"    Dataset        : {data['dataset']}")
    print(f"    Model          : {data['model']}")
    print(f"    Non-IID        : {data['if_noniid']} (Alpha: {data['alpha']})")
    
    print(f" [Attack]")
    poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    print(f"    Poison Ratio   : {poison_ratio}")
    if poison_ratio > 0:
        attacks = full_config['attack'].get('active_attacks', [])
        print(f"    Active Attacks : {attacks}")
        for atk in attacks:
            params = full_config['attack'].get('params', {}).get(atk, {})
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"      - {atk:<12}: {param_str}")
    else:
        print(f"    Active Attacks : None (Benign)")

    print(f" [Defense]")
    method = current_mode_config.get('defense_method', 'none')
    print(f"    Method         : {method}")
    if method != 'none' and 'defense' in full_config:
        def_conf = full_config['defense']
        print(f"    Projection Dim : {def_conf.get('projection_dim', 1024)}")
        print(f"    Target Layers  : {def_conf.get('target_layers', [])}")
        print(f"    Params         :")
        for k, v in def_conf.get('params', {}).items():
            print(f"      {k:<22}: {v}")
    
    print("=" * 65 + "\n")

# =======================================================
# 核心任务函数 (修复版)
# =======================================================
def task_client_train_and_project(client, global_params_cpu, proj_path, target_layers, device_str):
    """
    Phase 1: 本地训练 & 投影
    """
    try:
        # 1. 在子进程内重建设备对象
        device = torch.device(device_str)
        
        # 2. [修复点] 初始化模型
        # 如果 model 为 None (首次运行或多进程不共享状态)，则先实例化
        if client.model is None:
            client.model = client.model_class().to(device)
        else:
            client.model = client.model.to(device)
        
        # 3. 接收参数 (将 CPU 参数转到 GPU)
        # receive_model_and_proj 内部会调用 load_state_dict
        # 我们需要确保传入的 params 已经在目标 device 上，或者 rely on load_state_dict 的自动处理(通常需要同一 device)
        global_params_device = {k: v.to(device) for k, v in global_params_cpu.items()}
        
        client.receive_model_and_proj(global_params_device, proj_path)
        
        # 4. 本地训练
        _ = client.local_train()
        
        # 5. 生成投影
        feature_dict = client.generate_gradient_projection(target_layers=target_layers)
        
        # 6. [关键] 将结果转回 CPU
        feature_dict_cpu = {
            'full': feature_dict['full'].cpu(),
            'layers': {}
        }
        if 'layers' in feature_dict:
            for k, v in feature_dict['layers'].items():
                feature_dict_cpu['layers'][k] = v.cpu()
                
        data_size = len(client.dataloader.dataset)
        
        # 7. 清理显存
        client.model.cpu() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return client.client_id, feature_dict_cpu, data_size, None
        
    except Exception as e:
        # 打印完整堆栈以便调试
        import traceback
        traceback.print_exc()
        return client.client_id, None, 0, str(e)

def task_client_upload(client, weight):
    """Phase 2: 计算加权参数"""
    try:
        if weight > 0:
            # 在 task_client_train_and_project 结束时模型已经回到了 CPU
            # 所以这里直接在 CPU 上操作即可
            weighted_params = client.prepare_upload_weighted_params(weight)
            # 确保是 CPU 张量
            weighted_params_cpu = {k: v.cpu() for k, v in weighted_params.items()}
            return client.client_id, weighted_params_cpu, None
        else:
            return client.client_id, None, None
    except Exception as e:
        return client.client_id, None, str(e)

def run_single_mode(full_config, mode_name, current_mode_config):
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    attack_conf = full_config['attack']
    exp_conf = full_config['experiment']
    
    verbose = exp_conf.get('verbose', False)
    log_interval = exp_conf.get('log_interval', 100)
    worker_count = exp_conf.get('thread_count', 1)
    use_multiprocessing = exp_conf.get('use_multiprocessing', False)
    
    # 结果检查
    exists, data = check_result_exists(
        save_dir=exp_conf['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config
    )
    if exists:
        print(f"模式 {mode_name} 已完成，直接返回结果。")
        return np.array(data['accuracy_history'])

    device = get_device(exp_conf.get('device', 'auto'))
    device_str = str(device)

    # 日志路径
    log_file_path = None
    defense_name = current_mode_config['defense_method']
    if any(k in defense_name for k in ["mesas", "projected", "layers_proj"]):
        log_filename = get_result_filename(
            mode_name, data_conf['model'], data_conf['dataset'], defense_name, current_mode_config
        ).replace('.npz', '_detection_log.csv')
        log_file_path = os.path.join(exp_conf['save_dir'], log_filename)

    # 数据准备
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )

    # Server 初始化
    model_class = LeNet5 if data_conf['model'] == 'lenet5' else CIFAR10Net
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    server = Server(
        init_model, 
        detection_method=current_mode_config['defense_method'], 
        defense_config=full_config['defense'],
        seed=exp_conf['seed'],
        verbose=verbose,
        log_file_path=log_file_path
    )
    
    config_proj_dim = full_config['defense'].get('projection_dim', 1024)
    final_output_dim = min(config_proj_dim, model_param_dim)
    matrix_path = f"proj/projection_matrix_{data_conf['dataset']}_{data_conf['model']}_{final_output_dim}.pt"
    server.generate_projection_matrix(model_param_dim, final_output_dim, matrix_path)

    # Client 初始化
    poison_client_ids = []
    current_poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    if current_poison_ratio > 0:
        poison_client_ids = random.sample(range(fed_conf['total_clients']), int(fed_conf['total_clients'] * current_poison_ratio))
    
    clients = []
    active_attacks = attack_conf.get('active_attacks', [])
    attack_params_dict = attack_conf.get('params', {})
    attack_idx = 0
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
        else:
            poison_loader = PoisonLoader([], {})

        clients.append(Client(cid, all_client_dataloaders[cid], model_class, poison_loader, verbose=verbose, log_interval=log_interval))

    # 打印配置
    print_configuration_summary(mode_name, current_mode_config, full_config)
    print(f"  [Init] Projection Matrix: {model_param_dim} -> {final_output_dim}")
    print("\n>>> Training Start...")

    # 选择执行器
    if worker_count <= 1:
        use_multiprocessing = False
    ExecutorClass = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    accuracy_history = []
    asr_history = []
    total_rounds = fed_conf['comm_rounds']
    target_layers_config = full_config['defense'].get('target_layers', [])
    
    start_time = time.time()
    
    for r in range(1, total_rounds + 1):
        global_params, proj_path = server.get_global_params_and_proj()
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        
        # 准备传给子进程的参数 (必须是 CPU 版本)
        if use_multiprocessing:
            global_params_for_worker = {k: v.cpu() for k, v in global_params.items()}
        else:
            global_params_for_worker = global_params
            
        round_results_buffer = {}
        
        # >>> Phase 1: 并行本地训练 <<<
        with ExecutorClass(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    task_client_train_and_project, 
                    clients[cid], 
                    global_params_for_worker, 
                    proj_path, 
                    target_layers_config,
                    device_str
                ): cid 
                for cid in active_ids
            }
            
            for future in as_completed(futures):
                cid, feature_dict, data_size, err = future.result()
                if err:
                    print(f"  [Error] Client {cid} training failed: {err}")
                else:
                    round_results_buffer[cid] = (feature_dict, data_size)
        
        # 整理数据
        round_features = []
        round_data_sizes = []
        for cid in active_ids:
            if cid in round_results_buffer:
                feat, size = round_results_buffer[cid]
                # 确保特征在 Server 计算时回到 GPU
                feat_device = {}
                feat_device['full'] = feat['full'].to(device)
                if 'layers' in feat:
                    feat_device['layers'] = {k: v.to(device) for k, v in feat['layers'].items()}
                
                round_features.append(feat_device)
                round_data_sizes.append(size)
            else:
                round_features.append({})
                round_data_sizes.append(0)

        # >>> Phase 2: 检测 <<<
        weights_map = server.calculate_weights(active_ids, round_features, round_data_sizes, current_round=r)
        
        # >>> Phase 3: 并行加权上传 <<<
        round_models_buffer = {}
        with ExecutorClass(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    task_client_upload, 
                    clients[cid], 
                    weights_map.get(cid, 0.0)
                ): cid 
                for cid in active_ids
            }
            
            for future in as_completed(futures):
                cid, w_params, err = future.result()
                if err:
                    print(f"  [Error] Client {cid} upload failed: {err}")
                else:
                    round_models_buffer[cid] = w_params

        valid_models = []
        valid_ids = []
        for cid in active_ids:
            m = round_models_buffer.get(cid)
            if m is not None:
                m_device = {k: v.to(device) for k, v in m.items()}
                valid_models.append(m_device)
                valid_ids.append(cid)
        
        # >>> Phase 4: 聚合 & 评估 <<<
        server.update_global_model(valid_models, valid_ids)
        acc = server.evaluate(test_loader)
        accuracy_history.append(acc)
        
        asr_str = ""
        if current_poison_ratio > 0 and primary_attack_type in ["label_flip", "backdoor"]:
            asr = server.evaluate_asr(test_loader, primary_attack_type, primary_attack_params)
            asr_history.append(asr)
            asr_str = f" | ASR: {asr:.2f}%"
        
        current_time = time.time()
        elapsed = current_time - start_time
        print(f"  Round {r}/{total_rounds} | Accuracy: {acc:.2f}%{asr_str} | Time: {elapsed:.1f}s")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 结束
    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "="*65)
    print(f"  Experiment Report | Mode: {mode_name}")
    print("="*65)
    print(f"  Total Time     : {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"  Final Accuracy : {accuracy_history[-1]:.2f}%")
    if asr_history:
        print(f"  Final ASR      : {asr_history[-1]:.2f}%")
        
    if server.detection_history:
        print("\n  [Defense Statistics]")
        print(f"  {'Client':<8} {'Suspect':<10} {'Kicked':<10} {'Latest Events'}")
        print("  " + "-"*60)
        for cid, stats in sorted(server.detection_history.items()):
            if stats['suspect_cnt'] > 0 or stats['kicked_cnt'] > 0:
                events_str = ",".join(stats['events'][-3:])
                print(f"  {cid:<8} {stats['suspect_cnt']:<10} {stats['kicked_cnt']:<10} {events_str}...")
    print("="*65 + "\n")

    save_result_with_config(
        save_dir=exp_conf['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config,
        accuracy_history=accuracy_history,
        asr_history=asr_history
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
    
    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed'],
        'model_type': config['data']['model'],
        'dataset_type': config['data']['dataset']
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