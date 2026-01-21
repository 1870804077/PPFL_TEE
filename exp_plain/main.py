import torch
import numpy as np
import random
import os
import yaml
import argparse
import sys

# 设置项目根目录以便导入模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.poison_loader import PoisonLoader
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import check_result_exists, save_result_with_config, plot_comparison_curves
from entity.Server import Server
from entity.Client import Client

# 设备配置
def get_device(config_device):
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)

# =============================================================================
# 1. 单模式运行逻辑
# =============================================================================
def run_single_mode(full_config, mode_name, current_mode_config):
    """
    运行单个实验模式
    :param full_config: 完整的原始配置字典 (用于保存记录)
    :param mode_name: 当前模式名称 (如 pure_training)
    :param current_mode_config: 当前模式的具体配置 (即修改了 poison_ratio 等参数后的扁平配置)
    """
    device = get_device(full_config['experiment'].get('device', 'auto'))
    
    # 提取常用参数
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    attack_conf = full_config['attack']
    
    # 1. 检查结果是否存在
    exists, acc_history = check_result_exists(
        save_dir=full_config['experiment']['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config # 传入合并后的配置用于哈希校验
    )
    if exists:
        print(f"模式 {mode_name} 结果已存在，跳过训练。")
        return np.array(acc_history)

    # 2. 数据准备
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )

    # 3. 模型与服务端初始化
    model_class = LeNet5 if data_conf['model'] == 'lenet5' else CIFAR10Net
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    # 初始化服务端
    server = Server(
        init_model, 
        detection_method=current_mode_config['defense_method'], 
        seed=full_config['experiment']['seed']
    )
    
    # LSH 投影矩阵路径
    matrix_path = f"proj/projection_matrix_{data_conf['dataset']}_{data_conf['model']}.pt"
    server.generate_projection_matrix(model_param_dim, min(1024, model_param_dim), matrix_path)

    # 4. 客户端初始化与攻击分配
    poison_client_ids = []
    current_poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    
    if current_poison_ratio > 0:
        poison_client_ids = random.sample(
            range(fed_conf['total_clients']), 
            int(fed_conf['total_clients'] * current_poison_ratio)
        )
    
    clients = []
    attack_idx = 0
    # 获取启用的攻击类型
    active_attacks = attack_conf.get('active_attacks', [])
    # 获取具体的攻击参数字典
    attack_params_dict = attack_conf.get('params', {})

    for cid in range(fed_conf['total_clients']):
        poison_loader = None
        
        # 只有在恶意ID列表里，且当前配置允许攻击时才注入 PoisonLoader
        if cid in poison_client_ids and active_attacks:
            # 轮询分配攻击类型
            a_type = active_attacks[attack_idx % len(active_attacks)]
            attack_idx += 1
            
            # 从配置中读取该攻击的具体参数
            a_params = attack_params_dict.get(a_type, {})
            poison_loader = PoisonLoader([a_type], a_params)
            
            print(f"  [Client {cid}] 恶意客户端: {a_type} (参数: {a_params})")
        else:
            # 正常客户端
            poison_loader = PoisonLoader([], {})

        clients.append(Client(cid, all_client_dataloaders[cid], model_class, poison_loader))

    # 5. 联邦学习主循环
    accuracy_history = []
    total_rounds = fed_conf['comm_rounds']
    
    print(f"\n开始训练: {mode_name} | 总轮数: {total_rounds} | 恶意比例: {current_poison_ratio}")
    
    for r in range(1, total_rounds + 1):
        # 5.1 参数分发
        global_params, proj_path = server.get_global_params_and_proj()

        # 5.2 客户端选择
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        
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
            
            # 释放内存
            del grad_flat

        # 5.4 服务端聚合
        server.aggregate()

        # 5.5 评估与记录
        acc = server.evaluate(test_loader)
        accuracy_history.append(acc)
        
        if r % 10 == 0 or r == 1:
            print(f"  Round {r}/{total_rounds} | Accuracy: {acc:.2f}%")
        
        # 垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 保存结果
    save_result_with_config(
        save_dir=full_config['experiment']['save_dir'],
        mode_name=mode_name,
        model_type=data_conf['model'],
        dataset_type=data_conf['dataset'],
        detection_method=current_mode_config['defense_method'],
        config=current_mode_config, # 保存当前模式的特定配置
        accuracy_history=accuracy_history
    )

    return np.array(accuracy_history)

# =============================================================================
# 2. 配置文件加载与入口
# =============================================================================
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="联邦学习投毒防御实验 (Configurable)")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    config = load_config(args.config)
    
    # 提取基础信息
    model_type = config['data']['model']
    dataset_type = config['data']['dataset']
    default_defense = config['defense']['method']
    default_poison_ratio = config['attack']['poison_ratio']
    
    print("="*60)
    print(f"PPFL_TEE 实验启动")
    print(f"  Config File: {args.config}")
    print(f"  Model: {model_type} | Dataset: {dataset_type}")
    print(f"  Attacks: {config['attack']['active_attacks']}")
    print("="*60)

    # 2. 定义对比实验组
    # 为了保留原来的对比逻辑，我们需要构建针对每个模式的“特定配置字典”
    # 这个字典会被 save_config 用来生成文件名 hash，所以需要包含区分模式的关键参数
    
    # 提取需要传递给 save_config 的关键参数做扁平化，方便哈希
    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed']
    }

    modes = [
        # 1. 纯净训练
        {
            'name': 'pure_training',
            'mode_config': {
                **base_flat_config,
                'poison_ratio': 0.0,
                'defense_method': 'none'
            }
        },
        # 2. 投毒 + 无防御
        {
            'name': 'poison_no_detection',
            'mode_config': {
                **base_flat_config,
                'poison_ratio': default_poison_ratio,
                'defense_method': 'none'
            }
        },
        # 3. 投毒 + 有防御
        {
            'name': 'poison_with_detection',
            'mode_config': {
                **base_flat_config,
                'poison_ratio': default_poison_ratio,
                'defense_method': default_defense
            }
        }
    ]

    all_results = {}
    for mode in modes:
        print(f"\n>>> Running Mode: {mode['name']}")
        hist = run_single_mode(
            full_config=config,
            mode_name=mode['name'],
            current_mode_config=mode['mode_config']
        )
        all_results[mode['name']] = hist

    # 3. 绘图
    plot_path = os.path.join(
        config['experiment']['save_dir'], 
        f"comparison_{default_defense}_{dataset_type}.png"
    )
    # 使用 base_flat_config 绘图即可，或者合并任意一个 mode config
    plot_config = {**base_flat_config, 'poison_ratio': default_poison_ratio}
    plot_comparison_curves(plot_config, config['experiment']['save_dir'], plot_path)
    
    print(f"\n实验完成! 对比图已保存至: {plot_path}")

if __name__ == "__main__":
    main()