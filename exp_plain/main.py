import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import gc
import random
import os
import json

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.LSH_proj_extra import SuperBitLSH
from _utils_.poison_loader import PoisonLoader
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import *

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client:
    def __init__(self, client_id, dataloader, model_class, poison_loader=None):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model_class = model_class
        self.poison_loader = poison_loader or PoisonLoader()
        self.model = None
        self.optimizer = None
        self.superbit_lsh = SuperBitLSH()

    def receive_model_and_proj(self, model_params, projection_matrix_path):
        """接收服务端的模型参数和投影矩阵文件路径"""
        if self.model is None:
            self.model = self.model_class().to(DEVICE)
        self.model.load_state_dict(model_params)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.superbit_lsh.set_projection_matrix_path(projection_matrix_path)

    def local_train(self):
        """本地训练并返回模型参数和梯度"""
        # 使用PoisonLoader执行攻击
        if self.poison_loader is not None and self.poison_loader.attack_methods:
            # 恶意客户端：执行指定的攻击
            trained_params, grad_flat = self.poison_loader.execute_attack(
                self.model, self.dataloader, self.model_class, DEVICE, self.optimizer
            )
        else:
            # 正常客户端：执行标准训练
            import torch.nn as nn
            import copy
            import gc
            
            self.model.train()
            initial_params = copy.deepcopy(self.model.state_dict())
            initial_model = self.model_class().to(DEVICE)
            initial_model.load_state_dict(initial_params)

            criterion = nn.CrossEntropyLoss()
            for epoch in range(5):  # LOCAL_EPOCHS
                for data, target in self.dataloader:
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

            # 计算梯度 - 只对浮点参数计算梯度
            initial_flat = initial_model.get_flat_params()
            trained_flat = self.model.get_flat_params()
            grad_flat = trained_flat - initial_flat

            # 释放内存
            del initial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 应用梯度投毒
            grad_flat = self.poison_loader.apply_gradient_poison(grad_flat)

            # 返回训练后的参数和梯度
            trained_params = copy.deepcopy(self.model.state_dict())
        
        return trained_params, grad_flat


    def extract_gradient_feature(self, grad_flat):
        """提取梯度特征"""
        feature = self.superbit_lsh.extract_feature(grad_flat, batch_size=512)  # 分批处理
        # 应用特征投毒
        feature = self.poison_loader.apply_feature_poison(feature)

        # 释放梯度内存
        del grad_flat
        gc.collect()

        return feature


class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42):
        """服务端类"""
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.client_models = []
        self.client_data_sizes = []
        self.client_features = []
        self.client_ids = []
        self.detection_method = detection_method
        self.seed = seed 
        
        # 根据检测方法初始化组件
        if self.detection_method in ["lsh_score_kickout", "only_score"]:
            self.score_calculator = ScoreCalculator()
        if self.detection_method in ["lsh_score_kickout", "only_kickout"]:
            self.kickout_manager = KickoutManager()

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        """生成投影矩阵并保存到文件"""
        # 如果没有指定路径，使用proj文件夹
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def send_model_and_proj(self):
        """发送模型参数和投影矩阵文件路径"""
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def receive_client_data(self, model_params, data_size, feature, client_id):
        """接收客户端数据"""
        self.client_models.append(model_params)
        self.client_data_sizes.append(data_size)
        self.client_features.append(feature)
        self.client_ids.append(client_id)

    def aggregate_without_detection(self):
        """不带检测的聚合（FedAvg）"""
        if not self.client_models:
            return

        total_data_size = sum(self.client_data_sizes)
        if total_data_size == 0:
            return

        # 获取第一个客户端的模型参数作为模板
        first_params = self.client_models[0]
        agg_params = {}
        
        for key, param in first_params.items():
            # 确保聚合参数的数据类型与原始参数完全一致
            agg_params[key] = torch.zeros_like(param, dtype=param.dtype, device=param.device)

        for i, params in enumerate(self.client_models):
            weight = self.client_data_sizes[i] / total_data_size
            for key in agg_params.keys():
                # 确保权重和参数的数据类型匹配
                client_param = params[key].to(agg_params[key].device)
                
                # 如果参数是整数类型（Long, Int等），权重需要转换为合适的类型
                if agg_params[key].dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                    # 对于整数参数，直接相加或根据需要进行特殊处理
                    # 一般情况下，整数参数不应该参与加权平均，它们通常是索引或计数
                    # 这里我们跳过整数参数的聚合，只聚合浮点参数
                    continue
                else:
                    # 对于浮点参数，进行正常的加权聚合
                    agg_params[key] += client_param.float() * float(weight)

        # 更新全局模型
        self.global_model.load_state_dict(agg_params)
        self._clear_client_data()

    def aggregate_with_detection(self):
        """带检测的聚合"""
        if not self.client_models:
            return

        # 获取第一个客户端的模型参数作为模板
        first_params = self.client_models[0]
        agg_params = {}
        
        for key, param in first_params.items():
            # 确保聚合参数的数据类型与原始参数完全一致
            agg_params[key] = torch.zeros_like(param, dtype=param.dtype, device=param.device)

        if self.detection_method == "lsh_score_kickout":
            # 完整流程：打分 + 剔除
            client_scores = {}
            for i, client_id in enumerate(self.client_ids):
                client_scores[client_id] = self.score_calculator.calculate_scores(
                    client_id, self.client_features[i], self.client_data_sizes[i]
                )
            weights = self.kickout_manager.determine_weights(client_scores)
            
            for i, (client_id, params) in enumerate(zip(self.client_ids, self.client_models)):
                weight = weights.get(client_id, 0.0)
                if weight > 0:
                    for key in agg_params.keys():
                        # 确保权重和参数的数据类型匹配
                        client_param = params[key].to(agg_params[key].device)
                        
                        # 只对浮点参数进行加权聚合
                        if agg_params[key].dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                            continue
                        else:
                            agg_params[key] += client_param.float() * float(weight)
        
        elif self.detection_method == "only_score":
            # 仅打分（权重=分数，不剔除）
            client_scores = {}
            for i, client_id in enumerate(self.client_ids):
                client_scores[client_id] = self.score_calculator.calculate_scores(
                    client_id, self.client_features[i], self.client_data_sizes[i]
                )
            total_score = sum([s['final_score'] for s in client_scores.values()])
            weights = {cid: s['final_score']/total_score for cid, s in client_scores.items()}
            
            for i, (client_id, params) in enumerate(zip(self.client_ids, self.client_models)):
                weight = weights.get(client_id, 0.0)
                for key in agg_params.keys():
                    client_param = params[key].to(agg_params[key].device)
                    
                    # 只对浮点参数进行加权聚合
                    if agg_params[key].dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                        continue
                    else:
                        agg_params[key] += client_param.float() * float(weight)
        
        elif self.detection_method == "only_kickout":
            # 仅剔除（按固定阈值，不打分）
            threshold = 1.0 / len(self.client_ids)
            weights = {}
            total_data = sum(self.client_data_sizes)
            for i, client_id in enumerate(self.client_ids):
                data_ratio = self.client_data_sizes[i] / total_data
                weights[client_id] = data_ratio if data_ratio >= threshold else 0.0
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {cid: w/total_weight for cid, w in weights.items()}
            
            for i, (client_id, params) in enumerate(zip(self.client_ids, self.client_models)):
                weight = weights.get(client_id, 0.0)
                if weight > 0:
                    for key in agg_params.keys():
                        client_param = params[key].to(agg_params[key].device)
                        
                        # 只对浮点参数进行加权聚合
                        if agg_params[key].dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8]:
                            continue
                        else:
                            agg_params[key] += client_param.float() * float(weight)

        # 更新全局模型
        self.global_model.load_state_dict(agg_params)
        self._clear_client_data()

    def _clear_client_data(self):
        """清空客户端数据"""
        self.client_models = []
        self.client_data_sizes = []
        self.client_features = []
        self.client_ids = []

    def evaluate_model(self, test_loader):
        """评估模型准确率"""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

def run_single_mode(model_type, dataset_type, config, mode_name, detection_method, seed):
    """运行单个训练模式"""
    # 检查结果是否已存在（核心逻辑：避免重复训练）
    exists, acc_history = check_result_exists(
        save_dir="results",
        mode_name=mode_name,
        model_type=model_type,
        dataset_type=dataset_type,
        detection_method=detection_method,
        config=config
    )
    print(f"exists: {exists}")
    if exists:
        return np.array(acc_history)

    # 加载数据集
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=dataset_type,
        num_clients=config['total_clients'],
        batch_size=config['batch_size'],
        if_noniid=config['if_noniid'],
        alpha=config['alpha'],
        data_dir="./data"
    )

    # 选择模型
    model_class = LeNet5 if model_type == 'lenet5' else CIFAR10Net

    # 初始化服务端
    init_model = model_class()
    model_param_dim = sum(p.numel() for p in init_model.parameters())
    
    # 为每个模式生成唯一的投影矩阵文件到proj文件夹
    matrix_file_path = f"proj/projection_matrix_{dataset_type}.pt"
    server = Server(init_model, detection_method=detection_method, seed=seed)
    server.generate_projection_matrix(model_param_dim, min(1024, model_param_dim), matrix_file_path)

    # 选择投毒客户端
    poison_client_ids = random.sample(
        range(config['total_clients']), 
        int(config['total_clients'] * config['poison_ratio'])
    )
    ATTACK_TYPES = config['attack_types']  # 从配置获取攻击类型列表

    # 初始化客户端（应用选定的攻击类型）
    all_clients = []
    attack_type_idx = 0
    # 攻击参数映射表（与poison_loader.py对应）
    attack_params_map = {
        "random_poison": {"noise_std": 0.5},
        "label_flip": {"flip_ratio": 1},
        "model_compress": {"compress_ratio": 0.95},
        "backdoor": {"backdoor_ratio": 0.08, "backdoor_target": 9, "trigger_size": 2},
        "gradient_inversion": {"inversion_strength": 1000.0},
        "gradient_amplify": {"amplify_factor": 5.0},
        "feature_poison": {"poison_strength": 0.3, "perturb_dim": 100},
        "batch_poison": {"poison_ratio": 0.2, "batch_noise_std": 0.1}
    }

    for client_id in range(config['total_clients']):
        if client_id in poison_client_ids:
            # 循环分配攻击类型
            attack_type = ATTACK_TYPES[attack_type_idx % len(ATTACK_TYPES)]
            attack_type_idx += 1
            attack_params = attack_params_map[attack_type]
            poison_loader = PoisonLoader([attack_type], attack_params)
            print(f"客户端{client_id}：投毒攻击（{attack_type}，参数：{attack_params}）")
        else:
            poison_loader = None
            print(f"客户端{client_id}：正常客户端")

        client = Client(
            client_id=client_id,
            dataloader=all_client_dataloaders[client_id],
            model_class=model_class,
            poison_loader=poison_loader
        )
        all_clients.append(client)

    # 联邦训练主循环
    accuracy_history = []
    for round_num in range(1, config['comm_rounds'] + 1):
        print(f"\n===== {mode_name} - 第{round_num}/{config['comm_rounds']}轮 =====")

        # 下发模型和投影矩阵
        global_model_params, global_proj_matrix_path = server.send_model_and_proj()

        # 选择活跃客户端
        active_client_ids = random.sample(
            range(config['total_clients']), 
            config['active_clients']
        )
        active_clients = [all_clients[i] for i in active_client_ids]

        print(f"参与客户端：{active_client_ids} | 恶意客户端：{[cid for cid in active_client_ids if cid in poison_client_ids]}")

        # 客户端本地训练
        for client in active_clients:
            client.receive_model_and_proj(global_model_params, global_proj_matrix_path)
            trained_params, grad_flat = client.local_train()
            feature = client.extract_gradient_feature(grad_flat)
            data_size = len(client.dataloader.dataset)
            server.receive_client_data(trained_params, data_size, feature, client.client_id)

        # 聚合
        if detection_method == "none":
            server.aggregate_without_detection()
        else:
            server.aggregate_with_detection()

        # 评估
        acc = server.evaluate_model(test_loader)
        accuracy_history.append(acc)
        print(f"本轮准确率: {acc:.2f}%")

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 清理投影矩阵文件
    # if os.path.exists(matrix_file_path):
    #     os.remove(matrix_file_path)
    #     print(f"投影矩阵文件已清理: {matrix_file_path}")

    # 保存结果
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


# ---------------------- 主训练函数 ----------------------
def main_train(
    model_type='lenet5',
    dataset_type='mnist',
    detection_method="lsh_score_kickout",
    save_dir="results",
    **kwargs
):
    """统一的主训练函数"""
    # 默认配置
    base_config = {
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
        'attack_types': ["random_poison"],  # 默认攻击类型
        'seed': 42
    }
    # 覆盖默认配置
    base_config.update(kwargs)
    config = base_config

    print("===== 联邦学习投毒防御实验 =====")
    print(f"配置参数：{json.dumps(config, indent=2)}")
    print(f"检测方法：{detection_method}")
    print(f"数据集：{dataset_type} | 模型：{model_type}")
    print(f"使用的投毒攻击类型：{config['attack_types']}")

    # 定义训练模式
    modes = [
        {
            'name': 'pure_training',
            'config': {** config, 'poison_ratio': 0.0},
            'detection_method': 'none'
        },
        {
            'name': 'poison_no_detection',
            'config': {**config},
            'detection_method': 'none'
        },
        {
            'name': 'poison_with_detection',
            'config': {** config},
            'detection_method': detection_method
        }
    ]

    # 运行所有模式
    all_results = {}
    for mode in modes:
        print(f"\n=== 开始训练：{mode['name']} ===")
        acc_history = run_single_mode(
            model_type=model_type,
            dataset_type=dataset_type,
            config=mode['config'],
            mode_name=mode['name'],
            detection_method=mode['detection_method'],
            seed=config['seed']
        )
        all_results[mode['name']] = acc_history

    # 可视化对比结果
    plot_comparison_curves(
        base_config,
        result_dir=save_dir,
        save_path=os.path.join(save_dir, f"comparison_{detection_method}_{base_config['if_noniid']}_{base_config['attack_types']}_{base_config['poison_ratio']}.png")
    )

    # 输出总结
    print("\n===== 训练总结 =====")
    for mode_name, acc_history in all_results.items():
        final_acc = acc_history[-1]
        max_acc = np.max(acc_history)
        avg_acc = np.mean(acc_history)
        print(f"{mode_name}:")
        print(f"  最终准确率: {final_acc:.2f}% | 最高准确率: {max_acc:.2f}% | 平均准确率: {avg_acc:.2f}%")

    return all_results

# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    # 打印检测方法选项
    print("===== 联邦学习投毒防御实验 =====")
    # 检测方法设置
    # 可选检测方法：
    # 1. "none" - 无检测（纯FedAvg）
    # 2. "lsh_score_kickout" - LSH+三层打分+剔除（默认）
    # 3. "only_score" - 仅打分不剔除
    # 4. "only_kickout" - 仅剔除不打分
    detection_method = "lsh_score_kickout"  # 此处设置检测方法


    # 投毒攻击类型设置
    # 可选投毒攻击类型（可多选，填写攻击名称字符串）：
    # 1. "random_poison"
    # 2. "label_flip"
    # 3. "model_compress"
    # 4. "backdoor"
    # 5. "gradient_inversion"
    # 6. "gradient_amplify"
    # 7. "feature_poison"
    # 8. "batch_poison"
    selected_attacks = ["random_poison"]
    
    # 确保至少选择一种攻击类型（如果投毒比例>0）
    if not selected_attacks:
        print("未选择攻击类型，默认使用random_poison")
        selected_attacks = ["random_poison"]
    
    # 运行主训练函数
    results = main_train(
        model_type='cifar10',
        dataset_type='cifar10',
        detection_method=detection_method,
        comm_rounds=300,
        poison_ratio=0.2,
        attack_types=selected_attacks,  # 传入选定的攻击类型
        if_noniid=True,
        seed=42
    )
    
    print("\n🎉 训练完成！结果已保存至 results 目录")



