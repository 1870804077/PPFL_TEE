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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.seed = seed
        
        # 临时存储本轮客户端上传的数据
        self.client_uploads = {
            "models": [],
            "data_sizes": [],
            "features": [],
            "ids": []
        }
        
        # 初始化防御组件
        self.score_calculator = None
        self.kickout_manager = None
        
        if "score" in self.detection_method:
            self.score_calculator = ScoreCalculator()
        if "kickout" in self.detection_method:
            self.kickout_manager = KickoutManager()

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        """生成并保存投影矩阵"""
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def get_global_params_and_proj(self):
        """获取分发给客户端的参数"""
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def receive_client_upload(self, model_params, data_size, feature, client_id):
        """接收单个客户端上传"""
        self.client_uploads["models"].append(model_params)
        self.client_uploads["data_sizes"].append(data_size)
        self.client_uploads["features"].append(feature)
        self.client_uploads["ids"].append(client_id)

    def aggregate(self):
        """根据检测策略执行聚合"""
        if not self.client_uploads["models"]:
            return

        method = self.detection_method
        weights = {}

        # 1. 计算权重
        if method == "none":
            # FedAvg: 基于数据量加权
            total_size = sum(self.client_uploads["data_sizes"])
            weights = {
                cid: size / total_size 
                for cid, size in zip(self.client_uploads["ids"], self.client_uploads["data_sizes"])
            }
            
        else:
            # 防御机制：计算分数或剔除
            client_scores = {}
            if self.score_calculator:
                for i, cid in enumerate(self.client_uploads["ids"]):
                    client_scores[cid] = self.score_calculator.calculate_scores(
                        cid, 
                        self.client_uploads["features"][i], 
                        self.client_uploads["data_sizes"][i]
                    )

            if method == "lsh_score_kickout":
                weights = self.kickout_manager.determine_weights(client_scores)
            elif method == "only_score":
                total_score = sum(s['final_score'] for s in client_scores.values())
                weights = {cid: s['final_score']/total_score for cid, s in client_scores.items()}
            elif method == "only_kickout":
                # 简单剔除逻辑（如果不使用 score 模块）
                # 这里为了简化，假设只有 score 计算了才能剔除，或者使用简单的统计剔除
                # 实际代码中通常 kickout 依赖 score，所以这里复用 kickout_manager 逻辑
                if self.kickout_manager:
                     weights = self.kickout_manager.determine_weights(client_scores)
                else:
                    # Fallback: 平均聚合
                    n = len(self.client_uploads["ids"])
                    weights = {cid: 1.0/n for cid in self.client_uploads["ids"]}

        # 2. 执行加权聚合
        self._apply_aggregation(weights)
        
        # 3. 清理本轮数据
        self._clear_buffer()

    def _apply_aggregation(self, weights):
        """底层聚合逻辑"""
        first_params = self.client_uploads["models"][0]
        agg_params = {
            k: torch.zeros_like(v, dtype=v.dtype, device=DEVICE) 
            for k, v in first_params.items()
        }

        # 检查是否所有权重都为0（例如都被踢出了）
        total_weight = sum(weights.values())
        if total_weight <= 1e-6:
            print("  [Warning] 所有客户端均被剔除或权重为0，本轮不更新模型。")
            return

        for i, cid in enumerate(self.client_uploads["ids"]):
            w = weights.get(cid, 0.0)
            if w > 0:
                client_params = self.client_uploads["models"][i]
                for k in agg_params.keys():
                    # 只聚合浮点参数
                    if agg_params[k].dtype in [torch.float32, torch.float64]:
                        agg_params[k] += client_params[k].to(DEVICE) * w
                    else:
                        # 整数参数（如 BN 的 num_batches_tracked）通常取第一个模型的或累加
                        # 简单起见，这里保持累加或取模版
                        if i == 0: 
                            agg_params[k] = client_params[k].to(DEVICE)

        self.global_model.load_state_dict(agg_params)

    def _clear_buffer(self):
        for k in self.client_uploads:
            self.client_uploads[k] = []

    def evaluate(self, test_loader):
        """模型评估"""
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
        return 100 * correct / total