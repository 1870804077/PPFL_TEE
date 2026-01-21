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

class Client:
    def __init__(self, client_id, dataloader, model_class, poison_loader=None):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model_class = model_class
        self.poison_loader = poison_loader or PoisonLoader() # 默认为空加载器（正常训练）
        self.model = None
        self.optimizer = None
        self.superbit_lsh = SuperBitLSH()

    def receive_model_and_proj(self, model_params, projection_matrix_path):
        """接收下发的全局模型和LSH投影矩阵"""
        if self.model is None:
            self.model = self.model_class().to(DEVICE)
        self.model.load_state_dict(model_params)
        # 每次接收新模型后重置优化器
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.superbit_lsh.set_projection_matrix_path(projection_matrix_path)

    def local_train(self):
        """执行本地训练（包含潜在的攻击逻辑）"""
        # 调用 PoisonLoader 统一处理正常训练或攻击训练
        trained_params, grad_flat = self.poison_loader.execute_attack(
            self.model, self.dataloader, self.model_class, DEVICE, self.optimizer
        )
        return trained_params, grad_flat

    def extract_gradient_feature(self, grad_flat):
        """提取 LSH 梯度特征"""
        # 分批处理以节省显存
        feature = self.superbit_lsh.extract_feature(grad_flat, batch_size=512)
        # 如果有特征层攻击，在此应用
        feature = self.poison_loader.apply_feature_poison(feature)
        
        # 显式清理内存
        del grad_flat
        return feature