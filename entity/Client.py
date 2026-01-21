import torch
import torch.nn as nn
import torch.optim as optim
import copy
from _utils_.poison_loader import PoisonLoader
from _utils_.LSH_proj_extra import SuperBitLSH

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
        
        # 临时存储本地梯度/更新量，用于投影
        self.local_grad_flat = None 

    def receive_model_and_proj(self, model_params, projection_matrix_path):
        """步骤3: 接收模型和投影矩阵"""
        if self.model is None:
            self.model = self.model_class().to(DEVICE)
        self.model.load_state_dict(model_params)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.superbit_lsh.set_projection_matrix_path(projection_matrix_path)

    def local_train(self):
        """步骤4: 执行本地训练，返回更新量(伪梯度)"""
        # 调用 PoisonLoader 统一处理
        trained_params, grad_flat = self.poison_loader.execute_attack(
            self.model, self.dataloader, self.model_class, DEVICE, self.optimizer
        )
        
        # 暂存梯度用于后续投影
        self.local_grad_flat = grad_flat
        
        # 显存优化：如果不立即使用 trained_params，可以先不存，
        # 但为了后续加权上传，我们需要模型保持在 trained 状态
        return grad_flat

    def generate_gradient_projection(self, start_idx=0):
        """步骤5 & 6: 对暂存的梯度进行投影 (支持按需分层，这里默认全量)"""
        if self.local_grad_flat is None:
            raise ValueError("No gradient computed yet!")
        
        # 调用 LSH 工具进行投影
        # 如果是全量投影，start_idx=0
        feature = self.superbit_lsh.extract_feature(self.local_grad_flat, start_idx=start_idx)
        
        # 特征层投毒 (如果有)
        feature = self.poison_loader.apply_feature_poison(feature)
        
        return feature

    def prepare_upload_weighted_params(self, weight):
        """步骤8: 计算 全量参数 * 权重"""
        # 获取当前训练后的参数
        current_params = self.model.state_dict()
        weighted_params = {}
        
        for key, param in current_params.items():
            if param.dtype in [torch.float32, torch.float64]:
                # 乘上服务器下发的权重
                weighted_params[key] = param * weight
            else:
                # 整数参数(如step)保持不变或置零，取决于聚合策略
                # 这里简单起见，跟随参数传递，但在聚合时需注意
                weighted_params[key] = param 
                
        # 可以在这里清理显存
        self.local_grad_flat = None
        return weighted_params