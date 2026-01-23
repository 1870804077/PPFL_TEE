import torch
import torch.nn as nn
import torch.optim as optim
from _utils_.poison_loader import PoisonLoader
from _utils_.LSH_proj_extra import SuperBitLSH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client:
    def __init__(self, client_id, dataloader, model_class, poison_loader=None, verbose=False, log_interval=100):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model_class = model_class
        self.poison_loader = poison_loader or PoisonLoader()
        self.model = None
        self.optimizer = None
        self.superbit_lsh = SuperBitLSH()
        
        self.local_grad_flat = None 
        self.layer_indices = None
        
        # 保存日志配置
        self.verbose = verbose
        self.log_interval = log_interval

    def receive_model_and_proj(self, model_params, projection_matrix_path):
        if self.model is None:
            self.model = self.model_class().to(DEVICE)
        self.model.load_state_dict(model_params)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.superbit_lsh.set_projection_matrix_path(projection_matrix_path)
        
        if self.layer_indices is None:
            self._calculate_layer_indices()

    def _calculate_layer_indices(self):
        """辅助函数：计算每层参数的 (start, length)"""
        self.layer_indices = {}
        current_idx = 0
        for name, param in self.model.named_parameters():
            length = param.numel()
            self.layer_indices[name] = (current_idx, length)
            current_idx += length

    def local_train(self):
        """训练并返回梯度"""
        if self.verbose:
            print(f"  > [Client {self.client_id}] Start Local Training...")

        trained_params, grad_flat = self.poison_loader.execute_attack(
            self.model, 
            self.dataloader, 
            self.model_class, 
            DEVICE, 
            self.optimizer,
            verbose=self.verbose,
            uid=self.client_id,
            log_interval=self.log_interval
        )
        self.local_grad_flat = grad_flat
        return grad_flat

    def generate_gradient_projection(self, target_layers=None):
        """生成全量及指定层的投影"""
        if self.local_grad_flat is None:
            raise ValueError("No gradient computed yet!")
        
        projections = {}
        
        # 1. 全量投影
        full_proj = self.superbit_lsh.extract_feature(self.local_grad_flat, start_idx=0)
        projections['full'] = self.poison_loader.apply_feature_poison(full_proj)
        
        # 2. 指定层投影
        projections['layers'] = {}
        if target_layers:
            for layer_name in target_layers:
                if layer_name in self.layer_indices:
                    start, length = self.layer_indices[layer_name]
                    
                    # 切片并投影
                    layer_grad_chunk = self.local_grad_flat[start : start + length]
                    layer_proj = self.superbit_lsh.extract_feature(layer_grad_chunk, start_idx=start)
                    projections['layers'][layer_name] = layer_proj
        
        return projections

    def prepare_upload_weighted_params(self, weight):
        """计算加权参数"""
        current_params = self.model.state_dict()
        weighted_params = {}
        for key, param in current_params.items():
            if param.dtype in [torch.float32, torch.float64]:
                weighted_params[key] = param * weight
            else:
                weighted_params[key] = param 
        self.local_grad_flat = None # 清理显存
        return weighted_params