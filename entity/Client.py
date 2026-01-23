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
        self.layer_indices = {}
        current_idx = 0
        for name, param in self.model.named_parameters():
            length = param.numel()
            self.layer_indices[name] = (current_idx, length)
            current_idx += length

    def local_train(self):
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
        """
        [修改] 生成全量及指定层的投影
        :param target_layers: list of str, 指定需要单独投影的层名称 (例如 ['conv1.weight', 'fc3.bias'])
                             如果为 None，则只做全量投影或默认关键层投影
        :return: dict {'full': tensor, 'layers': {name: tensor}}
        """
        if self.local_grad_flat is None:
            raise ValueError("No gradient computed yet!")
        
        projections = {}
        
        # 1. 全量投影 (Full Gradient Projection)
        # 这里的 start_idx=0, length=auto
        full_proj = self.superbit_lsh.extract_feature(self.local_grad_flat, start_idx=0)
        # 应用特征投毒 (针对全量)
        projections['full'] = self.poison_loader.apply_feature_poison(full_proj)
        
        # 2. 指定层投影 (Layer-wise Projection)
        projections['layers'] = {}
        if target_layers:
            for layer_name in target_layers:
                if layer_name in self.layer_indices:
                    start, length = self.layer_indices[layer_name]
                    # 调用 LSH 对该段数据投影 (LSH类已支持 start_idx 和 length 自动推导)
                    # 注意：extract_feature 内部需要支持传入 explicit length 或者通过 slice 传入
                    # 为了复用之前的代码，我们需要传入切片后的 tensor 或者让 extract_feature 支持 length 参数
                    # 这里假设我们传入切片后的 tensor 给 extract_feature
                    
                    # 切片梯度
                    layer_grad_chunk = self.local_grad_flat[start : start + length]
                    
                    # 投影 (注意 start_idx 对应投影矩阵的列位置)
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
        self.local_grad_flat = None # 清理
        return weighted_params