import torch
import numpy as np
import gc
import os


class SuperBitLSH:
    def __init__(self, seed=42):
        self.projection_matrix_path = None  # 存储投影矩阵文件路径
        self.seed = seed  # 随机种子

    def set_projection_matrix_path(self, projection_matrix_path):
        """设置投影矩阵文件路径"""
        self.projection_matrix_path = projection_matrix_path

    def generate_projection_matrix(self, input_dim, output_dim, device='cpu', matrix_file_path=None):
        """生成随机投影矩阵并保存到文件"""
        # 如果没有指定路径，生成默认路径到proj文件夹
        if matrix_file_path is None:
            os.makedirs("proj", exist_ok=True)  # 创建proj文件夹
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}_seed{self.seed}.pt"
        else:
            # 确保路径的目录存在
            os.makedirs(os.path.dirname(matrix_file_path), exist_ok=True)
        
        print(f"正在生成投影矩阵，尺寸: ({input_dim}, {output_dim})...")
        
        # 设置随机种子以确保一致性
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # 生成标准正态分布的随机矩阵
        projection = torch.randn(input_dim, output_dim, device=device)
        
        # 正交化处理（仅在CPU上进行以避免GPU内存问题）
        if device != 'cpu':
            projection_cpu = projection.cpu()
        else:
            projection_cpu = projection
            
        # 在CPU上执行QR分解
        try:
            projection_cpu, _ = torch.linalg.qr(projection_cpu)
        except:
            # 如果QR分解失败，使用SVD分解
            U, S, V = torch.svd(projection_cpu)
            projection_cpu = U @ V.t()
        
        # 保存到文件
        torch.save(projection_cpu, matrix_file_path)
        self.projection_matrix_path = matrix_file_path
        
        print(f"投影矩阵已生成并保存到: {matrix_file_path}")
        
        # 释放内存
        del projection, projection_cpu
        gc.collect()
        
        return matrix_file_path

    def extract_feature(self, grad_flat, batch_size=1024):
        """使用投影矩阵提取梯度特征（分批处理）"""
        if self.projection_matrix_path is None:
            raise ValueError("投影矩阵未初始化，请先设置投影矩阵文件路径")

        # 检查投影矩阵文件是否存在
        if not os.path.exists(self.projection_matrix_path):
            raise FileNotFoundError(f"投影矩阵文件不存在: {self.projection_matrix_path}")

        # 从文件加载投影矩阵
        try:
            projection_matrix = torch.load(self.projection_matrix_path, map_location='cpu')
        except Exception as e:
            print(f"加载投影矩阵失败: {e}")
            print(f"尝试重新生成投影矩阵...")
            # 重新生成投影矩阵（需要知道原始维度）
            # 这里我们基于grad_flat的维度重新生成，使用相同的随机种子
            input_dim = grad_flat.shape[-1]  # 获取梯度维度
            output_dim = min(1024, input_dim)  # 输出维度
            # 保持原始的文件名格式，但确保使用相同的种子
            if f"_seed{self.seed}.pt" not in self.projection_matrix_path:
                # 如果原文件名没有种子信息，添加种子信息
                base_path = self.projection_matrix_path.replace('.pt', f'_seed{self.seed}.pt')
            else:
                base_path = self.projection_matrix_path
            self.generate_projection_matrix(input_dim, output_dim, device='cpu', matrix_file_path=base_path)
            projection_matrix = torch.load(base_path, map_location='cpu')

        # 确保投影矩阵在与梯度相同的设备上
        projection_matrix = projection_matrix.to(grad_flat.device)
        
        # 确保grad_flat是二维的 [batch_size, feature_dim]
        if grad_flat.dim() == 1:
            grad_flat = grad_flat.unsqueeze(0)  # [1, feature_dim]
        
        # 计算投影 - 分批处理以避免内存溢出
        feature = self._batch_matmul(grad_flat, projection_matrix, batch_size)
        
        # 二值化处理
        feature = torch.sign(feature)
        
        # 如果输入是一维的，输出也应该是对应的一维
        if feature.size(0) == 1:
            feature = feature.squeeze(0)
        
        # 释放投影矩阵内存
        del projection_matrix
        gc.collect()
        
        return feature

    def _batch_matmul(self, grad_flat, projection_matrix, batch_size):
        """分批矩阵乘法以避免内存溢出"""
        total_size = grad_flat.size(0)
        result_parts = []
        
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            grad_batch = grad_flat[start_idx:end_idx]
            
            # 执行矩阵乘法: [batch, input_dim] @ [input_dim, output_dim] = [batch, output_dim]
            result_batch = torch.matmul(grad_batch, projection_matrix)
            result_parts.append(result_batch)
            
            # 清理临时变量
            del grad_batch
            gc.collect()
        
        # 合并结果
        result = torch.cat(result_parts, dim=0)
        
        # 清理临时列表
        del result_parts
        gc.collect()
        
        return result