import torch
import os
import numpy as np

class SuperBitLSH:
    def __init__(self, seed=42):
        self.seed = seed
        self.projection_matrix = None
        self.input_dim = 0
        self.output_dim = 0

    def generate_projection_matrix(self, input_dim, output_dim, device='cpu', matrix_file_path=None):
        """生成并加载投影矩阵"""
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 尝试从文件加载
        if matrix_file_path and os.path.exists(matrix_file_path):
            try:
                # print(f"加载投影矩阵: {matrix_file_path}")
                self.projection_matrix = torch.load(matrix_file_path, map_location=device)
                if self.projection_matrix.shape != (output_dim, input_dim):
                    print("矩阵维度不匹配，重新生成...")
                else:
                    return matrix_file_path
            except Exception as e:
                print(f"加载失败: {e}，重新生成...")

        # 生成正交随机矩阵 (Super-Bit LSH 核心)
        torch.manual_seed(self.seed)
        # 生成一个随机高斯矩阵
        random_matrix = torch.randn(output_dim, input_dim, device=device)
        # 对行进行正交化 (Gram-Schmidt 或 QR分解)，保证投影保持角度特性
        # 注意：对于极高维度，完整QR可能很慢，这里使用随机高斯矩阵近似（在高维下近似正交）
        # 如果追求严格 Super-Bit，需要做正交化。考虑到速度，这里保持高斯随机投影。
        self.projection_matrix = random_matrix

        # 保存
        if matrix_file_path:
            os.makedirs(os.path.dirname(matrix_file_path), exist_ok=True)
            torch.save(self.projection_matrix, matrix_file_path)
        
        return matrix_file_path

    def set_projection_matrix_path(self, path):
        """客户端加载矩阵"""
        if path and os.path.exists(path):
            self.projection_matrix = torch.load(path, map_location='cpu') # 默认加载到CPU，按需转GPU

    def extract_feature(self, data_vector, start_idx=0):
        """
        提取特征 (支持分层/部分投影)
        :param data_vector: 输入向量 (1D Tensor)
        :param start_idx: 该向量在原始全量参数中的起始位置
        """
        device = data_vector.device
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not initialized!")

        # 确保矩阵在正确的设备上
        if self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        # 1. 获取输入长度
        length = data_vector.numel()
        
        # 2. 从投影矩阵中切片：取出对应的列 [start_idx : start_idx + length]
        # 矩阵形状: [Output_Dim, Total_Input_Dim]
        # 切片形状: [Output_Dim, Length]
        matrix_slice = self.projection_matrix[:, start_idx : start_idx + length]

        # 3. 执行投影乘法: Matrix_Slice * Vector
        # [Output_Dim, Length] x [Length] -> [Output_Dim]
        feature = torch.matmul(matrix_slice, data_vector)

        # 4. 返回 Float 向量 (根据要求：不要变成 bit/sign)
        # 如果需要归一化，可以在这里做，但 raw projection 包含幅度信息
        return feature