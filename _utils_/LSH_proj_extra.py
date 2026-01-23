import torch
import os
import numpy as np
import gc

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
                # 默认加载到 CPU，避免一开始就占满 GPU 显存
                self.projection_matrix = torch.load(matrix_file_path, map_location='cpu')
                if self.projection_matrix.shape != (output_dim, input_dim):
                    print("矩阵维度不匹配，重新生成...")
                else:
                    return matrix_file_path
            except Exception as e:
                print(f"加载失败: {e}，重新生成...")

        # 生成矩阵 (高斯随机矩阵)
        torch.manual_seed(self.seed)
        # 建议直接在 CPU 上生成，防止 OOM
        self.projection_matrix = torch.randn(output_dim, input_dim, device='cpu')

        # 保存
        if matrix_file_path:
            os.makedirs(os.path.dirname(matrix_file_path), exist_ok=True)
            torch.save(self.projection_matrix, matrix_file_path)
        
        return matrix_file_path

    def set_projection_matrix_path(self, path):
        """客户端加载矩阵"""
        if path and os.path.exists(path):
            # 强制加载到 CPU，只在计算时分块送入 GPU
            self.projection_matrix = torch.load(path, map_location='cpu')

    def extract_feature(self, data_vector, start_idx=0, batch_size=256):
        """
        提取特征 (支持分层 + 显存优化批量计算)
        
        :param data_vector: 输入向量 (1D Tensor)
        :param start_idx: 该向量在原始全量参数中的起始位置
        :param batch_size: 每次计算输出向量的多少个维度 (控制显存占用)
        """
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not initialized!")

        device = data_vector.device
        length = data_vector.numel()
        
        # 结果容器
        results = []
        
        # 投影矩阵的总行数 (即最终特征维度，如 1024)
        total_rows = self.projection_matrix.shape[0]
        
        # === 批量计算逻辑 (Row Batching) ===
        # 我们遍历投影矩阵的"行" (Rows)，每次只计算一小部分特征
        # 这样避免了一次性进行 [1024, Huge_Dim] x [Huge_Dim] 的大矩阵乘法
        
        for i in range(0, total_rows, batch_size):
            end_row = min(i + batch_size, total_rows)
            
            # 1. 切片：取出一批行，以及对应的列范围
            # Slice Shape: [Batch_Size, Length]
            # 注意：self.projection_matrix 在 CPU 上
            proj_chunk = self.projection_matrix[i:end_row, start_idx : start_idx + length]
            
            # 2. 搬运：将这一小块矩阵移动到数据所在的设备 (GPU)
            if proj_chunk.device != device:
                proj_chunk = proj_chunk.to(device)
            
            # 3. 计算：小矩阵乘法
            # [Batch, Length] x [Length] -> [Batch]
            chunk_res = torch.matmul(proj_chunk, data_vector)
            
            results.append(chunk_res)
            
            # 及时释放临时显存
            del proj_chunk
            del chunk_res
        
        # 4. 拼接：将所有批次的结果拼起来 -> [Total_Rows]
        final_feature = torch.cat(results)
        
        return final_feature