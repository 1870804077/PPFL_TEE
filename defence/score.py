import torch
import numpy as np
from collections import defaultdict

class ScoreCalculator:
    def __init__(self, history_window=5, similarity_threshold=0.7):
        self.client_history = defaultdict(list)  # 存储客户端历史特征
        self.history_window = history_window  # 历史窗口大小
        self.global_features = []  # 全局特征集合
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        self.client_consistency = defaultdict(float)  # 客户端一致性记录

    def calculate_scores(self, client_id, current_feature, data_size):
        """计算三层分数：历史一致性、全局一致性和贡献度"""
        # 确保current_feature在正确的设备上
        if current_feature.device != torch.device("cpu"):
            current_feature = current_feature.cpu()
        
        # 1. 历史一致性分数：与自身历史特征的相似度
        history_sim = self._calculate_history_similarity(client_id, current_feature)

        # 2. 全局一致性分数：与其他客户端特征的相似度
        global_sim = self._calculate_global_similarity(current_feature)

        # 3. 贡献度分数：基于数据量的权重
        contribution_score = self._calculate_contribution_score(data_size)

        # 综合分数 (调整权重以提高稳定性)
        final_score = 0.3 * history_sim + 0.5 * global_sim + 0.2 * contribution_score

        # 更新历史记录
        self._update_history(client_id, current_feature)
        self._update_global_features(current_feature)

        return {
            'history_similarity': history_sim,
            'global_similarity': global_sim,
            'contribution_score': contribution_score,
            'final_score': final_score
        }

    def _calculate_history_similarity(self, client_id, current_feature):
        """计算与历史特征的余弦相似度"""
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        history = self.client_history[client_id]
        if len(history) < 1:
            return 1.0  # 没有历史记录时默认为最高相似度

        # 计算与所有历史特征的平均相似度
        similarities = []
        for hist_feature in history:
            # 确保hist_feature和current_feature在同一设备上
            hist_feature = hist_feature.to(current_feature.device)
            sim = torch.nn.functional.cosine_similarity(
                current_feature.unsqueeze(0),
                hist_feature.unsqueeze(0)
            ).item()
            similarities.append(sim)

        avg_sim = np.mean(similarities)
        
        # 计算一致性（相似度的稳定性）
        if len(similarities) > 1:
            consistency = 1.0 - np.std(similarities)  # 相似度越稳定，一致性越高
            self.client_consistency[client_id] = max(0.1, consistency)  # 保持最小一致性
        
        return avg_sim

    def _calculate_global_similarity(self, current_feature):
        """计算与全局特征的余弦相似度"""
        if len(self.global_features) < 2:  # 至少需要两个特征才能计算全局相似度
            return 1.0

        # 计算与所有全局特征的平均相似度
        similarities = []
        for global_feature in self.global_features:
            # 确保global_feature和current_feature在同一设备上
            global_feature = global_feature.to(current_feature.device)
            sim = torch.nn.functional.cosine_similarity(
                current_feature.unsqueeze(0),
                global_feature.unsqueeze(0)
            ).item()
            similarities.append(sim)

        return np.mean(similarities)

    def _calculate_contribution_score(self, data_size):
        """基于数据量计算贡献度分数"""
        if not self.global_features:
            return 1.0

        # 使用相对数据量而不是绝对数据量
        if hasattr(self, '_total_data_size'):
            relative_size = data_size / self._total_data_size if self._total_data_size > 0 else 1.0
        else:
            relative_size = min(1.0, data_size / 1000.0)  # 假设1000为一个参考数据量
        
        return max(0.1, relative_size)  # 保持最小贡献度

    def _update_history(self, client_id, feature):
        """更新客户端历史特征"""
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        self.client_history[client_id].append(feature.detach().cpu())
        # 保持历史窗口大小
        if len(self.client_history[client_id]) > self.history_window:
            self.client_history[client_id].pop(0)

    def _update_global_features(self, feature):
        """更新全局特征集合"""
        self.global_features.append(feature.detach().cpu())
        # 限制全局特征集合大小，避免内存占用过大
        if len(self.global_features) > 50:  # 减少全局特征数量以提高稳定性
            self.global_features.pop(0)
