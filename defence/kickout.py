import torch
import numpy as np

class KickoutManager:
    def __init__(self, threshold=0.3, max_kick_ratio=0.4, adaptive=True):
        self.threshold = threshold  # 分数阈值，降低以避免过度剔除
        self.max_kick_ratio = max_kick_ratio  # 最大可剔除比例
        self.adaptive = adaptive  # 是否使用自适应阈值
        self.round_count = 0  # 记录轮次
        self.score_history = []  # 记录历史分数分布

    def determine_weights(self, client_scores):
        """
        根据分数确定客户端权重，被剔除的客户端权重为0
        client_scores: 字典，key为客户端ID，value为分数字典
        """
        if not client_scores:
            return {}

        # 提取最终分数
        scores = [info['final_score'] for info in client_scores.values()]
        client_ids = list(client_scores.keys())

        # 使用自适应阈值以避免过度剔除
        if self.adaptive:
            # 计算动态阈值，考虑历史分数分布
            if len(self.score_history) > 0:
                # 使用历史平均值和标准差来确定阈值
                hist_mean = np.mean(self.score_history[-20:])  # 最近20轮的平均值
                hist_std = np.std(self.score_history[-20:]) if len(self.score_history) > 1 else 0.1
                # 动态阈值：均值减去0.5个标准差
                dynamic_threshold = max(hist_mean - 0.5 * hist_std, 0.1)
            else:
                dynamic_threshold = self.threshold
            
            # 更新历史记录
            self.score_history.extend(scores)
            if len(self.score_history) > 100:  # 只保留最近100个分数
                self.score_history = self.score_history[-100:]
            
            # 使用分位数作为阈值，确保最多剔除max_kick_ratio比例的客户端
            valid_kick_ratio = min(self.max_kick_ratio, 1.0 - 1.0 / len(scores)) if scores else 0
            if valid_kick_ratio > 0:
                score_threshold = np.percentile(scores, valid_kick_ratio * 100)
                # 使用动态阈值和分位数阈值中的较大值，避免过度剔除
                score_threshold = max(score_threshold, dynamic_threshold)
            else:
                score_threshold = max(dynamic_threshold, min(scores)) if scores else 0
        else:
            # 使用固定阈值
            valid_kick_ratio = min(self.max_kick_ratio, 1.0 - 1.0 / len(scores)) if scores else 0
            if valid_kick_ratio > 0:
                score_threshold = np.percentile(scores, valid_kick_ratio * 100)
            else:
                score_threshold = min(self.threshold, min(scores)) if scores else 0

        # 计算权重
        weights = {}
        for client_id in client_ids:
            score = client_scores[client_id]['final_score']
            if score >= score_threshold:
                # 权重与分数正相关，但加入平滑处理
                weights[client_id] = max(score, 0.1)  # 防止权重过小
            else:
                # 被剔除，权重为0
                weights[client_id] = 0.0

        # 归一化权重，但保留一定比例的最小权重以避免完全剔除
        total_weight = sum(weights.values())
        if total_weight > 0:
            # 计算非零权重的数量
            non_zero_count = sum(1 for w in weights.values() if w > 0)
            if non_zero_count > 0:
                # 如果所有客户端都被剔除，给所有客户端分配最小权重
                if total_weight == 0:
                    min_weight = 0.1
                    for client_id in weights:
                        weights[client_id] = min_weight
                    total_weight = min_weight * len(weights)
                
                for client_id in weights:
                    if weights[client_id] > 0:
                        weights[client_id] /= total_weight
            else:
                # 如果没有客户端通过检测，给所有客户端分配相等权重
                equal_weight = 1.0 / len(weights)
                for client_id in weights:
                    weights[client_id] = equal_weight
        else:
            # 如果总权重为0，给所有客户端分配相等权重
            equal_weight = 1.0 / len(weights)
            for client_id in weights:
                weights[client_id] = equal_weight

        self.round_count += 1
        return weights
