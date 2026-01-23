import torch
import numpy as np

class ProjectedMesasDetector:
    def __init__(self):
        pass

    def detect(self, client_projections, global_update_direction, suspect_counters):
        """
        执行检测逻辑
        :param client_projections: dict, {cid: {'full': tensor, 'layers': {...}}}
        :param global_update_direction: tensor, 上一轮的全局更新方向 (Sum g_i * W)
        :param suspect_counters: dict, {cid: int}, 记录每个客户端的累计嫌疑次数
        :return: 
            weights: dict {cid: float}, 分配的权重 (0.0 或 1.0 或 0.x)
            metrics: dict, 用于日志记录的中间指标
        """
        
        client_ids = list(client_projections.keys())
        weights = {cid: 1.0 for cid in client_ids} # 默认权重
        metrics_log = {}

        # 如果是第一轮 (没有全局方向)，或者全局方向未初始化，跳过基于方向的检测
        if global_update_direction is None:
            return weights, {"info": "First round, skipping direction check"}

        # --- 1. 计算指标 (Interfaces) ---
        for cid in client_ids:
            proj_data = client_projections[cid]
            full_proj = proj_data['full']
            
            # (1) 计算与全局历史方向的 Cosine Similarity
            # global_update_direction 是上一轮所有投影的和，近似代表良性方向
            cos_sim = self._calc_cosine_similarity(full_proj, global_update_direction)
            
            # (2) 计算 L2 范数 (检测 Scaling 攻击)
            l2_norm = torch.norm(full_proj).item()
            
            # (3) 预留接口：分层指标检测
            # layer_anomalies = self._check_layer_consistency(proj_data['layers'])

            metrics_log[cid] = {'cos': cos_sim, 'l2': l2_norm}

        # --- 2. 判定离群点 (Outlier Detection) ---
        # 这里先写一个简单的统计学判定逻辑作为占位
        # 例如：Cos 相似度 < 均值 - 2*标准差，或者 Cos < 0
        
        cos_values = [m['cos'] for m in metrics_log.values()]
        mean_cos = np.mean(cos_values)
        std_cos = np.std(cos_values)
        threshold = mean_cos - 1.5 * std_cos # 阈值可调
        
        # --- 3. 标记与惩罚 ---
        for cid in client_ids:
            score = 1.0
            is_suspect = False
            
            # 判定规则示例
            if metrics_log[cid]['cos'] < threshold:
                is_suspect = True
                
            # 更新嫌疑计数
            if is_suspect:
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
                # 标记为疑似者，分配低权重 (例如 0.1)
                score = 0.1 
            else:
                # 表现良好，尝试减少计数 (可选，给予改过自新的机会)
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] -= 0.5 

            # --- 4. 累计剔除 (Kickout) ---
            # 累计三次标记则分配零权重
            if suspect_counters.get(cid, 0) >= 3:
                score = 0.0
                metrics_log[cid]['status'] = 'KICKED'
            elif is_suspect:
                metrics_log[cid]['status'] = 'SUSPECT'
            else:
                metrics_log[cid]['status'] = 'NORMAL'
                
            weights[cid] = score

        return weights, metrics_log

    def _calc_cosine_similarity(self, v1, v2):
        """计算余弦相似度"""
        # 确保 float 类型
        v1 = v1.float()
        v2 = v2.float()
        return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    def _check_layer_consistency(self, layers_dict):
        """接口：检查各层投影的一致性"""
        pass