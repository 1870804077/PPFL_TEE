import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances

class Layers_Proj_Detector:
    def __init__(self, config=None):
        """
        初始化检测器
        :param config: 包含防御参数的字典 (对应 config.yaml 中的 defense.params)
        """
        self.config = config or {}
        
        # --- 1. 读取聚类参数 (带默认值) ---
        self.clustering_method = self.config.get('clustering_method', 'dbscan')
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 3)
        
        # --- 2. 读取评分参数 (带默认值) ---
        self.base_good_score = self.config.get('base_good_score', 10.0)
        self.suspect_score = self.config.get('suspect_score', 1.0)
        self.max_bonus = self.config.get('max_bonus', 10.0)
        self.strike_threshold = self.config.get('strike_threshold', 3)
        
        # --- 3. 读取统计阈值倍数 ---
        self.l2_multiplier = self.config.get('l2_threshold_multiplier', 3.0)
        self.var_multiplier = self.config.get('var_threshold_multiplier', 3.0)

    def detect(self, client_projections, global_history_projections, suspect_counters):
        """
        执行检测流程
        :param client_projections: dict, 本轮客户端投影数据
        :param global_history_projections: tensor, 上一轮的全局累加方向 (用于历史一致性检测)
        :param suspect_counters: dict, 历史嫌疑计数器 (引用传递，将在内部更新)
        :return: (weights, raw_metrics)
                 weights: dict {cid: score}
                 raw_metrics: dict {cid: {details...}} 用于日志记录
        """
        active_client_ids = list(client_projections.keys())
        # 结果容器 {cid: {metric_name: value}}
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # =========================================================
        # 1. 提取全量投影并处理 (Full Projection)
        # =========================================================
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        
        # 计算基础统计量 (L2, Variance) & 历史相似度
        # 注意：global_history_projections 在 Server 中可能直接是 tensor (full summation)
        # 如果架构支持分层历史，这里需要适配。目前假设 Server 传的是 full tensor。
        history_tensor = global_history_projections if isinstance(global_history_projections, torch.Tensor) else None
        
        self._compute_basic_metrics(
            raw_metrics, 
            full_proj_dict, 
            history_tensor,
            prefix='full'
        )
        
        # 执行聚类 (全量)
        self._perform_clustering(raw_metrics, full_proj_dict, prefix='full')

        # =========================================================
        # 2. 提取分层投影并处理 (Layer-wise Projection)
        # =========================================================
        if active_client_ids:
            first_cid = active_client_ids[0]
            # 检查是否有分层数据
            if 'layers' in client_projections[first_cid]:
                layer_keys = client_projections[first_cid]['layers'].keys()
                
                for layer_name in layer_keys:
                    # 提取该层的投影数据
                    layer_proj_dict = {
                        cid: client_projections[cid]['layers'][layer_name] 
                        for cid in active_client_ids
                    }
                    
                    # 目前暂不支持分层历史对比 (除非 Server 也维护分层历史)，这里传 None
                    # 如果需要，可以在 Server 维护一个 dict 类型的 global_history_projections
                    self._compute_basic_metrics(
                        raw_metrics, 
                        layer_proj_dict, 
                        None, 
                        prefix=f'layer_{layer_name}'
                    )
                    
                    # 执行聚类
                    self._perform_clustering(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        # =========================================================
        # 3. 计算最终分数 (Scoring)
        # =========================================================
        weights = self.calculate_final_scores(raw_metrics, suspect_counters)

        return weights, raw_metrics

    def _compute_basic_metrics(self, metrics_container, proj_dict, history_tensor, prefix):
        """
        辅助函数：计算 L2, Variance, Cosine Similarity
        """
        for cid, proj_tensor in proj_dict.items():
            # 确保是 float 类型，避免半精度问题
            vec = proj_tensor.float()
            
            # 1. 计算 L2 范数
            metrics_container[cid][f'{prefix}_l2'] = torch.norm(vec, p=2).item()
            
            # 2. 计算方差
            metrics_container[cid][f'{prefix}_var'] = torch.var(vec).item()
            
            # 3. 计算与上一轮历史的余弦相似度
            if history_tensor is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), 
                    history_tensor.float().to(vec.device).unsqueeze(0)
                ).item()
                metrics_container[cid][f'{prefix}_hist_cos'] = cos_sim
            else:
                # 第一轮或无历史，给默认值 1.0 (表示无异议)
                metrics_container[cid][f'{prefix}_hist_cos'] = 1.0

    def _perform_clustering(self, metrics_container, proj_dict, prefix):
        """
        辅助函数：执行聚类并记录标签
        """
        cids = list(proj_dict.keys())
        if len(cids) < 3:
            # 样本太少，跳过聚类，默认都是一类 (0)
            for cid in cids:
                metrics_container[cid][f'{prefix}_cluster'] = 0
            return

        # 准备数据矩阵 [N_samples, N_features]
        matrix = torch.stack([proj_dict[cid] for cid in cids])
        # L2 归一化 (因为 DBSCAN 用余弦距离)
        matrix_norm = torch.nn.functional.normalize(matrix.float(), p=2, dim=1).cpu().numpy()
        
        # 计算余弦距离矩阵
        distance_matrix = cosine_distances(matrix_norm)
        
        labels = []
        if self.clustering_method == 'dbscan':
            clustering = DBSCAN(
                eps=self.dbscan_eps, 
                min_samples=self.dbscan_min_samples, 
                metric='precomputed'
            ).fit(distance_matrix)
            labels = clustering.labels_
            
        elif self.clustering_method == 'kmeans':
            clustering = KMeans(n_clusters=2, random_state=42).fit(matrix_norm)
            labels = clustering.labels_
            # 简单的启发式：假设样本多的一类是正常的 (Label 0)
            counts = np.bincount(labels)
            major_label = np.argmax(counts)
            labels = np.where(labels == major_label, 0, -1)

        # 记录结果
        for i, cid in enumerate(cids):
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])

    def calculate_final_scores(self, raw_metrics, suspect_counters):
        """
        基于 Hard Filtering + Soft Scoring 计算权重
        :return: weights dict
        """
        weights = {}
        cids = list(raw_metrics.keys())
        if not cids:
            return {}
        
        # 1. 准备全局统计量 (使用 Median 和 MAD)
        full_l2_values = [raw_metrics[cid]['full_l2'] for cid in cids]
        full_var_values = [raw_metrics[cid]['full_var'] for cid in cids]
        
        l2_median, l2_mad = self._calc_robust_stats(full_l2_values)
        var_median, var_mad = self._calc_robust_stats(full_var_values)

        # 设定动态阈值 (使用 config 中的 multiplier)
        l2_threshold = l2_median + self.l2_multiplier * max(l2_mad, 1e-5)
        var_threshold = var_median + self.var_multiplier * max(var_mad, 1e-5)

        for cid in cids:
            metrics = raw_metrics[cid]
            is_suspect = False
            suspect_reasons = []

            # --- A. 硬筛除 (Hard Filtering) ---
            
            # 规则1: L2 范数异常
            if metrics['full_l2'] > l2_threshold:
                is_suspect = True
                suspect_reasons.append("L2")
            
            # 规则2: 方差异常
            if metrics['full_var'] > var_threshold:
                is_suspect = True
                suspect_reasons.append("Var")
                
            # 规则3: 聚类筛除 (DBSCAN Noise)
            if metrics.get('full_cluster') == -1:
                is_suspect = True
                suspect_reasons.append("Cluster")
            
            # --- B. 权重计算 ---
            
            if is_suspect:
                # 判定为疑似: 使用配置中的 suspect_score
                final_score = self.suspect_score 
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                # 判定为良性: 计算软得分
                cos_sim = metrics.get('full_hist_cos', 0)
                
                # Formula: Base + Max * tanh(sim)
                bonus = self.max_bonus * np.tanh(max(0, cos_sim)) 
                final_score = self.base_good_score + bonus 
                
                # 表现良好减少计数
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            # --- C. 最终累计判定 (Strike System) ---
            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                raw_metrics[cid]['status'] = f'SUSPECT({",".join(suspect_reasons)})'
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score

        return weights

    def _calc_robust_stats(self, values):
        """计算中位数和绝对中位差 (MAD)"""
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad