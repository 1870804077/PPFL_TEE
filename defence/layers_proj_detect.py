import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances

class Layers_Proj_Detector:
    def __init__(self, config=None):
        self.config = config or {}
        
        # 聚类方法
        self.clustering_method = self.config.get('clustering_method', 'kmeans')
        
        # 基础分与惩罚
        self.suspect_score = self.config.get('suspect_score', 1.0)      # 疑似分 (1.0)
        self.base_good_score = self.config.get('base_good_score', 1.0)  # 基础分 (1.0)
        self.max_bonus = self.config.get('max_bonus', 10.0)             # 最大奖励
        self.strike_threshold = self.config.get('strike_threshold', 3)
        
        # 阈值倍数 (Median + k * MAD)
        self.l2_multiplier = self.config.get('l2_threshold_multiplier', 3.0)
        self.var_multiplier = self.config.get('var_threshold_multiplier', 3.0)
        self.dist_multiplier = self.config.get('dist_threshold_multiplier', 2.0)
        
        # 非线性映射参数
        self.score_decay = self.config.get('score_decay_rate', 2.0)
        
        # DBSCAN参数 (备用)
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 3)

    def detect(self, client_projections, global_history_projections, suspect_counters, verbose=False):
        """
        执行检测流程
        """
        active_client_ids = list(client_projections.keys())
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # =========================================================
        # 1. 提取特征 & 聚类 (Full + Layers)
        # =========================================================
        
        # 1.1 全量处理 (Full)
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        self._compute_stats_metrics(raw_metrics, full_proj_dict, prefix='full')
        self._perform_clustering_and_dist(raw_metrics, full_proj_dict, prefix='full')

        # 1.2 分层处理 (Layers)
        if active_client_ids:
            first_cid = active_client_ids[0]
            if 'layers' in client_projections[first_cid]:
                for layer_name in client_projections[first_cid]['layers'].keys():
                    layer_proj_dict = {cid: client_projections[cid]['layers'][layer_name] for cid in active_client_ids}
                    
                    self._compute_stats_metrics(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')
                    self._perform_clustering_and_dist(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        # =========================================================
        # 2. 计算最终分数 (Hard + Soft Screening)
        # =========================================================
        weights, suspect_counters, global_stats = self.calculate_final_scores(raw_metrics, suspect_counters, verbose=verbose)

        return weights, raw_metrics, global_stats

    def _compute_stats_metrics(self, metrics_container, proj_dict, prefix):
        """计算 L2 和 Variance (硬筛查指标)"""
        for cid, proj_tensor in proj_dict.items():
            vec = proj_tensor.float()
            metrics_container[cid][f'{prefix}_l2'] = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_var'] = torch.var(vec).item()

    def _perform_clustering_and_dist(self, metrics_container, proj_dict, prefix):
        """
        执行聚类并计算到【本轮多数类中心】的余弦距离 (软筛查指标)
        """
        cids = list(proj_dict.keys())
        if len(cids) < 3:
            for cid in cids: 
                metrics_container[cid][f'{prefix}_cluster'] = 0
                metrics_container[cid][f'{prefix}_dist'] = 0.0
            return

        # 准备数据 (L2归一化)
        matrix = torch.stack([proj_dict[cid] for cid in cids])
        # [N, dim] normalized
        matrix_norm = torch.nn.functional.normalize(matrix.float(), p=2, dim=1).cpu().numpy()
        
        labels = []
        dists = np.zeros(len(cids))

        # === K-Means ===
        if self.clustering_method == 'kmeans':
            # 强制分2类
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(matrix_norm)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # 多数投票：找出"良性中心"
            counts = np.bincount(labels)
            major_label = np.argmax(counts)
            major_center = centers[major_label]
            
            # [关键修改] 计算 Cosine Distance = 1 - Cosine Similarity
            # 1. 必须对中心进行归一化，因为 K-Means 的中心是均值，模长不一定是 1
            major_center_norm = major_center / (np.linalg.norm(major_center) + 1e-9)
            
            # 2. 计算点积 (因为 matrix_norm 已经是单位向量，点积即为余弦相似度)
            # matrix_norm: (N, D), center: (D,) -> (N,)
            cos_sims = np.dot(matrix_norm, major_center_norm)
            
            # 3. 转换为距离 (范围 0 ~ 2)
            dists = 1.0 - cos_sims
            # 防止浮点误差出现负数
            dists = np.maximum(dists, 0.0)
            
            # 标记少数派为 -1 (仅作记录)
            labels = np.where(labels == major_label, 0, -1)

        # === DBSCAN (备用) ===
        elif self.clustering_method == 'dbscan':
            distance_matrix = cosine_distances(matrix_norm)
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='precomputed').fit(distance_matrix)
            labels = clustering.labels_
            # DBSCAN 难以直接定义中心，这里简单处理：噪声距离设为 1.0，正常设为 0.0
            dists = np.where(labels == -1, 1.0, 0.0)

        # 记录结果
        for i, cid in enumerate(cids):
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])
            metrics_container[cid][f'{prefix}_dist'] = float(dists[i])

    def calculate_final_scores(self, raw_metrics, suspect_counters, verbose=False):
        weights = {}
        cids = list(raw_metrics.keys())
        global_stats = {}
        
        if not cids: return {}, suspect_counters, global_stats
        
        # =========================================================
        # 1. 识别 Metrics Key
        # =========================================================
        sample_metrics = raw_metrics[cids[0]]
        l2_keys = [k for k in sample_metrics.keys() if k.endswith('_l2')]
        var_keys = [k for k in sample_metrics.keys() if k.endswith('_var')]
        dist_keys = [k for k in sample_metrics.keys() if k.endswith('_dist')]

        # =========================================================
        # 2. 计算全局统计阈值 (Median & MAD)
        # =========================================================
        stats_cache = {} 

        # 2.1 硬筛查指标 (L2, Var)
        for key in l2_keys + var_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            multiplier = self.l2_multiplier if 'l2' in key else self.var_multiplier
            thresh = med + multiplier * max(mad, 1e-6)
            stats_cache[key] = {'med': med, 'mad': mad, 'thresh': thresh}
            
            if key == 'full_l2': 
                global_stats['l2_threshold'] = thresh
                global_stats['l2_median'] = med
                global_stats['l2_mad'] = mad
            if key == 'full_var': 
                global_stats['var_threshold'] = thresh

        # 2.2 软筛查指标 (Cosine Distance)
        for key in dist_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            # 使用 dist_multiplier
            thresh = med + self.dist_multiplier * max(mad, 1e-6)
            stats_cache[key] = {'med': med, 'mad': mad, 'thresh': thresh}
            
            if key == 'full_dist':
                global_stats['dist_threshold'] = thresh

        # =========================================================
        # 3. 评分 (Scores)
        # =========================================================
        for cid in cids:
            metrics = raw_metrics[cid]
            scores_list = []      
            suspect_reasons = []
            
            # --- Part A: 硬筛查 (L2 & Var) ---
            for key in l2_keys + var_keys:
                val = metrics[key]
                info = stats_cache[key]
                
                if val > info['thresh']:
                    scores_list.append(self.suspect_score) # 1.0
                    tag = "L2" if "l2" in key else "Var"
                    suspect_reasons.append(f"{key.replace(f'_{tag.lower()}','')}:{tag}")
                else:
                    # 使用标准化偏差 Z-score 进行非线性映射
                    z_score = abs(val - info['med']) / (info['mad'] + 1e-6)
                    score = self.base_good_score + self.max_bonus * np.exp(-self.score_decay * z_score)
                    scores_list.append(score)

            # --- Part B: 软筛查 (Cosine Distance) ---
            for key in dist_keys:
                val = metrics[key] # 这是 Cosine Distance
                info = stats_cache[key]
                
                if val > info['thresh']:
                    scores_list.append(self.suspect_score) # 1.0
                    suspect_reasons.append(f"{key.replace('_dist','')}:Dist({val:.3f})")
                else:
                    # 同样使用高斯衰减
                    # 这里可以直接用距离值 val 进行衰减，或者用 Z-score
                    # 考虑到 Cosine Distance 范围固定且较小，直接用 val 可能更好控制，
                    # 但为了与硬筛查保持参数一致性，这里也使用 Z-score 映射。
                    # 如果想要更严格，可以直接用 val: np.exp(-self.score_decay * val * 10) (系数需调整)
                    # 此处维持统一逻辑:
                    z_score = abs(val - info['med']) / (info['mad'] + 1e-6)
                    score = self.base_good_score + self.max_bonus * np.exp(-self.score_decay * z_score)
                    scores_list.append(score)

            # --- Part C: 综合判定 (短板效应) ---
            final_score = min(scores_list) if scores_list else self.base_good_score
            
            # 判定疑似 (考虑浮点误差)
            is_suspect = (final_score <= self.suspect_score + 1e-4)

            # --- Part D: 惩罚机制 ---
            if is_suspect:
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            # --- Part E: 状态标记 ---
            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                reason_str = ",".join(suspect_reasons)
                if len(reason_str) > 60: reason_str = reason_str[:57] + "..."
                raw_metrics[cid]['status'] = f'SUSPECT' 
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score
            
            if verbose and is_suspect:
                print(f"      [Alert] Client {cid} suspect: {suspect_reasons}")
                print(f"              Score: {final_score:.4f}")

        return weights, suspect_counters, global_stats

    def _calc_robust_stats(self, values):
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad