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
        self.suspect_score = self.config.get('suspect_score', 1.0)
        self.base_good_score = self.config.get('base_good_score', 1.0)
        self.max_bonus = self.config.get('max_bonus', 10.0)
        self.strike_threshold = self.config.get('strike_threshold', 3)
        
        # 阈值倍数
        self.l2_multiplier = self.config.get('l2_threshold_multiplier', 3.0)
        self.var_multiplier = self.config.get('var_threshold_multiplier', 3.0)
        self.dist_multiplier = self.config.get('dist_threshold_multiplier', 2.0)
        
        # 非线性映射参数
        self.score_decay = self.config.get('score_decay_rate', 2.0)
        
        # DBSCAN参数
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 3)

    def detect(self, client_projections, global_history_projections, suspect_counters, verbose=False):
        active_client_ids = list(client_projections.keys())
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # 1. 全量处理 (Full)
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        self._compute_stats_metrics(raw_metrics, full_proj_dict, prefix='full')
        self._perform_clustering_and_dist(raw_metrics, full_proj_dict, prefix='full')

        # 2. 分层处理 (Layers)
        if active_client_ids:
            first_cid = active_client_ids[0]
            if 'layers' in client_projections[first_cid]:
                for layer_name in client_projections[first_cid]['layers'].keys():
                    layer_proj_dict = {cid: client_projections[cid]['layers'][layer_name] for cid in active_client_ids}
                    
                    self._compute_stats_metrics(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')
                    self._perform_clustering_and_dist(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        # 3. 计算最终分数
        weights, suspect_counters, global_stats = self.calculate_final_scores(raw_metrics, suspect_counters, verbose=verbose)

        return weights, raw_metrics, global_stats

    def _compute_stats_metrics(self, metrics_container, proj_dict, prefix):
        """计算 L2 和 Variance"""
        for cid, proj_tensor in proj_dict.items():
            vec = proj_tensor.float()
            metrics_container[cid][f'{prefix}_l2'] = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_var'] = torch.var(vec).item()

    def _perform_clustering_and_dist(self, metrics_container, proj_dict, prefix):
        """执行聚类并计算 Cosine Distance"""
        cids = list(proj_dict.keys())
        if len(cids) < 3:
            for cid in cids: 
                metrics_container[cid][f'{prefix}_cluster'] = 0
                metrics_container[cid][f'{prefix}_dist'] = 0.0
            return

        matrix = torch.stack([proj_dict[cid] for cid in cids])
        # L2 归一化
        matrix_norm = torch.nn.functional.normalize(matrix.float(), p=2, dim=1).cpu().numpy()
        
        labels = []
        dists = np.zeros(len(cids))

        if self.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(matrix_norm)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # 多数投票
            counts = np.bincount(labels)
            major_label = np.argmax(counts)
            major_center = centers[major_label]
            
            # 计算 Cosine Distance = 1 - CosineSimilarity
            major_center_norm = major_center / (np.linalg.norm(major_center) + 1e-9)
            cos_sims = np.dot(matrix_norm, major_center_norm)
            dists = 1.0 - cos_sims
            dists = np.maximum(dists, 0.0)
            
            labels = np.where(labels == major_label, 0, -1)

        elif self.clustering_method == 'dbscan':
            distance_matrix = cosine_distances(matrix_norm)
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='precomputed').fit(distance_matrix)
            labels = clustering.labels_
            dists = np.where(labels == -1, 1.0, 0.0)

        for i, cid in enumerate(cids):
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])
            metrics_container[cid][f'{prefix}_dist'] = float(dists[i])

    def calculate_final_scores(self, raw_metrics, suspect_counters, verbose=False):
        weights = {}
        cids = list(raw_metrics.keys())
        global_stats = {} # 存储阈值信息
        
        if not cids: return {}, suspect_counters, global_stats
        
        sample_metrics = raw_metrics[cids[0]]
        l2_keys = [k for k in sample_metrics.keys() if k.endswith('_l2')]
        var_keys = [k for k in sample_metrics.keys() if k.endswith('_var')]
        dist_keys = [k for k in sample_metrics.keys() if k.endswith('_dist')]

        # 计算阈值
        stats_cache = {} 

        # 硬筛查 (L2, Var)
        for key in l2_keys + var_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            multiplier = self.l2_multiplier if 'l2' in key else self.var_multiplier
            thresh = med + multiplier * max(mad, 1e-6)
            stats_cache[key] = {'med': med, 'mad': mad, 'thresh': thresh}
            
            # [修改] 记录每一个指标的阈值到 global_stats，不仅仅是 full
            # key 可能是 'full_l2' 或 'layer_conv1.weight_l2'
            global_stats[f'{key}_threshold'] = thresh

        # 软筛查 (Distance)
        for key in dist_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            thresh = med + self.dist_multiplier * max(mad, 1e-6)
            stats_cache[key] = {'med': med, 'mad': mad, 'thresh': thresh}
            
            # [修改] 记录阈值
            global_stats[f'{key}_threshold'] = thresh

        # 评分
        for cid in cids:
            metrics = raw_metrics[cid]
            scores_list = []      
            suspect_reasons = []
            
            # Hard Screening
            for key in l2_keys + var_keys:
                val = metrics[key]
                info = stats_cache[key]
                if val > info['thresh']:
                    scores_list.append(self.suspect_score)
                    tag = "L2" if "l2" in key else "Var"
                    suspect_reasons.append(f"{key.replace(f'_{tag.lower()}','')}:{tag}")
                else:
                    z_score = abs(val - info['med']) / (info['mad'] + 1e-6)
                    score = self.base_good_score + self.max_bonus * np.exp(-self.score_decay * z_score)
                    scores_list.append(score)

            # Soft Screening
            for key in dist_keys:
                val = metrics[key]
                info = stats_cache[key]
                if val > info['thresh']:
                    scores_list.append(self.suspect_score)
                    suspect_reasons.append(f"{key.replace('_dist','')}:Dist({val:.3f})")
                else:
                    z_score = abs(val - info['med']) / (info['mad'] + 1e-6)
                    score = self.base_good_score + self.max_bonus * np.exp(-self.score_decay * z_score)
                    scores_list.append(score)

            final_score = min(scores_list) if scores_list else self.base_good_score
            is_suspect = (final_score <= self.suspect_score + 1e-4)

            if is_suspect:
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                reason_str = ",".join(suspect_reasons)
                # 日志如果不限制长度，可以记录更多细节
                raw_metrics[cid]['status'] = f'SUSPECT({reason_str})'
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score
            
            if verbose and is_suspect:
                print(f"      [Alert] Client {cid} suspect: {suspect_reasons} (Score: {final_score:.2f})")

        return weights, suspect_counters, global_stats

    def _calc_robust_stats(self, values):
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad