import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances

class Layers_Proj_Detector:
    def __init__(self, config=None):
        self.config = config or {}
        
        self.clustering_method = self.config.get('clustering_method', 'dbscan')
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 3)
        self.base_good_score = self.config.get('base_good_score', 10.0)
        self.suspect_score = self.config.get('suspect_score', 1.0)
        self.max_bonus = self.config.get('max_bonus', 10.0)
        self.strike_threshold = self.config.get('strike_threshold', 3)
        self.l2_multiplier = self.config.get('l2_threshold_multiplier', 3.0)
        self.var_multiplier = self.config.get('var_threshold_multiplier', 3.0)

    def detect(self, client_projections, global_history_projections, suspect_counters, verbose=False):
        active_client_ids = list(client_projections.keys())
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # 1. 全量处理
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        
        # 获取 Full History (如果 global_history 是字典)
        full_hist = None
        if isinstance(global_history_projections, dict):
            full_hist = global_history_projections.get('full')
        elif isinstance(global_history_projections, torch.Tensor):
            full_hist = global_history_projections
            
        self._compute_basic_metrics(raw_metrics, full_proj_dict, full_hist, prefix='full')
        self._perform_clustering(raw_metrics, full_proj_dict, prefix='full')

        # 2. 分层处理
        if active_client_ids:
            first_cid = active_client_ids[0]
            if 'layers' in client_projections[first_cid]:
                for layer_name in client_projections[first_cid]['layers'].keys():
                    layer_proj_dict = {cid: client_projections[cid]['layers'][layer_name] for cid in active_client_ids}
                    
                    # [关键] 获取对应层的历史
                    layer_hist = None
                    if isinstance(global_history_projections, dict) and 'layers' in global_history_projections:
                        layer_hist = global_history_projections['layers'].get(layer_name)
                    
                    self._compute_basic_metrics(raw_metrics, layer_proj_dict, layer_hist, prefix=f'layer_{layer_name}')
                    self._perform_clustering(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        # 3. 计算最终分数
        weights, suspect_counters, global_stats = self.calculate_final_scores(raw_metrics, suspect_counters, verbose=verbose)

        return weights, raw_metrics, global_stats

    def _compute_basic_metrics(self, metrics_container, proj_dict, history_tensor, prefix):
        for cid, proj_tensor in proj_dict.items():
            vec = proj_tensor.float()
            metrics_container[cid][f'{prefix}_l2'] = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_var'] = torch.var(vec).item()
            if history_tensor is not None:
                # 确保在同一 device
                hist = history_tensor.float().to(vec.device)
                metrics_container[cid][f'{prefix}_hist_cos'] = torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), hist.unsqueeze(0)
                ).item()
            else:
                metrics_container[cid][f'{prefix}_hist_cos'] = 1.0

    def _perform_clustering(self, metrics_container, proj_dict, prefix):
        cids = list(proj_dict.keys())
        if len(cids) < 3:
            for cid in cids: metrics_container[cid][f'{prefix}_cluster'] = 0
            return

        matrix = torch.stack([proj_dict[cid] for cid in cids])
        matrix_norm = torch.nn.functional.normalize(matrix.float(), p=2, dim=1).cpu().numpy()
        distance_matrix = cosine_distances(matrix_norm)
        
        labels = []
        if self.clustering_method == 'dbscan':
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='precomputed').fit(distance_matrix)
            labels = clustering.labels_
        elif self.clustering_method == 'kmeans':
            clustering = KMeans(n_clusters=2, random_state=42, n_init=10).fit(matrix_norm)
            raw_labels = clustering.labels_
            counts = np.bincount(raw_labels)
            major_label = np.argmax(counts)
            labels = np.where(raw_labels == major_label, 0, -1)

        for i, cid in enumerate(cids):
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])

    def calculate_final_scores(self, raw_metrics, suspect_counters, verbose=False):
        weights = {}
        cids = list(raw_metrics.keys())
        global_stats = {
            'l2_median': 0, 'l2_mad': 0, 'l2_threshold': 0,
            'var_median': 0, 'var_mad': 0, 'var_threshold': 0
        }
        
        if not cids: return {}, suspect_counters, global_stats
        
        # 1. 动态 Key
        sample_metrics = raw_metrics[cids[0]]
        l2_keys = [k for k in sample_metrics.keys() if k.endswith('_l2')]
        var_keys = [k for k in sample_metrics.keys() if k.endswith('_var')]
        clust_keys = [k for k in sample_metrics.keys() if k.endswith('_cluster')]
        cos_keys = [k for k in sample_metrics.keys() if k.endswith('_hist_cos')] # [新增]

        # 2. 计算阈值
        thresholds = {} 
        for key in l2_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            thresh = med + self.l2_multiplier * max(mad, 1e-5)
            thresholds[key] = thresh
            if key == 'full_l2':
                global_stats['l2_median'] = med
                global_stats['l2_mad'] = mad
                global_stats['l2_threshold'] = thresh

        for key in var_keys:
            values = [raw_metrics[cid][key] for cid in cids]
            med, mad = self._calc_robust_stats(values)
            thresh = med + self.var_multiplier * max(mad, 1e-5)
            thresholds[key] = thresh
            if key == 'full_var':
                global_stats['var_median'] = med
                global_stats['var_mad'] = mad
                global_stats['var_threshold'] = thresh

        # 3. 评分
        for cid in cids:
            metrics = raw_metrics[cid]
            is_suspect = False
            suspect_reasons = []

            # --- A. 指标检测 (One-Vote Veto) ---
            for key in l2_keys:
                if metrics[key] > thresholds[key]:
                    is_suspect = True
                    suspect_reasons.append(f"{key.replace('_l2','')}:L2")

            for key in var_keys:
                if metrics[key] > thresholds[key]:
                    is_suspect = True
                    suspect_reasons.append(f"{key.replace('_var','')}:Var")

            # --- B. 综合聚类状态 ---
            combined_cluster = 0
            for key in clust_keys:
                if metrics.get(key) == -1:
                    is_suspect = True
                    combined_cluster = -1 # 记录为异常
                    suspect_reasons.append(f"{key.replace('_cluster','')}:Clust")
            
            # 记录到 metrics 以便日志写入
            raw_metrics[cid]['combined_cluster'] = combined_cluster

            # --- C. 综合相似度 (最低分) ---
            min_cos_sim = 1.0
            if cos_keys:
                # 找出所有 cosine 中的最小值
                all_cos = [metrics.get(k, 1.0) for k in cos_keys]
                min_cos_sim = min(all_cos)
            raw_metrics[cid]['min_hist_cos'] = min_cos_sim

            # --- D. 算分 ---
            if is_suspect:
                final_score = self.suspect_score 
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                # [关键] 使用最低相似度计算 Bonus
                bonus = self.max_bonus * np.tanh(max(0, min_cos_sim)) 
                final_score = self.base_good_score + bonus 
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                reason_str = ",".join(suspect_reasons)
                if len(reason_str) > 50: reason_str = reason_str[:47] + "..."
                raw_metrics[cid]['status'] = f'SUSPECT({reason_str})'
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score
            
            # 调试打印：客户端详细数据
            if verbose:
                mark = "🔴" if is_suspect else "🟢"
                print(f"      {mark} Client {cid:<2}: L2={metrics['full_l2']:.4f} | Var={metrics['full_var']:.4f} | "
                      f"Cos={min_cos_sim:.4f} | Clust={metrics.get('full_cluster')} | "
                      f"Score={final_score:.2f} -> {raw_metrics[cid]['status']}")

        return weights, suspect_counters, global_stats

    def _calc_robust_stats(self, values):
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad