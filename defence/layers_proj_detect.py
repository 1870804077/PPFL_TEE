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
        
        # --- 1. 读取聚类参数 ---
        self.clustering_method = self.config.get('clustering_method', 'dbscan')
        self.dbscan_eps = self.config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = self.config.get('dbscan_min_samples', 3)
        
        # --- 2. 读取评分参数 ---
        self.base_good_score = self.config.get('base_good_score', 10.0)
        self.suspect_score = self.config.get('suspect_score', 1.0)
        self.max_bonus = self.config.get('max_bonus', 10.0)
        self.strike_threshold = self.config.get('strike_threshold', 3)
        
        # --- 3. 读取统计阈值倍数 ---
        self.l2_multiplier = self.config.get('l2_threshold_multiplier', 3.0)
        self.var_multiplier = self.config.get('var_threshold_multiplier', 3.0)

    def detect(self, client_projections, global_history_projections, suspect_counters, verbose=False):
        """
        执行检测流程 (增加 verbose 参数)
        """
        active_client_ids = list(client_projections.keys())
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # 1. 全量处理 (Full Projection)
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        history_tensor = global_history_projections if isinstance(global_history_projections, torch.Tensor) else None
        
        self._compute_basic_metrics(
            raw_metrics, 
            full_proj_dict, 
            history_tensor,
            prefix='full'
        )
        self._perform_clustering(raw_metrics, full_proj_dict, prefix='full')

        # 2. 分层处理 (Layer-wise) - 目前主要用于聚类辅助，评分逻辑暂主要依赖 Full
        if active_client_ids:
            first_cid = active_client_ids[0]
            if 'layers' in client_projections[first_cid]:
                layer_keys = client_projections[first_cid]['layers'].keys()
                for layer_name in layer_keys:
                    layer_proj_dict = {cid: client_projections[cid]['layers'][layer_name] for cid in active_client_ids}
                    self._compute_basic_metrics(raw_metrics, layer_proj_dict, None, prefix=f'layer_{layer_name}')
                    self._perform_clustering(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        # 3. 计算最终分数 (传递 verbose)
        weights = self.calculate_final_scores(raw_metrics, suspect_counters, verbose=verbose)

        return weights, raw_metrics

    def _compute_basic_metrics(self, metrics_container, proj_dict, history_tensor, prefix):
        for cid, proj_tensor in proj_dict.items():
            vec = proj_tensor.float()
            metrics_container[cid][f'{prefix}_l2'] = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_var'] = torch.var(vec).item()
            
            if history_tensor is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), 
                    history_tensor.float().to(vec.device).unsqueeze(0)
                ).item()
                metrics_container[cid][f'{prefix}_hist_cos'] = cos_sim
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
            clustering = DBSCAN(
                eps=self.dbscan_eps, 
                min_samples=self.dbscan_min_samples, 
                metric='precomputed'
            ).fit(distance_matrix)
            labels = clustering.labels_
        elif self.clustering_method == 'kmeans':
            clustering = KMeans(n_clusters=2, random_state=42).fit(matrix_norm)
            labels = clustering.labels_
            counts = np.bincount(labels)
            major_label = np.argmax(counts)
            labels = np.where(labels == major_label, 0, -1)

        for i, cid in enumerate(cids):
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])

    def calculate_final_scores(self, raw_metrics, suspect_counters, verbose=False):
        weights = {}
        cids = list(raw_metrics.keys())
        if not cids: return {}
        
        # 1. 准备全局统计量
        full_l2_values = [raw_metrics[cid]['full_l2'] for cid in cids]
        full_var_values = [raw_metrics[cid]['full_var'] for cid in cids]
        
        l2_median, l2_mad = self._calc_robust_stats(full_l2_values)
        var_median, var_mad = self._calc_robust_stats(full_var_values)

        # 计算阈值
        l2_threshold = l2_median + self.l2_multiplier * max(l2_mad, 1e-5)
        var_threshold = var_median + self.var_multiplier * max(var_mad, 1e-5)

        # 调试打印：全局阈值信息
        if verbose:
            print(f"    [Debug] 统计阈值详情:")
            print(f"      > L2 Norm : Median={l2_median:.4f} | MAD={l2_mad:.4f} | Multiplier={self.l2_multiplier} => Threshold={l2_threshold:.4f}")
            print(f"      > Variance: Median={var_median:.4f} | MAD={var_mad:.4f} | Multiplier={self.var_multiplier} => Threshold={var_threshold:.4f}")

        for cid in cids:
            metrics = raw_metrics[cid]
            is_suspect = False
            suspect_reasons = []

            # A. 硬筛除
            if metrics['full_l2'] > l2_threshold:
                is_suspect = True
                suspect_reasons.append("L2")
            
            if metrics['full_var'] > var_threshold:
                is_suspect = True
                suspect_reasons.append("Var")
                
            if metrics.get('full_cluster') == -1:
                is_suspect = True
                suspect_reasons.append("Clust")
            
            # B. 权重计算
            cos_sim = metrics.get('full_hist_cos', 0)
            
            if is_suspect:
                final_score = self.suspect_score 
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                bonus = self.max_bonus * np.tanh(max(0, cos_sim)) 
                final_score = self.base_good_score + bonus 
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            # C. 判定状态
            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                raw_metrics[cid]['status'] = f'SUSPECT({",".join(suspect_reasons)})'
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score
            
            # 调试打印：客户端详细数据
            if verbose:
                mark = "🔴" if is_suspect else "🟢"
                print(f"      {mark} Client {cid:<2}: L2={metrics['full_l2']:.4f} | Var={metrics['full_var']:.4f} | "
                      f"Cos={cos_sim:.4f} | Clust={metrics.get('full_cluster')} | "
                      f"Score={final_score:.2f} -> {raw_metrics[cid]['status']}")

        return weights

    def _calc_robust_stats(self, values):
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad