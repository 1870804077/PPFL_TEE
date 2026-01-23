import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances

class Layers_Proj_Detector:
    def __init__(self, clustering_method='dbscan', dbscan_eps=0.5, dbscan_min_samples=3):
        """
        初始化检测器
        :param clustering_method: 'dbscan' 或 'kmeans'
        :param dbscan_eps: DBSCAN 的邻域半径 (基于余弦距离，范围0-2)
        :param dbscan_min_samples: DBSCAN 的核心点最小样本数
        """
        self.clustering_method = clustering_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        # --- 评分超参数配置 ---
        self.base_good_score = 10.0   # 良性客户端的基础分
        self.suspect_score = 1.0      # 疑似投毒者的惩罚分 (最低权重)
        self.max_bonus = 10.0         # 软打分的最大加分上限
        self.strike_threshold = 3     # 累计几次疑似后判定为投毒并剔除

    def detect(self, client_projections, global_history_projections, active_client_ids):
        """
        执行 MESAS 风格的投影检测
        
        :param client_projections: dict, {cid: {'full': tensor, 'layers': {'layer_name': tensor, ...}}}
        :param global_history_projections: dict, {'full': tensor, 'layers': {...}} (上一轮的累加方向)
        :param active_client_ids: list, 本轮参与的客户端 ID 列表
        :return: 
            raw_metrics: dict, 包含每位客户端的所有计算指标（L2, Var, Cos, Cluster_Label等），用于后续打分
        """
        
        # 结果容器 {cid: {metric_name: value}}
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # =========================================================
        # 1. 提取全量投影并处理 (Full Projection)
        # =========================================================
        full_proj_dict = {cid: client_projections[cid]['full'] for cid in active_client_ids}
        
        # 计算基础统计量 (L2, Variance) & 历史相似度
        self._compute_basic_metrics(
            raw_metrics, 
            full_proj_dict, 
            global_history_projections.get('full') if global_history_projections else None,
            prefix='full'
        )
        
        # 执行聚类 (全量)
        self._perform_clustering(raw_metrics, full_proj_dict, prefix='full')


        # =========================================================
        # 2. 提取分层投影并处理 (Layer-wise Projection)
        # =========================================================
        # 假设所有客户端上传的 layers 结构一致，取第一个客户端的 keys
        first_cid = active_client_ids[0]
        layer_keys = client_projections[first_cid].get('layers', {}).keys()
        
        for layer_name in layer_keys:
            # 提取该层的投影数据
            layer_proj_dict = {
                cid: client_projections[cid]['layers'][layer_name] 
                for cid in active_client_ids
            }
            
            # 获取该层的历史方向 (如果有)
            layer_history = None
            if global_history_projections and 'layers' in global_history_projections:
                layer_history = global_history_projections['layers'].get(layer_name)
            
            # 计算基础统计量
            self._compute_basic_metrics(
                raw_metrics, 
                layer_proj_dict, 
                layer_history, 
                prefix=f'layer_{layer_name}'
            )
            
            # 执行聚类
            self._perform_clustering(raw_metrics, layer_proj_dict, prefix=f'layer_{layer_name}')

        return raw_metrics

    def _compute_basic_metrics(self, metrics_container, proj_dict, history_tensor, prefix):
        """
        辅助函数：计算 L2, Variance, Cosine Similarity
        """
        for cid, proj_tensor in proj_dict.items():
            # 确保是 float 类型，避免半精度问题
            vec = proj_tensor.float()
            
            # 1. 计算 L2 范数 (检测 Scaling 攻击 / 梯度爆炸)
            l2_norm = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_l2'] = l2_norm
            
            # 2. 计算方差 (检测分布异常 / 噪声注入)
            variance = torch.var(vec).item()
            metrics_container[cid][f'{prefix}_var'] = variance
            
            # 3. 计算与上一轮历史的余弦相似度 (检测方向突变)
            if history_tensor is not None:
                # 归一化后计算点积即为余弦相似度
                # history_tensor 也需要转 float
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), 
                    history_tensor.float().unsqueeze(0)
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
        # 注意：这里进行 L2 归一化，因为 DBSCAN 使用的是余弦距离
        matrix = torch.stack([proj_dict[cid] for cid in cids])
        matrix_norm = torch.nn.functional.normalize(matrix.float(), p=2, dim=1).cpu().numpy()
        
        # 计算余弦距离矩阵 (DBSCAN precomputed 需要距离矩阵)
        # Cosine Distance = 1 - Cosine Similarity
        distance_matrix = cosine_distances(matrix_norm)
        
        labels = []
        if self.clustering_method == 'dbscan':
            # metric='precomputed' 表示输入是距离矩阵
            clustering = DBSCAN(
                eps=self.dbscan_eps, 
                min_samples=self.dbscan_min_samples, 
                metric='precomputed'
            ).fit(distance_matrix)
            labels = clustering.labels_
            
        elif self.clustering_method == 'kmeans':
            # KMeans 简单分为 2 类 (正常/异常)
            clustering = KMeans(n_clusters=2, random_state=42).fit(matrix_norm)
            labels = clustering.labels_
            
            # 简单的启发式：假设样本多的一类是正常的 (Label 0)
            counts = np.bincount(labels)
            major_label = np.argmax(counts)
            # 将主类标记为 0，少数类标记为 -1 (模拟 DBSCAN 的噪声)
            labels = np.where(labels == major_label, 0, -1)

        # 记录结果
        for i, cid in enumerate(cids):
            # Label: 0 (Normal Cluster), -1 (Noise/Outlier), 1, 2... (Other Clusters)
            metrics_container[cid][f'{prefix}_cluster'] = int(labels[i])

    # -------------------------------------------------------------------------
    # 核心评分逻辑
    # -------------------------------------------------------------------------
    def calculate_final_scores(self, raw_metrics, suspect_counters):
        """
        基于 Hard Filtering (统计离群) + Soft Scoring (非线性映射) 计算权重
        :param raw_metrics: detect 返回的指标字典
        :param suspect_counters: dict {cid: count}, 历史嫌疑计数
        :return: weights_dict {cid: score}, updated_suspect_counters
        """
        weights = {}
        cids = list(raw_metrics.keys())
        if not cids:
            return {}, suspect_counters
        
        # 1. 准备全局统计量 (使用 Median 和 MAD 替代 Mean/Std 以防投毒干扰)
        # 我们主要关注全量投影的 L2 和 Var 统计分布
        full_l2_values = [raw_metrics[cid]['full_l2'] for cid in cids]
        full_var_values = [raw_metrics[cid]['full_var'] for cid in cids]
        
        l2_median, l2_mad = self._calc_robust_stats(full_l2_values)
        var_median, var_mad = self._calc_robust_stats(full_var_values)

        # 设定硬阈值 (例如 Median + 3 * MAD)
        # 如果 MAD 极小(大家都一样)，给一个极小底数防止过于敏感
        l2_threshold = l2_median + 3 * max(l2_mad, 1e-5)
        var_threshold = var_median + 3 * max(var_mad, 1e-5)

        for cid in cids:
            metrics = raw_metrics[cid]
            is_suspect = False
            suspect_reasons = []

            # --- A. 硬筛除 (Hard Filtering) ---
            
            # 规则1: L2 范数异常 (梯度爆炸/Scaling攻击)
            if metrics['full_l2'] > l2_threshold:
                is_suspect = True
                suspect_reasons.append("L2_Outlier")
            
            # 规则2: 方差异常 (噪声注入)
            if metrics['full_var'] > var_threshold:
                is_suspect = True
                suspect_reasons.append("Var_Outlier")
                
            # 规则3: 聚类筛除 (DBSCAN Noise)
            # 判定为噪声 (-1) 即为离群
            if metrics.get('full_cluster') == -1:
                is_suspect = True
                suspect_reasons.append("Cluster_Noise")
            
            # --- B. 权重计算 ---
            
            if is_suspect:
                # 判定为疑似: 权重降级为 1.0，并增加嫌疑计数
                final_score = self.suspect_score 
                suspect_counters[cid] = suspect_counters.get(cid, 0) + 1
            else:
                # 判定为良性: 计算软得分 (Soft Scoring)
                # 核心依据: 与历史全局方向的余弦相似度 (越高越好)
                cos_sim = metrics.get('full_hist_cos', 0)
                
                # 非线性映射: Tanh
                # Formula: Base(10) + Max(10) * tanh(sim)
                # 范围约 [10, 20]，不仅远大于疑似者，且对高相似度有收益递减效应
                bonus = self.max_bonus * np.tanh(max(0, cos_sim)) 
                final_score = self.base_good_score + bonus 
                
                # 软策略：表现良好的可以缓慢减少嫌疑计数 (Allow redemption)
                if suspect_counters.get(cid, 0) > 0:
                    suspect_counters[cid] = max(0, suspect_counters[cid] - 0.5)

            # --- C. 最终累计判定 (Strike System) ---
            # 累计达到阈值则剔除 (Score = 0)
            if suspect_counters.get(cid, 0) >= self.strike_threshold:
                final_score = 0.0
                raw_metrics[cid]['status'] = 'KICKED'
            elif is_suspect:
                raw_metrics[cid]['status'] = f'SUSPECT({",".join(suspect_reasons)})'
            else:
                raw_metrics[cid]['status'] = 'NORMAL'

            weights[cid] = final_score

        return weights, suspect_counters

    def _calc_robust_stats(self, values):
        """
        计算中位数和绝对中位差 (MAD)
        MAD = median(|x - median|)
        """
        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return median, mad