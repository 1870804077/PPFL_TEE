import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances

class Layers_Proj_Detector:
    def __init__(self, clustering_method='dbscan', dbscan_eps=0.5, dbscan_min_samples=3):
        """
        初始化检测器
        :param clustering_method: 'dbscan' or 'kmeans'
        :param dbscan_eps: DBSCAN 的邻域半径 (基于余弦距离，范围0-2)
        :param dbscan_min_samples: DBSCAN 的核心点最小样本数
        """
        self.clustering_method = clustering_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def detect(self, client_projections, global_history_projections, active_client_ids):
        """
        执行 MESAS 风格的投影检测
        
        :param client_projections: dict, {cid: {'full': tensor, 'layers': {'layer_name': tensor, ...}}}
        :param global_history_projections: dict, {'full': tensor, 'layers': {...}} (上一轮的累加结果)
        :param active_client_ids: list, 本轮参与的客户端 ID 列表
        :return: 
            raw_metrics: dict, 包含每位客户端的所有计算指标（L2, Var, Cos, Cluster_Label等），用于后续打分
        """
        
        # 结果容器 {cid: {metric_name: value}}
        raw_metrics = {cid: {} for cid in active_client_ids}
        
        # 1. 提取全量投影并处理
        # ------------------------------------------------------
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


        # 2. 提取分层投影并处理 (如果有)
        # ------------------------------------------------------
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
            # 确保是 float 类型
            vec = proj_tensor.float()
            
            # 1. 计算 L2 范数 (Scaling 检测)
            l2_norm = torch.norm(vec, p=2).item()
            metrics_container[cid][f'{prefix}_l2'] = l2_norm
            
            # 2. 计算方差 (分布异常检测)
            variance = torch.var(vec).item()
            metrics_container[cid][f'{prefix}_var'] = variance
            
            # 3. 计算与上一轮历史的余弦相似度 (方向突变检测)
            if history_tensor is not None:
                # 归一化后计算点积即为余弦相似度
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
        # 注意：这里需要归一化，因为我们关注的是方向一致性 (Cosine Distance)
        matrix = torch.stack([proj_dict[cid] for cid in cids])
        matrix_norm = torch.nn.functional.normalize(matrix, p=2, dim=1).cpu().numpy()
        
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
            # KMeans 假设 2 类 (正常/异常)，但通常不知道比例，效果不如 DBSCAN
            # 这里简单实现为 2 类
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
    # 评分逻辑接口 (留待下一步完善)
    # -------------------------------------------------------------------------
    def calculate_final_scores(self, raw_metrics, suspect_counters):
        """
        基于 raw_metrics 和 suspect_counters 计算最终权重
        :return: weights_dict {cid: score}, updated_suspect_counters
        """
        # 暂时返回全 1，等待讨论细则
        weights = {cid: 1.0 for cid in raw_metrics.keys()}
        return weights, suspect_counters