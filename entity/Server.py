import torch
import copy
from _utils_.LSH_proj_extra import SuperBitLSH
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
# 确保引用路径正确
from defence.layers_proj_detect import Layers_Proj_Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42, verbose=False):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.verbose = verbose  # 保存日志配置
        
        self.suspect_counters = {} 
        self.global_update_direction = None 
        
        self.mesas_detector = Layers_Proj_Detector()
        
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
            
        self.current_round_weights = {}

    # ... (generate_projection_matrix, get_global_params_and_proj 保持不变) ...

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes):
        client_projections = {
            cid: feat 
            for cid, feat in zip(client_id_list, client_features_dict_list)
        }

        self._update_global_direction_feature(client_projections)
        weights = {}

        if "mesas" in self.detection_method or "projected" in self.detection_method:
            if self.verbose:
                print(f"  [Server] Executing {self.detection_method} detection...")

            raw_weights, logs = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, 
                self.suspect_counters
            )
            
            # [新增] 详细打印检测结果
            if self.verbose:
                print(f"  [Server] Detection Results:")
                for cid, status in logs.items():
                    # 假设 logs 返回的是 raw_metrics，其中包含 status 字段
                    st = status.get('status', 'UNKNOWN') if isinstance(status, dict) else 'N/A'
                    score = raw_weights.get(cid, 0.0)
                    print(f"    - Client {cid}: Score={score:.2f} | Status={st}")

            total_score = sum(raw_weights.values())
            if total_score > 0:
                weights = {cid: s / total_score for cid, s in raw_weights.items()}
            else:
                print("  [Warning] All clients kicked out this round!")
                weights = {cid: 0.0 for cid in raw_weights}
            
            self._update_global_direction_feature(client_projections)
            
        else:
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        self.current_round_weights = weights
        return weights

    def _update_global_direction_feature(self, client_projections):
        """
        步骤3: 更新全局方向
        """
        if not client_projections:
            return

        first_proj = list(client_projections.values())[0]['full']
        agg_proj = torch.zeros_like(first_proj, device=first_proj.device)
        
        # 简单累加
        for cid, proj_data in client_projections.items():
            agg_proj += proj_data['full']
            
        if self.global_update_direction is None:
            self.global_update_direction = agg_proj
        else:
            # 直接替换 (无动量)
            self.global_update_direction = agg_proj 

    def update_global_model(self, weighted_client_models_list, client_ids_list):
        """步骤9: 聚合已加权的参数"""
        if not weighted_client_models_list:
            return

        first_params = weighted_client_models_list[0]
        agg_params = {k: torch.zeros_like(v, dtype=v.dtype, device=DEVICE) for k, v in first_params.items()}

        valid_updates = 0
        for i, cid in enumerate(client_ids_list):
            w = self.current_round_weights.get(cid, 0.0)
            if w > 0:
                valid_updates += 1
                client_params = weighted_client_models_list[i]
                for k in agg_params.keys():
                    if agg_params[k].dtype in [torch.float32, torch.float64]:
                        agg_params[k] += client_params[k].to(DEVICE)
                    elif i == 0:
                         agg_params[k] = client_params[k].to(DEVICE)
        
        if valid_updates > 0:
            self.global_model.load_state_dict(agg_params)
        else:
            print("  [Warning] 本轮无有效更新。")
        
        
    def evaluate(self, test_loader):
        """
        模型评估
        :param test_loader: 测试数据集加载器
        :return: 测试准确率 (0-100)
        """
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total
        
    def _fallback_old_detection(self, ids, features, sizes):
        """
        兼容旧的 score + kickout 逻辑，以及无防御模式(FedAvg)
        """
        # Case 1: 无防御模块 (FedAvg)
        if not self.score_calculator and not self.kickout_manager:
            total_size = sum(sizes)
            if total_size > 0:
                return {cid: size / total_size for cid, size in zip(ids, sizes)}
            else:
                return {cid: 1.0 / len(ids) for cid in ids}

        # Case 2: 只有 Kickout (无 Score) - 极其少见，暂不支持或退化为均权
        if self.kickout_manager and not self.score_calculator:
             return {cid: 1.0 / len(ids) for cid in ids}

        # Case 3: 正常流程 Score -> Kickout
        # 3.1 计算分数
        client_scores = {}
        for i, cid in enumerate(ids):
            client_scores[cid] = self.score_calculator.calculate_scores(
                cid, features[i], sizes[i]
            )

        # 3.2 确定权重
        weights = {}
        if self.kickout_manager:
            # KickoutManager 内部已经包含了归一化逻辑 (return normalized weights)
            weights = self.kickout_manager.determine_weights(client_scores)
        else:
            # Case 4: 只有 Score (无 Kickout)
            # 直接使用 final_score 并归一化
            raw_scores = {cid: s['final_score'] for cid, s in client_scores.items()}
            total_s = sum(raw_scores.values())
            if total_s > 0:
                weights = {cid: s / total_s for cid, s in raw_scores.items()}
            else:
                weights = {cid: 1.0 / len(ids) for cid in ids}
                
        return weights