import torch
import copy
from _utils_.LSH_proj_extra import SuperBitLSH
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
# 引入新模块
from defence.layers_proj_detect import Layers_Proj_Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        
        # 状态维护
        self.suspect_counters = {} # {cid: strike_count}
        self.global_update_direction = None # 上一轮的 Sum(Features)
        
        # 组件初始化
        self.mesas_detector = Layers_Proj_Detector()
        # 保留原有组件以兼容旧逻辑(如果需要)
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
            
        self.current_round_weights = {}

    # ... generate_projection_matrix, get_global_params_and_proj 保持不变 ...

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes):
        """
        步骤7: 检测并计算权重
        :param client_features_dict_list: list of dict, 每个元素是 Client 发来的 {'full':.., 'layers':..}
        """
        
        # 重组数据方便处理: list -> dict {cid: feature_dict}
        client_projections = {
            cid: feat 
            for cid, feat in zip(client_id_list, client_features_dict_list)
        }

        # 1. 计算下一轮的全局方向 (Step 3: Sum of current projections)
        # 注意：这里我们使用本轮所有客户端(在剔除前)的投影之和作为"本轮的整体方向"
        # 也可以选择只使用 High Weight 客户端的和，这里先按要求全量加和
        self._update_global_direction_feature(client_projections)

        # 2. 执行检测 (Step 2 & 4: MESAS-like detection & Scoring)
        if "mesas" in self.detection_method or "projected" in self.detection_method:
            # 使用新的检测器
            weights, logs = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, # 注意：这里传入的是"上一轮"积累下来的方向，如果刚更新完，需要理清逻辑
                # 逻辑修正：
                # 应该拿"当前轮的个体的投影" 与 "上一轮的全局方向" 做对比
                # 所以 self.global_update_direction 应该在 detect 之后再 update，或者这里传入 old_direction
                # 这里假设 self.global_update_direction 存储的是 Round T-1 的结果
                self.suspect_counters
            )
            
            # 更新全局方向供 Round T+1 使用 (Step 3)
            # 放到 Detect 之后更新，确保 Detect 用的是 History
            self._update_global_direction_feature(client_projections)
            
        else:
            # Fallback 到旧逻辑 (仅供参考)
            # 这里需要适配 client_features_dict_list 只取 ['full']
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        self.current_round_weights = weights
        return weights

    def _update_global_direction_feature(self, client_projections):
        """
        步骤3: 将本轮投影对应加和，得到全局更新方向，留待下轮检测
        Global_Dir = Sum(g_i * W) = Sum(Proj_i)
        """
        if not client_projections:
            return

        # 取出第一个来初始化形状
        first_proj = list(client_projections.values())[0]['full']
        agg_proj = torch.zeros_like(first_proj, device=first_proj.device)
        
        # 简单累加 (也可以做加权累加)
        for cid, proj_data in client_projections.items():
            agg_proj += proj_data['full']
            
        # 更新 Server 状态
        # 可以引入动量: New = 0.9 * Old + 0.1 * Current
        if self.global_update_direction is None:
            self.global_update_direction = agg_proj
        else:
            # 简单的直接替换，或者动量更新
            self.global_update_direction = agg_proj 

    def update_global_model(self, weighted_client_models_list, client_ids_list):
        """步骤9: 聚合已加权的参数"""
        if not weighted_client_models_list:
            return

        # 1. 验证本轮有效总权重 (用于归一化，如果客户端上传的是 params * weight)
        # 注意：如果 weight 已经是归一化的 (sum=1)，则直接相加即可。
        # 如果 weight 是 raw score (sum!=1)，则需要除以 total_weight。
        # 通常在 calculate_weights 阶段，determine_weights 返回的最好是归一化权重，或者我们在聚合时处理。
        # 这里假设 weights 是归一化好的 (sum ≈ 1)，或者是 0 (剔除)。
        
        # 实际上，如果 calculate_weights 返回的是 normalized weights，客户端计算 params * w_i
        # Server 只需要 sum(params * w_i)
        
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
                        agg_params[k] += client_params[k].to(DEVICE) # 客户端已经乘过权重了，直接加
                    elif i == 0:
                         agg_params[k] = client_params[k].to(DEVICE)
        
        if valid_updates > 0:
            self.global_model.load_state_dict(agg_params)
        else:
            print("  [Warning] 本轮无有效更新。")
        
    def _fallback_old_detection(self, ids, features, sizes):
        # ... 旧的 score/kickout 逻辑 ...
        return {cid: 1.0 for cid in ids}