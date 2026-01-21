import torch
import copy
from _utils_.LSH_proj_extra import SuperBitLSH
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.seed = seed
        
        # 初始化防御组件
        self.score_calculator = None
        self.kickout_manager = None
        
        if "score" in self.detection_method:
            self.score_calculator = ScoreCalculator()
        if "kickout" in self.detection_method:
            self.kickout_manager = KickoutManager()
            
        # 存储本轮的临时权重
        self.current_round_weights = {}

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        """步骤1: 生成投影矩阵"""
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def get_global_params_and_proj(self):
        """步骤3: 获取分发参数"""
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def calculate_weights(self, client_id_list, client_features, client_data_sizes):
        """步骤7: 基于投影特征计算权重"""
        if self.detection_method == "none":
            # 无防御：FedAvg 权重 (基于数据量)
            total_size = sum(client_data_sizes)
            weights = {cid: size / total_size for cid, size in zip(client_id_list, client_data_sizes)}
            self.current_round_weights = weights
            return weights

        # 有防御：计算分数
        client_scores = {}
        if self.score_calculator:
            for i, cid in enumerate(client_id_list):
                client_scores[cid] = self.score_calculator.calculate_scores(
                    cid, client_features[i], client_data_sizes[i]
                )

        # 决定权重
        weights = {}
        if self.detection_method == "lsh_score_kickout":
            weights = self.kickout_manager.determine_weights(client_scores)
        elif self.detection_method == "only_score":
            total_score = sum(s['final_score'] for s in client_scores.values())
            weights = {cid: s['final_score']/total_score for cid, s in client_scores.items()}
        elif self.detection_method == "only_kickout":
            if self.kickout_manager:
                weights = self.kickout_manager.determine_weights(client_scores)
        
        self.current_round_weights = weights
        return weights

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