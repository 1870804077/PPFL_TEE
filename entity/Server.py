import torch
import copy
from collections import defaultdict
from _utils_.LSH_proj_extra import SuperBitLSH
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from defence.layers_proj_detect import Layers_Proj_Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", seed=42, verbose=False):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.verbose = verbose
        
        # 状态维护
        self.suspect_counters = {} 
        self.global_update_direction = None 
        
        # [新增] 检测历史统计 {cid: {'suspect_cnt': 0, 'kicked_cnt': 0, 'rounds': []}}
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        
        # 组件初始化
        self.mesas_detector = Layers_Proj_Detector()
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        self.current_round_weights = {}

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def get_global_params_and_proj(self):
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        """
        步骤7: 检测并计算权重
        """
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
            
            # 详细打印检测结果
            if self.verbose:
                print(f"  [Server] Detection Results (Round {current_round}):")
                
            for cid in sorted(logs.keys()):
                status = logs[cid].get('status', 'NORMAL')
                score = raw_weights.get(cid, 0.0)
                
                # 记录统计
                if "SUSPECT" in status:
                    self.detection_history[cid]['suspect_cnt'] += 1
                    self.detection_history[cid]['events'].append(f"R{current_round}:Suspect")
                if "KICKED" in status:
                    self.detection_history[cid]['kicked_cnt'] += 1
                    self.detection_history[cid]['events'].append(f"R{current_round}:Kicked")

                # 实时打印非正常状态
                if self.verbose and status != 'NORMAL':
                     print(f"    [ALERT] Client {cid}: Score={score:.2f} | Status={status}")

            total_score = sum(raw_weights.values())
            if total_score > 0:
                weights = {cid: s / total_score for cid, s in raw_weights.items()}
            else:
                if self.verbose:
                    print("  [Warning] All clients kicked out this round!")
                weights = {cid: 0.0 for cid in raw_weights}
            
            self._update_global_direction_feature(client_projections)
            
        else:
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        self.current_round_weights = weights
        return weights

    def _update_global_direction_feature(self, client_projections):
        if not client_projections:
            return
        first_proj = list(client_projections.values())[0]['full']
        agg_proj = torch.zeros_like(first_proj, device=first_proj.device)
        for cid, proj_data in client_projections.items():
            agg_proj += proj_data['full']
        if self.global_update_direction is None:
            self.global_update_direction = agg_proj
        else:
            self.global_update_direction = agg_proj 

    def update_global_model(self, weighted_client_models_list, client_ids_list):
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
            if self.verbose:
                print("  [Warning] No valid updates for aggregation.")

    def evaluate(self, test_loader):
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

    
    def evaluate_asr(self, test_loader, attack_type, attack_params):
        """
        计算攻击成功率 (ASR)
        :param attack_type: "label_flip" or "backdoor"
        :param attack_params: 攻击的具体参数字典
        """
        self.global_model.eval()
        correct_attack = 0
        total_attack = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # --- 针对 Label Flip 的 ASR ---
                # ASR = (原 Source 类样本被预测为 Target 类的比例)
                if attack_type == "label_flip":
                    source_class = attack_params.get("source_class", 5)
                    target_class = attack_params.get("target_class", 7)
                    
                    # 筛选出属于源类别的样本
                    mask = (target == source_class)
                    if mask.sum() == 0:
                        continue
                        
                    data_source = data[mask]
                    
                    outputs = self.global_model(data_source)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # 成功攻击 = 预测为目标类
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += mask.sum().item()

                # --- 针对 Backdoor 的 ASR ---
                # ASR = (所有样本加上 Trigger 后被预测为 Target 类的比例)
                # 注意：通常排除掉原本就是 Target 类的样本，或者直接算全部
                elif attack_type == "backdoor":
                    target_class = attack_params.get("backdoor_target", 0)
                    trigger_size = attack_params.get("trigger_size", 3)
                    
                    # 给整个 batch 加 trigger
                    # 假设数据是 [B, C, H, W]
                    data_poisoned = data.clone()
                    if data_poisoned.dim() == 4:
                         data_poisoned[:, :, -trigger_size:, -trigger_size:] = data_poisoned.max()
                    
                    outputs = self.global_model(data_poisoned)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # 成功攻击 = 预测为后门目标类
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += data.size(0)
                    
        if total_attack == 0:
            return 0.0
        return 100 * correct_attack / total_attack

    def _fallback_old_detection(self, ids, features, sizes):
        # ... (保持不变) ...
        if not self.score_calculator and not self.kickout_manager:
            total_size = sum(sizes)
            if total_size > 0:
                return {cid: size / total_size for cid, size in zip(ids, sizes)}
            else:
                return {cid: 1.0 / len(ids) for cid in ids}
        if self.kickout_manager and not self.score_calculator:
             return {cid: 1.0 / len(ids) for cid in ids}
        client_scores = {}
        for i, cid in enumerate(ids):
            client_scores[cid] = self.score_calculator.calculate_scores(
                cid, features[i], sizes[i]
            )
        weights = {}
        if self.kickout_manager:
            weights = self.kickout_manager.determine_weights(client_scores)
        else:
            raw_scores = {cid: s['final_score'] for cid, s in client_scores.items()}
            total_s = sum(raw_scores.values())
            if total_s > 0:
                weights = {cid: s / total_s for cid, s in raw_scores.items()}
            else:
                weights = {cid: 1.0 / len(ids) for cid in ids}
        return weights