import torch
import copy
import csv
import os
from collections import defaultdict
from _utils_.LSH_proj_extra import SuperBitLSH
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from defence.layers_proj_detect import Layers_Proj_Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server:
    def __init__(self, model, detection_method="lsh_score_kickout", defense_config=None, seed=42, verbose=False, log_file_path=None):
        self.global_model = model.to(DEVICE)
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.verbose = verbose
        self.log_file_path = log_file_path
        
        self.suspect_counters = {} 
        self.global_update_direction = None 
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        
        det_params = defense_config.get('params', {}) if defense_config else {}
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        self.current_round_weights = {}

        if self.log_file_path and "mesas" in self.detection_method:
            self._init_log_file()

    def _init_log_file(self):
        """初始化详细日志 CSV 文件"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        
        headers = [
            "Round", "Client_ID", 
            "Full_L2", "L2_Median", "L2_MAD", "L2_Threshold",
            "Full_Var", "Var_Median", "Var_MAD", "Var_Threshold",
            "Cos_Sim", "Cluster", 
            "Score", "Status"
        ]
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def get_global_params_and_proj(self):
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
        self._update_global_direction_feature(client_projections)
        weights = {}

        if "mesas" in self.detection_method or "projected" in self.detection_method:
            if self.verbose:
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")

            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, 
                self.suspect_counters,
                verbose=self.verbose 
            )
            
            # 写入日志
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)

            # 更新历史记录
            for cid in sorted(logs.keys()):
                status = logs[cid].get('status', 'NORMAL')
                if "SUSPECT" in status:
                    self.detection_history[cid]['suspect_cnt'] += 1
                    self.detection_history[cid]['events'].append(f"R{current_round}:Suspect")
                if "KICKED" in status:
                    self.detection_history[cid]['kicked_cnt'] += 1
                    self.detection_history[cid]['events'].append(f"R{current_round}:Kicked")

            total_score = sum(raw_weights.values())
            if total_score > 0:
                weights = {cid: s / total_score for cid, s in raw_weights.items()}
            else:
                weights = {cid: 0.0 for cid in raw_weights}
            
            self._update_global_direction_feature(client_projections)
            
        else:
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        self.current_round_weights = weights
        return weights

    def _write_detection_log(self, round_num, logs, raw_weights, stats):
        """将本轮检测数据写入 CSV"""
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for cid in sorted(logs.keys()):
                metrics = logs[cid]
                row = [
                    round_num, cid,
                    f"{metrics.get('full_l2', 0):.4f}", 
                    f"{stats['l2_median']:.4f}", f"{stats['l2_mad']:.4f}", f"{stats['l2_threshold']:.4f}",
                    f"{metrics.get('full_var', 0):.4f}", 
                    f"{stats['var_median']:.4f}", f"{stats['var_mad']:.4f}", f"{stats['var_threshold']:.4f}",
                    f"{metrics.get('full_hist_cos', 0):.4f}",
                    metrics.get('full_cluster', 0),
                    f"{raw_weights.get(cid, 0):.2f}",
                    metrics.get('status', 'UNKNOWN')
                ]
                writer.writerow(row)

    def _update_global_direction_feature(self, client_projections):
        if not client_projections: return
        first_proj = list(client_projections.values())[0]['full']
        agg_proj = torch.zeros_like(first_proj, device=first_proj.device)
        for cid, proj_data in client_projections.items():
            agg_proj += proj_data['full']
        if self.global_update_direction is None:
            self.global_update_direction = agg_proj
        else:
            self.global_update_direction = agg_proj 

    def update_global_model(self, weighted_client_models_list, client_ids_list):
        if not weighted_client_models_list: return
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
        self.global_model.eval()
        correct_attack = 0
        total_attack = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                if attack_type == "label_flip":
                    source_class = attack_params.get("source_class", 5)
                    target_class = attack_params.get("target_class", 7)
                    mask = (target == source_class)
                    if mask.sum() == 0: continue
                    data_source = data[mask]
                    outputs = self.global_model(data_source)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += mask.sum().item()
                elif attack_type == "backdoor":
                    target_class = attack_params.get("backdoor_target", 0)
                    trigger_size = attack_params.get("trigger_size", 3)
                    data_poisoned = data.clone()
                    if data_poisoned.dim() == 4:
                         data_poisoned[:, :, -trigger_size:, -trigger_size:] = data_poisoned.max()
                    outputs = self.global_model(data_poisoned)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += data.size(0)
        if total_attack == 0: return 0.0
        return 100 * correct_attack / total_attack

    def _fallback_old_detection(self, ids, features, sizes):
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