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
        
        # 保存防御配置以便后续生成日志表头
        self.defense_config = defense_config or {}
        
        self.suspect_counters = {} 
        self.global_update_direction = None 
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        
        det_params = self.defense_config.get('params', {})
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        self.current_round_weights = {}

        should_log = any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"])
        if self.log_file_path and should_log:
            self._init_log_file()

    def _init_log_file(self):
        """初始化详细日志 CSV 文件 (支持动态列)"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        
        # 1. 基础列
        headers = ["Round", "Client_ID", "Score", "Status"]
        
        # 2. 动态生成指标列
        # 我们要监控的目标层
        target_layers = self.defense_config.get('target_layers', [])
        # 监控范围: Full + 每一层
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        # 监控指标
        metrics = ['l2', 'var', 'dist']
        
        for scope in scopes:
            for metric in metrics:
                # 添加 Value 列和 Threshold 列
                # e.g., Full_L2, Full_L2_Thresh, layer_conv1_dist, layer_conv1_dist_thresh
                headers.append(f"{scope}_{metric}")
                headers.append(f"{scope}_{metric}_threshold")
        
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"  [Warning] 无法初始化日志文件: {e}")

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

        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"]):
            if self.verbose:
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")

            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, 
                self.suspect_counters,
                verbose=self.verbose 
            )
            
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)

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

    def _update_global_direction_feature(self, client_projections):
        if not client_projections: return
        first_cid = list(client_projections.values())[0]
        # 简单累加用于历史记录 (虽然本版本逻辑主要依赖本轮统计)
        # 保持结构一致性即可
        pass 

    def _write_detection_log(self, round_num, logs, raw_weights, stats):
        """将全量+分层的详细数据写入 CSV"""
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics_list = ['l2', 'var', 'dist']
        
        try:
            with open(self.log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                for cid in sorted(logs.keys()):
                    metrics = logs[cid]
                    
                    # 1. 基础信息
                    row = [
                        round_num, 
                        cid, 
                        f"{raw_weights.get(cid, 0):.2f}",
                        metrics.get('status', 'UNKNOWN')
                    ]
                    
                    # 2. 动态提取所有指标
                    for scope in scopes:
                        for metric in metrics_list:
                            # 提取 Value (从 logs[cid] 中)
                            # e.g. key = full_l2, layer_conv1.weight_dist
                            val_key = f"{scope}_{metric}"
                            val = metrics.get(val_key, 0)
                            
                            # 提取 Threshold (从 stats 中)
                            # e.g. key = full_l2_threshold
                            thresh_key = f"{val_key}_threshold"
                            thresh = stats.get(thresh_key, 0)
                            
                            row.append(f"{val:.4f}")
                            row.append(f"{thresh:.4f}")
                            
                    writer.writerow(row)
        except Exception as e:
            # 避免日志写入错误卡死训练，只打印警告
            pass

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