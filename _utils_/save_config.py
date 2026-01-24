import matplotlib.pyplot as plt
import numpy as np
import os
import json

def get_result_filename(mode_name, model_type, dataset_type, detection_method, config):
    """
    生成具有可读性的唯一结果文件名
    """
    attacks = config.get('attack_types', [])
    if isinstance(attacks, list):
        if not attacks or config.get('poison_ratio', 0) == 0:
            attack_str = "NoAttack"
        else:
            # 简化攻击名称，避免文件名过长
            attack_str = "+".join(sorted([str(a) for a in attacks]))
    else:
        attack_str = str(attacks)

    poison_ratio = config.get('poison_ratio', 0.0)
    pr_str = f"p{poison_ratio:.2f}"

    is_noniid = config.get('if_noniid', False)
    alpha = config.get('alpha', '')
    if is_noniid:
        dist_str = f"NonIID_a{alpha}"
    else:
        dist_str = "IID"

    # [修改] 增加维度信息到文件名（如果存在），避免不同维度的矩阵结果混淆
    proj_dim = config.get('defense', {}).get('projection_dim', 1024) if 'defense' in config else 1024
    if "mesas" in detection_method or "projected" in detection_method:
        dim_str = f"_dim{proj_dim}"
    else:
        dim_str = ""

    filename = f"{mode_name}_{model_type}_{dataset_type}_{detection_method}{dim_str}_{attack_str}_{pr_str}_{dist_str}.npz"
    filename = filename.replace(" ", "").replace("'", "").replace('"', "")
    return filename

def check_result_exists(save_dir, mode_name, model_type, dataset_type, detection_method, config):
    """检查结果是否已存在"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"✅ [Skip] 结果已存在: {filename}")
        try:
            data = np.load(filepath)
            # [修改] 返回整个 data 对象，以便提取 asr_history
            return True, data
        except Exception as e:
            print(f"⚠️ 文件存在但读取失败 ({e})，将重新训练。")
            return False, None
    return False, None

def save_result_with_config(save_dir, mode_name, model_type, dataset_type, detection_method, config, accuracy_history, asr_history=None):
    """保存结果(.npz)和配置(.json)，新增 asr_history 支持"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    
    # [修改] 保存 asr_history
    save_dict = {'accuracy_history': accuracy_history}
    if asr_history is not None and len(asr_history) > 0:
        save_dict['asr_history'] = asr_history
        
    np.savez(filepath, **save_dict)
    
    config_file = filepath.replace('.npz', '_config.json')
    
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4, default=convert)
    
    print(f"💾 结果已保存: {filename}")
    return filepath

def plot_comparison_curves(config=None, result_dir="results", save_path="comparison.png"):
    """绘制对比曲线"""
    if not os.path.exists(result_dir):
        print(f"⚠️ 结果目录 {result_dir} 不存在。")
        return

    files = [f for f in os.listdir(result_dir) if f.endswith('.npz')]
    if not files:
        print(f"⚠️ 结果目录为空，跳过绘图")
        return
    
    if config:
        m_type = config.get('model_type', '')
        d_type = config.get('dataset_type', '')
        if m_type and d_type:
            target_token = f"{m_type}_{d_type}"
            files = [f for f in files if target_token in f]

    if not files:
        print("⚠️ 未找到匹配当前配置的结果文件。")
        return
    
    plt.figure(figsize=(12, 8))
    
    styles = {
        'pure_training': {'color': 'green', 'label': 'Benign (Baseline)', 'style': '--'},
        'poison_no_detection': {'color': 'red', 'label': 'Attack (No Defense)', 'style': '-'},
        'poison_with_detection': {'color': 'blue', 'label': 'Attack + Defense (Ours)', 'style': '-'}
    }
    
    has_data = False
    files.sort()

    for file in files:
        try:
            mode = None
            for k in styles.keys():
                if file.startswith(k):
                    mode = k
                    break
            
            if mode:
                data = np.load(os.path.join(result_dir, file))
                acc_hist = data['accuracy_history']
                rounds = np.arange(1, len(acc_hist) + 1)
                
                style = styles[mode]
                
                # Accuracy 曲线
                plt.plot(rounds, acc_hist, 
                         color=style['color'], 
                         linestyle=style['style'], 
                         label=f"{style['label']} (Final Acc: {acc_hist[-1]:.1f}%)",
                         linewidth=2 if mode == 'poison_with_detection' else 1.5)
                
                # [新增] 如果有 ASR，也可以选择画出来 (这里暂不画，避免图太乱，只在 label 中体现或另画图)
                if 'asr_history' in data:
                     final_asr = data['asr_history'][-1]
                     # print(f"  > {file} has ASR: {final_asr:.2f}%")
                     
                has_data = True
                
        except Exception as e:
            print(f"Skip file {file}: {e}")

    if not has_data:
        print("⚠️ 找到文件但未匹配到任何已知模式。")
        return

    title = "Defensive Performance Comparison"
    if config:
        attack = config.get('attack_types', ['Unknown'])
        title += f"\nAttack: {attack} | Poison Ratio: {config.get('poison_ratio')} | { 'Non-IID' if config.get('if_noniid') else 'IID' }"
    
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300)
    print(f"📊 对比图已保存: {save_path}")
    plt.close()