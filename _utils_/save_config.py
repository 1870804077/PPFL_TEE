import matplotlib.pyplot as plt
import numpy as np
import os
import json

# =============================================================================
# 1. 文件名生成逻辑 (修改核心)
# =============================================================================
def get_result_filename(mode_name, model_type, dataset_type, detection_method, config):
    """
    生成具有可读性的唯一结果文件名
    格式: 模式_模型_数据集_防御_攻击类型_投毒率_分布.npz
    """
    # 1. 处理攻击类型 (List -> String)
    # 将列表 ['label_flip', 'backdoor'] 转换为 'label_flip+backdoor'
    attacks = config.get('attack_types', [])
    if isinstance(attacks, list):
        if not attacks or config.get('poison_ratio', 0) == 0:
            attack_str = "NoAttack"
        else:
            # 排序以确保 ['a', 'b'] 和 ['b', 'a'] 生成相同的文件名
            attack_str = "+".join(sorted([str(a) for a in attacks]))
    else:
        attack_str = str(attacks)

    # 2. 处理投毒比例 (float -> string)
    # 例如 0.2 -> 'p0.2'
    poison_ratio = config.get('poison_ratio', 0.0)
    pr_str = f"p{poison_ratio:.2f}"

    # 3. 处理数据分布 (IID/Non-IID)
    is_noniid = config.get('if_noniid', False)
    alpha = config.get('alpha', '')
    if is_noniid:
        dist_str = f"NonIID_a{alpha}" # 例如 NonIID_a0.5
    else:
        dist_str = "IID"

    # 4. 组合文件名
    # 示例: poison_with_detection_lenet5_cifar10_lsh_score_kickout_label_flip_p0.2_NonIID_a0.5.npz
    filename = f"{mode_name}_{model_type}_{dataset_type}_{detection_method}_{attack_str}_{pr_str}_{dist_str}.npz"
    
    # 清理非法字符 (防止配置中有空格或引号)
    filename = filename.replace(" ", "").replace("'", "").replace('"', "")
    
    return filename

# =============================================================================
# 2. 检查结果是否存在
# =============================================================================
def check_result_exists(save_dir, mode_name, model_type, dataset_type, detection_method, config):
    """检查结果是否已存在（基于生成的可读文件名）"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    
    # 同时也检查配套的 json 配置是否存在
    config_file = filepath.replace('.npz', '_config.json')
    
    if os.path.exists(filepath):
        print(f"✅ [Skip] 结果已存在: {filename}")
        try:
            data = np.load(filepath)
            return True, data['accuracy_history']
        except Exception as e:
            print(f"⚠️ 文件存在但读取失败 ({e})，将重新训练。")
            return False, None
    
    # print(f"ℹ️ 准备生成: {filename}")
    return False, None

# =============================================================================
# 3. 保存结果
# =============================================================================
def save_result_with_config(save_dir, mode_name, model_type, dataset_type, detection_method, config, accuracy_history):
    """保存结果(.npz)和配置(.json)"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    
    # 保存准确率数据
    np.savez(filepath, accuracy_history=accuracy_history)
    
    # 保存详细配置 (方便后续查看参数)
    config_file = filepath.replace('.npz', '_config.json')
    # 将 numpy 类型转换为原生类型以支持 JSON 序列化
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4, default=convert)
    
    print(f"💾 结果已保存: {filename}")
    return filepath

# =============================================================================
# 4. 绘图函数 (保持原有逻辑，稍作增强)
# =============================================================================
def plot_single_curve_from_file(file_path, mode_name=None, save_path=None, model_name="unknown", attack_types=["none"], poison_ratio=0.0, is_iid=True):
    """从文件中绘制单个准确率曲线"""
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return

    try:
        data = np.load(file_path)
        if 'accuracy_history' not in data:
            print(f"⚠️ 文件中没有找到 'accuracy_history'")
            return
        
        accuracy_history = data['accuracy_history']
        
        # 尝试从文件名解析模式名称，如果未提供
        if mode_name is None:
            base_name = os.path.basename(file_path)
            # 简单的解析尝试
            if base_name.startswith("pure_training"): mode_name = "Pure Training"
            elif base_name.startswith("poison_no_detection"): mode_name = "No Defense"
            elif base_name.startswith("poison_with_detection"): mode_name = "With Defense"
            else: mode_name = "Unknown Mode"
        
        plt.figure(figsize=(10, 6))
        rounds = np.arange(1, len(accuracy_history) + 1)

        plt.plot(rounds, accuracy_history, linewidth=1.5, label=mode_name)

        attack_str = "+".join(attack_types) if isinstance(attack_types, list) else str(attack_types)
        distribution_str = "IID" if is_iid else "Non-IID"
        title = f"{model_name.upper()} | {distribution_str} | Attack: {attack_str} (Ratio: {poison_ratio}) | Mode: {mode_name}"
        
        plt.xlabel("Communication Rounds", fontsize=12)
        plt.ylabel("Test Accuracy (%)", fontsize=12)
        plt.title(title, fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加最终结果标记
        plt.text(0.02, 0.95, f"Final Acc: {accuracy_history[-1]:.2f}%", transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Single mode curve saved to: {save_path}")
        # plt.show() # 根据环境决定是否显示
        plt.close()
        
    except Exception as e:
        print(f"⚠️ 绘图出错: {e}")

def plot_comparison_curves(config=None, result_dir="results", save_path="comparison.png"):
    """绘制对比曲线 - 读取目录下的所有相关文件"""
    files = [f for f in os.listdir(result_dir) if f.endswith('.npz')]
    if not files:
        print(f"⚠️ 结果目录为空，跳过绘图")
        return
    
    # 简单过滤：只画当前数据集和模型的图
    if config:
        target_token = f"{config.get('model_type')}_{config.get('dataset_type')}"
        files = [f for f in files if target_token in f]

    if not files:
        print("⚠️ 未找到匹配当前配置的结果文件。")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 定义颜色和线型，区分不同模式
    styles = {
        'pure_training': {'color': 'green', 'label': 'Benign (Baseline)', 'style': '--'},
        'poison_no_detection': {'color': 'red', 'label': 'Attack (No Defense)', 'style': '-'},
        'poison_with_detection': {'color': 'blue', 'label': 'Attack + Defense (Ours)', 'style': '-'}
    }
    
    has_data = False
    
    for file in files:
        try:
            # 识别模式
            mode = None
            for k in styles.keys():
                if file.startswith(k):
                    mode = k
                    break
            
            if mode:
                data = np.load(os.path.join(result_dir, file))
                acc_hist = data['accuracy_history']
                rounds = np.arange(1, len(acc_hist) + 1)
                
                # 绘制
                style = styles[mode]
                plt.plot(rounds, acc_hist, 
                         color=style['color'], 
                         linestyle=style['style'], 
                         label=f"{style['label']} (Final: {acc_hist[-1]:.1f}%)",
                         linewidth=2 if mode == 'poison_with_detection' else 1.5)
                has_data = True
                
        except Exception as e:
            print(f"Skip file {file}: {e}")

    if not has_data:
        return

    # 设置图表装饰
    title = "Defensive Performance Comparison"
    if config:
        title += f"\nAttack: {config.get('attack_types')} | Poison Ratio: {config.get('poison_ratio')} | { 'Non-IID' if config.get('if_noniid') else 'IID' }"
    
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300)
    print(f"📊 对比图已保存: {save_path}")
    plt.close()