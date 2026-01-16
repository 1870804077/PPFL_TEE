import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_single_curve_from_file(file_path, mode_name=None, save_path=None, model_name="unknown", attack_types=["none"], poison_ratio=0.0, is_iid=True):
    """
    从文件中绘制单个准确率曲线

    参数:
        file_path: 包含准确率历史的npz文件路径
        mode_name: 模式名称（如果为None，则从文件名推断）
        save_path: 图片保存路径（None则不保存）
        model_name: 模型名称
        attack_types: 攻击类型列表
        poison_ratio: 投毒比例
        is_iid: 是否为IID数据分布
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return

    try:
        # 加载数据
        data = np.load(file_path)
        if 'accuracy_history' not in data:
            print(f"⚠️ 文件中没有找到 'accuracy_history'")
            return
        
        accuracy_history = data['accuracy_history']
        
        # 如果没有提供模式名称，从文件名推断
        if mode_name is None:
            base_name = os.path.basename(file_path).replace('.npz', '').replace('_config', '')
            parts = base_name.split('_')
            if parts[0] in ['pure_training', 'poison_no_detection', 'poison_with_detection']:
                mode_name = parts[0]
            else:
                mode_name = base_name
        
        # 绘图
        plt.figure(figsize=(10, 6))
        rounds = np.arange(1, len(accuracy_history) + 1)  # 通信轮次

        # 绘制曲线 - 使用细线
        plt.plot(rounds, accuracy_history, linewidth=1.0, label=mode_name)

        # 设置图表属性
        attack_str = ", ".join(attack_types)
        distribution_str = "IID" if is_iid else "Non-IID"
        title = f"{model_name.upper()} | {distribution_str} | Attack: {attack_str} | Poison Ratio: {poison_ratio*100:.0f}% | Mode: {mode_name}"
        
        plt.xlabel("Communication Rounds", fontsize=12)
        plt.ylabel("Test Accuracy (%)", fontsize=12)
        plt.title(title, fontsize=10, fontweight="normal")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.xlim(1, len(rounds))  # x轴范围

        # 添加统计信息（最终/最高/平均准确率）
        final_acc = accuracy_history[-1]
        max_acc = np.max(accuracy_history)
        avg_acc = np.mean(accuracy_history)
        plt.text(
            0.02, 0.98,
            f"Final: {final_acc:.2f}%\nMax: {max_acc:.2f}%\nAvg: {avg_acc:.2f}%",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor='gray', pad=5),
            verticalalignment='top',
            fontsize=9
        )

        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Single mode curve saved to: {save_path}")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️ 读取或绘制文件时出错: {e}")


def plot_comparison_curves(config=None, result_dir="results", save_path="comparison.png"):
    """绘制对比曲线 - 只绘制相同模型和检测方法的结果"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 获取所有结果文件
    files = [f for f in os.listdir(result_dir) if f.endswith('.npz') and not f.endswith('_config.json')]
    if not files:
        print(f"⚠️ No result files found, skipping plotting")
        return
    
    # 如果提供了config，基于config过滤结果
    if config:
        model_type = config.get('model_type', '')
        dataset_type = config.get('dataset_type', '')
        
        # 过滤出匹配当前配置的结果文件
        filtered_files = []
        for file in files:
            # 检查文件名是否包含模型类型、数据集类型和检测方法
            if model_type in file and dataset_type in file in file:
                filtered_files.append(file)
        
        files = filtered_files
    
    if not files:
        print("⚠️ No matching result files found, skipping plotting")
        return
    
    accuracy_dict = {}
    for file in files:
        try:
            data = np.load(os.path.join(result_dir, file))
            if 'accuracy_history' in data:
                # 提取模式名称（从文件名中提取）
                base_name = file.replace('.npz', '').replace('_config', '')
                # 从文件名中提取模式名称（通常是文件名的前缀部分）
                parts = base_name.split('_')
                # 通常模式名称是第一个部分：pure_training, poison_no_detection, poison_with_detection
                if parts[0] in ['pure_training', 'poison_no_detection', 'poison_with_detection']:
                    mode_name = parts[0]
                else:
                    # 如果第一个部分不是标准模式名称，则使用整个文件名
                    mode_name = base_name
                accuracy_dict[mode_name] = data['accuracy_history']
            else:
                print(f"⚠️ No 'accuracy_history' found in file {file}")
        except Exception as e:
            print(f"⚠️ Error reading file {file}: {e}")
    
    if not accuracy_dict:
        print("⚠️ No valid accuracy data, skipping plotting")
        return
    
    # 绘图
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (mode_name, acc_history) in enumerate(accuracy_dict.items()):
        if len(acc_history) > 0:
            rounds = np.arange(1, len(acc_history) + 1)
            color = colors[i % len(colors)]
            plt.plot(rounds, acc_history, label=mode_name, linewidth=1.0, color=color)
    
    # 生成标题
    if config:
        model_name = config.get('model_type', 'unknown').upper()
        dataset_name = config.get('dataset_type', 'unknown').upper()
        det_method = config.get('detection_method', 'unknown')
        is_noniid = config.get('if_noniid', True)
        distribution_str = "Non-IID" if is_noniid else "IID"
        poison_ratio = config.get('poison_ratio', 0.0)
        attack_types = config.get('attack_types', ['none'])
        
        attack_str = ", ".join(attack_types)
        title = f"{model_name} | {dataset_name} | {distribution_str} | Attack: {attack_str} | Poison Ratio: {poison_ratio*100:.0f}% | Method: {det_method}"
    else:
        title = "Accuracy Comparison"
    
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加整体统计信息
    all_final_accs = [acc_hist[-1] for acc_hist in accuracy_dict.values() if len(acc_hist) > 0]
    if all_final_accs:
        avg_final = np.mean(all_final_accs)
        plt.text(0.02, 0.02, f"Average Final Acc: {avg_final:.2f}%", 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor='gray', pad=5),
                 fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Comparison chart saved to: {save_path}")



def get_result_filename(mode_name, model_type, dataset_type, detection_method, config):
    """生成唯一的结果文件名"""
    filename = f"{mode_name}_{model_type}_{dataset_type}_{detection_method}_{config['attack_types']}_{config['poison_ratio']}_{config['if_noniid']}.npz"
    return filename


def check_result_exists(save_dir, mode_name, model_type, dataset_type, detection_method, config):
    """检查结果是否已存在（包含攻击类型校验）"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    config_file = filepath.replace('.npz', '_config.json')
    
    if os.path.exists(filepath) and os.path.exists(config_file):
        print(f"✅ 检测到{mode_name}已有有效结果，跳过训练")
        print("使用文件" + filename + "结果")
        data = np.load(filepath)
        return True, data['accuracy_history']
    print("期望文件" + filename + "不存在")
    return False, None


def save_result_with_config(save_dir, mode_name, model_type, dataset_type, detection_method, config, accuracy_history):
    """保存结果和配置"""
    os.makedirs(save_dir, exist_ok=True)
    filename = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    filepath = os.path.join(save_dir, filename)
    
    # 保存准确率
    np.savez(filepath, accuracy_history=accuracy_history)
    # 保存配置
    config_file = filepath.replace('.npz', '_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"💾 结果已保存至: {filepath}")
    return filepath

