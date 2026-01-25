import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="联邦学习防御日志分析工具")
    # parser.add_argument('--file', type=str, required=True, help='CSV日志文件的路径')
    parser.add_argument('--malicious', type=str, default="", help='恶意客户端ID列表，用逗号分隔 (例如: "0,4,13,17")')
    return parser.parse_args()

def analyze_log(file_path, malicious_ids):
    # 1. 读取数据
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='gbk') # 尝试不同编码

    print(f"成功读取日志，共 {len(df)} 条记录")
    
    # 标记良性/恶意
    df['Type'] = df['Client_ID'].apply(lambda x: 'Malicious' if x in malicious_ids else 'Benign')

    # 设置绘图风格
    sns.set(style="whitegrid")
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # =======================================================
    # 图表 1: 评分分离度分析 (Score Separation)
    # =======================================================
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Round', y='Score', hue='Type', style='Type', markers=True, dashes=False, palette={'Malicious': 'red', 'Benign': 'blue'})
    plt.title('Score Trend: Malicious vs Benign')
    plt.ylabel('Final Weight Score')
    plt.savefig(os.path.join(output_dir, f'{base_name}_scores.png'))
    plt.close()
    print(f"[1/4] 生成评分趋势图 -> {base_name}_scores.png")

    # =======================================================
    # 图表 2: 软筛查指标分析 (Min Cosine/Distance)
    # =======================================================
    # CSV中记录的是 Min_Cos (旧) 或 Min_Dist (新逻辑可能未完全对应列名，视Server.py而定)
    # 假设 Server.py 记录的是 "Min_Cos" (即 min similarity) 或 "Min_Dist"
    # 根据之前的代码，Server.py 里的 header 是 "Min_Cos"，但存的值是 最小相似度
    # 我们之前的逻辑是：相似度越低越坏。
    
    if 'Min_Cos' in df.columns:
        plt.figure(figsize=(12, 6))
        # 画出每个点的 Min_Cos
        sns.lineplot(data=df, x='Round', y='Min_Cos', hue='Type', palette={'Malicious': 'red', 'Benign': 'blue'})
        plt.title('Minimum Cosine Similarity (Layer-wise Worst Case)')
        plt.ylabel('Cosine Similarity (Lower is worse)')
        plt.savefig(os.path.join(output_dir, f'{base_name}_cosine.png'))
        plt.close()
        print(f"[2/4] 生成相似度分析图 -> {base_name}_cosine.png")

    # =======================================================
    # 图表 3: 硬筛查指标分析 (Full L2 vs Threshold)
    # =======================================================
    if 'Full_L2' in df.columns and 'L2_Threshold' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # 绘制阈值线 (取每一轮的阈值均值，虽然每轮一样，但画成线更直观)
        threshold_data = df.groupby('Round')['L2_Threshold'].mean().reset_index()
        plt.plot(threshold_data['Round'], threshold_data['L2_Threshold'], 'k--', label='Dynamic Threshold', linewidth=2)
        
        # 绘制客户端 L2
        sns.scatterplot(data=df, x='Round', y='Full_L2', hue='Type', alpha=0.6, palette={'Malicious': 'red', 'Benign': 'blue'})
        
        plt.title('Full L2 Norm vs Dynamic Threshold')
        plt.yscale('log') # 使用对数坐标，因为 L2 差异可能很大
        plt.ylabel('L2 Norm (Log Scale)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{base_name}_l2.png'))
        plt.close()
        print(f"[3/4] 生成 L2 范数分析图 -> {base_name}_l2.png")

    # =======================================================
    # 图表 4: 拦截原因统计 (词云/柱状图) - 验证分层策略
    # =======================================================
    # 解析 Status 列，提取 "SUSPECT(reason)" 中的 reason
    suspects = df[df['Status'].str.contains('SUSPECT', na=False)]
    
    if not suspects.empty:
        # 提取括号内的内容
        reasons = []
        for status in suspects['Status']:
            match = re.search(r'SUSPECT\((.*?)\)', status)
            if match:
                # 可能有多个原因，用逗号分隔
                r_list = match.group(1).split(',')
                # 提取具体的层名或指标名 (例如 "layer_conv1.weight:Dist" -> "conv1:Dist")
                reasons.extend([r.split(':')[0].replace('layer_', '').strip() for r in r_list])
        
        if reasons:
            reason_counts = pd.Series(reasons).value_counts()
            
            plt.figure(figsize=(10, 6))
            reason_counts.plot(kind='bar', color='orange')
            plt.title('Frequency of Detection Triggers (Why were they caught?)')
            plt.xlabel('Layer / Metric Name')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_reasons.png'))
            plt.close()
            print(f"[4/4] 生成拦截原因统计图 -> {base_name}_reasons.png")
            print("\n[统计] 拦截原因分布:")
            print(reason_counts)
    else:
        print("[4/4] 未检测到 SUSPECT 状态，跳过原因分析图。")

if __name__ == "__main__":
    args = parse_args()
    
    # 处理恶意 ID 输入
    if args.malicious:
        m_ids = [int(x.strip()) for x in args.malicious.split(',')]
    else:
        m_ids = []
        print("提示: 未指定恶意ID (--malicious)，图表中将无法区分红蓝颜色。")
    
    file = "/home/xd/sjh/PPFL_TEE/exp_plain/results/cifar-10/backdoor/poison_with_detection_cifar10_cifar10_layers_proj_detect_backdoor_p0.20_IID_detection_log.csv"
    
    analyze_log(file, m_ids)