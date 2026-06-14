# -*- coding: utf-8 -*-
"""
Kleiber定律验证 - 可视化分析
生成标度指数对比图和血管网络示意图
"""

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_scaling_exponents():
    """标度指数对比图"""
    scaling_params = {
        'BMR': {'observed': 0.7705, 'theoretical': 0.75, 'color': '#1f77b4'},
        '心率': {'observed': -0.2620, 'theoretical': -0.25, 'color': '#2ca02c'},
        '主动脉长度': {'observed': 0.25, 'theoretical': 0.25, 'color': '#ff7f0e'},
        '毛细血管数': {'observed': 0.75, 'theoretical': 0.75, 'color': '#9467bd'}
    }
    
    labels = list(scaling_params.keys())
    observed = [scaling_params[k]['observed'] for k in labels]
    theoretical = [scaling_params[k]['theoretical'] for k in labels]
    colors = [scaling_params[k]['color'] for k in labels]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], observed, width, label='观测值', color=colors, alpha=0.8)
    rects2 = ax.bar([i + width/2 for i in x], theoretical, width, label='理论值', color='gray', alpha=0.5)
    
    ax.set_ylabel('标度指数', fontsize=14)
    ax.set_title('标度指数理论值与观测值对比', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    for i, (obs, theo) in enumerate(zip(observed, theoretical)):
        ax.text(i - width/2, obs + 0.01, f'{obs:.4f}', ha='center', va='bottom', fontsize=11)
        ax.text(i + width/2, theo + 0.01, f'{theo:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('../../figure/metabolic_rate/scaling_exponents.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: scaling_exponents.png")

def plot_vascular_network():
    """血管网络分形结构示意图"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def draw_vessel(x, y, length, angle, depth, max_depth=4):
        if depth > max_depth:
            return
        
        end_x = x + length * np.cos(angle)
        end_y = y + length * np.sin(angle)
        
        ax.plot([x, end_x], [y, end_y], 'k-', linewidth=max(1, 4 - depth))
        
        if depth < max_depth:
            num_branches = 2 if depth < 2 else 3
            branch_angle = np.pi / 6 if depth < 2 else np.pi / 5
            
            for i in range(num_branches):
                offset_angle = (i - (num_branches - 1) / 2) * branch_angle
                draw_vessel(end_x, end_y, length * 0.7, angle + offset_angle, depth + 1)
    
    draw_vessel(0.5, 0.1, 0.35, np.pi/2, 1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('血管网络分形结构示意图', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.02, '主动脉 (k=1)', ha='center', fontsize=12)
    ax.text(0.25, 0.75, '分支血管', fontsize=10, rotation=30)
    ax.text(0.75, 0.75, '分支血管', fontsize=10, rotation=-30)
    ax.text(0.5, 0.95, '毛细血管网络', ha='center', fontsize=12)
    ax.axis('off')
    
    plt.savefig('../../figure/metabolic_rate/vascular_network.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: vascular_network.png")

def main():
    print("=" * 70)
    print("Kleiber定律验证 - 可视化分析")
    print("=" * 70)
    
    plot_scaling_exponents()
    plot_vascular_network()
    
    print("\n" + "=" * 70)
    print("可视化图像生成完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()