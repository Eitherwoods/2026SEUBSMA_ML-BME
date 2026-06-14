# -*- coding: utf-8 -*-
"""
Kleiber定律验证 - 幂律拟合分析
生成BMR和心率拟合图、残差分析图、代谢率对比图
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def power_law(x, a, b):
    """幂律函数：y = a * x^b"""
    return a * np.power(x, b)

def plot_bmr_fitting(weights, bmr, species, popt_bmr):
    """BMR与体重的幂律拟合"""
    plt.figure(figsize=(10, 7))
    plt.loglog(weights, bmr, 'bo', markersize=14, label='实际数据', 
               markeredgecolor='black', markeredgewidth=1.5)
    x_fit = np.logspace(-3, 3, 200)
    plt.loglog(x_fit, power_law(x_fit, *popt_bmr), 'r-', linewidth=3,
               label='拟合曲线: y = %.2fx^{%.4f}' % (popt_bmr[0], popt_bmr[1]))
    plt.loglog(x_fit, power_law(x_fit, 3.45, 0.75), 'g--', linewidth=2, alpha=0.8,
               label='Kleiber定律: y = 3.45x^{0.75}')
    
    for i, sp in enumerate(species):
        plt.annotate(sp, (weights[i], bmr[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight='bold')
    
    plt.xlabel('体重 M (kg)', fontsize=14)
    plt.ylabel('基础代谢率 BMR (W)', fontsize=14)
    plt.title('BMR与体重的幂律关系 (Kleiber定律)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('../../figure/metabolic_rate/bmr_fitting.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: bmr_fitting.png")

def plot_hr_fitting(weights, heart_rates, species, popt_hr):
    """心率与体重的幂律拟合"""
    plt.figure(figsize=(10, 7))
    plt.loglog(weights, heart_rates, 'gs', markersize=14, label='实际数据',
               markeredgecolor='black', markeredgewidth=1.5)
    x_fit = np.logspace(-3, 3, 200)
    plt.loglog(x_fit, power_law(x_fit, *popt_hr), 'r-', linewidth=3,
               label='拟合曲线: y = %.2fx^{%.4f}' % (popt_hr[0], popt_hr[1]))
    plt.loglog(x_fit, power_law(x_fit, 241, -0.25), 'g--', linewidth=2, alpha=0.8,
               label='理论预测: y = 241x^{-0.25}')
    
    for i, sp in enumerate(species):
        plt.annotate(sp, (weights[i], heart_rates[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight='bold')
    
    plt.xlabel('体重 M (kg)', fontsize=14)
    plt.ylabel('静息心率 HR (bpm)', fontsize=14)
    plt.title('心率与体重的幂律关系', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('../../figure/metabolic_rate/hr_fitting.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: hr_fitting.png")

def plot_residual_analysis(weights, bmr, heart_rates, popt_bmr, popt_hr):
    """残差分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bmr_pred = power_law(weights, *popt_bmr)
    residuals_bmr = (bmr - bmr_pred) / bmr_pred * 100
    ax1.scatter(weights, residuals_bmr, s=100, c='b', marker='o', edgecolor='black')
    ax1.axhline(0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('体重 (kg)', fontsize=12)
    ax1.set_ylabel('BMR残差 (%)', fontsize=12)
    ax1.set_title('BMR拟合残差分析', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    hr_pred = power_law(weights, *popt_hr)
    residuals_hr = (heart_rates - hr_pred) / hr_pred * 100
    ax2.scatter(weights, residuals_hr, s=100, c='g', marker='s', edgecolor='black')
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('体重 (kg)', fontsize=12)
    ax2.set_ylabel('心率残差 (%)', fontsize=12)
    ax2.set_title('心率拟合残差分析', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../figure/metabolic_rate/residual_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: residual_analysis.png")

def plot_metabolic_comparison(weights, bmr, species):
    """不同动物代谢率对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(species, bmr, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    ax.set_xlabel('动物种类', fontsize=14)
    ax.set_ylabel('基础代谢率 BMR (W)', fontsize=14)
    ax.set_title('不同动物基础代谢率对比', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, w in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{w} kg', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../../figure/metabolic_rate/metabolic_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存: metabolic_comparison.png")

def main():
    print("=" * 70)
    print("Kleiber定律验证 - 幂律拟合分析")
    print("=" * 70)
    
    data = {
        'species': ['小鼠', '大鼠', '兔', '猫', '狗', '人'],
        'weight_kg': [0.02, 0.25, 2.5, 3.5, 10.0, 70.0],
        'heart_rate_bpm': [624, 362, 213, 130, 96, 65],
        'bmr_watts': [0.25, 2.0, 7.4, 7.4, 18.0, 82.0]
    }
    
    weights = np.array(data['weight_kg'])
    heart_rates = np.array(data['heart_rate_bpm'])
    bmr = np.array(data['bmr_watts'])
    species = data['species']
    
    popt_bmr, _ = curve_fit(power_law, weights, bmr, p0=[3.4, 0.75], maxfev=5000)
    popt_hr, _ = curve_fit(power_law, weights, heart_rates, p0=[240, -0.25], maxfev=5000)
    
    print(f"\nBMR拟合结果: BMR = {popt_bmr[0]:.4f} x M^({popt_bmr[1]:.4f})")
    print(f"理论预测: BMR = 3.45 x M^0.75")
    print(f"偏差: {abs(popt_bmr[1] - 0.75)/0.75*100:.2f}%")
    
    print(f"\n心率拟合结果: HR = {popt_hr[0]:.4f} x M^({popt_hr[1]:.4f})")
    print(f"理论预测: HR = 241 x M^(-0.25)")
    print(f"偏差: {abs(popt_hr[1] - (-0.25))/0.25*100:.2f}%")
    
    plot_bmr_fitting(weights, bmr, species, popt_bmr)
    plot_hr_fitting(weights, heart_rates, species, popt_hr)
    plot_residual_analysis(weights, bmr, heart_rates, popt_bmr, popt_hr)
    plot_metabolic_comparison(weights, bmr, species)
    
    print("\n" + "=" * 70)
    print("拟合分析图像生成完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()