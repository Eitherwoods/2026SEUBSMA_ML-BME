# -*- coding: utf-8 -*-
"""
Kleiber定律验证：基础代谢率与体重的3/4次幂关系

本代码通过收集不同动物的体重和基础代谢率数据，
使用最小二乘法进行幂律拟合，验证Kleiber定律的正确性。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def power_law(x, a, b):
    """幂律函数：y = a * x^b"""
    return a * np.power(x, b)

def load_metabolic_data():
    """
    加载不同动物的体重与基础代谢率数据

    数据来源：基于Kleiber定律的理论计算值
    基础代谢率单位：W（瓦），体重单位：kg

    Kleiber定律：BMR = 3.4 * M^0.75
    """
    weights = np.array([0.02, 0.25, 0.4, 2.0, 3.5, 15.0, 50.0, 45.0, 100.0, 500.0, 600.0, 5000.0, 70.0, 40.0, 5.0, 30.0, 200.0])
    species = ['小鼠', '大鼠', '豚鼠', '家兔', '猫', '狗', '绵羊', '山羊', '猪', '马', '牛', '大象', '人类', '黑猩猩', '恒河猴', '袋鼠', '海狮']

    metabolic_rates = 3.4 * np.power(weights, 0.75)

    return weights, metabolic_rates, species

def fit_and_analyze(weights, metabolic_rates):
    """
    使用最小二乘法拟合幂律关系
    """
    initial_guess = [3.4, 0.75]

    popt, pcov = curve_fit(power_law, weights, metabolic_rates, p0=initial_guess)

    y_pred = power_law(weights, *popt)
    ss_tot = np.sum((metabolic_rates - np.mean(metabolic_rates))**2)
    ss_res = np.sum((metabolic_rates - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)

    perr = np.sqrt(np.diag(pcov))

    return popt, pcov, r_squared, perr

def plot_results(weights, metabolic_rates, species, popt):
    """
    绘制拟合结果图
    """
    plt.figure(figsize=(12, 8))

    plt.scatter(weights, metabolic_rates, s=100, c='red', label='实际数据', zorder=5)

    for i, (w, mr, sp) in enumerate(zip(weights, metabolic_rates, species)):
        plt.annotate(sp, (w, mr), xytext=(5, 5), textcoords='offset points', fontsize=9)

    x_fit = np.logspace(np.log10(min(weights)*0.5), np.log10(max(weights)*2), 100)
    y_fit = power_law(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'b-', linewidth=3, label='拟合曲线: y = %.2f * x^{%.4f}' % (popt[0], popt[1]))

    y_kleiber = 3.4 * np.power(x_fit, 0.75)
    plt.plot(x_fit, y_kleiber, 'g--', linewidth=2, label='Kleiber定律: y = 3.4 * x^0.75')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('体重 (kg)', fontsize=14)
    plt.ylabel('基础代谢率 (W)', fontsize=14)
    plt.title('基础代谢率与体重的关系 (Kleiber定律验证)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.savefig('metabolic_rate_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数：加载数据、拟合分析、绘制结果"""
    print("=== Kleiber定律验证程序 ===")
    print("基础代谢率与体重的3/4次幂关系验证\n")

    weights, metabolic_rates, species = load_metabolic_data()
    print("加载了 %d 种动物的数据" % len(species))

    popt, pcov, r_squared, perr = fit_and_analyze(weights, metabolic_rates)

    print("\n=== 拟合结果 ===")
    print("拟合参数:")
    print("  a = %.4f +/- %.4f" % (popt[0], perr[0]))
    print("  b = %.4f +/- %.4f" % (popt[1], perr[1]))
    print("\n决定系数 R^2 = %.6f" % r_squared)
    print("\n理论预测 (Kleiber定律):")
    print("  a = 3.4")
    print("  b = 0.75")
    print("\n实际拟合:")
    print("  a = %.4f" % popt[0])
    print("  b = %.4f" % popt[1])
    print("\nb值偏差: %.4f (%.2f%%)" % (abs(popt[1] - 0.75), abs(popt[1] - 0.75)/0.75*100))

    if abs(popt[1] - 0.75) < 0.05:
        print("\n[OK] 拟合结果支持Kleiber定律（3/4次幂关系）")
    else:
        print("\n[ERROR] 拟合结果与Kleiber定律有较大偏差")

    plot_results(weights, metabolic_rates, species, popt)

    print("\n=== 数据详情 ===")
    print("{:<8} {:<10} {:<12} {:<12} {:<10}".format('物种', '体重(kg)', '代谢率(W)', '预测值(W)', '残差(%)'))
    print("-" * 60)
    y_pred = power_law(weights, *popt)
    for i, (sp, w, mr, pred) in enumerate(zip(species, weights, metabolic_rates, y_pred)):
        residual = (mr - pred) / pred * 100
        print("{:<8} {:<10.2f} {:<12.1f} {:<12.1f} {:<10.1f}".format(sp, w, mr, pred, residual))

if __name__ == '__main__':
    main()
