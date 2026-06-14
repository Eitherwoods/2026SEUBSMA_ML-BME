# -*- coding: utf-8 -*-
"""
交叉验证分析脚本
用于验证幂律模型的泛化能力
"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

def run_cross_validation():
    """
    执行二折和五折交叉验证，评估幂律模型的泛化能力
    """
    # 数据
    weights = np.array([0.02, 0.25, 2.5, 3.5, 10.0, 70.0])
    bmr = np.array([0.25, 2.0, 7.4, 7.4, 18.0, 82.0])
    heart_rates = np.array([624, 362, 213, 130, 96, 65])

    # 对数变换（幂律模型在对数空间为线性）
    log_w = np.log(weights).reshape(-1, 1)
    log_bmr = np.log(bmr)
    log_hr = np.log(heart_rates)

    print("=" * 60)
    print("交叉验证分析结果")
    print("=" * 60)

    # 二折交叉验证
    kf2 = KFold(n_splits=2, shuffle=True, random_state=42)
    model = LinearRegression()

    print("\n【二折交叉验证】")
    cv_scores_bmr_2 = cross_val_score(model, log_w, log_bmr, cv=kf2, scoring='r2')
    cv_scores_hr_2 = cross_val_score(model, log_w, log_hr, cv=kf2, scoring='r2')

    print(f"BMR CV-R² (2-fold): {cv_scores_bmr_2}")
    print(f"BMR CV-R² mean: {cv_scores_bmr_2.mean():.4f}")
    print(f"心率 CV-R² (2-fold): {cv_scores_hr_2}")
    print(f"心率 CV-R² mean: {cv_scores_hr_2.mean():.4f}")

    # 五折交叉验证
    print("\n【五折交叉验证】")
    print("注意：样本量n=6，五折交叉验证每折只有1个测试样本，结果不稳定")

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_bmr_5 = cross_val_score(model, log_w, log_bmr, cv=kf5, scoring='r2')
    cv_scores_hr_5 = cross_val_score(model, log_w, log_hr, cv=kf5, scoring='r2')

    print(f"BMR CV-R² (5-fold): {cv_scores_bmr_5}")
    print(f"心率 CV-R² (5-fold): {cv_scores_hr_5}")

    print("\n" + "=" * 60)
    print("结论：由于样本量较小(n=6)，交叉验证结果不稳定。")
    print("建议：不采用交叉验证作为模型评估指标。")
    print("=" * 60)

    return {
        'bmr_2fold': cv_scores_bmr_2,
        'hr_2fold': cv_scores_hr_2,
        'bmr_5fold': cv_scores_bmr_5,
        'hr_5fold': cv_scores_hr_5
    }

if __name__ == '__main__':
    run_cross_validation()
