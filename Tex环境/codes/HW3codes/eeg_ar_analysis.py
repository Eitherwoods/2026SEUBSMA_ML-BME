# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pyedflib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class EEGARAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subject_info = self._load_subject_info()
    
    def _load_subject_info(self):
        info_path = os.path.join(self.data_dir, 'subject-info.csv')
        df = pd.read_csv(info_path)
        df['Group'] = df['Count quality'].apply(lambda x: 'G' if x == 0 else 'B')
        return df
    
    def get_subjects_by_group(self):
        group_g = self.subject_info[self.subject_info['Group'] == 'G']['Subject'].tolist()
        group_b = self.subject_info[self.subject_info['Group'] == 'B']['Subject'].tolist()
        return group_g, group_b
    
    def load_eeg_data(self, subject_id, task_type='pre'):
        suffix = '_1' if task_type == 'pre' else '_2'
        filename = f"{subject_id}{suffix}.edf"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file
            signal_labels = f.getSignalLabels()
            signals = []
            for i in range(n_channels):
                signals.append(f.readSignal(i))
            f.close()
            return signals, signal_labels
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None
    
    def estimate_ar_order(self, signal, max_order=20):
        n = len(signal)
        best_order = 1
        best_aic = float('inf')
        
        for p in range(1, max_order + 1):
            X = np.zeros((n - p, p))
            y = signal[p:]
            
            for i in range(p):
                X[:, i] = signal[p - 1 - i : n - 1 - i]
            
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            aic = n * np.log(mse) + 2 * p
            
            if aic < best_aic:
                best_aic = aic
                best_order = p
        
        return best_order
    
    def fit_ar_model(self, signal, order):
        n = len(signal)
        X = np.zeros((n - order, order))
        y = signal[order:]
        
        for i in range(order):
            X[:, i] = signal[order - 1 - i : n - 1 - i]
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        return model.coef_, model.intercept_
    
    def compute_ar_spectrum(self, coefficients, fs=250, n_freq=512):
        coef = np.array([1.0] + list(-coefficients))
        freq = np.linspace(0, fs/2, n_freq)
        spectrum = np.zeros(n_freq)
        
        for i, f in enumerate(freq):
            w = 2 * np.pi * f / fs
            z = np.exp(1j * w)
            denom = np.polyval(coef, z)
            spectrum[i] = 1 / np.abs(denom) ** 2
        
        return freq, spectrum
    
    def compute_psd(self, signal, fs=250):
        freq, psd = welch(signal, fs=fs, nperseg=512)
        return freq, psd
    
    def extract_features(self, signals, signal_labels):
        features = []
        for i, signal in enumerate(signals):
            if len(signal) > 0:
                ar_order = self.estimate_ar_order(signal)
                coefs, intercept = self.fit_ar_model(signal, ar_order)
                freq, ar_spec = self.compute_ar_spectrum(coefs)
                freq_welch, psd_welch = self.compute_psd(signal)
                
                theta_power = np.mean(psd_welch[(freq_welch >= 4) & (freq_welch <= 8)])
                alpha_power = np.mean(psd_welch[(freq_welch >= 8) & (freq_welch <= 13)])
                beta_power = np.mean(psd_welch[(freq_welch >= 13) & (freq_welch <= 30)])
                
                features.append({
                    'channel': signal_labels[i],
                    'ar_order': ar_order,
                    'theta_power': theta_power,
                    'alpha_power': alpha_power,
                    'beta_power': beta_power,
                    'ar_spectrum': ar_spec,
                    'psd': psd_welch,
                    'freq': freq_welch
                })
        return features

def main():
    data_dir = r"D:\2026BSMA\eeg-during-mental-arithmetic-tasks-1.0.0"
    analyzer = EEGARAnalyzer(data_dir)
    
    group_g, group_b = analyzer.get_subjects_by_group()
    print(f"Group G (good): {len(group_g)} subjects")
    print(f"Group B (bad): {len(group_b)} subjects")
    
    np.random.seed(42)
    selected_g = np.random.choice(group_g, 2, replace=False)
    selected_b = np.random.choice(group_b, 2, replace=False)
    
    print(f"\nSelected Group G subjects: {selected_g}")
    print(f"Selected Group B subjects: {selected_b}")
    
    results = {'G': {'pre': [], 'post': []}, 'B': {'pre': [], 'post': []}}
    
    for subject in selected_g:
        signals_pre, labels_pre = analyzer.load_eeg_data(subject, 'pre')
        signals_post, labels_post = analyzer.load_eeg_data(subject, 'post')
        
        if signals_pre:
            features_pre = analyzer.extract_features(signals_pre, labels_pre)
            results['G']['pre'].append({'subject': subject, 'features': features_pre})
        
        if signals_post:
            features_post = analyzer.extract_features(signals_post, labels_post)
            results['G']['post'].append({'subject': subject, 'features': features_post})
    
    for subject in selected_b:
        signals_pre, labels_pre = analyzer.load_eeg_data(subject, 'pre')
        signals_post, labels_post = analyzer.load_eeg_data(subject, 'post')
        
        if signals_pre:
            features_pre = analyzer.extract_features(signals_pre, labels_pre)
            results['B']['pre'].append({'subject': subject, 'features': features_pre})
        
        if signals_post:
            features_post = analyzer.extract_features(signals_post, labels_post)
            results['B']['post'].append({'subject': subject, 'features': features_post})
    
    fig_dir = r"d:\2026BSMA\2026SEUBSMA_ML-BME\Tex环境\figure\HW3figure"
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    orders_g_pre = []
    orders_g_post = []
    orders_b_pre = []
    orders_b_post = []
    
    for sub in results['G']['pre']:
        for feat in sub['features']:
            orders_g_pre.append(feat['ar_order'])
    
    for sub in results['G']['post']:
        for feat in sub['features']:
            orders_g_post.append(feat['ar_order'])
    
    for sub in results['B']['pre']:
        for feat in sub['features']:
            orders_b_pre.append(feat['ar_order'])
    
    for sub in results['B']['post']:
        for feat in sub['features']:
            orders_b_post.append(feat['ar_order'])
    
    plt.boxplot([orders_g_pre, orders_g_post, orders_b_pre, orders_b_post],
                labels=['G Pre', 'G Post', 'B Pre', 'B Post'])
    plt.title('AR Model Order Distribution by Group and Task')
    plt.ylabel('AR Order')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'ar_order_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_theta_g_pre = np.mean([np.mean([f['theta_power'] for f in sub['features']]) for sub in results['G']['pre']])
    avg_theta_g_post = np.mean([np.mean([f['theta_power'] for f in sub['features']]) for sub in results['G']['post']])
    avg_theta_b_pre = np.mean([np.mean([f['theta_power'] for f in sub['features']]) for sub in results['B']['pre']])
    avg_theta_b_post = np.mean([np.mean([f['theta_power'] for f in sub['features']]) for sub in results['B']['post']])
    
    avg_alpha_g_pre = np.mean([np.mean([f['alpha_power'] for f in sub['features']]) for sub in results['G']['pre']])
    avg_alpha_g_post = np.mean([np.mean([f['alpha_power'] for f in sub['features']]) for sub in results['G']['post']])
    avg_alpha_b_pre = np.mean([np.mean([f['alpha_power'] for f in sub['features']]) for sub in results['B']['pre']])
    avg_alpha_b_post = np.mean([np.mean([f['alpha_power'] for f in sub['features']]) for sub in results['B']['post']])
    
    avg_beta_g_pre = np.mean([np.mean([f['beta_power'] for f in sub['features']]) for sub in results['G']['pre']])
    avg_beta_g_post = np.mean([np.mean([f['beta_power'] for f in sub['features']]) for sub in results['G']['post']])
    avg_beta_b_pre = np.mean([np.mean([f['beta_power'] for f in sub['features']]) for sub in results['B']['pre']])
    avg_beta_b_post = np.mean([np.mean([f['beta_power'] for f in sub['features']]) for sub in results['B']['post']])
    
    freq = results['G']['pre'][0]['features'][0]['freq']
    avg_psd_g_pre = np.mean([np.mean([f['psd'] for f in sub['features']], axis=0) for sub in results['G']['pre']], axis=0)
    avg_psd_g_post = np.mean([np.mean([f['psd'] for f in sub['features']], axis=0) for sub in results['G']['post']], axis=0)
    avg_psd_b_pre = np.mean([np.mean([f['psd'] for f in sub['features']], axis=0) for sub in results['B']['pre']], axis=0)
    avg_psd_b_post = np.mean([np.mean([f['psd'] for f in sub['features']], axis=0) for sub in results['B']['post']], axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(freq, avg_psd_g_pre, label='G Pre', alpha=0.7)
    plt.semilogy(freq, avg_psd_g_post, label='G Post', alpha=0.7)
    plt.semilogy(freq, avg_psd_b_pre, label='B Pre', alpha=0.7)
    plt.semilogy(freq, avg_psd_b_post, label='B Post', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Average PSD Comparison')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 30)
    plt.savefig(os.path.join(fig_dir, 'psd_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_ar_spec_g_pre = np.mean([np.mean([f['ar_spectrum'] for f in sub['features']], axis=0) for sub in results['G']['pre']], axis=0)
    avg_ar_spec_g_post = np.mean([np.mean([f['ar_spectrum'] for f in sub['features']], axis=0) for sub in results['G']['post']], axis=0)
    avg_ar_spec_b_pre = np.mean([np.mean([f['ar_spectrum'] for f in sub['features']], axis=0) for sub in results['B']['pre']], axis=0)
    avg_ar_spec_b_post = np.mean([np.mean([f['ar_spectrum'] for f in sub['features']], axis=0) for sub in results['B']['post']], axis=0)
    
    ar_freq = np.linspace(0, 125, len(avg_ar_spec_g_pre))
    
    plt.figure(figsize=(12, 6))
    plt.plot(ar_freq, avg_ar_spec_g_pre, label='G Pre', alpha=0.7)
    plt.plot(ar_freq, avg_ar_spec_g_post, label='G Post', alpha=0.7)
    plt.plot(ar_freq, avg_ar_spec_b_pre, label='B Pre', alpha=0.7)
    plt.plot(ar_freq, avg_ar_spec_b_post, label='B Post', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('AR Spectrum')
    plt.title('Average AR Spectrum Comparison')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 30)
    plt.savefig(os.path.join(fig_dir, 'ar_spectrum_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    bar_width = 0.15
    indices = np.arange(4)
    
    plt.figure(figsize=(10, 6))
    plt.bar(indices - bar_width, [avg_theta_g_pre, avg_theta_g_post, avg_theta_b_pre, avg_theta_b_post], 
            width=bar_width, label='Theta (4-8 Hz)')
    plt.bar(indices, [avg_alpha_g_pre, avg_alpha_g_post, avg_alpha_b_pre, avg_alpha_b_post], 
            width=bar_width, label='Alpha (8-13 Hz)')
    plt.bar(indices + bar_width, [avg_beta_g_pre, avg_beta_g_post, avg_beta_b_pre, avg_beta_b_post], 
            width=bar_width, label='Beta (13-30 Hz)')
    
    plt.xticks(indices, ['G Pre', 'G Post', 'B Pre', 'B Post'])
    plt.title('EEG Band Power Comparison')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(fig_dir, 'band_power_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    stats_df = pd.DataFrame({
        'Group': ['G', 'G', 'B', 'B'],
        'Task': ['Pre', 'Post', 'Pre', 'Post'],
        'AR_Order_Mean': [np.mean(orders_g_pre), np.mean(orders_g_post), 
                         np.mean(orders_b_pre), np.mean(orders_b_post)],
        'AR_Order_Std': [np.std(orders_g_pre), np.std(orders_g_post), 
                        np.std(orders_b_pre), np.std(orders_b_post)],
        'Theta_Power': [avg_theta_g_pre, avg_theta_g_post, avg_theta_b_pre, avg_theta_b_post],
        'Alpha_Power': [avg_alpha_g_pre, avg_alpha_g_post, avg_alpha_b_pre, avg_alpha_b_post],
        'Beta_Power': [avg_beta_g_pre, avg_beta_g_post, avg_beta_b_pre, avg_beta_b_post]
    })
    
    stats_path = os.path.join(fig_dir, 'statistics_results.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\nAnalysis completed. Results saved to {fig_dir}")
    print("\nSummary Statistics:")
    print(stats_df)

if __name__ == '__main__':
    main()