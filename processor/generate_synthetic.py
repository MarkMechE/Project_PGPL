# Fixed Synthetic Generator (No np.where Bug)
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'sigma': np.random.uniform(0, 35, n_samples),
    'noise': np.random.uniform(0, 1, n_samples),
    'rms_raw': np.random.uniform(100, 900, n_samples),
    'pers_raw': np.random.uniform(0.1, 1.0, n_samples)
}

data['rmsn'] = data['rms_raw'] / 1023.0
data['persm'] = data['pers_raw']
data['thresh'] = 150 + data['sigma'] * 2 + data['noise'] * 80

# FIXED: np.select for ternary
data['ev'] = np.select(
    [data['rmsn'] > 0.6, data['persm'] > 0.3],
    ['Burst', 'Micro'],
    default='Tidal'
)
data['typeW'] = np.select(
    [data['ev'] == 'Burst', data['ev'] == 'Micro'],
    [1.0, 0.6],
    default=0.0
)
data['sev'] = 0.4 * data['rmsn'] + 0.3 * data['typeW'] + 0.3 * 0.5
data['tier'] = np.select(
    [data['sev'] >= 0.8, data['sev'] >= 0.6],
    ['P1', 'P2'],
    default='P4'
)
data['lat'] = 35.1796 + np.random.normal(0, 0.001, n_samples)
data['lon'] = 129.0756 + np.random.normal(0, 0.001, n_samples)

df = pd.DataFrame(data)
df.to_csv('../../data/synthetic/tinkercad_logs.csv', index=False)
print("Synthetic Generated: 1000 samples")
print(df.head())
print(f"Synthetic Acc: {(df['tier'] != 'P4').mean()*100:.1f}%")