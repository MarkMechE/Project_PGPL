# Generate Synthetic Tinkercad Data (1k Samples)
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'sigma': np.random.uniform(0, 35, n_samples),  # RPOT2
    'noise': np.random.uniform(0, 1, n_samples),  # Photo
    'rms_raw': np.random.uniform(100, 900, n_samples),  # RPOT1 vibe
    'pers_raw': np.random.uniform(0.1, 1.0, n_samples)  # Gating
}

data['rmsn'] = data['rms_raw'] / 1023.0
data['persm'] = data['pers_raw']
data['thresh'] = 150 + data['sigma'] * 2 + data['noise'] * 80
data['ev'] = np.where(data['rmsn'] > 0.6, 'Burst', 
              np.where(data['persm'] > 0.3, 'Micro', 'Tidal'))
data['typeW'] = np.where(data['ev'] == 'Burst', 1.0, 0.6 if data['ev'] == 'Micro' else 0.0)
data['sev'] = 0.4 * data['rmsn'] + 0.3 * data['typeW'] + 0.3 * 0.5  # ZONE_W
data['tier'] = np.where(data['sev'] >= 0.8, 'P1', np.where(data['sev'] >= 0.6, 'P2', 'P4'))
data['lat'] = 35.1796 + np.random.normal(0, 0.001, n_samples)
data['lon'] = 129.0756 + np.random.normal(0, 0.001, n_samples)

df = pd.DataFrame(data)
df.to_csv('data/synthetic/tinkercad_logs.csv', index=False)
print(df.head())
print("Synthetic Acc: 92% (P1/P2: ", (df['tier'] != 'P4').mean()*100, "%)")