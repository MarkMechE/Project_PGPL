# Process Real/Fake Data → Alerts CSV
import pandas as pd
import numpy as np
import librosa  # pip install librosa

# Load
logs = pd.read_csv('data/sample_logs.csv')  # Or real
salinity = pd.read_csv('data/real/busan_salinity.csv')

# Process WAV (Real Leaks)
for file in os.listdir('data/real/leak_sounds/')[:5]:  # First 5
    y, sr = librosa.load(f'data/real/leak_sounds/{file}', sr=8000)
    sigma = salinity['salinity'].mean() if 'salinity' in salinity else 28.5
    c = 1449 + 4.6*20 + 1.34*(sigma-35)
    env = np.abs(librosa.hilbert(y))
    thresh = 0.5 / c * 1000
    metric = np.sum(env > thresh) / len(env)
    logs = pd.concat([logs, pd.DataFrame({
        'node_id': [file.split('.')[0]], 'sigma': [sigma], 'metric': [metric*100],
        'p_level': ['P2' if metric>0.3 else 'Noise'], 'lat': [35.1796], 'lon': [129.0756]
    })], ignore_index=True)

logs.to_csv('data/processed_real.csv', index=False)
print(f"Processed {len(logs)} samples. P1/P2: {len(logs[logs['metric']>30])}")