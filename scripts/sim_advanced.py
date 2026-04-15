import numpy as np

def generate_sensor_data():
    # Simulated Pressure (Mu=3.2 bar), Flow (13 L/s), and Vibration
    sensors = {
        'pres1': np.random.normal(3.2, 0.1),
        'pres2': np.random.normal(3.2, 0.1),
        'flow': np.random.normal(13, 0.5),
        'var_z': np.random.uniform(2.5, 3.5), # Vibration intensity
        'sensor_id': 'node_01'
    }
    return sensors

# Test Gating Score
s = generate_sensor_data()
z_score = abs(s['flow'] - 13) / 0.5
print(f"Sensor Data: {s}")
print(f"Gating Score: {z_score:.2f}")
