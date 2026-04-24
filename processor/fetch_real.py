# Fetch Real Data
import requests
import pandas as pd
import os

# Busan Salinity (KOHO – Free API Key at khoa.go.kr)
os.makedirs('data/real', exist_ok=True)
url = "https://api.khoaa.go.kr/api/salinity/latest?lat=35.1796&lon=129.0756&serviceKey=YOUR_FREE_KEY"
r = requests.get(url)
if r.status_code == 200:
    df = pd.DataFrame(r.json()['response']['body']['items'])
    df.to_csv('data/real/busan_salinity.csv', index=False)
    print("Real Busan Salinity Saved")
else:
    print("Get free key at khoa.go.kr")

print("Kaggle Leaks: Manual download 'acoustic-water-leaks' → data/real/leak_sounds/")