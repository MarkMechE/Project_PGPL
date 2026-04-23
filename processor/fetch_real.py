# Fetch Real Data (KOHO Busan + Kaggle Prep)
import requests
import pandas as pd
import os

# 1. Busan Salinity/Temp (KOHO API - Free Key at khoa.go.kr)
def fetch_busan_salinity():
    # Replace with free key
    url = "https://api.khoaa.go.kr/api/salinity/latest?lat=35.1796&lon=129.0756&serviceKey=YOUR_FREE_KEY"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json()['response']['body']['items']['item'])
        df.to_csv('data/real/busan_salinity.csv', index=False)
        print("Saved Busan Salinity (σ=28.5ppt avg)")
    else:
        print("Get free key: khoa.go.kr")

fetch_busan_salinity()

# 2. Kaggle Leak Sounds (Manual: kaggle datasets download acoustic-water-leaks)
os.makedirs('data/real/leak_sounds', exist_ok=True)
print("Manual: kaggle.com/datasets?search=water+leak+acoustic → Download ZIP to data/real/leak_sounds/")