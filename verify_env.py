"""
verify_env.py — environment check
python verify_env.py
"""
import sys

required = [
    ("numpy",    "import numpy as np;     print(f'  numpy        {np.__version__}')"),
    ("scipy",    "import scipy;            print(f'  scipy        {scipy.__version__}')"),
    ("pandas",   "import pandas as pd;    print(f'  pandas       {pd.__version__}')"),
    ("networkx", "import networkx as nx;  print(f'  networkx     {nx.__version__}')"),
]
optional = [
    ("streamlit",        "import streamlit as st;  print(f'  streamlit    {st.__version__}')"),
    ("folium",           "import folium;           print(f'  folium       {folium.__version__}')"),
    ("streamlit_folium", "from streamlit_folium import st_folium; print('  streamlit-folium  ok')"),
]

print("\nPGL Environment Check")
print("─" * 36)
all_ok = True

print("Required:")
for name, stmt in required:
    try:
        exec(stmt)
    except ImportError:
        print(f"  {name:<20} MISSING  →  pip install {name}")
        all_ok = False

print("\nOptional (dashboard):")
for name, stmt in optional:
    try:
        exec(stmt)
    except ImportError:
        print(f"  {name:<20} not installed")

print("\nSrc modules:")
for mod in ["src.biot_velocity", "src.data_generator",
            "src.anomaly_classifier", "src.pgpl_brain"]:
    try:
        __import__(mod)
        print(f"  {mod:<35} ok")
    except ImportError as e:
        print(f"  {mod:<35} MISSING: {e}")
        all_ok = False

print()
if all_ok:
    print("✅  Environment ready — run: python run_pipeline.py")
else:
    print("⚠️   Fix the items above, then re-run.")
    sys.exit(1)