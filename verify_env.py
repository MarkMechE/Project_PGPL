"""
verify_env.py  —  Environment check
Run: python verify_env.py
All imports must succeed before running the pipeline or dashboard.
"""

import sys

checks = [
    ("numpy",        "import numpy as np; print(f'  numpy       {np.__version__}')"),
    ("scipy",        "import scipy; print(f'  scipy       {scipy.__version__}')"),
    ("pandas",       "import pandas as pd; print(f'  pandas      {pd.__version__}')"),
    ("streamlit",    "import streamlit as st; print(f'  streamlit   {st.__version__}')"),
    ("folium",       "import folium; print(f'  folium      {folium.__version__}')"),
    ("networkx",     "import networkx as nx; print(f'  networkx    {nx.__version__}')"),
]

optional = [
    ("osmnx",        "import osmnx as ox; print(f'  osmnx       {ox.__version__}')"),
    ("geopandas",    "import geopandas as gpd; print(f'  geopandas   {gpd.__version__}')"),
    ("streamlit_folium", "from streamlit_folium import st_folium; print('  streamlit_folium  ok')"),
]

print("\nPGL Environment Check")
print("─" * 35)

all_ok = True
print("Required:")
for name, stmt in checks:
    try:
        exec(stmt)
    except ImportError:
        print(f"  {name:<20} MISSING  →  pip install {name}")
        all_ok = False

print("\nOptional (Phase 2+):")
for name, stmt in optional:
    try:
        exec(stmt)
    except ImportError:
        print(f"  {name:<20} not installed (Phase 2)")

print("\nCore modules:")
for mod in ["biot_velocity", "data_generator",
            "anomaly_classifier", "pgpl_brain",
            "main_pipeline", "pulse_at_bridge"]:
    try:
        __import__(mod)
        print(f"  {mod:<30} ok")
    except ImportError as e:
        print(f"  {mod:<30} MISSING: {e}")
        all_ok = False

print()
if all_ok:
    print("✅  Environment ready — run: python main_pipeline.py")
else:
    print("⚠️  Fix missing packages above, then re-run this check.")
    sys.exit(1)