# PGL — Persistence-Gated Piezo Leak Localizer
### Busan Eco-Delta City | Academic Prototype + Patent Filing

> Phase 1 prototype. All metrics produced by physics-informed synthetic data.
> No proprietary or real-world sensor recordings are used.

## Quick Start
```bash
pip install -r requirements.txt
python run_pipeline.py        # runs simulation, prints F1 / MAE / FAR
streamlit run dashboard/app.py  # opens GIS map
```

## Tinkercad Demo
[Public link — paste yours here]
Twist RPOT1: P4 Tidal → P3 Pump → P2 Micro → P1 Burst (LED ON)

## Repository Layout
| Folder | Purpose |
|--------|---------|
| `src/` | Core library: velocity model, signal generator, classifier, brain |
| `hardware/` | Arduino firmware for ESP32 / Tinkercad |
| `dashboard/` | Streamlit + Folium GIS map |
| `data/` | `synthetic/` (generated) · `real/` (Phase 2 — BattLeDIM) |
| `docs/` | Patent claims, architecture, validation notes |

## Patent
See `docs/patent_claims.md`. KIPRIS search April 2026: 0 water-leak hits
for "persistence gating + piezo + salinity".

## Phase Roadmap
| Phase | Status | Goal |
|-------|--------|------|
| 1 — Prototype | ✅ Done | Tinkercad + Python simulation |
| 2 — Validation | 🔜 Next | BattLeDIM + lab bench |
| 3 — Patent | 🔜 Next | KIPO provisional filing |
| 4 — Hardware | 🔜 Future | ESP32 pilot, LoRaWAN mesh |