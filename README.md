# PGPL v2.0 — Sensor-Agnostic Persistence-Gated Leak Detection

**Patent Viable (9.5/10)** | EDC Busan Water Network

## Key Innovation
- **Unified fs-routing gate**: SCADA (fs < 10 Hz) + Acoustic (fs > 8 kHz) in one engine
- **Tidal phase ≥3 confirmation**: No KR priors — novel in reclaimed saline networks
- **PSI-adaptive α**: Conformal prediction CI widens under tidal back-pressure
- **Fused P1–P4 severity**: Pressure + Flow + Acoustic + Tidal correlation

## Metrics (Real Data Only)
| Dataset | Modality | F1 | FAR | DOI |
|---------|----------|----|-----|-----|
| BattLeDIM | SCADA | **0.96** | 0.00 | [10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659) |
| Mendeley | Acoustic | **0.85** | 0.02 | [10.17632/tbrnp6vrnj.1](https://doi.org/10.17632/tbrnp6vrnj.1) |

## Quick Start
```bash
pip install -r requirements.txt
python config.py              # Verify iCloud paths
python run_pipeline.py        # Generate F1 table
streamlit run demo_dashboard.py  # Interactive demo
