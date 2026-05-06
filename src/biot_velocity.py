"""
biot_velocity.py — Biot acoustic velocity for EDC saline pipes
PGPL v2.0 | Reused + cleaned from PROJECT_PGPL
"""
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
RHO_FRESH   = 998.0    # kg/m³ freshwater
RHO_SALINE  = 1025.0   # kg/m³ seawater (EDC Busan)
K_WATER     = 2.15e9   # Bulk modulus water (Pa)
K_SALINE    = 2.34e9   # Bulk modulus saline (Pa)  ← EDC-specific
E_STEEL     = 200e9    # Young's modulus steel (Pa)
E_HDPE      = 0.8e9    # Young's modulus HDPE (Pa)
NU_STEEL    = 0.30
NU_HDPE     = 0.46

PIPE_MATERIALS = {
    "steel": (E_STEEL, NU_STEEL),
    "hdpe":  (E_HDPE,  NU_HDPE),
    "pvc":   (3.0e9,   0.38),
    "ci":    (170e9,   0.26),   # cast iron
}


def biot_wave_speed(
    diameter_m: float,
    thickness_m: float,
    material: str = "hdpe",
    saline: bool = True,
    tidal_psi: float = 0.0,
) -> float:
    """
    Compute acoustic wave speed via Biot thin-shell model.

    Parameters
    ----------
    diameter_m   : Inner pipe diameter (m)
    thickness_m  : Pipe wall thickness (m)
    material     : One of {steel, hdpe, pvc, ci}
    saline       : True → EDC saline props; False → freshwater
    tidal_psi    : Tidal back-pressure (PSI) — adjusts bulk modulus

    Returns
    -------
    c : float  wave speed (m/s)
    """
    if material not in PIPE_MATERIALS:
        raise ValueError(f"Unknown material '{material}'. Choose: {list(PIPE_MATERIALS)}")

    E, nu   = PIPE_MATERIALS[material]
    rho     = RHO_SALINE if saline else RHO_FRESH
    K_fluid = K_SALINE   if saline else K_WATER

    # Tidal pressure correction (EDC novelty)
    tidal_pa = tidal_psi * 6894.76
    K_eff    = K_fluid * (1.0 + tidal_pa / K_fluid)

    # Biot compliance factor
    D     = diameter_m
    e     = thickness_m
    psi_b = (K_eff * D) / (E * e) * (1.0 - nu ** 2)

    c = np.sqrt(K_eff / (rho * (1.0 + psi_b)))
    return float(c)


def tdoa_distance(
    c: float,
    dt_sec: float,
    sensor_spacing_m: float,
) -> float:
    """
    Compute leak location from Time Difference of Arrival.

    Returns distance from sensor A (m).
    x = (L - c * Δt) / 2
    """
    x = (sensor_spacing_m - c * dt_sec) / 2.0
    return float(np.clip(x, 0.0, sensor_spacing_m))