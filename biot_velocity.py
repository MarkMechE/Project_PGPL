"""
biot_velocity.py  —  Biot-Gassmann acoustic velocity for saline reclaimed soil
Nakdong Estuary, EDC | valid range: 3-10 psu
Returns c in m/s, clipped to [1450, 1520].
"""

import numpy as np


def get_biot_velocity(salinity_psu: float = 7.0,
                      porosity: float = 0.40,
                      temperature_c: float = 20.0) -> float:
    """
    Gassmann substitution for saturated sandy reclaimed soil.

    Parameters
    ----------
    salinity_psu  : groundwater salinity (3-10 psu for Nakdong Estuary)
    porosity      : soil porosity (0.30-0.50 typical for reclaimed land)
    temperature_c : porewater temperature (°C)

    Returns
    -------
    float : acoustic P-wave velocity in m/s
    """
    s  = float(np.clip(salinity_psu, 0.0, 35.0))
    n  = float(np.clip(porosity, 0.30, 0.50))

    # Dry frame bulk modulus (Pa) — quartz sand, porosity-scaled
    K_frame = 8e8 * (1.0 - n / 0.40)

    # Fluid bulk modulus: freshwater base + salinity correction
    # +2.4 MPa per psu (empirical Biot-Stoll for brackish water)
    K_fluid = 2.25e9 + s * 2.4e6

    # Solid grain modulus (quartz)
    K_grain = 3.6e10

    # Fluid + bulk density
    rho_fluid = 1000.0 + 0.8 * s
    rho_bulk  = (1.0 - n) * 2650.0 + n * rho_fluid

    # Gassmann undrained bulk modulus
    A     = (1.0 - K_frame / K_grain) ** 2
    B     = n / K_fluid + (1.0 - n) / K_grain - K_frame / K_grain ** 2
    K_sat = K_frame + A / B

    # Shear modulus unchanged by fluid
    G = 4e8 * (1.0 - n / 0.40)
    M = K_sat + 4.0 / 3.0 * G

    # Temperature correction (~0.4 m/s per °C above 20)
    v = np.sqrt(M / rho_bulk) + (temperature_c - 20.0) * 0.4

    return float(np.clip(v, 1450.0, 1520.0))


if __name__ == "__main__":
    print("Biot velocity over 3-10 psu range:")
    for s in [3, 5, 7, 9, 10]:
        print(f"  {s:2d} psu → {get_biot_velocity(s):.1f} m/s")