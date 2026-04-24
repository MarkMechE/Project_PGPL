"""
Biot-Gassmann acoustic velocity for saturated sandy reclaimed soil.
Valid range: 3-10 psu (Nakdong Estuary, EDC reclaimed aquifer).
"""
import numpy as np


def get_biot_velocity(
    salinity_psu: float = 7.0,
    porosity: float = 0.40,
    temperature_c: float = 20.0,
) -> float:
    """
    Returns P-wave velocity in m/s, clipped to [1450, 1520].

    Parameters
    ----------
    salinity_psu  : groundwater salinity (3-10 psu)
    porosity      : soil porosity (0.30-0.50)
    temperature_c : porewater temperature °C
    """
    s = float(np.clip(salinity_psu, 0.0, 35.0))
    n = float(np.clip(porosity, 0.30, 0.50))

    K_frame = 8e8 * (1.0 - n / 0.40)
    K_fluid = 2.25e9 + s * 2.4e6          # +2.4 MPa/psu (Biot-Stoll)
    K_grain = 3.6e10                        # quartz

    rho_fluid = 1000.0 + 0.8 * s
    rho_bulk  = (1.0 - n) * 2650.0 + n * rho_fluid

    A     = (1.0 - K_frame / K_grain) ** 2
    B     = n / K_fluid + (1.0 - n) / K_grain - K_frame / K_grain ** 2
    K_sat = K_frame + A / B

    G = 4e8 * (1.0 - n / 0.40)
    M = K_sat + 4.0 / 3.0 * G

    v = np.sqrt(M / rho_bulk) + (temperature_c - 20.0) * 0.4
    return float(np.clip(v, 1450.0, 1520.0))


if __name__ == "__main__":
    for s in [3, 5, 7, 9, 10]:
        print(f"  {s:2d} psu → {get_biot_velocity(s):.1f} m/s")