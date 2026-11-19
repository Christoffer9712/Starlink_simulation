"""WGS-84 coordinate helpers (all angles in degrees, distances in metres).

Every function is vectorised: inputs may be scalars or arrays of any shape,
with xyz on the last axis.
"""
import numpy as np

WGS84_A = 6_378_137.0                  # semi-major axis [m]
WGS84_F = 1 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)     # first eccentricity squared


def lla_to_ecef(lat, lon, alt=0.0):
    """Geodetic (lat, lon, alt) -> Earth-fixed xyz, shape (..., 3)."""
    lat, lon, alt = np.radians(lat), np.radians(lon), np.asarray(alt, float)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_lla(xyz):
    """Earth-fixed xyz (..., 3) -> (lat, lon, alt) arrays. Bowring's method."""
    x, y, z = np.moveaxis(np.asarray(xyz, float), -1, 0)
    b = WGS84_A * (1 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / b**2
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3,
                     p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), alt


def up_vector(lat, lon):
    """Local 'up' unit vector (geodetic normal) in Earth-fixed axes, (..., 3)."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def elevation_deg(observer_xyz, observer_up, target_xyz):
    """Elevation angle of target(s) seen from observer(s), degrees.

    Shapes broadcast, e.g. observer (T, 1, 3) against targets (T, N, 3)
    gives a (T, N) result.
    """
    rel = target_xyz - observer_xyz
    rng = np.linalg.norm(rel, axis=-1)
    sin_el = np.sum(rel * observer_up, axis=-1) / rng
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
