"""
feeder_links.py

Gateway-satellite feeder links over time.

At every network update each gateway selects the closest satellites above
the minimum elevation (optionally at most max_per_gateway). Between updates
the links are kept; a link breaks when its satellite drops below the minimum
elevation, and is not replaced until the next update.

Network updates happen either
  * at given ground-truth times (update_times, e.g. derived from measured
    RTT steps), or
  * every update_interval_s seconds, counted from the first time step.

Results use the same pair format as find_isl_pairs, one list per time step:
    feeder_links[i] = [[gateway_name, satellite_name], ...]  at unix_s[i]

Usage:
    feeder_links = find_feeder_links(bb_operating_satrec_by_name, sc.gateways,
                                     sc.unix_s, max_per_gateway=8)
    feeder_links = find_feeder_links(bb_operating_satrec_by_name, sc.gateways,
                                     sc.unix_s, max_per_gateway=8,
                                     update_times=update_times)
"""
import math
from typing import Optional, Sequence

import numpy as np
from sgp4.api import SatrecArray
from sgp4.propagation import gstime

from simulation_param import *

WGS84_A_KM = 6378.137
WGS84_E2 = 6.69437999014e-3
UNIX_EPOCH_JD = 2440587.5


def find_feeder_links(satrec_by_name, gateways, unix_s,
                      min_elev_deg: float = MIN_GW_ELEV_DEG,
                      max_per_gateway: Optional[int] = None,
                      update_interval_s: float = NETWORK_UPDATE_INTERVAL,
                      update_times: Optional[Sequence[float]] = None):
    """Return a list with one entry per time step: [[gateway, satellite], ...].

    satrec_by_name    -- {name: Satrec} of the satellites to consider
    gateways          -- DataFrame with columns 'name', 'lat', 'lon' (degrees),
                         optionally 'alt_m'
    unix_s            -- times (seconds since 1970-01-01 UTC), e.g. sc.unix_s
    min_elev_deg      -- minimum elevation of the satellite seen from the gateway
    max_per_gateway   -- at each update, keep at most this many (closest)
                         satellites per gateway; None keeps all visible ones
    update_interval_s -- interval between updates when update_times is None
    update_times      -- ground-truth update times (Unix seconds). Gateways
                         reselect at the first time step at or after each of
                         them. The first time step always selects links.
    """
    unix_s = np.atleast_1d(np.asarray(unix_s, dtype=float))
    if not satrec_by_name or len(gateways) == 0:
        return [[] for _ in unix_s]

    update_index = make_update_index(unix_s, update_interval_s, update_times)

    sat_names = np.array(list(satrec_by_name))
    sat_array = SatrecArray([satrec_by_name[n] for n in sat_names])
    gw_names = np.asarray(gateways["name"])

    lat, lon = gateways["lat"].to_numpy(), gateways["lon"].to_numpy()
    alt_km = gateways["alt_m"].to_numpy() / 1000.0 if "alt_m" in gateways else 0.0
    gw_xyz = geodetic_to_ecef_km(lat, lon, alt_km)          # (M, 3)
    gw_up = local_up(lat, lon)                               # (M, 3)
    min_sin_elev = math.sin(math.radians(min_elev_deg))

    links_per_step = []
    sat_idx = gw_idx = np.array([], dtype=int)               # current links
    last_update = None
    for t in unix_s:
        sat_xyz, ok = satellite_positions_ecef(sat_array, t)            # (N, 3), (N,)
        update = update_index(t)

        if update != last_update:
            # Network update: every gateway picks its closest visible satellites
            sin_elev, dist2 = elevation_and_distance(sat_xyz, gw_xyz, gw_up)   # (N, M)
            visible = (sin_elev >= min_sin_elev) & ok[:, None]
            sat_idx, gw_idx = closest_links(dist2, visible, max_per_gateway)
            last_update = update
        else:
            # Between updates: keep the links, drop those whose satellite has set
            still_up = (link_sin_elevation(sat_xyz[sat_idx], gw_xyz[gw_idx], gw_up[gw_idx])
                        >= min_sin_elev) & ok[sat_idx]
            sat_idx, gw_idx = sat_idx[still_up], gw_idx[still_up]

        links_per_step.append(sorted([[gw_names[g], sat_names[s]]
                                      for s, g in zip(sat_idx, gw_idx)]))
    return links_per_step


def make_update_index(unix_s, update_interval_s, update_times):
    """Function t -> number of the network update period that t falls in.

    A change of this number between two time steps triggers a reselection.
    """
    if update_times is None:
        start = unix_s[0]
        return lambda t: math.floor((t - start) / update_interval_s)

    times = np.sort(np.asarray(update_times, dtype=float))
    return lambda t: int(np.searchsorted(times, t, side="right"))   # updates at or before t


def satellite_positions_ecef(sat_array, unix_time):
    """ECEF positions (km) of all satellites at one time, and a mask of valid ones."""
    jd = UNIX_EPOCH_JD + np.floor(unix_time / 86400.0)
    fr = (unix_time % 86400.0) / 86400.0
    error, r, _ = sat_array.sgp4(np.array([jd]), np.array([fr]))
    r_teme = r[:, 0, :]                                       # (N, 3)

    theta = gstime(jd + fr)                                   # Earth rotation angle
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = r_teme[:, 0], r_teme[:, 1], r_teme[:, 2]
    r_ecef = np.column_stack([c * x + s * y, -s * x + c * y, z])
    return r_ecef, error[:, 0] == 0


def elevation_and_distance(sat_xyz, gw_xyz, gw_up):
    """sin(elevation) and squared distance for every satellite (rows) and
    gateway (columns), without building an (N, M, 3) array."""
    height = sat_xyz @ gw_up.T - np.sum(gw_xyz * gw_up, axis=1)          # (N, M)
    dist2 = (np.sum(sat_xyz ** 2, axis=1)[:, None]
             + np.sum(gw_xyz ** 2, axis=1)[None, :]
             - 2.0 * sat_xyz @ gw_xyz.T)
    return height / np.sqrt(dist2), dist2


def link_sin_elevation(sat_xyz, gw_xyz, gw_up):
    """sin(elevation) for matched rows: satellite k seen from gateway k."""
    d = sat_xyz - gw_xyz
    return np.sum(d * gw_up, axis=1) / np.linalg.norm(d, axis=1)


def closest_links(dist2, visible, max_per_gateway):
    """(satellite indices, gateway indices) of the closest visible satellites."""
    if max_per_gateway is None:
        return np.nonzero(visible)
    score = np.where(visible, -dist2, -np.inf)                # closer = higher score
    k = min(max_per_gateway, score.shape[0])
    top = np.argpartition(-score, k - 1, axis=0)[:k]          # (k, M): k closest per gateway
    sat_idx = top.ravel()
    gw_idx = np.tile(np.arange(score.shape[1]), k)
    keep = np.isfinite(score[sat_idx, gw_idx])                # drop slots with no visible satellite
    return sat_idx[keep], gw_idx[keep]


def geodetic_to_ecef_km(lat_deg, lon_deg, alt_km=0.0):
    """WGS-84 geodetic coordinates to ECEF (km); arrays in, (n, 3) out."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A_KM / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt_km) * np.cos(lat) * np.cos(lon)
    y = (n + alt_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.column_stack([x, y, z])


def local_up(lat_deg, lon_deg):
    """Unit vector normal to the WGS-84 ellipsoid (local vertical); (n, 3)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])